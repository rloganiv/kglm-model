"""
Discriminative version of EntityNLM for importance sampling.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import masked_softmax
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.modules import DynamicEmbedding,  WeightDroppedLstm
from kglm.nn.util import sample_from_logp

logger = logging.getLogger(__name__)


StateDict = Dict[str, Union[torch.Tensor]]  # pylint: disable=invalid-name


@Model.register('entitydisc')
class EntityNLMDiscriminator(Model):
    """
    Implementation of the discriminative model from:
        https://arxiv.org/abs/1708.00781
    used to draw importance samples.

    Parameters
    ----------
    vocab : ``Vocabulary``
        The model vocabulary.
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed tokens.
    encoder : ``Seq2SeqEncoder``
        Used to encode the sequence of token embeddings.
    embedding_dim : ``int``
        The dimension of entity / length embeddings. Should match the encoder output size.
    max_mention_length : ``int``
        Maximum entity mention length.
    max_embeddings : ``int``
        Maximum number of embeddings.
    variational_dropout_rate : ``float``, optional
        Dropout rate of variational dropout applied to input embeddings. Default: 0.0
    dropout_rate : ``float``, optional
        Dropout rate applied to hidden states. Default: 0.0
    initializer : ``InitializerApplicator``, optional
        Used to initialize model parameters.
    """
    # pylint: disable=line-too-long
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 max_mention_length: int,
                 max_embeddings: int,
                 variational_dropout_rate: float = 0.0,
                 dropout_rate: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(EntityNLMDiscriminator, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._max_mention_length = max_mention_length
        self._max_embeddings = max_embeddings
        self._sos_token = self.vocab.get_token_index('@@START@@', 'tokens')
        self._eos_token = self.vocab.get_token_index('@@END@@', 'tokens')
        self._rnn = WeightDroppedLstm(num_layers, embedding_dim, hidden_size, variational_dropout_rate)
        self._state: Optional[StateDict] = None

        # Input variational dropout
        self._variational_dropout = InputVariationalDropout(variational_dropout_rate)
        self._dropout = torch.nn.Dropout(dropout_rate)

        # For entity type prediction
        self._entity_type_projection = torch.nn.Linear(in_features=embedding_dim,
                                                       out_features=2,
                                                       bias=False)
        self._dynamic_embeddings = DynamicEmbedding(embedding_dim=embedding_dim,
                                                    max_embeddings=max_embeddings)

        # For mention length prediction
        self._mention_length_projection = torch.nn.Linear(in_features=2*embedding_dim,
                                                          out_features=max_mention_length)

        # Metrics
        self._entity_type_accuracy = CategoricalAccuracy()
        self._entity_id_accuracy = CategoricalAccuracy()
        self._mention_length_accuracy = CategoricalAccuracy()

        initializer(self)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                source: Dict[str, torch.Tensor],
                entity_types: Optional[torch.Tensor] = None,
                entity_ids: Optional[torch.Tensor] = None,
                mention_lengths: Optional[torch.Tensor] = None,
                reset: torch.ByteTensor=None)-> Dict[str, torch.Tensor]:
        """
        Computes the loss during training / validation.

        Parameters
        ----------
        tokens : ``Dict[str, torch.Tensor]``
            A tensor of shape ``(batch_size, sequence_length)`` containing the sequence of
            tokens.
        entity_types : ``torch.Tensor``
            A tensor of shape ``(batch_size, sequence_length)`` indicating whether or not the
            corresponding token belongs to a mention.
        entity_ids : ``torch.Tensor``
            A tensor of shape ``(batch_size, sequence_length)`` containing the ids of the
            entities the corresponding token is mentioning.
        mention_lengths : ``torch.Tensor``
            A tensor of shape ``(batch_size, sequence_length)`` tracking how many remaining
            tokens (including the current one) there are in the mention.
        reset : ``torch.ByteTensor``
            Whether or not to reset the model's state. This should be done at the start of each
            new sequence.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.Tensor``
            The combined loss.
        """
        tokens = source  # MONKEY PATCH
        batch_size = tokens['tokens'].shape[0]

        if reset is not None:
            self.reset_states(reset)

        if entity_types is not None:
            output_dict = self._forward_loop(tokens=tokens,
                                             entity_types=entity_types,
                                             entity_ids=entity_ids,
                                             mention_lengths=mention_lengths)
        else:
            output_dict = {}

        self.detach_states()

        return output_dict

    def sample(self,  # pylint: disable=unused-argument
               source: Dict[str, torch.Tensor],
               reset: torch.ByteTensor = None,
               temperature: float = 1.0,
               offset: bool = False,
               **kwargs) -> Dict[str, torch.Tensor]:
        """
        Generates a sample from the discriminative model.

        WARNING: Unlike during training, this function expects the full (unsplit) sequence of
        tokens.

        Parameters
        ----------
        source : ``Dict[str, torch.Tensor]``
            A tensor of shape ``(batch_size, sequence_length)`` containing the sequence of
            tokens.
        reset : ``torch.ByteTensor``
            Whether or not to reset the model's state. This should be done at the start of each
            new sequence.

        Returns
        -------
        An output dictionary consisting of:
        logp : ``torch.Tensor``
            A tensor containing the log-probability of the sample (averaged over time)
        entity_types : ``torch.Tensor``
            A tensor of shape ``(batch_size, sequence_length)`` indicating whether or not the
            corresponding token belongs to a mention.
        entity_ids : ``torch.Tensor``
            A tensor of shape ``(batch_size, sequence_length)`` containing the ids of the
            entities the corresponding token is mentioning.
        mention_lengths : ``torch.Tensor``
            A tensor of shape ``(batch_size, sequence_length)`` tracking how many remaining
            tokens (including the current one) there are in the mention.
        """
        batch_size, sequence_length = source['tokens'].shape
        if reset is not None:
            self.reset_states(reset)

        if self._state is None:
            prev_mention_lengths = source['tokens'].new_ones(batch_size)
        else:
            prev_mention_lengths = self._state['prev_mention_lengths']

        # Embed tokens and get RNN hidden state.
        mask = get_text_field_mask(source)

        # If not offsetting, we need to ignore contribution of @@START@@ token annotation since it
        # is never used.

        if not offset:
            sos_mask = source['tokens'].ne(self._sos_token)
            mask = mask.byte() & sos_mask
        # If offsetting, we need to ignore contribution of @@END@@ token annotation since it is
        # never used.
        if offset:
            eos_mask = source['tokens'].ne(self._eos_token)
            mask = mask.byte() & eos_mask

        embeddings = self._text_field_embedder(source)
        hidden = self._rnn(embeddings)

        # Initialize outputs
        logp = hidden.new_zeros(batch_size) # Track total logp for **each** generated sample
        entity_types = torch.zeros_like(source['tokens'], dtype=torch.uint8)
        entity_ids = torch.zeros_like(source['tokens'])
        mention_lengths = torch.ones_like(source['tokens'])

        for timestep in range(sequence_length):

            current_hidden = hidden[:, timestep]

            # We only predict types / ids / lengths if the previous mention is terminated.
            predict_mask = prev_mention_lengths == 1
            predict_mask = predict_mask & mask[:, timestep].byte()

            if predict_mask.any():

                # Predict entity types
                entity_type_logits = self._entity_type_projection(current_hidden[predict_mask]) / temperature
                entity_type_logp = F.log_softmax(entity_type_logits, dim=-1)
                entity_type_prediction_logp, entity_type_predictions = sample_from_logp(entity_type_logp)
                entity_type_predictions = entity_type_predictions.byte()
                entity_types[predict_mask, timestep] = entity_type_predictions
                logp[predict_mask] += entity_type_prediction_logp

                # Only predict entity and mention lengths if we predicted that there was a mention
                predict_em = entity_types[:, timestep] & predict_mask
                if predict_em.any():
                    # Predict entity ids
                    entity_id_prediction_outputs = self._dynamic_embeddings(hidden=current_hidden,
                                                                            timestep=timestep,
                                                                            mask=predict_em)
                    entity_id_logits = entity_id_prediction_outputs['logits'] / temperature
                    entity_id_mask = entity_id_prediction_outputs['logit_mask']
                    entity_id_probs = masked_softmax(entity_id_logits,
                                                     entity_id_mask)
                    entity_id_predictions = torch.multinomial(entity_id_probs, 1)
                    entity_id_prediction_logp = entity_id_probs.gather(-1, entity_id_predictions).log()
                    entity_id_predictions = entity_id_predictions.squeeze(-1)
                    entity_id_prediction_logp = entity_id_prediction_logp.squeeze(-1)

                    # Predict mention lengths - we do this before writing the
                    # entity id predictions since we'll need to reindex the new
                    # entities, but need the null embeddings here.
                    predicted_entity_embeddings = self._dynamic_embeddings.embeddings[predict_em, entity_id_predictions]
                    concatenated = torch.cat((current_hidden[predict_em], predicted_entity_embeddings), dim=-1)
                    mention_length_logits = self._mention_length_projection(concatenated) / temperature
                    mention_length_logp = F.log_softmax(mention_length_logits, dim=-1)
                    mention_length_prediction_logp, mention_length_predictions = sample_from_logp(mention_length_logp)

                    # Write predictions
                    new_entity_mask = entity_id_predictions == 0
                    new_entity_labels = self._dynamic_embeddings.num_embeddings[predict_em]
                    entity_id_predictions[new_entity_mask] = new_entity_labels[new_entity_mask]
                    entity_ids[predict_em, timestep] = entity_id_predictions
                    logp[predict_em] += entity_id_prediction_logp


                    mention_lengths[predict_em, timestep] = mention_length_predictions
                    logp[predict_em] += mention_length_prediction_logp

                # Add / update entity embeddings
                new_entities = entity_ids[:, timestep] == self._dynamic_embeddings.num_embeddings
                self._dynamic_embeddings.add_embeddings(timestep, new_entities)

                self._dynamic_embeddings.update_embeddings(hidden=current_hidden,
                                                           update_indices=entity_ids[:, timestep],
                                                           timestep=timestep,
                                                           mask=predict_em)

            # If the previous mentions are ongoing, we assign the output deterministically. Mention
            # lengths decrease by 1, all other outputs are copied from the previous timestep. Do
            # not need to add anything to logp since these 'predictions' have probability 1 under
            # the model.
            deterministic_mask = prev_mention_lengths > 1
            deterministic_mask = deterministic_mask & mask[:, timestep].byte()
            if deterministic_mask.any():
                entity_types[deterministic_mask, timestep] = entity_types[deterministic_mask, timestep - 1]
                entity_ids[deterministic_mask, timestep] = entity_ids[deterministic_mask, timestep - 1]
                mention_lengths[deterministic_mask, timestep] = prev_mention_lengths[deterministic_mask] - 1

            # Update mention lengths for next timestep
            prev_mention_lengths = mention_lengths[:, timestep]

        # Update state
        self._state['prev_mention_lengths'] = prev_mention_lengths.detach()

        return {
                'logp': logp,
                'sample': {
                        'source': source,
                        'reset': reset,
                        'entity_types': entity_types,
                        'entity_ids': entity_ids,
                        'mention_lengths': mention_lengths
                }
        }

    @property
    def num_possible_annotations(self):
        # Number of ways to annotate an entity mention + 1 way to annotate a non-entity mention.
        return self._max_embeddings * self._max_mention_length + 1

    @property
    def entity_type_lookup(self):
        entity_type_lookup = [0] + [1] * self._max_embeddings * self._max_mention_length
        return torch.ByteTensor(entity_type_lookup)

    @property
    def entity_id_lookup(self):
        entity_id_lookup = [0] + list(range(self._max_embeddings) * self._max_mention_length)
        return torch.LongTensor(entity_id_lookup)

    @property
    def mention_length_lookup(self):
        mention_length_lookup = [1] + list([i] * self._max_embeddings for i in range(self._max_mention_length))
        return torch.LongTensor(mention_length_lookup)

    def _annotation_logp(self,
                         hidden: torch.FloatTensor,
                         timestep: int,
                         state_dict: Dict[str, Any] = None) -> torch.Tensor:
        """Computes the log-probability of all possible annotations for a single beam state.

        Parameters
        ==========
        TODO: Fill in

        Returns
        =======
        A tensor of log-probabilities for the possible annotations of shape
        (batch_size, num_annotations).
        """
        batch_size, hidden_dim =  hidden.shape
        if state_dict is not None:
            self._dynamic_embeddings.load_state_dict(state_dict['dynamic_embeddings_state_dict'])
        elif timestep > 0:
            raise RuntimeError('Dynamic embedding state_dict required.')

        # Entity type log probabilities: (batch_size, 2)
        entity_type_logits = self._entity_type_projection(hidden)
        entity_type_logp = F.log_softmax(entity_type_logits, -1)

        # Entity id log probabilities: (batch_size, max_embeddings)
        entity_id_logits = self._dynamic_embeddings(hidden, timestep)['logits']
        entity_id_logp = F.log_softmax(entity_id_logits, -1)

        # Mention length log probabilites: (batch_size, max_embeddings x max_mention_lengths)
        # NOTE: Entity id is guaranteed to be zero at initialization
        embeddings = self._dynamic_embeddings.embeddings
        concatenated = torch.cat((hidden.unsqueeze(1).expand_as(embeddings), embeddings), dim=-1)
        mention_length_logits = self._mention_length_projection(concatenated)
        mention_length_logp = F.log_softmax(mention_length_logits, -1).view(batch_size, -1)

        # Add together log probabilities
        logp = torch.zeros((batch_size, self.num_possible_annotations))
        logp[:, 0] += entity_type_logp[:, 0]
        logp[:, 1:] += entity_type_logp[:, 1:]
        logp[:, 1:] += entity_id_logp.repeat((1, self._max_mention_length))
        logp[:, 1:] += mention_length_logp

        return logp

    def _top_k_annotations(self, logp: torch.FloatTensor, k: int):
        """Extracts the top-k annotations.

        Parameters
        ==========
        logp : torch.Tensor
            A (batch_size, beam_width, num_annotations tensor)
        """
        batch_size = logp.shape[0]
        # Get the top canidates from each beam (makes math much eaiser)
        top_logp, top_indices = logp.topk(k, dim=-1)

        # Next flatten
        flat_logp = top_logp.view(batch_size, -1)
        flat_indices = top_indices.view(batch_size, -1)

        # Next get the top-k overall
        top_logp = flat_logp.topk(k, dim=-1)
        output_dict = {
            'entity_types': self.entity_type_lookup.take(top_indices),
            'entity_ids': self.entity_id_lookup.take(top_indices),
            'mention_lengths': self.mention_length_lookup.take(top_indices)
        }
        return top_logp, output_dict

    def _beam_step_fn(self,
                      hidden: torch.FloatTensor,
                      timestep: int,
                      k: int,
                      last_predictions: Optional[Dict[str, torch.Tensor]] = None,
                      state: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This function computes the top-k most likely next states for the beam.

        Parameters
        ==========

        Returns
        =======
        A tuple comprised of:
            next
        """
        batch_size, hidden_dim =  hidden.shape

        # Initial predictions are a special case since we only have to consider batch_size inputs
        # and don't need to deal with ongoing mentions.
        if last_predictions is None:
            # Step 1 - Compute type of next token
            entity_type_logits = self._entity_type_projection(hidden)

            # shape: (batch_size, 2)
            entity_type_logp = F.log_softmax(entity_type_logits, -1)

            # Step 2 - Compute probabilities for each entity
            # TODO: Take this shit out - its obvious...
            entity_id_prediction_outputs = self._dynamic_embeddings(hidden=hidden,
                                                                    timestep=timestep)
            entity_id_logits = entity_id_prediction_outputs['logits']

            # shape: (batch_size, num_entities)
            entity_id_logp = F.log_softmax(entity_id_logits, -1)  # Technically should mask

            # Step 3 - Compute mention length probabilities for each entity

            # shape: (batch_size, num_entities, embedding_dim)
            embeddings = self._dynamic_embeddings.embeddings[:, :1]  # Only one entity right now

            concatenated = torch.cat((hidden.unsqueeze(1), embeddings), dim=-1)
            mention_length_logits = self._mention_length_projection(concatenated)

            # shape: (batch_size, num_entities, max_mention_length)
            mention_length_logp = F.log_softmax(mention_length_logits, -1)

            # Step 4 - Combine log probabilities

            # Use meshgrid to get the Cartesian product of entity id and mention length options.
            # shape: (num_entities, max_mention_length)
            entity_ids, mention_lengths = torch.meshgrid(torch.arange(1, dtype=torch.int64),
                                                         torch.arange(self._max_mention_length, dtype=torch.int64))
            entity_ids = entity_ids.view(-1)
            mention_lengths = mention_lengths.view(-1)
            entity_types = torch.ones_like(entity_ids, dtype=torch.uint8)

            # When the type is not a mention there is only one possible annotation. Add it.
            entity_types = torch.cat((torch.ByteTensor([0]), entity_types))
            entity_ids = torch.cat((torch.LongTensor([0]), entity_ids))
            mention_lengths = torch.cat((torch.LongTensor([1]), mention_lengths))
            num_annotations = entity_types.shape[0]

            logp = torch.zeros(size=(batch_size, num_annotations),
                               dtype=torch.float32,
                               device=hidden.device)
            logp[:, 0] = entity_type_logp[:, 0]
            logp[:, 1:] = entity_type_logp[:, 1].unsqueeze(-1)
            logp[:, 1:] += entity_id_logp.view(batch_size, -1)
            logp[:, 1:] += mention_length_logp.view(batch_size, -1)

            top_logp, top_indices = logp.topk(k, dim=-1)
            output_dict = {
                'entity_types': torch.take(entity_types, top_indices),
                'entity_ids': torch.take(entity_ids, top_indices),
                'mention_lengths': torch.take(mention_lengths, top_indices)
            }
            return top_logp, output_dict

    def predict_top_k(self,
                      source: Dict[str, torch.Tensor],
                      reset: torch.ByteTensor,
                      k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtain the top-k (approximately) most likely predictions from the model using beam
        search. Unlike typical beam search all of the beam states are returned instead of just
        the most likely.

        The returned candidates are intended to be marginalized over to obtain an upper bound for
        the token-level perplexity of the EntityNLM.

        Parameters
        ==========
        source : ``Dict[str, torch.Tensor]``
            A tensor of shape ``(batch_size, sequence_length)`` containing the sequence of
            tokens.
        reset : ``torch.ByteTensor``
            Whether or not to reset the model's state. This should be done at the start of each
            new sequence.
        k : ``int``
            Number of predictions to return.

        Returns
        =======
        predictions : ``torch.Tensor``
            A tensor of shape ``(batch_size * k, sequence_length)`` containing the top-k
            predictions.
        logp : ``torch.Tensor``
            The log-probabilities of each prediction. WARNING: These are returned purely for
            diagnostic purposes and should not be factored in the the perplexity calculation.
        """
        batch_size, sequence_length = source['tokens'].shape

        # Reset the model's internal state.
        if not reset.all():
            raise RuntimeError('Detecting that not all states are being `reset` (e.g., that input '
                               'sequences have been split). Cannot predict top-K annotations in '
                               'this setting!')
        self.reset_states(reset)
        prev_mention_lengths = source['tokens'].new_ones(batch_size)

        # Embed and encode the tokens up front.
        embeddings = self._text_field_embedder(source['tokens'])
        hidden = self._rnn(embeddings)

        # The rest of the code will be a modified version of:
        #   https://github.com/allenai/allennlp/blob/master/allennlp/nn/beam_search.py

        # List of predictions. One for each time step. Unlike in allennlp's beam search, here each
        # element is a dictionary of (batch_size, beam_size) tensors whose keys are the annotation
        # variables.
        predictions: List[Dict[str, torch.Tensor]] = []

        # List of (batch_size, beam_size) tensors. One for each time step. Stores the index n of
        # the parent prediction it came from.
        backpointers: List[torch.LongTensor] = []

        # To keep things simple, we are going to store the beam states
        for timestep in range(sequence_length):
            pass

        # Trace backpointers to get output.

    def _forward_loop(self,
                      tokens: Dict[str, torch.Tensor],
                      entity_types: torch.Tensor,
                      entity_ids: torch.Tensor,
                      mention_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs the forward pass to calculate the loss on a chunk of training data.

        Parameters
        ----------
        tokens : ``Dict[str, torch.Tensor]``
            A tensor of shape ``(batch_size, sequence_length)`` containing the sequence of
            tokens.
        entity_types : ``torch.Tensor``
            A tensor of shape ``(batch_size, sequence_length)`` indicating whether or not the
            corresponding token belongs to a mention.
        entity_ids : ``torch.Tensor``
            A tensor of shape ``(batch_size, sequence_length)`` containing the ids of the
            entities the corresponding token is mentioning.
        mention_lengths : ``torch.Tensor``
            A tensor of shape ``(batch_size, sequence_length)`` tracking how many remaining
            tokens (including the current one) there are in the mention.

        Returns
        -------
        An output dictionary consisting of:
        entity_type_loss : ``torch.Tensor``
            The loss of entity type predictions.
        entity_id_loss : ``torch.Tensor``
            The loss of entity id predictions.
        mention_length_loss : ``torch.Tensor``
            The loss of mention length predictions.
        loss : ``torch.Tensor``
            The combined loss.
        """
        batch_size, sequence_length = tokens['tokens'].shape

        # Need to track previous mention lengths in order to know when to measure loss.
        if self._state is None:
            prev_mention_lengths = mention_lengths.new_ones(batch_size)
        else:
            prev_mention_lengths = self._state['prev_mention_lengths']

        # Embed tokens and get RNN hidden state.
        mask = get_text_field_mask(tokens)
        embeddings = self._text_field_embedder(tokens)
        embeddings = self._variational_dropout(embeddings)
        hidden = self._rnn(embeddings)

        # Initialize losses
        entity_type_loss = torch.tensor(0.0, requires_grad=True, device=hidden.device)
        entity_id_loss = torch.tensor(0.0, requires_grad=True, device=hidden.device)
        mention_length_loss = torch.tensor(0.0, requires_grad=True, device=hidden.device)

        for timestep in range(sequence_length):

            current_entity_types = entity_types[:, timestep]
            current_entity_ids = entity_ids[:, timestep]
            current_mention_lengths = mention_lengths[:, timestep]
            current_hidden = hidden[:, timestep]
            current_hidden = self._dropout(hidden[:, timestep])

            # We only predict types / ids / lengths if we are not currently in the process of
            # generating a mention (e.g. if the previous remaining mention length is 1). Indexing /
            # masking with ``predict_all`` makes it possible to do this in batch.
            predict_all = prev_mention_lengths == 1
            predict_all = predict_all & mask[:, timestep].byte()
            if predict_all.any():

                # Equation 3 in the paper.
                entity_type_logits = self._entity_type_projection(current_hidden[predict_all])
                entity_type_logp = F.log_softmax(entity_type_logits, -1)
                _entity_type_loss = -entity_type_logp.gather(-1, current_entity_types[predict_all].long().unsqueeze(-1))
                entity_type_loss = entity_type_loss + _entity_type_loss.sum()

                self._entity_type_accuracy(predictions=entity_type_logits,
                                           gold_labels=current_entity_types[predict_all].long())

                # Only proceed to predict entity and mention length if there is in fact an entity.
                predict_em = current_entity_types & predict_all

                if predict_em.any():
                    # Equation 4 in the paper. We want new entities to correspond to a prediction of
                    # zero, their embedding should be added after they've been predicted for the first
                    # time.
                    modified_entity_ids = current_entity_ids.clone()
                    modified_entity_ids[modified_entity_ids == self._dynamic_embeddings.num_embeddings] = 0
                    entity_id_prediction_outputs = self._dynamic_embeddings(hidden=current_hidden,
                                                                            timestep=timestep,
                                                                            target=modified_entity_ids,
                                                                            mask=predict_em)
                    _entity_id_loss = -entity_id_prediction_outputs['loss']
                    entity_id_loss = entity_id_loss + _entity_id_loss.sum()
                    self._entity_id_accuracy(predictions=entity_id_prediction_outputs['logits'],
                                             gold_labels=modified_entity_ids[predict_em])

                    # Equation 5 in the paper.
                    predicted_entity_embeddings = self._dynamic_embeddings.embeddings[predict_em, modified_entity_ids[predict_em]]
                    predicted_entity_embeddings = self._dropout(predicted_entity_embeddings)
                    concatenated = torch.cat((current_hidden[predict_em], predicted_entity_embeddings), dim=-1)

                    mention_length_logits = self._mention_length_projection(concatenated)
                    mention_length_logp = F.log_softmax(mention_length_logits, -1)
                    _mention_length_loss = -mention_length_logp.gather(-1, current_mention_lengths[predict_em].unsqueeze(-1))
                    mention_length_loss = mention_length_loss + _mention_length_loss.sum()
                    self._mention_length_accuracy(predictions=mention_length_logits,
                                                  gold_labels=current_mention_lengths[predict_em])

            # We add new entities to any sequence where the current entity id matches the number of
            # embeddings that currently exist for that sequence (this means we need a new one since
            # there is an additional dummy embedding).
            new_entities = current_entity_ids == self._dynamic_embeddings.num_embeddings
            self._dynamic_embeddings.add_embeddings(timestep, new_entities)

            # We also perform updates of the currently observed entities.
            self._dynamic_embeddings.update_embeddings(hidden=current_hidden,
                                                       update_indices=current_entity_ids,
                                                       timestep=timestep,
                                                       mask=current_entity_types)

            prev_mention_lengths = current_mention_lengths

        # Normalize the losses
        entity_type_loss = entity_type_loss / mask.sum()
        entity_id_loss = entity_id_loss / mask.sum()
        mention_length_loss = mention_length_loss / mask.sum()
        logger.debug('et: %f, eid: %f, ml: %f', entity_type_loss.item(),
                     entity_id_loss.item(), mention_length_loss.item())
        total_loss = entity_type_loss + entity_id_loss + mention_length_loss

        output_dict = {
                'entity_type_loss': entity_type_loss,
                'entity_id_loss': entity_id_loss,
                'mention_length_loss': mention_length_loss,
                'loss': total_loss
        }

        # Update the model state
        self._state = {
            'prev_mention_lengths': mention_lengths[:, -1].detach()
        }

        return output_dict


    def reset_states(self, reset: torch.ByteTensor) -> None:
        """Resets the model's internals. Should be called at the start of a new batch."""
        if reset.any() and (self._state is not None):
            # Zero out any previous elements
            self._state['prev_mention_lengths'][reset] = 1

        # Reset the dynamic embeddings and lstm
        self._dynamic_embeddings.reset_states(reset)
        self._rnn.reset(reset)

    def detach_states(self):
        """Detaches the model's state to enforce truncated backpropagation."""
        self._dynamic_embeddings.detach_states()

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'et_acc': self._entity_type_accuracy.get_metric(reset),
            'eid_acc': self._entity_id_accuracy.get_metric(reset),
            'ml_acc': self._mention_length_accuracy.get_metric(reset)
        }
