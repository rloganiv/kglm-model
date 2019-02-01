"""
Discriminative version of EntityNLM for importance sampling.
"""
import logging
from typing import Dict, Optional, Union

from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.modules import DynamicEmbedding
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
                 encoder: Seq2SeqEncoder,
                 embedding_dim: int,
                 max_mention_length: int,
                 max_embeddings: int,
                 variational_dropout_rate: float = 0.0,
                 dropout_rate: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(EntityNLMDiscriminator, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._embedding_dim = embedding_dim
        self._max_mention_length = max_mention_length
        self._max_embeddings = max_embeddings

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

        self._entity_type_accuracy = CategoricalAccuracy()
        self._entity_id_accuracy = CategoricalAccuracy()
        self._mention_length_accuracy = CategoricalAccuracy()

        initializer(self)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                tokens: Dict[str, torch.Tensor],
                entity_types: Optional[torch.Tensor] = None,
                entity_ids: Optional[torch.Tensor] = None,
                mention_lengths: Optional[torch.Tensor] = None,
                reset: bool = False)-> Dict[str, torch.Tensor]:
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
        reset : ``bool``
            Whether or not to reset the model's state. This should be done at the start of each
            new sequence.

        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.Tensor``
            The combined loss.
        """
        batch_size = tokens['tokens'].shape[0]

        if reset:
            self.reset_states(batch_size)
        else:
            self.detach_states()

        if entity_types is not None:
            output_dict = self._forward_loop(tokens=tokens,
                                             entity_types=entity_types,
                                             entity_ids=entity_ids,
                                             mention_lengths=mention_lengths)
        else:
            output_dict = {}

        return output_dict

    def sample(self,
               tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generates a sample from the discriminative model.

        WARNING: Unlike during training, this function expects the full (unsplit) sequence of
        tokens.

        Parameters
        ----------
        tokens : ``Dict[str, torch.Tensor]``
            A tensor of shape ``(batch_size, sequence_length)`` containing the sequence of
            tokens.

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
        batch_size, sequence_length = tokens['tokens'].shape

        # We will use a standard iterator during evaluation instead of a split iterator. Otherwise
        # it will be a pain to handle generating multiple samples for a sequence since there's no
        # way to get back to the first split.
        self.reset_states(batch_size)

        # Embed tokens and get RNN hidden state.
        mask = get_text_field_mask(tokens)
        embeddings = self._text_field_embedder(tokens)
        hidden = self._encoder(embeddings, mask)
        prev_mention_lengths = tokens['tokens'].new_ones(batch_size)

        # Initialize outputs
        logp = hidden.new_zeros(batch_size) # Track total logp for **each** generated sample
        entity_types = torch.zeros_like(tokens['tokens'], dtype=torch.uint8)
        entity_ids = torch.zeros_like(tokens['tokens'])
        mention_lengths = torch.ones_like(tokens['tokens'])

        # Generate outputs
        for timestep in range(sequence_length):

            current_hidden = hidden[:, timestep]

            # We only predict types / ids / lengths if the previous mention is terminated.
            predict_mask = prev_mention_lengths == 1
            predict_mask = predict_mask * mask[:, timestep].byte()
            if predict_mask.sum() > 0:

                # Predict entity types
                entity_type_logits = self._entity_type_projection(current_hidden[predict_mask])
                entity_type_logp = F.log_softmax(entity_type_logits, dim=-1)
                entity_type_prediction_logp, entity_type_predictions = sample_from_logp(entity_type_logp)
                entity_type_predictions = entity_type_predictions.byte()
                entity_types[predict_mask, timestep] = entity_type_predictions
                logp[predict_mask] += entity_type_prediction_logp

                # Only predict entity and mention lengths if we predicted that there was a mention
                predict_em = entity_types[:, timestep] * predict_mask
                if predict_em.sum() > 0:
                    # Predict entity ids
                    entity_id_prediction_outputs = self._dynamic_embeddings(hidden=current_hidden,
                                                                            mask=predict_em)
                    entity_id_logits = entity_id_prediction_outputs['logits']
                    entity_id_logp = F.log_softmax(entity_id_logits, dim=-1)
                    entity_id_prediction_logp, entity_id_predictions = sample_from_logp(entity_id_logp)

                    # Predict mention lengths - we do this before writing the
                    # entity id predictions since we'll need to reindex the new
                    # entities, but need the null embeddings here.
                    predicted_entity_embeddings = self._dynamic_embeddings.embeddings[predict_em, entity_id_predictions]
                    concatenated = torch.cat((current_hidden[predict_em], predicted_entity_embeddings), dim=-1)
                    mention_length_logits = self._mention_length_projection(concatenated)
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
            deterministic_mask = deterministic_mask * mask[:, timestep].byte()
            if deterministic_mask.sum() > 1:
                entity_types[deterministic_mask, timestep] = entity_types[deterministic_mask, timestep - 1]
                entity_ids[deterministic_mask, timestep] = entity_ids[deterministic_mask, timestep - 1]
                mention_lengths[deterministic_mask, timestep] = mention_lengths[deterministic_mask, timestep - 1] - 1

            # Update mention lengths for next timestep
            prev_mention_lengths = mention_lengths[:, timestep]

        return {
                'logp': logp,
                'sample': {
                        'entity_types': entity_types,
                        'entity_ids': entity_ids,
                        'mention_lengths': mention_lengths
                }
        }

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
        hidden = self._encoder(embeddings, mask)

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
            predict_all = predict_all * mask[:, timestep].byte()
            if predict_all.sum() > 0:

                # Equation 3 in the paper.
                entity_type_logits = self._entity_type_projection(current_hidden[predict_all])
                entity_type_loss = entity_type_loss + F.cross_entropy(
                        entity_type_logits,
                        current_entity_types[predict_all].long(),
                        reduction='sum')
                self._entity_type_accuracy(predictions=entity_type_logits,
                                           gold_labels=current_entity_types[predict_all].long())

                # Only proceed to predict entity and mention length if there is in fact an entity.
                predict_em = current_entity_types * predict_all

                if predict_em.sum() > 0:
                    # Equation 4 in the paper. We want new entities to correspond to a prediction of
                    # zero, their embedding should be added after they've been predicted for the first
                    # time.
                    modified_entity_ids = current_entity_ids.clone()
                    modified_entity_ids[modified_entity_ids == self._dynamic_embeddings.num_embeddings] = 0
                    entity_id_prediction_outputs = self._dynamic_embeddings(hidden=current_hidden,
                                                                            target=modified_entity_ids,
                                                                            mask=predict_em)
                    entity_id_loss = entity_id_loss + entity_id_prediction_outputs['loss'].sum()
                    self._entity_id_accuracy(predictions=entity_id_prediction_outputs['logits'],
                                             gold_labels=modified_entity_ids[predict_em])

                    # Equation 5 in the paper.
                    predicted_entity_embeddings = self._dynamic_embeddings.embeddings[predict_em, modified_entity_ids[predict_em]]
                    predicted_entity_embeddings = self._dropout(predicted_entity_embeddings)
                    concatenated = torch.cat((current_hidden[predict_em], predicted_entity_embeddings), dim=-1)
                    mention_length_logits = self._mention_length_projection(concatenated)
                    mention_length_loss = mention_length_loss + F.cross_entropy(
                            mention_length_logits,
                            current_mention_lengths[predict_em])
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
        total_loss = entity_type_loss + entity_id_loss + mention_length_loss

        output_dict = {
                'entity_type_loss': entity_type_loss,
                'entity_id_loss': entity_id_loss,
                'mention_length_loss': mention_length_loss,
                'loss': total_loss
        }

        # Update state
        self._state = {
                'prev_mention_lengths': prev_mention_lengths.detach()
        }

        return output_dict

    def reset_states(self, batch_size: int) -> None:
        """Resets the model's internals. Should be called at the start of a new batch."""
        self._encoder.reset_states()
        self._dynamic_embeddings.reset_states(batch_size)
        self._state = None

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
