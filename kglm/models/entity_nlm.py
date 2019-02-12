"""
Implementation of the EntityNLM from: https://arxiv.org/abs/1708.00781
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
from torch.nn import Parameter
import torch.nn.functional as F

from kglm.modules import DynamicEmbedding
# from kglm.training.metrics import Perplexity, UnknownPenalizedPerplexity

logger = logging.getLogger(__name__)


StateDict = Dict[str, Union[torch.Tensor]]  # pylint: disable=invalid-name


@Model.register('entitynlm')
class EntityNLM(Model):
    """
    Implementation of the Entity Neural Language Model from:
        https://arxiv.org/abs/1708.00781

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
    tie_weights : ``bool``
        Whether to tie embedding and output weights.
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
                 tie_weights: bool,
                 variational_dropout_rate: float = 0.0,
                 dropout_rate: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(EntityNLM, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._embedding_dim = embedding_dim
        self._max_mention_length = max_mention_length
        self._max_embeddings = max_embeddings
        self._tie_weights = tie_weights
        self._variational_dropout_rate = variational_dropout_rate
        self._dropout_rate = dropout_rate

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

        # For next word prediction
        self._dummy_context_embedding = Parameter(F.normalize(torch.randn(1, embedding_dim))) # TODO: Maybe squeeze
        self._entity_output_projection = torch.nn.Linear(in_features=embedding_dim,
                                                         out_features=embedding_dim,
                                                         bias=False)
        self._context_output_projection = torch.nn.Linear(in_features=embedding_dim,
                                                          out_features=embedding_dim,
                                                          bias=False)
        self._vocab_projection = torch.nn.Linear(in_features=embedding_dim,
                                                 out_features=vocab.get_vocab_size('tokens'))
        if tie_weights:
            self._vocab_projection.weight = self._text_field_embedder._token_embedders['tokens'].weight  # pylint: disable=W0212

        # self._perplexity = Perplexity()
        # self._unknown_penalized_perplexity = UnknownPenalizedPerplexity(self.vocab)
        self._entity_type_accuracy = CategoricalAccuracy()
        self._entity_id_accuracy = CategoricalAccuracy()
        self._mention_length_accuracy = CategoricalAccuracy()

        if tie_weights:
            self._vocab_projection.weight = self._text_field_embedder._token_embedders['tokens'].weight  # pylint: disable=W0212

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

        if not self.training:
            # TODO Some evaluation stuff
            pass

        return output_dict

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
        vocab_loss : ``torch.Tensor``
            The loss of vocab word predictions.
        loss : ``torch.Tensor``
            The combined loss.
        logp : ``torch.Tensor``
            Instance level log-probabilities
        """
        batch_size, sequence_length = tokens['tokens'].shape

        # The model state allows us to recover the last timestep from the previous chunk in the
        # split. If it does not exist, then we are processing a new batch.
        if self._state is not None:
            tokens = {field: torch.cat((self._state['prev_tokens'][field], tokens[field]), dim=1)
                      for field in tokens}
            entity_types = torch.cat((self._state['prev_entity_types'], entity_types), dim=1)
            entity_ids = torch.cat((self._state['prev_entity_ids'], entity_ids), dim=1)
            mention_lengths = torch.cat((self._state['prev_mention_lengths'], mention_lengths), dim=1)
            contexts = self._state['prev_contexts']
            sequence_length += 1
        else:
            contexts = self._dummy_context_embedding.repeat(batch_size, 1)

        # Embed tokens and get RNN hidden state.
        mask = get_text_field_mask(tokens)
        embeddings = self._text_field_embedder(tokens)
        embeddings = self._variational_dropout(embeddings)
        hidden = self._encoder(embeddings, mask)

        # Initialize losses
        entity_type_loss = 0.0
        entity_id_loss = 0.0
        mention_length_loss = 0.0
        vocab_loss = 0.0
        logp = hidden.new_zeros(batch_size)

        # We dynamically add entities and update their representations in sequence. The following
        # loop is designed to imitate as closely as possible lines 219-313 in:
        #   https://github.com/jiyfeng/entitynlm/blob/master/entitynlm.h
        # while still being carried out in batch.
        for timestep in range(sequence_length - 1):

            current_entity_types = entity_types[:, timestep]
            current_entity_ids = entity_ids[:, timestep]
            current_mention_lengths = mention_lengths[:, timestep]
            current_hidden = self._dropout(hidden[:, timestep])

            next_entity_types = entity_types[:, timestep + 1]
            next_entity_ids = entity_ids[:, timestep + 1]
            next_mention_lengths = mention_lengths[:, timestep + 1]
            next_mask = mask[:, timestep + 1]
            next_tokens = tokens['tokens'][:, timestep + 1]

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

            # This part is a little counter-intuitive. Because the above code adds a new embedding
            # whenever the **current** entity id matches the number of embeddings, we are one
            # embedding short if the **next** entity id has not been seen before. To deal with
            # this, we use the null embedding (e.g. the first one we created) as a proxy for the
            # new entity's embedding (since it is on average what the new entity's embedding will
            # be initialized in the next timestep). It might seem more sensible to just create the
            # embedding now, but we cannot because of the subsequent update (since this would
            # require access to the **next** hidden state, which does not exist during generation).
            next_entity_ids = next_entity_ids.clone()  # This prevents mutating the source data.
            next_entity_ids[next_entity_ids == self._dynamic_embeddings.num_embeddings] = 0

            # We only predict the types / ids / lengths of the next mention if we are not currently
            # in the process of generating it (e.g. if the current remaining mention length is 1).
            # Indexing / masking with ``predict_all`` makes it possible to do this in batch.
            predict_all = (current_mention_lengths == 1) * next_mask.byte()
            if predict_all.sum() > 0:

                # Equation 3 in the paper.
                entity_type_logits = self._entity_type_projection(current_hidden[predict_all])
                _entity_type_loss = F.cross_entropy(entity_type_logits,
                                                    next_entity_types[predict_all].long(),
                                                    reduction='none')
                entity_type_loss += _entity_type_loss.sum()

                entity_type_logp = torch.zeros_like(next_entity_types, dtype=torch.float32)
                entity_type_logp[predict_all] = -_entity_type_loss
                logp += entity_type_logp

                self._entity_type_accuracy(predictions=entity_type_logits,
                                           gold_labels=next_entity_types[predict_all].long())

                # Only proceed to predict entity and mention length if there is in fact an entity.
                predict_em = next_entity_types * predict_all
                if predict_em.sum() > 0:
                    # Equation 4 in the paper.
                    entity_id_prediction_outputs = self._dynamic_embeddings(hidden=current_hidden,
                                                                            target=next_entity_ids,
                                                                            mask=predict_em)
                    _entity_id_loss = entity_id_prediction_outputs['loss']
                    entity_id_loss += _entity_id_loss.sum()

                    entity_id_logp = torch.zeros_like(next_entity_ids, dtype=torch.float32)
                    entity_id_logp[predict_em] = -_entity_id_loss
                    logp += entity_id_logp

                    self._entity_id_accuracy(predictions=entity_id_prediction_outputs['logits'],
                                             gold_labels=next_entity_ids[predict_em])

                    # Equation 5 in the paper.
                    next_entity_embeddings = self._dynamic_embeddings.embeddings[predict_em, next_entity_ids[predict_em]]
                    next_entity_embeddings = self._dropout(next_entity_embeddings)
                    concatenated = torch.cat((current_hidden[predict_em], next_entity_embeddings), dim=-1)
                    mention_length_logits = self._mention_length_projection(concatenated)
                    _mention_length_loss = F.cross_entropy(mention_length_logits,
                                                           next_mention_lengths[predict_em],
                                                           reduction='none')
                    mention_length_loss += _mention_length_loss.sum()

                    mention_length_logp = torch.zeros_like(next_mention_lengths, dtype=torch.float32)
                    mention_length_logp[predict_em] = -_mention_length_loss
                    logp += mention_length_logp

                    self._mention_length_accuracy(predictions=mention_length_logits,
                                                  gold_labels=next_mention_lengths[predict_em])

            # Always predict the next word. This is done using the hidden state and contextual bias.
            entity_embeddings = self._dynamic_embeddings.embeddings[next_entity_types, next_entity_ids[next_entity_types]]
            entity_embeddings = self._entity_output_projection(entity_embeddings)
            context_embeddings = contexts[1 - next_entity_types]
            context_embeddings = self._context_output_projection(context_embeddings)

            # The checks in the following block of code are required to prevent adding empty
            # tensors to vocab_features (which causes a floating point error).
            vocab_features = current_hidden.clone()
            if next_entity_types.sum() > 0:
                vocab_features[next_entity_types] = vocab_features[next_entity_types] + entity_embeddings
            if (1 - next_entity_types.sum()) > 0:
                vocab_features[1 - next_entity_types] = vocab_features[1 - next_entity_types] + context_embeddings
            vocab_logits = self._vocab_projection(vocab_features)

            _vocab_loss = F.cross_entropy(vocab_logits, next_tokens, reduction='none')
            _vocab_loss = _vocab_loss * next_mask.float()
            vocab_loss += _vocab_loss.sum()
            logp += -_vocab_loss

            # self._perplexity(logits=vocab_logits,
            #                  labels=next_tokens,
            #                  mask=next_mask.float())
            # self._unknown_penalized_perplexity(logits=vocab_logits,
            #                                    labels=next_tokens,
            #                                    mask=next_mask.float())

            # Lastly update contexts
            contexts = current_hidden

        # Normalize the losses
        entity_type_loss /= mask.sum()
        entity_id_loss /= mask.sum()
        mention_length_loss /= mask.sum()
        vocab_loss /= mask.sum()
        total_loss = entity_type_loss + entity_id_loss + mention_length_loss + vocab_loss

        output_dict = {
                'entity_type_loss': entity_type_loss,
                'entity_id_loss': entity_id_loss,
                'mention_length_loss': mention_length_loss,
                'vocab_loss': vocab_loss,
                'loss': total_loss,
                'logp': logp
        }

        # Update the model state
        self._state = {
                'prev_tokens': {field: tokens[field][:, -1].unsqueeze(1).detach() for field in tokens},
                'prev_entity_types': entity_types[:, -1].unsqueeze(1).detach(),
                'prev_entity_ids': entity_ids[:, -1].unsqueeze(1).detach(),
                'prev_mention_lengths': mention_lengths[:, -1].unsqueeze(1).detach(),
                'prev_contexts': contexts.detach()
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
                # 'ppl': self._perplexity.get_metric(reset),
                # 'upp': self._unknown_penalized_perplexity.get_metric(reset),
                'et_acc': self._entity_type_accuracy.get_metric(reset),
                'eid_acc': self._entity_id_accuracy.get_metric(reset),
                'ml_acc': self._mention_length_accuracy.get_metric(reset)
        }
