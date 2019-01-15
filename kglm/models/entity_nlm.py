"""
Implementation of the EntityNLM from: https://arxiv.org/abs/1708.00781
"""
import logging
from typing import Dict, List, Optional, Union

from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn import InitializerApplicator
from overrides import overrides
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from kglm.modules import DynamicEmbedding

logger = logging.getLogger(__name__)


StateDict = Dict[str, Union[torch.Tensor]]  # pylint: disable=invalid-name


@Model.register('entitynlm')
class EntityNLM(Model):
    """
    Implementation of the Entity Neural Language Model from:
        https://arxiv.org/abs/1708.00781

    Parameters
    ----------
    vocab : Vocabulary
    dim : int
        Dimension of embeddings.
    max_mention_length : int
        Maximum entity mention length.
    max_embeddings : int
        Maximum number of embeddings.
    text_field_embedder : TextFieldEmbedder
        Used to embed tokens.
    initializer : InitializerApplicator
        Used to initialize parameters.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 embedding_dim: int,
                 max_mention_length: int,
                 max_embeddings: int,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(EntityNLM, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._embedding_dim = embedding_dim
        self._max_mention_length = max_mention_length
        self._max_embeddings = max_embeddings

        self._state: Optional[StateDict] = None

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

        initializer(self)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                tokens: Dict[str, torch.Tensor],
                entity_types: Optional[torch.Tensor] = None,
                entity_ids: Optional[torch.Tensor] = None,
                mention_lengths: Optional[torch.Tensor] = None,
                reset: bool = False)-> Dict[str, torch.Tensor]:
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
                      mention_lengths: torch.Tensor) -> torch.Tensor:
        # The model state is updated at the end of every split. In order to
        if self._state is not None:
            tokens = {field: torch.cat((self._state['prev_tokens'][field], tokens[field]), dim=1) for field in tokens}
            entity_types = torch.cat((self._state['prev_entity_types'], entity_types), dim=1)
            entity_ids = torch.cat((self._state['prev_entity_ids'], entity_ids), dim=1)
            mention_lengths = torch.cat((self._state['prev_mention_lengths'], mention_lengths), dim=1)
            contexts = self._state['prev_contexts']
        else:
            batch_size = tokens['tokens'].shape[0]
            contexts = self._dummy_context_embedding.repeat(batch_size, 1)

        # Embed tokens and get RNN hidden state.
        sequence_length = tokens['tokens'].shape[1]
        mask = get_text_field_mask(tokens)
        embeddings = self._text_field_embedder(tokens)
        hidden = self._encoder(embeddings, mask)

        # Initialize losses
        entity_type_loss = 0.0
        entity_id_loss = 0.0
        mention_length_loss = 0.0
        vocab_loss = 0.0

        for timestep in range(sequence_length - 1):

            current_entity_types = entity_types[:, timestep]
            current_entity_ids = entity_ids[:, timestep]
            current_mention_lengths = mention_lengths[:, timestep]
            current_hidden = hidden[:, timestep]

            next_entity_types = entity_types[:, timestep + 1]
            next_entity_ids = entity_ids[:, timestep + 1]
            next_mention_lengths = mention_lengths[:, timestep + 1]
            next_mask = mask[:, timestep + 1]
            next_tokens = tokens['tokens'][:, timestep + 1]

            # A new entity embedding is added if the current entity id matches the number of
            # existing embeddings for the sequence.
            new_entities = current_entity_ids == self._dynamic_embeddings.num_embeddings
            self._dynamic_embeddings.add_embeddings(timestep, new_entities)

            # We also perform updates of the currently observed entities.
            self._dynamic_embeddings.update_embeddings(hidden=current_hidden,
                                                       update_indices=current_entity_ids,
                                                       timestep=timestep,
                                                       mask=current_entity_types)

            print(self._dynamic_embeddings.embeddings)

            # This is kind of stupid, but because we only update when the current entity id equals
            # the number of entities we do not have a well defined embeddings when next entity is a
            # new entity. The approach in the original code is to use the null embedding in this
            # case.
            next_entity_ids = next_entity_ids.clone()  # Prevent manipulating source data...
            next_entity_ids[next_entity_ids == self._dynamic_embeddings.num_embeddings] = 0

            # The only time we predict entity types, ids and mention lengths is when the current
            # mention length is 1.
            predict_all = current_mention_lengths == 1
            if predict_all.sum() > 0:

                # Index with predict all to omit any irrelevant elements.
                entity_type_logits = self._entity_type_projection(current_hidden[predict_all])
                entity_type_loss += F.cross_entropy(entity_type_logits,
                                                    next_entity_types[predict_all].long(),
                                                    reduction='sum')

                # Somewhat strangely, we use the null embedding to predict new entities. This is
                # because their embedding won't be added until the next timestep.
                embedding_output_dict = self._dynamic_embeddings(hidden=current_hidden,
                                                                 target=next_entity_ids,
                                                                 mask=next_entity_types * predict_all)
                entity_id_loss += embedding_output_dict['loss']

                # Use the next entity embeddings to predict mention length.
                next_entity_embeddings = self._dynamic_embeddings.embeddings[predict_all, next_entity_ids[predict_all]]
                concatenated = torch.cat((current_hidden[predict_all], next_entity_embeddings), dim=-1)
                mention_length_logits = self._mention_length_projection(concatenated)
                mention_length_loss += F.cross_entropy(mention_length_logits, next_mention_lengths[predict_all])

            # Always predict the next word. This is done using the hidden state and contextual bias.
            entity_embeddings = self._dynamic_embeddings.embeddings[next_entity_types, next_entity_ids[next_entity_types]]
            entity_embeddings = self._entity_output_projection(entity_embeddings)
            context_embeddings = contexts[1 - next_entity_types]
            context_embeddings = self._context_output_projection(context_embeddings)

            vocab_features = current_hidden.clone()
            if next_entity_types.sum() > 0:
                vocab_features[next_entity_types] = vocab_features[next_entity_types] + entity_embeddings
            if (1 - next_entity_types.sum()) > 0:
                vocab_features[1 - next_entity_types] = vocab_features[1 - next_entity_types] + context_embeddings
            vocab_logits = self._vocab_projection(vocab_features)
            _vocab_loss = F.cross_entropy(vocab_logits, next_tokens, reduction='none')
            _vocab_loss = _vocab_loss * next_mask.float()
            vocab_loss += _vocab_loss.sum()

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
            'loss': total_loss
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
        # Reset stateful modules
        self._encoder.reset_states()
        self._dynamic_embeddings.reset_states(batch_size)
        self._state = None

    def detach_states(self):
        self._dynamic_embeddings.detach_states()
