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


DynamicEmbeddingList = List[DynamicEmbedding]  # pylint: disable=invalid-name
StateDict = Dict[str, Union[torch.Tensor, DynamicEmbeddingList]]  # pylint: disable=invalid-name


def scalar_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.view(1, -1)
    target = target.view(1)
    return F.cross_entropy(logits, target)

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
    max_length : int
        Maximum entity mention length.
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
                 initializer: InitializerApplicator) -> None:
        super(EntityNLM, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self._encoder = encoder
        self._embedding_dim = embedding_dim
        self._max_mention_length = max_mention_length

        self._state: Optional[StateDict] = None

        # For entity type prediction
        self._entity_type_projection = torch.nn.Linear(in_features=embedding_dim,
                                                       out_features=2,
                                                       bias=False)

        # For entity prediction / updates
        self._dummy_entity_embedding = Parameter(torch.empty(1, embedding_dim))
        self._entity_projection = torch.nn.Linear(in_features=embedding_dim,
                                                  out_features=embedding_dim,
                                                  bias=False)
        self._distance_scalar = Parameter(torch.tensor(1e-6))  # pylint: disable=not-callable
        self._delta_projection = torch.nn.Linear(in_features=embedding_dim,
                                                 out_features=embedding_dim,
                                                 bias=False)

        # For mention length prediction
        self._length_projection = torch.nn.Linear(in_features=2*embedding_dim,
                                                  out_features=max_mention_length)

        # For next word prediction
        self._dummy_context_embedding = Parameter(torch.empty(1, embedding_dim))
        self._entity_output_projection = torch.nn.Linear(in_features=embedding_dim,
                                                         out_features=embedding_dim,
                                                         bias=False)
        self._context_output_projection = torch.nn.Linear(in_features=embedding_dim,
                                                          out_features=embedding_dim,
                                                          bias=False)
        self._vocab_projection = torch.nn.Linear(in_features=embedding_dim,
                                                 out_features=vocab.get_vocab_size("tokens"))

        initializer(self)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                reset_states: bool,
                inputs: Dict[str, torch.Tensor],
                outputs: Dict[str, torch.Tensor] = None,
                entity_types: torch.Tensor = None,
                entity_ids: torch.Tensor = None,
                mention_lengths: torch.Tensor = None)-> Dict[str, torch.Tensor]:
        # TODO: Inference
        batch_size = inputs['tokens'].shape[0]

        if reset_states:
            # Seq2SeqEncoder tracks its own state, it just needs to know to reset.
            self._encoder.reset_states()
            # In addition, dynamic entity embeddings need to be initialized for each sequence in
            # the batch.
            self._reset_states(batch_size)
        else:
            self._detach_states()

        out = {}

        if self.training:

            # Make MyPy happy
            assert outputs is not None
            assert entity_types is not None
            assert entity_ids is not None
            assert mention_lengths is not None

            # TODO: Make ArrayField able to use int dtypes.
            entity_types = entity_types.long()
            entity_ids = entity_ids.long()
            mention_lengths = mention_lengths.long()

            loss = self._compute_loss(inputs=inputs,
                                      outputs=outputs,
                                      entity_types=entity_types,
                                      entity_ids=entity_ids,
                                      mention_lengths=mention_lengths)
            out['loss'] = loss

        return out

    def _compute_loss(self,
                      inputs: Dict[str, torch.Tensor],
                      outputs: Dict[str, torch.Tensor],
                      entity_types: torch.Tensor,
                      entity_ids: torch.Tensor,
                      mention_lengths: torch.Tensor) -> torch.Tensor:

        assert self._state is not None

        batch_size, sequence_length = inputs['tokens'].shape

        mask = get_text_field_mask(inputs)
        embeddings = self._text_field_embedder(inputs)
        encoded = self._encoder(embeddings, mask)

        total_loss = 0.0
        for j in range(sequence_length):
            for i in range(batch_size):
                if mask[i, j] == 0:
                    continue

                dynamic_embedding = self._state['dynamic_embeddings'][i]

                # Note: We need to access self._state to prevent issues when continuing sequences
                # across splits.
                current_entity_type = self._state['entity_types'][i]
                current_entity_id = self._state['entity_ids'][i]
                current_mention_length = self._state['mention_lengths'][i]
                context = self._state['context'][i]

                next_token = outputs['tokens'][i, j]
                next_entity_type = entity_types[i, j]
                next_entity_id = entity_ids[i, j]
                next_mention_length = mention_lengths[i, j]

                hidden = encoded[i, j]

                zero = torch.tensor(0, device=hidden.device)  # pylint: disable=not-callable

                # Update entity at end of mention (This could also go at the end...)
                if current_entity_type > 0 and current_entity_id > 0:
                    if current_entity_id == len(dynamic_embedding.embeddings):
                        dynamic_embedding.add_entity(j)
                    dynamic_embedding.update_entity(hidden, current_entity_id, j)

                # Make entity/type predictions if we are not currently in the middle of a mention
                if  current_mention_length == 1:

                    # Next entity type prediction
                    entity_type_logits = self._entity_type_projection(hidden)
                    entity_type_loss = scalar_cross_entropy(entity_type_logits, next_entity_type)
                    total_loss += entity_type_loss

                    if next_entity_type == 1:

                        # Entity prediction
                        entity_logits = dynamic_embedding(hidden, j)
                        if next_entity_id < len(dynamic_embedding.embeddings):
                            entity_loss = scalar_cross_entropy(entity_logits, next_entity_id)
                        else:
                            entity_loss = scalar_cross_entropy(entity_logits, zero)  # Weird...
                        total_loss += entity_loss

                        # Length prediction
                        if next_entity_id < len(dynamic_embedding.embeddings):
                            embedding = dynamic_embedding.embeddings[next_entity_id].clone().squeeze()
                        else:
                            embedding = dynamic_embedding.embeddings[0].clone().squeeze()
                        concatenated = torch.cat((hidden, embedding))
                        length_logits = self._length_projection(concatenated)
                        length_loss = scalar_cross_entropy(length_logits, next_mention_length)
                        total_loss += length_loss

                # Word prediction
                if next_entity_type == 1:
                    if next_entity_id < len(dynamic_embedding.embeddings):
                        entity_embedding = dynamic_embedding.embeddings[next_entity_id].clone()
                    else:
                        entity_embedding = dynamic_embedding.embeddings[0].clone()
                    combined = hidden + self._entity_output_projection(entity_embedding).squeeze()
                else:
                    combined = hidden + self._context_output_projection(context).squeeze()
                token_logits = self._vocab_projection(combined)
                token_loss = scalar_cross_entropy(token_logits, next_token)
                total_loss += token_loss

            self._state['entity_types'] = entity_types[:, j]
            self._state['entity_ids'] = entity_ids[:, j]
            self._state['mention_lengths'] = mention_lengths[:, j]
            self._state['context'] = encoded[:, j]

        return total_loss

    def _reset_states(self, batch_size: int) -> None:
        self._state = {}

        # Initialize dynamic embeddings.
        dynamic_embeddings: List[DynamicEmbedding] = []
        for _ in range(batch_size):
            dynamic_embedding = DynamicEmbedding(mean_embedding=self._dummy_entity_embedding,
                                                 distance_scalar=self._distance_scalar,
                                                 embedding_projection=self._entity_projection,
                                                 delta_projection=self._delta_projection)
            dynamic_embeddings.append(dynamic_embedding)
        self._state['dynamic_embeddings'] = dynamic_embeddings

        # Initialize metadata - these are the `curr_` values in the original implementation,
        # applied to the <START> token.
        self._state['entity_types'] = torch.zeros(batch_size, dtype=torch.int64)
        self._state['entity_ids'] = torch.zeros(batch_size, dtype=torch.int64)
        self._state['mention_lengths'] = torch.ones(batch_size, dtype=torch.int64)
        self._state['context'] = self._dummy_context_embedding.repeat(batch_size, 1)

    def _detach_states(self):
        for dynamic_embedding in self._state['dynamic_embeddings']:
            dynamic_embedding.detach()
        self._state['context'] = self._state['context'].detach()
