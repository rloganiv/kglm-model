"""
Implementation of the EntityNLM from: https://arxiv.org/abs/1708.00781
"""
import logging
from typing import Dict, List, Optional, Union

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
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

        # Entity type log probability is proportional to a bilinear form applied to the current
        # hidden state and the entity type embedding. Currently there are only two entity types:
        # entities and non-entities. We compute the bilinear form by first projecting the entity
        # type embeddings (achieved by the linear layer), then computing the dot product with
        # the current hidden state (by using einsum).
        self._entity_type_embeddings = Parameter(torch.empty(2, embedding_dim))
        self._entity_type_projection = torch.nn.Linear(in_features=embedding_dim,
                                                       out_features=embedding_dim,
                                                       bias=False)

        self._null_entity_embedding = Parameter(torch.empty(1, embedding_dim))
        self._entity_projection = torch.nn.Linear(in_features=embedding_dim,
                                                  out_features=embedding_dim,
                                                  bias=False)
        self._delta_projection = torch.nn.Linear(in_features=embedding_dim,
                                                 out_features=embedding_dim,
                                                 bias=False)

        # TODO: Entity distance features - probably going to need to define a maximum cutoff...
        #self._length_projection = torch.nn.Linear(in_features=2*dim, out_features=max_length)

        initializer(self)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                reset_states: bool,
                inputs: Dict[str, torch.Tensor],
                outputs: Dict[str, torch.Tensor] = None,
                entity_types: torch.Tensor = None,
                entity_ids: torch.Tensor = None,
                entity_mention_lengths: torch.Tensor = None)-> Dict[str, torch.Tensor]:
        # TODO: Inference
        batch_size, sequence_length = inputs['tokens'].shape

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
            assert entity_mention_lengths is not None

            # TODO: Make ArrayField able to use int dtypes.
            entity_types = entity_types.long()
            entity_ids = entity_ids.long()
            entity_mention_lengths = entity_mention_lengths.long()

            loss = self._compute_loss(inputs=inputs,
                                    outputs=outputs,
                                    entity_types=entity_types,
                                    entity_ids=entity_ids,
                                    entity_mention_lengths=entity_mention_lengths)
            out['loss'] = loss

        return out

    def _compute_loss(self,
                      inputs: Dict[str, torch.Tensor],
                      outputs: Dict[str, torch.Tensor],
                      entity_types: torch.Tensor,
                      entity_ids: torch.Tensor,
                      entity_mention_lengths: torch.Tensor) -> torch.Tensor:

        assert self._state is not None

        batch_size, sequence_length = inputs['tokens'].shape

        mask = get_text_field_mask(inputs)
        embeddings = self._text_field_embedder(inputs)
        encoded = self._encoder(embeddings, mask)

        projected_type_embeddings = self._entity_type_projection(self._entity_type_embeddings)
        # TODO: Investigate if this slows things down (could use a different BLAS function if so)
        # THIS DOES NOT WORK - TYPES ALSO ARE ONLY PREDICTED WHEN L=1
        type_logits = torch.einsum('ijk,lk->ijl', encoded, projected_type_embeddings).contiguous()
        total_type_loss = sequence_cross_entropy_with_logits(type_logits, entity_types, mask, average=None).sum()

        total_entity_loss = 0.0
        total_length_loss = 0.0
        for i in range(batch_size):
            dynamic_embedding = self._state['dynamic_embeddings'][i]
            for j in range(sequence_length):
                if mask[i, j] == 0:
                    continue

                # current_token_embedding = embeddings[i,j]
                # current_type = entity_types[i,j]
                # current_entity_id = entity_ids[i,j]
                # current_length = entity_mention_lengths[i,j]
                # next_token_embedding = embeddings[i,j+1]
                # next_type = entity_types[i,j+1]
                # next_entity_id = entity_ids[i,j+1]
                # next_length = entity_mention_lengths[i,j+1]

                # Following Yangfeng's code we start by updating entities if needed

                hidden = encoded[i, j]
                # We only measure entity and length prediction loss when an entity is first observed.
                # (e.g. not when length deterministically decrements).
                if (self._state['prev_mention_lengths'][i] == 1) and (entity_types[i, j] == 1):
                    _, entity_loss = dynamic_embedding(hidden, entity_ids[i, j])
                    total_entity_loss += entity_loss

        total_type_loss = total_type_loss / mask.sum()
        total_entity_loss = total_entity_loss / mask.sum()

        loss = (total_type_loss + total_entity_loss) / 2

        return loss

    def _reset_states(self, batch_size: int) -> None:
        self._state = {}

        # Initialize dynamic embeddings
        mean_embedding = self._entity_type_embeddings[1]
        dynamic_embeddings: List[DynamicEmbedding] = []
        for _ in range(batch_size):
            dynamic_embedding = DynamicEmbedding(mean_embedding=mean_embedding,
                                                 embedding_projection=self._entity_projection,
                                                 delta_projection=self._delta_projection)
            dynamic_embeddings.append(dynamic_embedding)

        self._state['dynamic_embeddings'] = dynamic_embeddings

        self._state['prev_mention_lengths'] = torch.ones(batch_size, dtype=torch.int64)
        self._state['prev_entity_ids'] = torch.zeros(batch_size, dtype=torch.int64)

    def _detach_states(self):
        for dynamic_embedding in self._state['dynamic_embeddings']:
            dynamic_embedding.detach()