"""
Implementation of the EntityNLM from: https://arxiv.org/abs/1708.00781
"""
from typing import Dict

from allennlp.models import Model
from allennlp.nn import InitializerApplicator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from overrides import overrides
import torch
from torch.nn import Parameter

from kglm.modules import DynamicEmbedding


@Model.register('entitynlm')
class EntityNLM(Model):
    """
    Implementation of the Entity Neural Language Model from:
        TODO: Cite Yangfeng Ji

    This is a stateful model.

    WARNING: Must use StatefulIterator during training.

    Parameters
    ==========
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
                 dim: int,
                 max_length: int,
                 text_field_embedder: TextFieldEmbedder) -> None:
        super(EntityNLM, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder

        self._dim = dim
        self._max_length = max_length

        self._state = None

        # Entity type log probability is proportional to a bilinear form applied to the current
        # hidden state and the entity type embedding. Currently there are only two entity types:
        # entities and non-entities. We compute the bilinear form by first projecting the entity
        # type embeddings (achieved by the linear layer), then computing the dot product with
        # the current hidden state (by using einsum).
        self._entity_type_embeddings = Parameter(torch.empty(2, dim))
        self._entity_type_projection = torch.nn.Linear(in_features=dim, out_features=dim)

        # TODO: Entity distance features - probably going to need to define a maximum cutoff...

        self._length_projection = torch.nn.Linear(in_features=2*dim, out_features=max_length)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                input: Dict[str, torch.Tensor],
                output: Dict[str, torch.Tensor],
                reset_state: bool,
                z: torch.Tensor,
                e: torch.Tensor,
                l: torch.Tensor)-> Dict[str, torch.Tensor]:
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        return {'loss': w.sum()}

    def _init_state(self, batch_size: int):
        # State should have:
        #   1. The previous hidden states of the LSTM (detached)
        #   2. A list of DynamicEmbedding modules
        raise NotImplementedError
