"""
Implementation of the EntityNLM from: https://arxiv.org/abs/1708.00781
"""
from typing import Dict

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from overrides import overrides
import torch



@Model.register('entitynlm')
class EntityNLM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder):
        super(EntityNLM, self).__init__(vocab)
        self._text_field_embedder = text_field_embedder

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                input: Dict[str, torch.Tensor],
                output: Dict[str, torch.Tensor],
                z: torch.Tensor,
                e: torch.Tensor,
                l: torch.Tensor)-> Dict[str, torch.Tensor]:
        import pdb; pdb.set_trace()
        return {}

    def compute_loss(self):
        raise NotImplementedError
