"""
Implementation of perplexity and unknown penalized perplexity metrics.
"""
import math
from typing import Optional, Tuple

from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.training.metrics import Metric
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.data.extended_vocabulary import ExtendedVocabulary


@Metric.register('ppl')
class Perplexity(Metric):
    """Computes perplexity."""
    def __init__(self) -> None:
        self._sum_log_p = 0.0
        self._total_count = 0.0

    def __call__(self,
                 logits: torch.Tensor,
                 labels: torch.Tensor,
                 mask: Optional[torch.Tensor]) -> None:
        """
        Parameters
        ----------
        logits : ``torch.Tensor``, required.
            A tensor of class logits of shape (batch_size, k, sequence_length).
        labels : ``torch.Tensor``, required.
            A tensor of integer class labels of shape (batch_size, sequence_length).
        mask: ``torch.Tensor``, optional (default = None).
            A binary mask tensor of shape (batch_size, sequence_length).
        """
        logits, labels, mask = self.unwrap_to_tensors(logits, labels, mask)

        log_p = -F.cross_entropy(logits, labels, reduction='none')
        if mask is not None:
            self._sum_log_p += (mask * log_p).sum()
            self._total_count += mask.sum()
        else:
            self._sum_log_p += log_p.sum()
            self._total_count += torch.numel(labels)  # pylint: disable=no-member

    @overrides
    def get_metric(self, reset: bool) -> float:
        cross_entropy = -self._sum_log_p / self._total_count
        perplexity = math.exp(cross_entropy)
        if reset:
            self.reset()
        return perplexity

    @overrides
    def reset(self):
        self._sum_log_p = 0.0
        self._total_count = 0.0


@Metric.register('upp')
class UnknownPenalizedPerplexity(Metric):
    """
    Computes unknown penalized perplexity.

    Parameters
    ----------
    vocabulary : ``ExtendedVocabulary``, required.
        Vocabulary used by the model.
    namespace : ``str``, optional(default = 'tokens')
        Token namespace.
    oov_token : ``str``, optional(default = '@@UNKNOWN@@')
        Out of vocabulary token. AllenNLP's ``DEFAULT_OOV_TOKEN`` is used by default.
    """
    def __init__(self,
                 vocabulary: ExtendedVocabulary,
                 namespace: str = 'tokens',
                 oov_token: str = DEFAULT_OOV_TOKEN) -> None:
        # Compute the penalty weight applied to p(<unk>).
        vocab_size = vocabulary.get_vocab_size(namespace)
        unk_vocab_size = vocabulary.get_vocab_size(namespace + '_unk')
        if unk_vocab_size > 0:
            self._unk_penalty = math.log(unk_vocab_size)  # pylint: disable=no-member
        else:
            self._unk_penalty = 0.0

        # Identify the index of the <unk> token.
        self._unk_idx = vocabulary.get_token_index(oov_token, namespace=namespace)

        # Initialize the metric variables.
        self._sum_log_p = 0.0
        self._total_count = 0.0

    def __call__(self,
                 logits: torch.Tensor,
                 labels: torch.Tensor,
                 mask: Optional[torch.Tensor]) -> None:
        """
        Parameters
        ----------
        logits : ``torch.Tensor``, required.
            A tensor of class logits of shape (batch_size, k, sequence_length).
        labels : ``torch.Tensor``, required.
            A tensor of integer class labels of shape (batch_size, sequence_length).
        mask: ``torch.Tensor``, optional (default = None).
            A binary mask tensor of shape (batch_size, sequence_length).
        """
        logits, labels, mask = self.unwrap_to_tensors(logits, labels, mask)
        log_p = -F.cross_entropy(logits, labels, reduction='none')

        # Apply penalty to unks
        unk_ids = labels.eq(self._unk_idx)
        log_p[unk_ids] -= self._unk_penalty

        if mask is not None:
            self._sum_log_p += (mask * log_p).sum()
            self._total_count += mask.sum()
        else:
            self._sum_log_p += log_p.sum()
            self._total_count += torch.numel(labels)  # pylint: disable=no-member

    @overrides
    def get_metric(self, reset: bool) -> float:
        cross_entropy = -self._sum_log_p / self._total_count
        perplexity = math.exp(cross_entropy)
        if reset:
            self.reset()
        return perplexity

    @overrides
    def reset(self):
        self._sum_log_p = 0.0
        self._total_count = 0.0
