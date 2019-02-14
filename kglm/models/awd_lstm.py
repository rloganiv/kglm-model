import logging
import math
from typing import Any, Dict, List, Optional

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from overrides import overrides
import torch

from kglm.modules import embedded_dropout, LockedDropout, WeightDrop
from kglm.training.metrics import Ppl

logger = logging.getLogger(__name__)


@Model.register('awd-lstm-lm')
class AwdLstmLanguageModel(Model):
    """
    Port of the awd-lstm-lm model from:
        https://github.com/salesforce/awd-lstm-lm/

    Parameters
    ----------
    vocab : ``Vocabulary``
        The model vocabulary.
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed tokens.
    hidden_size : ``int``
        LSTM hidden layer size (note: not needed if num_layers == 1)
    num_layers : ``int``
        Number of LSTM layers to use in encoder.
    splits : ``List[int]``, optional (default=``[]``)
        Splits to use in adaptive softmax.

    A bunch of optional dropout parameters...

    tie_weights : ``bool``, optional (default=``False``)
        Whether to tie embedding and output projection weights.
    initializer: ``InitializerApplicator``,  optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 splits: List[int] = [],
                 dropout: float = 0.4,
                 dropouth: float = 0.3,
                 dropouti: float = 0.65,
                 dropoute: float = 0.1,
                 wdrop: float = 0.5,
                 alpha: float = 2.0,
                 beta: float = 1.0,
                 tie_weights: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(AwdLstmLanguageModel, self).__init__(vocab)

        # Model architecture
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_weights = tie_weights
        self.splits = splits
        self.alpha = alpha
        self.beta = beta

        # Dropout stuff
        self.locked_dropout = LockedDropout()
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.dropout = dropout

        # Initialize empty state dict
        self._state: Optional[Dict[str, Any]] = None

        # Tokens are manually embedded instead of using a TokenEmbedder to make using
        # embedding_dropout easier.
        self.embedder = torch.nn.Embedding(vocab.get_vocab_size(namespace='tokens'),
                                           embedding_size)

        rnns: List[torch.nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                input_size = embedding_size
            else:
                input_size = hidden_size
            if (i == num_layers - 1) and tie_weights:
                output_size = embedding_size
            else:
                output_size = hidden_size
            rnns.append(torch.nn.LSTM(input_size, output_size, batch_first=True))
        rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in rnns]
        self.rnns = torch.nn.ModuleList(rnns)

        self.decoder = torch.nn.Linear(output_size, vocab.get_vocab_size(namespace='tokens'))

        # Optionally tie weights
        if tie_weights:
            # pylint: disable=protected-access
            self.decoder.weight = self.embedder.weight

        initializer(self)

        self._unk_index = vocab.get_token_index(DEFAULT_OOV_TOKEN)
        self._unk_penalty = math.log(vocab.get_vocab_size('tokens_unk'))

        self.ppl = Ppl()
        self.upp = Ppl()

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                source: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                reset: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # THE BELOW ONLY NEEDS TO BE SATISFIED FOR THE FANCY ITERATOR, MERITY
        # ET AL JUST PROPOGATE THE HIDDEN STATE NO MATTER WHAT
        # To make life easier when evaluating the model we use a BasicIterator
        # so that we do not need to worry about the sequence truncation
        # performed by our splitting iterators. To accomodate this, we assume
        # that if reset is not given, then everything gets reset.
        if reset is None:
            self._state = None
        elif reset.all() and (self._state is not None):
            logger.debug('RESET')
            self._state = None
        elif reset.any() and (self._state is not None):
            for layer in range(self.num_layers):
                h, c = self._state['layer_%i' % layer]
                h[:, reset, :] = torch.zeros_like(h[:, reset, :])
                c[:, reset, :] = torch.zeros_like(c[:, reset, :])
                self._state['layer_%i' % layer] = (h, c)

        target_mask = get_text_field_mask(target)
        source = source['tokens']
        target = target['tokens']

        embeddings = embedded_dropout(self.embedder, source,
                                      dropout=self.dropoute if self.training else 0)
        embeddings = self.locked_dropout(embeddings, self.dropouti)

        # Iterate through RNN layers
        current_input = embeddings
        current_hidden = []
        outputs = []
        dropped_outputs = []
        for layer, rnn in enumerate(self.rnns):

            # Bookkeeping
            if self._state is not None:
                prev_hidden = self._state['layer_%i' % layer]
            else:
                prev_hidden = None

            # Forward-pass
            output, hidden = rnn(current_input, prev_hidden)

            # More bookkeeping
            output = output.contiguous()
            outputs.append(output)
            hidden = tuple(h.detach() for h in hidden)
            current_hidden.append(hidden)

            # Apply dropout
            if layer == self.num_layers - 1:
                current_input = self.locked_dropout(output, self.dropout)
                dropped_outputs.append(output)
            else:
                current_input = self.locked_dropout(output, self.dropouth)
                dropped_outputs.append(current_input)

        # Compute logits and loss
        logits = self.decoder(current_input)
        loss = sequence_cross_entropy_with_logits(logits, target.contiguous(),
                                                  target_mask,
                                                  average="token")
        num_tokens = target_mask.float().sum() + 1e-13

        # Activation regularization
        if self.alpha:
            loss = loss + self.alpha * current_input.pow(2).mean()
        # Temporal activation regularization (slowness)
        if self.beta:
            loss = loss + self.beta * (output[:, 1:] - output[:, :-1]).pow(2).mean()

        # Update metrics and state
        unks = target.eq(self._unk_index)
        unk_penalty = self._unk_penalty * unks.float().sum()

        self.ppl(loss * num_tokens, num_tokens)
        self.upp(loss * num_tokens + unk_penalty, num_tokens)
        self._state = {'layer_%i' % l: h for l, h in enumerate(current_hidden)}

        return {'loss': loss}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'ppl': self.ppl.get_metric(reset),
            'upp': self.upp.get_metric(reset)
        }
