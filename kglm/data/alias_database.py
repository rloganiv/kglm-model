import logging
import pickle
from typing import Dict, List, Optional, Union

from allennlp.data import Vocabulary
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
import torch

logger = logging.getLogger(__name__)


AliasList = List[List[Token]]


# TODO: Maybe someday we'll want a general ``Database`` of which this would be a specific type.
class AliasDatabase:
    """A Database of Aliases"""
    def __init__(self,
                 path: str,
                 alias_tokenizer: Tokenizer = None) -> None:
        self._path = path
        self._alias_tokenizer = alias_tokenizer or WordTokenizer()

        self._alias_lookup: Optional[Dict[Token, AliasList]] = None
        self._tensorized_lookup: Optional[List[torch.Tensor]] = None

    def read(self):
        # Right now we only support loading the alias database from a pickle file.
        logger.debug('Reading alias database from "%s"', self._path)
        with open(self._path, 'rb') as f:
            alias_lookup = pickle.load(f)

        # Since aliases originally are
        logger.debug('Tokenizing aliases')
        alias_lookup = {Token(entity): [self._alias_tokenizer.tokenize(alias) for alias in aliases]
                        for entity, aliases in alias_lookup.items()}

        self._alias_lookup = alias_lookup

    def tensorize(self, vocab: Vocabulary):
        """
        Creates a list of tensors from the alias lookup.

        After dataset creation, we'll mainly want to work with alias lists as lists of padded
        tensors and their associated masks. This needs to be done **after** the vocabulary has
        been created. Accordingly, in our current approach, this method must be called in the
        forward pass of the model (since the operation is rather expensive we'll make sure that
        it doesn't anything after the first time it is called).
        """

        assert self._alias_lookup is not None, 'Alias lookup must be built before it is tensorized'

        if self._tensorized_lookup is not None:
            return

        entity_idx_to_token = vocab.get_index_to_token_vocabulary('entities')
        self._tensorized_lookup = []
        for i in range(len(entity_idx_to_token)):  # pylint: disable=C0200
            entity = entity_idx_to_token[i]
            aliases = self._alias_lookup[entity]
            sequence_length = max(len(alias) for alias in aliases)
            alias_tensor = torch.zeros(len(aliases), sequence_length)
            for j, alias in enumerate(aliases):
                for k, token in enumerate(alias):
                    alias_tensor[j, k] = vocab.get_token_index(token, 'tokens')
            self._tensorized_lookup.append(alias_tensor)

    def __getitem__(self, idx: Union[torch.tensor, int]) -> torch.Tensor:
        assert self._tensorized_lookup is not None, 'Need to tensorize alias database first'
        if isinstance(idx, torch.Tensor):
            # TODO: Do we want to catch when idx has more than one elt?
            idx = idx.item()
        return self._tensorized_lookup[idx]
