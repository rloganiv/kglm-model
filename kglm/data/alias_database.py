import logging
import pickle
from typing import Dict, List, Optional

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
                 token_indexer: SingleIdTokenIndexer,
                 entity_indexer: SingleIdTokenIndexer,
                 alias_tokenizer: Tokenizer = None) -> None:
        self._path = path
        self._token_indexer = token_indexer
        self._entity_indexer = entity_indexer
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

    def tensorize(self):
        """
        Creates a list of tensors from the alias lookup.

        After dataset creation, we'll mainly want to work with alias lists as lists of padded
        tensors and their associated masks. This needs to be done **after** the vocabulary has
        been created. Accordingly, in our current approach, this method must be called in the
        forward pass of the model (since the operation is rather expensive we'll make sure that
        it doesn't anything after the first time it is called).
        """
        raise NotImplementedError
