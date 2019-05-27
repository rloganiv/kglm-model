from collections import defaultdict
import logging
import pickle
from typing import Any, Dict, List, Set, Tuple

from allennlp.common.tqdm import Tqdm
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
import numpy as np
import torch

logger = logging.getLogger(__name__)


AliasList = List[List[str]]
MAX_ALIASES = 4
MAX_TOKENS = 8


def tokenize_to_string(text: str, tokenizer: Tokenizer) -> List[str]:
    """Sigh"""
    return [token.text for token in tokenizer.tokenize(text)]


class AliasDatabase:
    """A Database of Aliases"""
    def __init__(self,
                 token_lookup: Dict[str, AliasList],
                 id_map_lookup: Dict[str, Dict[str, int]],
                 id_array_lookup: Dict[str, np.ndarray],
                 token_to_entity_lookup: Dict[str, Set[Any]]) -> None:
        self._token_lookup = token_lookup or {}
        self._id_map_lookup = id_map_lookup or {}
        self._id_array_lookup = id_array_lookup or {}
        self._token_to_entity_lookup = token_to_entity_lookup or {}

        self.is_tensorized = False
        self._global_id_lookup: List[torch.Tensor] = []
        self._local_id_lookup: List[torch.Tensor] = []
        self._token_id_to_entity_id_lookup: List[torch.Tensor] = []
        self._num_entities = -1

    @classmethod
    def load(cls, path: str):

        logger.info('Loading alias database from "%s". This will probably take a second.', path)
        # TODO: Pretokenize the database to match the tokenization of the data itself. This
        # shouldn't be an issue ATM since I believe WordTokenizer() also uses SpaCy. But better to
        # air on the side of caution...
        tokenizer = WordTokenizer()
        token_lookup: Dict[str, AliasList] = {}
        id_map_lookup: Dict[str, Dict[str, int]] = {}
        id_array_lookup: Dict[str, np.ndarray] = {}
        token_to_entity_lookup: Dict[str, Set[Any]] = defaultdict(set)

        # Right now we only support loading the alias database from a pickle file.
        with open(path, 'rb') as f:
            alias_lookup = pickle.load(f)

        for entity, aliases in Tqdm.tqdm(alias_lookup.items()):
            # Reverse token to potential entity lookup
            for alias in aliases:
                for token in tokenize_to_string(alias, tokenizer):
                    token_to_entity_lookup[token].add(entity)

            # Start by tokenizing the aliases
            tokenized_aliases: AliasList = [tokenize_to_string(alias, tokenizer)[:MAX_TOKENS] for alias in aliases]
            tokenized_aliases = tokenized_aliases[:MAX_ALIASES]
            token_lookup[entity] = tokenized_aliases

            # Next obtain the set of unqiue tokens appearing in aliases for this entity. Use this
            # to build a map from tokens to their unique id.
            unique_tokens = set()
            for tokenized_alias in tokenized_aliases:
                unique_tokens.update(tokenized_alias)
            id_map = {token: i + 1 for i, token in enumerate(unique_tokens)}
            id_map_lookup[entity] = id_map

            # Lastly create an array associating the tokens in the alias to their corresponding ids.
            num_aliases = len(tokenized_aliases)
            max_alias_length = max(len(tokenized_alias) for tokenized_alias in tokenized_aliases)
            id_array = np.zeros((num_aliases, max_alias_length), dtype=int)
            for i, tokenized_alias in enumerate(tokenized_aliases):
                for j, token in enumerate(tokenized_alias):
                    id_array[i, j] = id_map[token]
            id_array_lookup[entity] = id_array

        return cls(token_lookup=token_lookup,
                   id_map_lookup=id_map_lookup,
                   id_array_lookup=id_array_lookup,
                   token_to_entity_lookup=token_to_entity_lookup)

    def token_to_uid(self, entity: str, token: str) -> int:
        if entity in self._id_map_lookup:
            id_map = self._id_map_lookup[entity]
            if token in id_map:
                return id_map[token]
        return 0

    def tensorize(self, vocab: Vocabulary):
        """
        Creates a list of tensors from the alias lookup.

        After dataset creation, we'll mainly want to work with alias lists as lists of padded
        tensors and their associated masks. This needs to be done **after** the vocabulary has
        been created. Accordingly, in our current approach, this method must be called in the
        forward pass of the model (since the operation is rather expensive we'll make sure that
        it doesn't anything after the first time it is called).
        """
        # This operation is expensive, only do it once.
        if self.is_tensorized:
            return

        logger.debug('Tensorizing AliasDatabase')

        entity_idx_to_token = vocab.get_index_to_token_vocabulary('raw_entity_ids')
        for i in range(len(entity_idx_to_token)):  # pylint: disable=C0200
            entity = entity_idx_to_token[i]
            try:
                tokenized_aliases = self._token_lookup[entity]
            except KeyError:
                # If we encounter non-entity tokens (e.g. padding and null) then just add
                # a blank placeholder - these should not be encountered during training.
                self._global_id_lookup.append(None)
                self._local_id_lookup.append(None)
                continue

            # Construct tensor of alias token indices from the global vocabulary.
            num_aliases = len(tokenized_aliases)
            max_alias_length = max(len(tokenized_alias) for tokenized_alias in tokenized_aliases)
            global_id_tensor = torch.zeros(num_aliases, max_alias_length, dtype=torch.int64,
                                           requires_grad=False)
            for j, tokenized_alias in enumerate(tokenized_aliases):
                for k, token in enumerate(tokenized_alias):
                    # WARNING: Extremely janky cast to string
                    global_id_tensor[j, k] = vocab.get_token_index(str(token), 'tokens')
            self._global_id_lookup.append(global_id_tensor)

            # Convert array of local alias token indices into a tensor
            local_id_tensor = torch.tensor(self._id_array_lookup[entity], requires_grad=False)  # pylint: disable=not-callable
            self._local_id_lookup.append(local_id_tensor)

        # Build the tensorized token -> potential entities lookup.
        # NOTE: Initial approach will be to store just the necessary info to build one-hot vectors
        # on the fly since storing them will probably be way too expensive.
        token_idx_to_token = vocab.get_index_to_token_vocabulary('tokens')
        for i in range(len(token_idx_to_token)):
            token = token_idx_to_token[i]
            try:
                potential_entities = self._token_to_entity_lookup[token]
            except KeyError:
                self._token_id_to_entity_id_lookup.append(None)
            else:
                potential_entity_ids = torch.tensor([vocab.get_token_index(str(x), 'entity_ids') for x in potential_entities],
                                                    dtype=torch.int64,
                                                    requires_grad=False)
                self._token_id_to_entity_id_lookup.append(potential_entity_ids)
        self._num_entities = vocab.get_vocab_size('entity_ids')  # Needed to get one-hot vector length

        self.is_tensorized = True

        logger.debug('Done tensorizing AliasDatabase')

    def lookup(self, entity_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Looks up alias tokens for the given entities."""
        # Initialize empty tensors and fill them using the lookup
        batch_size, sequence_length = entity_ids.shape
        global_tensor = entity_ids.new_zeros(batch_size, sequence_length, MAX_ALIASES, MAX_TOKENS,
                                             requires_grad=False)
        local_tensor = entity_ids.new_zeros(batch_size, sequence_length, MAX_ALIASES, MAX_TOKENS,
                                            requires_grad=False)
        for i in range(batch_size):
            for j in range(sequence_length):
                entity_id = entity_ids[i, j]
                local_indices = self._local_id_lookup[entity_id]
                global_indices = self._global_id_lookup[entity_id]
                if local_indices is not None:
                    num_aliases, alias_length = local_indices.shape
                    local_tensor[i, j, :num_aliases, :alias_length] = local_indices
                    global_tensor[i, j, :num_aliases, :alias_length] = global_indices

        return global_tensor, local_tensor

    def reverse_lookup(self, tokens: torch.Tensor) -> torch.Tensor:
        """Looks up potential entity matches for the given token."""
        batch_size, sequence_length = tokens.shape
        logger.debug('Performing reverse lookup')
        output = tokens.new_zeros(batch_size, sequence_length, self._num_entities,
                                  dtype=torch.uint8,
                                  requires_grad=False)
        for i in range(batch_size):
            for j in range(sequence_length):
                token_id = tokens[i, j]
                potential_entities = self._token_id_to_entity_id_lookup[token_id]
                output[i, j, potential_entities] = 1
        return output
