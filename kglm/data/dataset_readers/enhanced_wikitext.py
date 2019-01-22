"""
Readers for the enhanced Wikitext dataset.
"""
from typing import Any, Dict, Iterable, Set
import json
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import numpy as np
from overrides import overrides

from kglm.data import AliasDatabase
from kglm.data.fields import SequentialArrayField

logger = logging.getLogger(__name__)


def _flatten(nested: Iterable[str]):
    return [x for seq in nested for x in seq]


@DatasetReader.register('enhanced-wikitext-entity-nlm')
class EnhancedWikitextEntityNlmReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 lazy: bool = False) -> None:
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        super().__init__(lazy)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                yield self.text_to_instance(data)

    @overrides
    def text_to_instance(self, data: Dict[str, Any]) -> Instance:  # pylint: disable=arguments-differ
        # Flatten and pad tokens
        tokens = _flatten(data['tokens'])
        tokens = ['@@START@@', *tokens, '@@END@@']
        tokens = [Token(x) for x in tokens]
        fields = {'tokens': TextField(tokens, self._token_indexers)}

        # If annotations are provided, process them into arrays.
        if 'annotations' in data:

            # Initialize arrays and book keeping data structures.
            seen_entities: Set[str] = set()
            entity_types = np.zeros(shape=(len(tokens),))
            if self._enumerate_entities:
                entity_ids = np.zeros(shape=(len(tokens),))
                mention_lengths = np.ones(shape=(len(tokens),))

            # Process annotations
            for annotation in data['annotations']:

                seen_entities.add(annotation['id'])
                start, end = annotation['span']
                length = end - start

                for i in range(*annotation['span']):
                    # Note: +1 offset to account for start token.
                    entity_types[i+1] = 1
                    entity_ids[i+1] = len(seen_entities)
                    mention_lengths[i+1] = length
                    length -= 1

            fields['entity_types'] = SequentialArrayField(entity_types, dtype=np.uint8)
            fields['entity_ids'] = SequentialArrayField(entity_ids, dtype=np.int64)
            fields['mention_lengths'] = SequentialArrayField(mention_lengths, dtype=np.int64)

        return Instance(fields)


@DatasetReader.register('enhanced-wikitext-kglm')
class EnhancedWikitextKglmReader(DatasetReader):

    def __init__(self,
                 alias_database_path: str,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 entity_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        """
        Parameters
        ----------
        alias_database_path : str
            Path to the alias database.
        """
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._entity_indexers = entity_indexers or {'entity_ids': SingleIdTokenIndexer(namespace='entity_ids')}
        if 'tokens' not in self._token_indexers or \
                not isinstance(self._token_indexers['tokens'], SingleIdTokenIndexer):
            raise ConfigurationError("EnhancedWikitextReader expects 'token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")
        if 'entity_ids' not in self._entity_indexers or \
                not isinstance(self._entity_indexers['entity_ids'], SingleIdTokenIndexer):
            raise ConfigurationError("EnhancedWikitextReader expects 'entity_indexers' to contain "
                                     "a 'single_id' token indexer called 'entities'.")
        self._alias_database = AliasDatabase.load(path=alias_database_path)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                yield self.text_to_instance(data)

    @overrides
    def text_to_instance(self, data: Dict[str, Any]) -> Instance:  # pylint: disable=arguments-differ
        # Flatten and pad tokens
        tokens = _flatten(data['tokens'])
        tokens = ['@@START@@', *tokens, '@@END@@']
        fields = {
                'tokens': TextField([Token(x) for x in tokens], self._token_indexers),
        }
        meta_fields = {
                'tokens': tokens,
                'alias_database': self._alias_database
        }

        # Process annotations
        if 'annotations' in data:

            entity_types = np.zeros(shape=(len(tokens),))
            entity_ids = ['@@NULL@@'] * len(tokens)
            alias_ids = np.zeros(shape=(len(tokens),))

            # TODO: This is just to make life easier when testing the copy mechanism - eventually
            # it should be removed.
            entity_id_set = set()

            # Process annotations
            for annotation in data['annotations']:

                entity_id = annotation['id']
                entity_id_set.add(entity_id)

                for i in range(*annotation['span']):
                    # Note: +1 offset to account for start token.
                    entity_types[i+1] = 1
                    entity_ids[i+1] = entity_id
                    alias_ids[i+1] = self._alias_database.token_to_uid(entity_id, tokens[i+1])

            entity_shortlist = list(entity_id_set)
            shortlist_map = {entity_id: target for target, entity_id in enumerate(entity_shortlist)}
            entity_shortlist_ids = np.array([shortlist_map[entity_id] if entity_id in shortlist_map else -1 for entity_id in entity_ids])

            fields['entity_types'] = SequentialArrayField(entity_types, dtype=np.uint8)
            fields['entity_ids'] = TextField([Token(x) for x in entity_ids],
                                             token_indexers=self._entity_indexers)
            fields['alias_ids'] = SequentialArrayField(alias_ids, dtype=np.int64)
            fields['entity_shortlist'] = TextField([Token(x) for x in entity_shortlist],
                                                   token_indexers=self._entity_indexers)
            fields['entity_shortlist_ids'] = SequentialArrayField(entity_shortlist_ids, dtype=np.int64)
            meta_fields['entity_ids'] = entity_ids

        fields['metadata'] = MetadataField(meta_fields)

        return Instance(fields)
