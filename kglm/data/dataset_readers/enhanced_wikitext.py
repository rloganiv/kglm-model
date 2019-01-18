"""
Readers for the enhanced Wikitext dataset.
"""
from typing import Any, Dict, Iterable, Set
import json
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import numpy as np
from overrides import overrides

from kglm.data import AliasDatabase
from kglm.data.fields import GlobalObject, SequentialArrayField

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
        self._entity_indexers = entity_indexers or {'entities': SingleIdTokenIndexer(namespace='entities')}
        if 'tokens' not in self._token_indexers or \
                not isinstance(self._token_indexers['tokens'], SingleIdTokenIndexer):
            raise ConfigurationError("EnhancedWikitextReader expects 'token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")
        if 'entities' not in self._entity_indexers or \
                not isinstance(self._entity_indexers['entities'], SingleIdTokenIndexer):
            raise ConfigurationError("EnhancedWikitextReader expects 'entity_indexers' to contain "
                                     "a 'single_id' token indexer called 'entities'.")
        self._alias_database = AliasDatabase(path=alias_database_path,
                                             token_indexer=self._token_indexers['tokens'],
                                             entity_indexer=self._token_indexers['entities'])

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # Read aliases in first
        self._alias_database.read()

        # Then start reading in the data
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
        fields = {
                'tokens': TextField(tokens, self._token_indexers),
                'alias_database': GlobalObject(self._alias_database)
        }

        # TODO: Annotation processing logic

        return Instance(fields)
