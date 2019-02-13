"""
Readers for the enhanced Wikitext dataset.
"""
from typing import Any, Dict, Iterable, List, Set
import json
import logging

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
import numpy as np
from overrides import overrides

from kglm.data import AliasDatabase
from kglm.data.fields import SequentialArrayField

logger = logging.getLogger(__name__)


def _flatten(nested: Iterable[str]):
    return [x for seq in nested for x in seq]


@DatasetReader.register('enhanced-wikitext')
class EnhancedWikitextReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        super().__init__(lazy)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                yield self.text_to_instance(data)

    @overrides
    def text_to_instance(self, data: Dict[str, Any]) -> Instance:  # pylint: disable=arguments-differ
        # Flatten and pad tokens
        tokens = [x + ['@@END@@'] for x in data['tokens']]
        tokens = _flatten(tokens)
        tokens = [Token(x) for x in tokens]
        # source = tokens[:-1]
        # target = tokens[1:]
        # fields = {
        #         'source': TextField(source, self._token_indexers),
        #         'target': TextField(target, self._token_indexers)
        # }
        fields = {'tokens': TextField(tokens, self._token_indexers)}
        return Instance(fields)


@DatasetReader.register('enhanced-wikitext-entity-nlm')
class EnhancedWikitextEntityNlmReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        super().__init__(lazy)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

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
        source = [Token(x) for x in tokens[:-1]]
        target = [Token(x) for x in tokens[1:]]
        fields = {
            'source': TextField(source, self._token_indexers),
            'target': TextField(target, self._token_indexers)
        }
        meta_fields = {
            'tokens': tokens,
            'alias_database': self._alias_database
        }

        # Process annotations
        if 'annotations' in data:

            # We maintain a "shortlist" of observed entities, that is used for baseline models
            # that only select entities from the set that appear in the document (as opposed to
            # the set of all possible entities).
            shortlist = [DEFAULT_PADDING_TOKEN]
            reverse_shortlist = {DEFAULT_PADDING_TOKEN: 0}

            entity_ids = [DEFAULT_PADDING_TOKEN] * len(target)
            shortlist_inds = np.zeros(shape=(len(target,)))
            alias_copy_inds = np.zeros(shape=(len(target),))

            # Process annotations
            for annotation in data['annotations']:

                # Obtain the entity identifier for the annotated span
                entity_id = annotation['id']

                # If neccessary, update the shortlist. Obtain the index of the entity identifier in
                # the shortlist.
                if entity_id not in reverse_shortlist:
                    reverse_shortlist[entity_id] = len(reverse_shortlist)
                    shortlist.append(entity_id)
                shortlist_ind = reverse_shortlist[entity_id]

                # Update the outputs
                for i in range(*annotation['span']):
                    # Note: +1 offset to account for start token.
                    entity_ids[i] = entity_id
                    alias_copy_inds[i] = self._alias_database.token_to_uid(entity_id, tokens[i+1])
                    shortlist_inds[i] = shortlist_ind

            # Convert to fields
            fields['entity_ids'] = TextField(
                [Token(x) for x in entity_ids],
                token_indexers=self._entity_indexers)
            fields['alias_copy_inds'] = SequentialArrayField(
                alias_copy_inds,
                dtype=np.int64)
            fields['shortlist'] = TextField(
                [Token(x) for x in shortlist],
                token_indexers=self._entity_indexers)
            fields['shortlist_inds'] = SequentialArrayField(
                shortlist_inds,
                dtype=np.int64)
            # meta_fields['entity_ids'] = entity_ids

        fields['metadata'] = MetadataField(meta_fields)

        return Instance(fields)


@DatasetReader.register('enhanced-wikitext-simple-kglm')
class EnhancedWikitextSimpleKglmReader(DatasetReader):

    def __init__(self,
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
        source = [Token(x) for x in tokens[:-1]]
        target = [Token(x) for x in tokens[1:]]
        fields = {
            'source': TextField(source, self._token_indexers),
            'target': TextField(target, self._token_indexers)
        }

        # Process annotations
        if 'annotations' in data:

            # We maintain a "shortlist" of observed entities, that is used for baseline models
            # that only select entities from the set that appear in the document (as opposed to
            # the set of all possible entities).
            shortlist = [DEFAULT_PADDING_TOKEN]
            reverse_shortlist = {DEFAULT_PADDING_TOKEN: 0}

            entity_ids = [DEFAULT_PADDING_TOKEN] * len(target)
            shortlist_inds = np.zeros(shape=(len(target,)))
            alias_copy_inds = np.zeros(shape=(len(target),))
            alias_tokens = [TextField([], self._token_indexers)] * len(target)
            alias_inds: List[List[int]] = [[]] * len(target)
            max_len = 0

            # Process annotations
            for annotation in data['annotations']:

                # Obtain the entity identifier for the annotated span
                entity_id = annotation['id']
                alias = annotation['alias']
                alias_map = {token: i+1 for i, token in enumerate(set(alias))}

                # If neccessary, update the shortlist. Obtain the index of the entity identifier in
                # the shortlist.
                if entity_id not in reverse_shortlist:
                    reverse_shortlist[entity_id] = len(reverse_shortlist)
                    shortlist.append(entity_id)
                shortlist_ind = reverse_shortlist[entity_id]

                # Update the outputs
                for i in range(*annotation['span']):
                    # Note: +1 offset to account for start token.
                    if tokens[i+1] not in alias_map:
                        continue
                    else:
                        entity_ids[i] = entity_id
                        shortlist_inds[i] = shortlist_ind
                        alias_copy_inds[i] = alias_map[tokens[i+1]]
                        alias_inds[i] = [alias_map[token] for token in alias]
                        alias_tokens[i] = TextField([Token(x) for x in alias],
                                                    self._token_indexers)
                        max_len = max(max_len, len(alias))

            # Make alias_inds into a numpy array
            alias_ind_array = np.zeros((len(target), max_len))
            for i, arr in enumerate(alias_inds):
                for j, ind in enumerate(arr):
                    alias_ind_array[i, j] = ind

            fields['entity_ids'] = TextField(
                [Token(x) for x in entity_ids],
                token_indexers=self._entity_indexers)
            fields['alias_copy_inds'] = SequentialArrayField(
                alias_copy_inds,
                dtype=np.int64)
            fields['shortlist'] = TextField(
                [Token(x) for x in shortlist],
                token_indexers=self._entity_indexers)
            fields['shortlist_inds'] = SequentialArrayField(
                shortlist_inds,
                dtype=np.int64)
            fields['alias_tokens'] = ListField(alias_tokens)
            fields['alias_inds'] = SequentialArrayField(
                alias_ind_array,
                dtype=np.int64)

        return Instance(fields)
