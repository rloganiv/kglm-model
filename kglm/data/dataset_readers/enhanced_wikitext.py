"""
Readers for the enhanced Wikitext dataset.
"""
from typing import Any, Dict, Iterable, Set
import json

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import numpy as np
from overrides import overrides

from kglm.data.fields import SequentialArrayField


def _flatten(nested: Iterable[str]):
    return [x for seq in nested for x in seq]


@DatasetReader.register('enhanced-wikitext')
class EnhancedWikitextReader(DatasetReader):

    def __init__(self,
                 enumerate_entities: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 field_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._enumerate_entities = enumerate_entities
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._claim_indexers = field_indexers or {"claim_tokens": SingleIdTokenIndexer()}

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
        fields = {
                'tokens': TextField(tokens, self._token_indexers),
        }

        # If annotations provided (e.g. during training)
        # TODO: Something smart regarding parents and knowledge graphs...
        if self._enumerate_entities:
            seen_entities: Set[str] = set()

        if 'annotations' in data:

            # Initialize fields.
            entity_types = np.zeros(shape=(len(tokens),))
            if self._enumerate_entities:
                entity_ids = np.zeros(shape=(len(tokens),))
                mention_lengths = np.zeros(shape=(len(tokens),))

            # Fill in annotations
            for annotation in data['annotations']:

                if self._enumerate_entities:
                    seen_entities.add(annotation['id'])
                    start, end = annotation['span']
                    length = end - start

                    for i in range(*annotation['span']):
                        # Note: +1 offset to account for start token.
                        entity_types[i+1] = 1
                        if self._enumerate_entities:
                            entity_ids[i+1] = len(seen_entities)
                            mention_lengths[i+1] = length
                            length -= 1

            fields['entity_types'] = SequentialArrayField(entity_types, dtype=np.uint8)
            if self._enumerate_entities:
                fields['entity_ids'] = SequentialArrayField(entity_ids, dtype=np.int64)
                fields['mention_lengths'] = SequentialArrayField(mention_lengths, dtype=np.int64)

        return Instance(fields)
