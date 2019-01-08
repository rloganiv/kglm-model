"""
Readers for the enhanced Wikitext dataset.
"""
from typing import Any, Dict, Iterable, Set
import json

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import ArrayField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import numpy as np
from overrides import overrides


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
                'input': TextField(tokens[:-1], self._token_indexers),
                'output': TextField(tokens[1:], self._token_indexers)
        }

        # If annotations provided (e.g. during training)
        # TODO: Something smart regarding parents and knowledge graphs...
        if self._enumerate_entities:
            seen_entities: Set[str] = set()

        if 'annotations' in data:

            # Initialize fields.
            # Note: +1 here is to make these fields line up with output.
            # tail_entities = [Token('NA')] * (len(tokens) + 1)
            z = np.zeros(shape=(len(tokens) + 1,))
            if self._enumerate_entities:
                e = np.zeros(shape=(len(tokens) + 1,))
                l = np.zeros(shape=(len(tokens) + 1,))

            # Fill in annotations
            for annotation in data['annotations']:

                if self._enumerate_entities:
                    seen_entities.add(annotation['id'])
                    start, end = annotation['span']
                    length = end - start

                for i in range(*annotation['span']):
                    z[i] = 1
                    if self._enumerate_entities:
                        e[i] = len(seen_entities) - 1
                        l[i] = length
                        length -= 1

            fields['z'] = ArrayField(z[1:])
            if self._enumerate_entities:
                fields['e'] = ArrayField(e[1:])
                fields['l'] = ArrayField(l[1:])

        return Instance(fields)
