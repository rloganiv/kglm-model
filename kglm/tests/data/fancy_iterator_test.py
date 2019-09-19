# pylint: disable=no-self-use
from typing import List

from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from kglm.data.iterators import FancyIterator


class FancyIteratorTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.vocab = Vocabulary()
        self.this_index = self.vocab.add_token_to_namespace('this')
        self.is_index = self.vocab.add_token_to_namespace('is')
        self.a_index = self.vocab.add_token_to_namespace('a')
        self.sentence_index = self.vocab.add_token_to_namespace('sentence')
        self.another_index = self.vocab.add_token_to_namespace('another')
        self.yet_index = self.vocab.add_token_to_namespace('yet')
        self.very_index = self.vocab.add_token_to_namespace('very')
        self.long_index = self.vocab.add_token_to_namespace('long')
        instances = [
                self.create_instance(["this", "is", "a", "sentence"]),
                self.create_instance(["this", "is", "another", "sentence"]),
                self.create_instance(["yet", "another", "sentence"]),
                self.create_instance(["this", "is", "a", "very", "very", "very", "very", "long", "sentence"]),
                self.create_instance(["sentence"]),
                ]

        self.instances = instances

    def create_instance(self, str_tokens: List[str]):
        tokens = [Token(t) for t in str_tokens]
        instance = Instance({'source': TextField(tokens, self.token_indexers)})
        return instance

    def test_truncate(self):
        # Checks that the truncate parameter works as intended.

        # Since split size is less than the length of the "very ... very long" sentence, the
        # iterator should return one batch when the truncation is enabled.
        split_size = 4
        truncated_iterator = FancyIterator(batch_size=5,
                                           split_size=split_size,
                                           splitting_keys=['source'],
                                           truncate=True)
        truncated_iterator.index_with(self.vocab)
        batches = list(truncated_iterator(self.instances, num_epochs=1))
        assert len(batches) == 1

        # When truncation is disabled the iterator should return 3 batches instead.
        non_truncated_iterator = FancyIterator(batch_size=5,
                                               split_size=split_size,
                                               splitting_keys=['source'],
                                               truncate=False)
        non_truncated_iterator.index_with(self.vocab)
        batches = list(non_truncated_iterator(self.instances, num_epochs=1))
        assert len(batches) == 3

        # When the batch size is larger than the number of instances, truncation will the iterator
        # to return zero batches of data (since some of the instances in the batch would consist
        # entirely of padding). Check that the iterator raises an error in this case.
        invalid_iterator = FancyIterator(batch_size=6,
                                         split_size=split_size,
                                         splitting_keys=['source'],
                                         truncate=True)
        invalid_iterator.index_with(self.vocab)
        with self.assertRaises(ConfigurationError):
            batches = list(invalid_iterator(self.instances, num_epochs=1))

        # If truncation is disabled then this should not cause an issue
        valid_iterator = FancyIterator(batch_size=6,
                                       split_size=split_size,
                                       splitting_keys=['source'],
                                       truncate=False)
        valid_iterator.index_with(self.vocab)
        batches = list(valid_iterator(self.instances, num_epochs=1))
        assert len(batches) == 3
