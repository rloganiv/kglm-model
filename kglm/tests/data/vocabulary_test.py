from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from kglm.data.extended_vocabulary import ExtendedVocabulary


class TestVocabulary(AllenNlpTestCase):
    # pylint: disable=no-self-use, invalid-name, too-many-public-methods, protected-access

    def setUp(self):
        token_indexer = SingleIdTokenIndexer("tokens")
        text_field = TextField([Token(t) for t in ["a", "a", "a", "a", "b", "b", "c", "c", "c"]],
                               {"tokens": token_indexer})
        self.instance = Instance({"text": text_field})
        self.dataset = Batch([self.instance])
        super(TestVocabulary, self).setUp()

    def test_unk_namespace_is_empty_if_vocab_unconstrained(self):
        vocab = ExtendedVocabulary.from_instances(self.dataset)
        words = vocab.get_index_to_token_vocabulary('tokens_unk')
        assert not words  # This checks that there's nothing in ``words`` w/out pylint complaining

    def test_from_dataset_respects_max_vocab_size_single_int(self):
        max_vocab_size = 1
        vocab = ExtendedVocabulary.from_instances(self.dataset, max_vocab_size=max_vocab_size)
        words = vocab.get_index_to_token_vocabulary().values()
        # Additional 2 tokens are '@@PADDING@@' and '@@UNKNOWN@@' by default
        assert len(words) == max_vocab_size + 2

        vocab = ExtendedVocabulary.from_instances(self.dataset, min_count=None)
        words = vocab.get_index_to_token_vocabulary().values()
        assert len(words) == 5

    def test_from_dataset_respects_min_count(self):
        vocab = ExtendedVocabulary.from_instances(self.dataset, min_count={'tokens': 4})
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' not in words
        assert 'c' not in words

        vocab = ExtendedVocabulary.from_instances(self.dataset, min_count=None)
        words = vocab.get_index_to_token_vocabulary().values()
        assert 'a' in words
        assert 'b' in words
        assert 'c' in words

    def test_saving_and_loading(self):
        # pylint: disable=protected-access
        vocab_dir = self.TEST_DIR / 'vocab_save'

        vocab = ExtendedVocabulary(non_padded_namespaces=["a", "c"])
        vocab.add_token_to_namespace("a0", namespace="a")  # non-padded, should start at 0
        vocab.add_token_to_namespace("a1", namespace="a")
        vocab.add_token_to_namespace("a2", namespace="a")
        vocab.add_token_to_namespace("b2", namespace="b")  # padded, should start at 2
        vocab.add_token_to_namespace("b3", namespace="b")

        vocab.save_to_files(vocab_dir)
        vocab2 = ExtendedVocabulary.from_files(vocab_dir)

        assert vocab2._non_padded_namespaces == {"a", "c"}

        # Check namespace a.
        assert vocab2.get_vocab_size(namespace='a') == 3
        assert vocab2.get_token_from_index(0, namespace='a') == 'a0'
        assert vocab2.get_token_from_index(1, namespace='a') == 'a1'
        assert vocab2.get_token_from_index(2, namespace='a') == 'a2'
        assert vocab2.get_token_index('a0', namespace='a') == 0
        assert vocab2.get_token_index('a1', namespace='a') == 1
        assert vocab2.get_token_index('a2', namespace='a') == 2

        # Check namespace b.
        assert vocab2.get_vocab_size(namespace='b') == 4  # (unk + padding + two tokens)
        assert vocab2.get_token_from_index(0, namespace='b') == vocab._padding_token
        assert vocab2.get_token_from_index(1, namespace='b') == vocab._oov_token
        assert vocab2.get_token_from_index(2, namespace='b') == 'b2'
        assert vocab2.get_token_from_index(3, namespace='b') == 'b3'
        assert vocab2.get_token_index(vocab._padding_token, namespace='b') == 0
        assert vocab2.get_token_index(vocab._oov_token, namespace='b') == 1
        assert vocab2.get_token_index('b2', namespace='b') == 2
        assert vocab2.get_token_index('b3', namespace='b') == 3

        # Check the dictionaries containing the reverse mapping are identical.
        assert vocab.get_index_to_token_vocabulary("a") == vocab2.get_index_to_token_vocabulary("a")
        assert vocab.get_index_to_token_vocabulary("b") == vocab2.get_index_to_token_vocabulary("b")
