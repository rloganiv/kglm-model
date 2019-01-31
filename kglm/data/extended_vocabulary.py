"""
Modified vocabulary for computing unknown penalized perplexity. The only difference is that
instead of eliminating tokens that would be mapped to <UNK>, we keep them and modify the indexing
functions to return <UNK>.
"""
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Union
from collections import defaultdict

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.tqdm import Tqdm
from allennlp.data import instance as adi  # pylint: disable=unused-import
from allennlp.data.vocabulary import _read_pretrained_tokens, namespace_match, pop_max_vocab_size
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

logger = logging.getLogger(__name__)


EXTENDED_NON_PADDED_NAMESPACES = ('*tags', '*labels', '*unk')


@Vocabulary.register('extended')
class ExtendedVocabulary(Vocabulary):
    """
    Modified vocabulary for computing unknown penalized perplexity. The only difference is that
    for each namespace, we create an additional "*unk" namespace which stores tokens that
    get mapped to <UNK>.
    """
    def __init__(self,
                 counter: Dict[set, Dict[str, int]] = None,
                 min_count: Dict[str, int] = None,
                 max_vocab_size: Union[int, Dict[str, int]] = None,
                 non_padded_namespaces: Iterable[str] = EXTENDED_NON_PADDED_NAMESPACES,
                 pretrained_files: Optional[Dict[str, str]] = None,
                 only_include_pretrained_words: bool = False,
                 tokens_to_add: Dict[str, List[str]] = None,
                 min_pretrained_embeddings: Dict[str, int] = None) -> None:
        super(ExtendedVocabulary, self).__init__(counter=counter,
                                                 min_count=min_count,
                                                 max_vocab_size=max_vocab_size,
                                                 non_padded_namespaces=non_padded_namespaces,
                                                 pretrained_files=pretrained_files,
                                                 only_include_pretrained_words=only_include_pretrained_words,
                                                 tokens_to_add=tokens_to_add,
                                                 min_pretrained_embeddings=min_pretrained_embeddings)

    def _extend(self,
                counter: Dict[str, Dict[str, int]] = None,
                min_count: Dict[str, int] = None,
                max_vocab_size: Union[int, Dict[str, int]] = None,
                non_padded_namespaces: Iterable[str] = EXTENDED_NON_PADDED_NAMESPACES,
                pretrained_files: Optional[Dict[str, str]] = None,
                only_include_pretrained_words: bool = False,
                tokens_to_add: Dict[str, List[str]] = None,
                min_pretrained_embeddings: Dict[str, int] = None) -> None:
        """
        Modifies the default ``Vocabulary._extend`` method so that tokens which would be
        eliminated are instead added to "*unk" namespaces.
        """
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)  # type: ignore
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        min_pretrained_embeddings = min_pretrained_embeddings or {}
        non_padded_namespaces = set(non_padded_namespaces)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}

        self._retained_counter = counter
        # Make sure vocabulary extension is safe.
        current_namespaces = {*self._token_to_index}
        extension_namespaces = {*counter, *tokens_to_add}

        for namespace in current_namespaces & extension_namespaces:
            # if new namespace was already present
            # Either both should be padded or none should be.
            original_padded = not any(namespace_match(pattern, namespace)
                                      for pattern in self._non_padded_namespaces)
            extension_padded = not any(namespace_match(pattern, namespace)
                                       for pattern in non_padded_namespaces)
            if original_padded != extension_padded:
                raise ConfigurationError("Common namespace {} has conflicting ".format(namespace)+
                                         "setting of padded = True/False. "+
                                         "Hence extension cannot be done.")

        # Add new non-padded namespaces for extension
        self._token_to_index.add_non_padded_namespaces(non_padded_namespaces)
        self._index_to_token.add_non_padded_namespaces(non_padded_namespaces)
        self._non_padded_namespaces.update(non_padded_namespaces)

        for namespace in counter:  # pylint: disable=too-many-nested-blocks
            if namespace in pretrained_files:
                pretrained_list = _read_pretrained_tokens(pretrained_files[namespace])
                min_embeddings = min_pretrained_embeddings.get(namespace, 0)
                if min_embeddings > 0:
                    tokens_old = tokens_to_add.get(namespace, [])
                    tokens_new = pretrained_list[:min_embeddings]
                    tokens_to_add[namespace] = tokens_old + tokens_new
                pretrained_set = set(pretrained_list)
            else:
                pretrained_set = set()
            token_counts = list(counter[namespace].items())
            token_counts.sort(key=lambda x: x[1], reverse=True)
            try:
                max_vocab = max_vocab_size[namespace]
                if max_vocab is not None:
                    unk_counts = token_counts[max_vocab:]  # Add these to *unk namespace
                    token_counts = token_counts[:max_vocab]
                else:
                    unk_counts = []
            except KeyError:
                unk_counts = []
            for token, count in token_counts:
                if pretrained_set is not None:
                    if only_include_pretrained_words:
                        if token in pretrained_set:
                            if count >= min_count.get(namespace, 1):
                                self.add_token_to_namespace(token, namespace)
                            else:
                                self.add_token_to_namespace(token, namespace + '_unk')
                    elif token in pretrained_set or count >= min_count.get(namespace, 1):
                        self.add_token_to_namespace(token, namespace)
                    else:
                        self.add_token_to_namespace(token, namespace + '_unk')
                elif count >= min_count.get(namespace, 1):
                    self.add_token_to_namespace(token, namespace)
                else:
                    self.add_token_to_namespace(token, namespace + '_unk')
            for token, count in unk_counts:
                self.add_token_to_namespace(token, namespace + '_unk')

        for namespace, tokens in tokens_to_add.items():
            for token in tokens:
                self.add_token_to_namespace(token, namespace)

    @classmethod
    @overrides
    def from_instances(cls,
                       instances: Iterable['adi.Instance'],
                       min_count: Dict[str, int] = None,
                       max_vocab_size: Union[int, Dict[str, int]] = None,
                       non_padded_namespaces: Iterable[str] = EXTENDED_NON_PADDED_NAMESPACES,
                       pretrained_files: Optional[Dict[str, str]] = None,
                       only_include_pretrained_words: bool = False,
                       tokens_to_add: Dict[str, List[str]] = None,
                       min_pretrained_embeddings: Dict[str, int] = None) -> 'ExtendedVocabulary':
        """
        Constructs a vocabulary given a collection of `Instances` and some parameters.
        We count all of the vocabulary items in the instances, then pass those counts
        and the other parameters, to :func:`__init__`.  See that method for a description
        of what the other parameters do.
        """
        logger.info("Fitting token dictionary from dataset.")
        namespace_token_counts: Dict[Set[Any], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for instance in Tqdm.tqdm(instances):
            instance.count_vocab_items(namespace_token_counts)

        return cls(counter=namespace_token_counts,
                   min_count=min_count,
                   max_vocab_size=max_vocab_size,
                   non_padded_namespaces=non_padded_namespaces,
                   pretrained_files=pretrained_files,
                   only_include_pretrained_words=only_include_pretrained_words,
                   tokens_to_add=tokens_to_add,
                   min_pretrained_embeddings=min_pretrained_embeddings)

    # There's enough logic here to require a custom from_params.
    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):  # type: ignore
        """
        There are two possible ways to build a vocabulary; from a
        collection of instances, using :func:`Vocabulary.from_instances`, or
        from a pre-saved vocabulary, using :func:`Vocabulary.from_files`.
        You can also extend pre-saved vocabulary with collection of instances
        using this method. This method wraps these options, allowing their
        specification from a ``Params`` object, generated from a JSON
        configuration file.
        Parameters
        ----------
        params: Params, required.
        instances: Iterable['adi.Instance'], optional
            If ``params`` doesn't contain a ``directory_path`` key,
            the ``Vocabulary`` can be built directly from a collection of
            instances (i.e. a dataset). If ``extend`` key is set False,
            dataset instances will be ignored and final vocabulary will be
            one loaded from ``directory_path``. If ``extend`` key is set True,
            dataset instances will be used to extend the vocabulary loaded
            from ``directory_path`` and that will be final vocabulary used.
        Returns
        -------
        A ``Vocabulary``.
        """
        # pylint: disable=arguments-differ
        # Vocabulary is ``Registrable`` so that you can configure a custom subclass,
        # but (unlike most of our registrables) almost everyone will want to use the
        # base implementation. So instead of having an abstract ``VocabularyBase`` or
        # such, we just add the logic for instantiating a registered subclass here,
        # so that most users can continue doing what they were doing.
        vocab_type = params.pop("type", None)
        if vocab_type is not None:
            return cls.by_name(vocab_type).from_params(params=params, instances=instances)

        extend = params.pop("extend", False)
        vocabulary_directory = params.pop("directory_path", None)
        if not vocabulary_directory and not instances:
            raise ConfigurationError("You must provide either a Params object containing a "
                                     "vocab_directory key or a Dataset to build a vocabulary from.")
        if extend and not instances:
            raise ConfigurationError("'extend' is true but there are not instances passed to extend.")
        if extend and not vocabulary_directory:
            raise ConfigurationError("'extend' is true but there is not 'directory_path' to extend from.")

        if vocabulary_directory and instances:
            if extend:
                logger.info("Loading Vocab from files and extending it with dataset.")
            else:
                logger.info("Loading Vocab from files instead of dataset.")

        if vocabulary_directory:
            vocab = Vocabulary.from_files(vocabulary_directory)
            if not extend:
                params.assert_empty("Vocabulary - from files")
                return vocab
        if extend:
            vocab.extend_from_instances(params, instances=instances)
            return vocab
        min_count = params.pop("min_count", None)
        max_vocab_size = pop_max_vocab_size(params)
        non_padded_namespaces = params.pop("non_padded_namespaces", EXTENDED_NON_PADDED_NAMESPACES)
        pretrained_files = params.pop("pretrained_files", {})
        min_pretrained_embeddings = params.pop("min_pretrained_embeddings", None)
        only_include_pretrained_words = params.pop_bool("only_include_pretrained_words", False)
        tokens_to_add = params.pop("tokens_to_add", None)
        params.assert_empty("Vocabulary - from dataset")
        return ExtendedVocabulary.from_instances(instances=instances,
                                                 min_count=min_count,
                                                 max_vocab_size=max_vocab_size,
                                                 non_padded_namespaces=non_padded_namespaces,
                                                 pretrained_files=pretrained_files,
                                                 only_include_pretrained_words=only_include_pretrained_words,
                                                 tokens_to_add=tokens_to_add,
                                                 min_pretrained_embeddings=min_pretrained_embeddings)
