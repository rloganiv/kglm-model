"""
Modified vocabulary for computing unknown penalized perplexity. The only difference is that
instead of eliminating tokens that would be mapped to <UNK>, we keep them and modify the indexing
functions to return <UNK>.
"""
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union
from collections import defaultdict

from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import _read_pretrained_tokens, namespace_match
from allennlp.data.vocabulary import Vocabulary


EXTENDED_NON_PADDED_NAMESPACES = ('*tags', '*labels', '*unk')


@Vocabulary.register('extended')
class ExtendedVocabulary(Vocabulary):
    """
    Modified vocabulary for computing unknown penalized perplexity. The only difference is that
    for each namespace, we create an additional "*unk" namespace which stores tokens that
    get mapped to <UNK>.
    """
    def __init__(self,
                 non_padded_namespaces: Iterable[str] = EXTENDED_NON_PADDED_NAMESPACES,
                 *args, **kwargs):
        # The only modification we make to the initialization is to override the default non-padded namespaces.
        super(ExtendedVocabulary, self).__init__(*args, **kwargs, non_padded_namespaces=non_padded_namespaces)

    def _extend(self,
                counter: Dict[str, Dict[str, int]] = None,
                min_count : Dict[str, int] = None,
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

        for namespace in counter:
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
            except KeyError:
                max_vocab = -1
            if max_vocab:
                token_counts = token_counts[:max_vocab]
                unk_counts = token_counts[max_vocab:]  # Add these to the corresponding *unk namespace
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
