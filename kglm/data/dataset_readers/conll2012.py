"""
Readers for the CoNLL 2012 dataset.
"""
import collections
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.fields import Field, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import numpy as np
from overrides import overrides

from kglm.data.fields import SequentialArrayField


def _flatten(nested: Iterable[str]):
    return [x for seq in nested for x in seq]


def canonicalize_clusters(clusters: DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The CONLL 2012 data includes 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]


@DatasetReader.register('conll2012')
class Conll2012DatasetReader(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Parameters
    ----------
    replace_numbers: ``bool``, optional
        Whether to replace numbers with a @@NUM@@ token.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 replace_numbers: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._replace_numbers = replace_numbers
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        ontonotes_reader = Ontonotes()
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens,
                                              end + total_tokens))
                total_tokens += len(sentence.words)

            canonical_clusters = canonicalize_clusters(clusters)
            yield self.text_to_instance([s.words for s in sentences], canonical_clusters)

    @overrides
    def text_to_instance(self,  # type: ignore
                         sentences: List[List[str]],
                         gold_clusters: Optional[List[List[Tuple[int, int]]]] = None) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentences : ``List[List[str]]``, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.

        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            tokens : ``TextField``
                The text of the full document.
            entity_types : ``SequentialArrayField``
                An array with 1's in positions corresponding to words in entities,
                and 0's in positions corresponding to words not in entities.
            entity_ids : ``SequentialArrayField``
                An array with an entity index in positions corresponding to words in
                entities, and 0's in positions corresponding to words not in entities.
                Words in coreferring entities share the same entity ID.
            mention_lengths : ``SequentialArrayField``
                An array with the remaining words in each entity. For words that aren't
                in an entity, the corresponding index is "1". Else, the corresponding
                index has the number of words remaining in the entity. If the entity
                is of length "1", it is assigned "1".
        """
        # Filter gold_clusters: for embedded mentions, only the
        # enclosing (outer) entity mention is kept.
        filtered_gold_clusters = []
        all_entity_spans = [span for gold_cluster in gold_clusters for span in gold_cluster]
        for cluster in gold_clusters:
            filtered_cluster = []
            for span in cluster:
                is_embedded_span = False
                for other_span in all_entity_spans:
                    # Skip if span is equal to other_span
                    if span == other_span:
                        continue
                    if span[0] >= other_span[0] and span[1] <= other_span[1]:
                        # span is embedded within other_span, so don't use it
                        is_embedded_span = True
                        break
                if not is_embedded_span:
                    filtered_cluster.append(span)
            if filtered_cluster:
                filtered_gold_clusters.append(filtered_cluster)

        # Sort the gold clusters, so the earlier-occurring clusters are earlier in the list
        filtered_gold_clusters = sorted(filtered_gold_clusters, key=lambda x: sorted(x)[0][0])

        flattened_sentences = [self._normalize_word(word, self._replace_numbers)
                               for sentence in sentences
                               for word in sentence]
        tokens = ['@@START@@', *flattened_sentences, '@@END@@']
        text_field = TextField([Token(word) for word in tokens],
                               self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        cluster_dict = {}
        if filtered_gold_clusters is not None:
            for cluster_id, cluster in enumerate(filtered_gold_clusters, 1):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        # Initialize fields.
        entity_types = np.zeros(shape=(len(tokens),))
        entity_ids = np.zeros(shape=(len(tokens),))
        mention_lengths = np.ones(shape=(len(tokens),))

        if cluster_dict:
            for cluster, entity_id in cluster_dict.items():
                # Fill in "1" for positions corresponding to words in entities
                # Need offset by 1 to account for @@START@@ token.
                entity_types[cluster[0] + 1:cluster[1] + 1 + 1] = 1
                # Fill in entity ID
                entity_ids[cluster[0] + 1:cluster[1] + 1 + 1] = entity_id
                entity_length = (cluster[1] + 1) - cluster[0]
                # Fill in mention length
                mention_lengths[cluster[0] + 1:cluster[1] + 1 + 1] = np.arange(entity_length, 0, step=-1)

        fields['entity_ids'] = SequentialArrayField(entity_ids, dtype=np.int64)
        fields['mention_lengths'] = SequentialArrayField(mention_lengths, dtype=np.int64)
        fields['entity_types'] = SequentialArrayField(entity_types, dtype=np.uint8)
        return Instance(fields)

    @staticmethod
    def _normalize_word(word, replace_numbers: bool) -> str:
        if word in ["/.", "/?"]:
            return word[1:]
        if replace_numbers:
            if word.replace('.', '', 1).isdigit():
                return "@@NUM@@"
        return word
