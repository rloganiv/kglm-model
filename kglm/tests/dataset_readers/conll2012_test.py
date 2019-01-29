from allennlp.common.util import ensure_list
import numpy as np
import pytest

from kglm.data.dataset_readers import Conll2012DatasetReader


class TestConll2012DatasetReader:
    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('enumerate_entities', (True, False))
    def test_read_from_file(self, lazy, enumerate_entities):
        reader = Conll2012DatasetReader(lazy=lazy, enumerate_entities=enumerate_entities)
        fixture_path = 'kglm/tests/fixtures/coref.gold_conll'
        instances = ensure_list(reader.read(fixture_path))
        assert len(instances) == 2

        first_instance_tokens = [x.text for x in instances[0]["tokens"].tokens]
        assert first_instance_tokens == [
            '@@START@@', 'In', 'the', 'summer', 'of', '@@NUM@@', ',', 'a', 'picture', 'that',
            'people', 'have', 'long', 'been', 'looking', 'forward', 'to',
            'started', 'emerging', 'with', 'frequency', 'in', 'various', 'major',
            'Hong', 'Kong', 'media', '.', 'With', 'their', 'unique', 'charm', ',',
            'these', 'well', '-', 'known', 'cartoon', 'images', 'once', 'again',
            'caused', 'Hong', 'Kong', 'to', 'be', 'a', 'focus', 'of', 'worldwide',
            'attention', '.', 'The', 'world', "'s", 'fifth', 'Disney', 'park',
            'will', 'soon', 'open', 'to', 'the', 'public', 'here', '.', '@@END@@']
        # {(41, 42): 1, (23, 24): 1, (28, 28): 2, (32, 37): 2}
        first_instance_entity_types = instances[0]["entity_types"].array
        if enumerate_entities:
            # Add 1 to both indices to account for @@START@@
            # Add 1 more to the end to work with slicing
            entity_indices = {(41+1, 42+1+1), (23+1, 24+1+1),
                              (28+1, 28+1+1), (32+1, 37+1+1)}
            assert first_instance_tokens[23+1:24+1+1] == ['Hong', 'Kong']
            assert first_instance_tokens[28+1:28+1+1] == ['their']
            assert first_instance_tokens[32+1:37+1+1] == ['these', 'well', '-',
                                                          'known', 'cartoon', 'images']
            assert first_instance_tokens[41+1:42+1+1] == ['Hong', 'Kong']
            np.testing.assert_allclose(first_instance_entity_types,
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                                        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                                        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            np.testing.assert_allclose(instances[0]["entity_ids"].array,
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2,
                                        0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0])
            np.testing.assert_allclose(instances[0]["mention_lengths"].array,
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1,
                                        1, 1, 1, 6, 5, 4, 3, 2, 1, 1, 1, 1, 2, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1])
        else:
            np.testing.assert_allclose(first_instance_entity_types, np.zeros(67))
            assert "entity_ids" not in instances[0]
            assert "mention_lengths" not in instances[0]
