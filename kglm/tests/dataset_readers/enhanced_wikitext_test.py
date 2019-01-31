from allennlp.common.util import ensure_list
import numpy as np
import pytest

from kglm.data.dataset_readers import EnhancedWikitextEntityNlmReader


@pytest.mark.parametrize('lazy', (True, False))
def test_read_from_file(lazy):
    reader = EnhancedWikitextEntityNlmReader(lazy=lazy)
    fixture_path = 'kglm/tests/fixtures/enhanced-wikitext.jsonl'
    instances = ensure_list(reader.read(fixture_path))

    first_instance_tokens = [x.text for x in instances[0]["tokens"].tokens]
    assert first_instance_tokens[:5] == ['@@START@@', 'State', 'Route', '127', '(']
    assert first_instance_tokens[-5:] == ['Elmer', 'Huntley', 'Bridge', '.', '@@END@@']
    second_instance_entity_types = instances[1]["entity_types"].array
    np.testing.assert_allclose(second_instance_entity_types[:5], [0, 0, 1, 1, 1])
    np.testing.assert_allclose(second_instance_entity_types[-5:], [0, 0, 0, 0, 0])
    np.testing.assert_allclose(instances[1]["entity_ids"].array[:5], [0, 0, 1, 1, 1])
    np.testing.assert_allclose(instances[1]["entity_ids"].array[-5:], [0, 0, 0, 0, 0])
    np.testing.assert_allclose(instances[1]["mention_lengths"].array[:5],
                               [1, 1, 5, 4, 3])
    np.testing.assert_allclose(instances[1]["mention_lengths"].array[-5:],
                               [1, 1, 1, 1, 1])
