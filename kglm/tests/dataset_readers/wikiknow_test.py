from allennlp.common.util import ensure_list
import numpy as np
import pytest

from kglm.data.dataset_readers import EnhancedWikitextReader


@pytest.mark.parametrize('lazy', (True, False))
@pytest.mark.parametrize('enumerate_entities', (True, False))
def test_read_from_file(lazy, enumerate_entities):
    reader = EnhancedWikitextReader(lazy=lazy, enumerate_entities=enumerate_entities)
    fixture_path = 'kglm/tests/fixtures/mini.train.jsonl'
    instances = ensure_list(reader.read(fixture_path))

    first_instance_tokens = [x.text for x in instances[0]["tokens"].tokens]
    assert first_instance_tokens[:5] == ['@@START@@', 'Aage', 'Niels', 'Bohr', '(']
    assert first_instance_tokens[-5:] == ['wife', 'and', 'children', '.', '@@END@@']
    first_instance_entity_types = instances[0]["entity_types"].array
    if enumerate_entities:
        np.testing.assert_allclose(first_instance_entity_types[:5], [0, 1, 1, 1, 0])
        np.testing.assert_allclose(first_instance_entity_types[-5:], [0, 0, 0, 0, 0])
        np.testing.assert_allclose(instances[0]["entity_ids"].array[:5], [0, 1, 1, 1, 0])
        np.testing.assert_allclose(instances[0]["entity_ids"].array[-5:], [0, 0, 0, 0, 0])
        np.testing.assert_allclose(instances[0]["mention_lengths"].array[:5],
                                   [1, 3, 2, 1, 1])
        np.testing.assert_allclose(instances[0]["mention_lengths"].array[-5:],
                                   [1, 1, 1, 1, 1])
    else:
        np.testing.assert_allclose(first_instance_entity_types[:5], np.zeros(5))
        np.testing.assert_allclose(first_instance_entity_types[-5:], np.zeros(5))
        assert "entity_ids" not in instances[0]
        assert "mention_lengths" not in instances[0]
    assert len(instances) == 32
