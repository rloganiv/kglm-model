from allennlp.common.util import ensure_list
import pytest

from kglm.data.dataset_readers import EnhancedWikitextReader


@pytest.mark.parametrize('lazy', (True, False))
@pytest.mark.parametrize('enumerate_entities', (True, False))
def test_read_from_file(lazy, enumerate_entities):
    reader = EnhancedWikitextReader(lazy=lazy, enumerate_entities=enumerate_entities)
    fixture_path = 'kglm/tests/fixtures/mini.train.jsonl'
    instances = ensure_list(reader.read(fixture_path))

    assert len(instances) == 32
