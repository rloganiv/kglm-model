from allennlp.common.util import ensure_list
import numpy as np
import pytest

from kglm.data.dataset_readers import (
    EnhancedWikitextEntityNlmReader,
    EnhancedWikitextKglmReader)


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


class TestEnhancedWikitextKglmReader:
    @pytest.mark.parametrize('lazy', (True, False))
    def test_read_from_file(self, lazy):
        alias_database_path = 'kglm/tests/fixtures/mini.alias.pkl'
        reader = EnhancedWikitextKglmReader(lazy=lazy,
                                            alias_database_path=alias_database_path)
        fixture_path = 'kglm/tests/fixtures/enhanced-wikitext.jsonl'
        instances = ensure_list(reader.read(fixture_path))

        # Test correct number of instances is being created
        assert len(instances) == 3

        # Test article tokens are being read properly
        first_instance_source_tokens = [x.text for x in instances[0]['source'].tokens]
        assert first_instance_source_tokens[:5] == ['@@START@@', 'State', 'Route', '127', '(']
        assert first_instance_source_tokens[-5:] == ['the', 'Elmer', 'Huntley', 'Bridge', '.']

        first_instance_target_tokens = [x.text for x in instances[0]['target'].tokens]
        assert first_instance_target_tokens[:5] == ['State', 'Route', '127', '(', 'SR']
        assert first_instance_target_tokens[-5:] == ['Elmer', 'Huntley', 'Bridge', '.', '@@END@@']

        # Test new entity mask is being generated properly
        # Non-mention tokens are not new entities
        first_instance_new_entity_mask = instances[0]['new_entity_mask'].array
        assert first_instance_new_entity_mask[0] == 0
        # "state highway" is a new entity mention
        assert first_instance_new_entity_mask[16] == 1
        # "Washington" is not since it has parents in the KG
        assert first_instance_new_entity_mask[27] == 0

        # Test entity id
        first_instance_entity_ids = [x.text for x in instances[0]['entity_ids'].tokens]
        # Non-mentions correspond to padding tokens
        assert first_instance_entity_ids[7:9] == ['@@PADDING@@', '@@PADDING@@']
        assert first_instance_entity_ids[7:9] == ['@@PADDING@@', '@@PADDING@@']
        # Mentions are the WikiData id
        assert first_instance_entity_ids[16:18] == ['Q831285', 'Q831285']

        # Test parent ids and relations
        first_instance_parent_ids = [[x.text for x in y.tokens] for y in instances[0]['parent_ids']]
        first_instance_relations = [[x.text for x in y.tokens] for y in instances[0]['relations']]
        # Non-mentions correspond to a singleton padding token
        assert first_instance_parent_ids[7] == ['@@PADDING@@']
        assert first_instance_relations[7] == ['@@PADDING@@']
        # "Washington" has two parents
        assert first_instance_parent_ids[27] == ['Q831285', 'Q3046581']
        assert first_instance_relations[27] == ['P131', 'P131']

        # Test that copy indices properly match tokens to their place in aliases
        alias_database = instances[0]['metadata']['alias_database']
        first_instance_alias_copy_inds = instances[0]['alias_copy_inds'].array
        entity_id = first_instance_entity_ids[16]
        first_mention_token = first_instance_target_tokens[16]
        uid = alias_database.token_to_uid(entity_id, first_mention_token)
        assert uid == first_instance_alias_copy_inds[16]

        # Test that shortlist is being properly generated
        first_instance_shortlist = [x.text for x in instances[0]['shortlist'].tokens]
        expected_entities = {
            '@@PADDING@@', 'Q831285', 'Q3046581', 'Q35657', 'Q695782', 'Q1223', 'Q452623',
            'Q272074', 'Q800459'
        }
        assert set(first_instance_shortlist) == expected_entities

        # Test that shortlist inds point to correct
        first_instance_shortlist = [x.text for x in instances[0]['shortlist'].tokens]
        first_instance_shortlist_inds = instances[0]['shortlist_inds'].array
        assert first_instance_shortlist_inds[16] == first_instance_shortlist.index('Q831285')
