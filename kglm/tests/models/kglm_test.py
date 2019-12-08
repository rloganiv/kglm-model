# pylint: disable=protected-access,not-callable,unused-import
from allennlp.common import Params
from allennlp.data import DataIterator, DatasetReader
import numpy as np
import torch

from kglm.training import LmTrainer
from kglm.common.testing import KglmModelTestCase
from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextKglmReader
from kglm.models.kglm import Kglm
from kglm.models.kglm_disc import KglmDisc, KglmBeamState
from kglm.modules import RecentEntities


class KglmTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/kglm.json",
                          "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

class KglmNoShortlistTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/kglm.no-shortlist.json",
                          "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

class KglmDiscTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/kglm-disc.json",
                          "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, gradients_to_ignore=['_overlap_weight'])

    def test_sample(self):
        generator_params = Params.from_file("kglm/tests/fixtures/training_config/kglm.json")
        params = Params.from_file(self.param_file)
        dataset_file = "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl"

        # Need instances from 'generative' reader!
        reader_params = generator_params['dataset_reader']
        reader = DatasetReader.from_params(reader_params)
        instances = list(reader.read(dataset_file))
        iterator = DataIterator.from_params(generator_params['iterator'])
        iterator.index_with(self.model.vocab)
        batch = next(iterator(instances, shuffle=False))
        self.model.sample(**batch)


class KglmDiscNoShortlistTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/kglm-disc.no-shortlist.json",
                          "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_sample(self):
        generator_params = Params.from_file("kglm/tests/fixtures/training_config/kglm.no-shortlist.json")
        params = Params.from_file(self.param_file)
        dataset_file = "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl"

        # Need instances from 'generative' reader!
        reader_params = generator_params['dataset_reader']
        reader_params['mode'] = 'generative'
        reader = DatasetReader.from_params(reader_params)
        instances = list(reader.read(dataset_file))

        iterator = DataIterator.from_params(generator_params['iterator'])
        iterator.index_with(self.model.vocab)
        batch = next(iterator(instances, shuffle=False))

        # Samples should match (we'll test by comparing logp)
        torch.manual_seed(123)
        logp1 = self.model.sample(**batch).get('logp', None)
        torch.manual_seed(123)
        logp2 = self.model.sample(**batch).get('logp', None)

    def test_beam_search(self):
        generator_params = Params.from_file("kglm/tests/fixtures/training_config/kglm.no-shortlist.json")
        params = Params.from_file(self.param_file)
        dataset_file = "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl"

        # Need instances from 'generative' reader!
        reader_params = generator_params['dataset_reader']
        reader_params['mode'] = 'generative'
        reader = DatasetReader.from_params(reader_params)
        instances = list(reader.read(dataset_file))

        iterator = DataIterator.from_params(generator_params['iterator'])
        iterator.index_with(self.model.vocab)
        batch = next(iterator(instances, shuffle=False))

        # Just want to check that function does not raise an error for now.
        self.model.beam_search(batch['source'],
                               batch['target'],
                               batch['reset'],
                               batch['metadata'],
                               k=5)

    def test_next_mention_type_logp(self):
        # Checks whether penalty correctly applied to ongoing mentions
        batch_size = 1
        num_classes = 2
        k = 2

        # All mention types have equal prob
        next_mention_type_logits = torch.ones(batch_size, num_classes)

        # First beam has an ongoing mention, second does not
        recent_entities_state = self.model._recent_entities.beam_state()
        ongoing_0 = torch.ones(batch_size, dtype=torch.uint8)
        ongoing_1 = torch.zeros(batch_size, dtype=torch.uint8)
        beam_states = [
            KglmBeamState(recent_entities=recent_entities_state, ongoing=ongoing_0),
            KglmBeamState(recent_entities=recent_entities_state, ongoing=ongoing_1)
        ]

        next_mention_type_logp = self.model._next_mention_type_logp(next_mention_type_logits, beam_states)
        # Log probabilities should be same on first beam, and different on second.
        assert torch.allclose(next_mention_type_logp[0, 0, 0], next_mention_type_logp[0, 0, 1])
        assert not torch.allclose(next_mention_type_logp[0, 1, 0], next_mention_type_logp[0, 1, 1])
        # Log probability of first state (e.g., non-ongoing) should be close to 0.0 on second beam.
        assert torch.allclose(next_mention_type_logp[0, 1, 0], torch.tensor(0.0))

    def test_next_new_entity_logp(self):
        # Checks whether penalty correctly applied to previously mentioned entities
        batch_size = 1
        num_entities = 2
        k = 2

        # All next entities have equal prob
        next_new_entity_logits = torch.ones(batch_size, num_entities)

        # First entity is previously mentioned on first beam.
        # No previous mentions on second beam.
        ongoing = None  # Value doesn't matter
        recent_entities_state_0 = {'remaining': [{0 : None}]}
        recent_entities_state_1 = {'remaining': [{}]}
        beam_states = [
            KglmBeamState(recent_entities=recent_entities_state_0, ongoing=ongoing),
            KglmBeamState(recent_entities=recent_entities_state_1, ongoing=ongoing)]

        next_new_entity_logp = self.model._next_new_entity_logp(next_new_entity_logits, beam_states)
        # Log probabilities should be different on first beam, and same on second.
        assert not torch.allclose(next_new_entity_logp[0, 0, 0], next_new_entity_logp[0, 0, 1])
        assert torch.allclose(next_new_entity_logp[0, 1, 0], next_new_entity_logp[0, 1, 1])
        # Log probability of non-recent entity should be close to 0.0 on first beam.
        assert torch.allclose(next_new_entity_logp[0, 0, 1], torch.tensor(0.0))

    def test_next_related_entity_logp(self):
        # Checks that:
        #   * There is no probability mass if there are no candidates
        #   * Probability distribution is valid if there are candidates
        #   * Annotations look correct (e.g., parents ids are consistent)
        batch_size = 1
        k = 2

        next_encoded_head = torch.randn((batch_size, self.model.entity_embedding_dim))
        next_encoded_relation = torch.randn((batch_size, self.model.entity_embedding_dim))
        ongoing = None  # Value doesn't matter

        # NOTE: `parent_id` = 5 chosen since this node in the knowledge graph has a relatively
        # small number of outgoing edges.
        recent_entities_state_0 = {'remaining': [{5 : None}]}
        recent_entities_state_1 = {'remaining': [{}]}
        recent_entities_state_2 = {'remaining': [{5: None, 6: None}]}
        beam_states = [
            KglmBeamState(recent_entities=recent_entities_state_0, ongoing=ongoing),
            KglmBeamState(recent_entities=recent_entities_state_1, ongoing=ongoing),
            KglmBeamState(recent_entities=recent_entities_state_2, ongoing=ongoing)
        ]

        logp, annotations = self.model._next_related_entity_logp(next_encoded_head,
                                                                 next_encoded_relation,
                                                                 beam_states)
        # Only first and last states will have probability mass
        assert torch.allclose(logp[0, 0].exp().sum(), torch.tensor(1.0))
        assert torch.allclose(logp[0, 1].exp().sum(), torch.tensor(0.0))
        assert torch.allclose(logp[0, 2].exp().sum(), torch.tensor(1.0))

        assert annotations['parent_ids'][0, 0].unique().size(0) == 2  # ids: 0, 5
        assert annotations['parent_ids'][0, 1].unique().size(0) == 1  # ids: 0
        assert annotations['parent_ids'][0, 2].unique().size(0) == 2  # ids: 0, 5, 6
