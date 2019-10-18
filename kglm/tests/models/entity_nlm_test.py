# pylint: disable=protected-access,not-callable,unused-import

from allennlp.common.testing import ModelTestCase
import numpy as np
import torch

from kglm.training import LmTrainer
from kglm.common.testing import KglmModelTestCase
from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextEntityNlmReader
from kglm.models.entity_nlm import EntityNLM
from kglm.models.entity_disc import EntityNLMDiscriminator


class EntityNLMTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/entity_nlm.json",
                          "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file,
                                                  gradients_to_ignore=['_dummy_context_embedding'])

class EntityDiscTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/entity_disc.json",
                          "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file,
                                                  gradients_to_ignore=['_dummy_context_embedding'])

    def test_annotation_logp(self):
        batch_size = 2

        # Need to reset the states
        reset = torch.ByteTensor([1] * batch_size)
        self.model.reset_states(reset)

        # Apply to random hidden state
        hidden = torch.randn(batch_size, self.model._embedding_dim)
        logp = self.model._annotation_logp(hidden, timestep=0)

        # Check that output has correct shape
        assert tuple(logp.shape) == (batch_size, self.model.num_possible_annotations)

        # Check that state dict can be fed to function...
        state_dict = {
            'dynamic_embeddings_state_dict': self.model._dynamic_embeddings.state_dict()
        }
        logp_prime = self.model._annotation_logp(hidden, timestep=0, state_dict=state_dict)

        # ...and that output is the same as before
        assert torch.allclose(logp, logp_prime)
