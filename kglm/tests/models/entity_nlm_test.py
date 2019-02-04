# pylint: disable=protected-access,not-callable,unused-import

from allennlp.common.testing import ModelTestCase
import numpy as np
import torch

from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextEntityNlmReader
from kglm.models.entity_nlm import EntityNLM


class EntityNLMTest(ModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/entity_nlm.json",
                          "kglm/tests/fixtures/enhanced-wikitext.jsonl")

    def test_model_can_train_save_and_load(self):
        # TODO: Construct test cases where we can obtain gradients for these components
        gradients_to_ignore = [
                '_dummy_context_embedding',
                '_dynamic_embeddings._distance_scalar',
                '_dynamic_embeddings._embedding_projection.weight'
        ]
        self.ensure_model_can_train_save_and_load(self.param_file,
                                                  gradients_to_ignore=gradients_to_ignore)
