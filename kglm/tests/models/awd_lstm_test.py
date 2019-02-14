# pylint: disable=protected-access,not-callable,unused-import
import numpy as np
import torch

from kglm.common.testing import KglmModelTestCase
from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextReader
from kglm.models.awd_lstm import AwdLstmLanguageModel


class AwdLstmLanguageModelTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/awd-lstm-lm.json",
                          "kglm/tests/fixtures/enhanced-wikitext.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
