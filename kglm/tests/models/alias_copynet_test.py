# pylint: disable=protected-access,not-callable,unused-import
import numpy as np
import torch

from kglm.common.testing import KglmModelTestCase
from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextKglmReader
from kglm.models.alias_copynet import AliasCopynet


class AliasCopynetTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/alias_copynet.json",
                          "kglm/tests/fixtures/enhanced-wikitext.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
