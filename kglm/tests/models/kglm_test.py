# pylint: disable=protected-access,not-callable

from allennlp.common.testing import ModelTestCase
import numpy as np
import torch

from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextKglmReader
from kglm.models.kglm import Kglm


class KglmTest(ModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/models/kglm/experiment.json",
                          "kglm/tests/fixtures/data/enhanced-wikitext.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
