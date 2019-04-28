# pylint: disable=protected-access,not-callable,unused-import
import numpy as np
import torch

from kglm.common.testing import KglmModelTestCase
from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextKglmReader
from kglm.models.no_story import NoStory
# from kglm.models.kglm_disc import KglmDisc


class KglmTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/no-story.json",
                          "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

# class KglmNoShortlistTest(KglmModelTestCase):
#
    # def setUp(self):
        # super().setUp()
        # self.set_up_model("kglm/tests/fixtures/training_config/kglm.no-shortlist.json",
                        #   "kglm/tests/fixtures/enhanced-wikitext.jsonl")
#
    # def test_model_can_train_save_and_load(self):
        # self.ensure_model_can_train_save_and_load(self.param_file)
#
# class KglmDiscTest(KglmModelTestCase):
#
    # def setUp(self):
        # super().setUp()
        # self.set_up_model("kglm/tests/fixtures/training_config/kglm-disc.json",
                        #   "kglm/tests/fixtures/enhanced-wikitext.jsonl")
#
    # def test_model_can_train_save_and_load(self):
        # self.ensure_model_can_train_save_and_load(self.param_file)
#
# class KglmDiscNoShortlistTest(KglmModelTestCase):
#
    # def setUp(self):
        # super().setUp()
        # self.set_up_model("kglm/tests/fixtures/training_config/kglm-disc.no-shortlist.json",
                        #   "kglm/tests/fixtures/enhanced-wikitext.jsonl")
#
    # def test_model_can_train_save_and_load(self):
        # self.ensure_model_can_train_save_and_load(self.param_file)
#
#
