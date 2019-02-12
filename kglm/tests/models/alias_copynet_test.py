# pylint: disable=protected-access,not-callable,unused-import

from allennlp.common.testing import ModelTestCase
import numpy as np
import torch

from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextKglmReader
from kglm.models.alias_copynet import AliasCopynet


class  AliasCopynetTest(ModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/alias_copynet.json",
                          "kglm/tests/fixtures/enhanced-wikitext.jsonl")

    def test_model_can_train_save_and_load(self):
        # TODO: Construct a test case where we can ensure that _copy_mode_projection is learning
        # something.
        self.ensure_model_can_train_save_and_load(self.param_file) #, gradients_to_ignore=['_copy_mode_projection'])
