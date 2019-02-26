# pylint: disable=protected-access,not-callable,unused-import
from allennlp.common import Params
from allennlp.data import DataIterator, DatasetReader
import numpy as np
import torch

from kglm.common.testing import KglmModelTestCase
from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextKglmReader
from kglm.models.kglm import Kglm
from kglm.models.kglm_disc import KglmDisc


class KglmTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/kglm.json",
                          "kglm/tests/fixtures/enhanced-wikitext.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

class KglmNoShortlistTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/kglm.no-shortlist.json",
                          "kglm/tests/fixtures/enhanced-wikitext.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

class KglmDiscTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/kglm-disc.json",
                          "kglm/tests/fixtures/enhanced-wikitext.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_sample(self):
        params = Params.from_file(self.param_file)
        dataset_file = "kglm/tests/fixtures/enhanced-wikitext.jsonl"

        # Need instances from 'generative' reader!
        reader_params = params['dataset_reader']
        reader_params['mode'] = 'generative'
        reader = DatasetReader.from_params(reader_params)
        instances = list(reader.read(dataset_file))
        iterator = DataIterator.from_params(params['iterator'])
        iterator.index_with(self.model.vocab)
        batch, _ = next(iterator(instances, shuffle=False))
        self.model.sample(**batch)


class KglmDiscNoShortlistTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/kglm-disc.no-shortlist.json",
                          "kglm/tests/fixtures/enhanced-wikitext.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_sample(self):
        params = Params.from_file(self.param_file)
        dataset_file = "kglm/tests/fixtures/enhanced-wikitext.jsonl"

        # Need instances from 'generative' reader!
        reader_params = params['dataset_reader']
        reader_params['mode'] = 'generative'
        reader = DatasetReader.from_params(reader_params)
        instances = list(reader.read(dataset_file))

        iterator = DataIterator.from_params(params['iterator'])
        iterator.index_with(self.model.vocab)
        batch, _ = next(iterator(instances, shuffle=False))
        self.model.sample(**batch)
