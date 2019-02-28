from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from kglm.data.dataset_readers import EnhancedWikitextKglmReader
from kglm.models.kglm import Kglm
from kglm.predictors.cloze import ClozePredictor


class ClozePredictorTest(AllenNlpTestCase):
    def test_works(self):
        inputs = {
            "prefix": ["Barack", "Obama", "'s", "wife", "is"],
            "expected_tail": "Michelle",
            "entity_id": "Q76",
            "entity_indices": [0, 2]
        }
        archive = load_archive('kglm/tests/fixtures/kglm.model.tar.gz')
        predictor = Predictor.from_archive(archive, 'cloze')
        predictor.predict_json(inputs)