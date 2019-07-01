from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from kglm.data.dataset_readers import LinkedWikiTextKglmReader
from kglm.models.kglm import Kglm
from kglm.predictors.cloze import ClozePredictor


class ClozePredictorTest(AllenNlpTestCase):
    def test_works(self):
        inputs = {
            "prefix": ["Benton", "Brindge", "is", "in"],
            "expected_tail": "Washington",
            "entity_id": "Q4890550",
            "entity_indices": [0, 2],
            "shortlist": ["Q4890550", "Q35657"]
        }
        archive = load_archive('kglm/tests/fixtures/kglm.model.tar.gz')
        predictor = Predictor.from_archive(archive, 'cloze')
        predictor.predict_json(inputs)
        predictor.predict_json(inputs)
