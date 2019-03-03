from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from kglm.data.dataset_readers import EnhancedWikitextKglmReader
from kglm.models.kglm import Kglm
from kglm.predictors.cloze import ClozePredictor


class ClozePredictorTest(AllenNlpTestCase):
    def test_works(self):
        inputs = {
            "prefix": ["Benton", "Brindge", "is", "in"],
            "expected_tail": "Washington",
            "entity_id": "Q4890550",
            "entity_indices": [0, 2]
        }
        archive = load_archive('kglm/tests/fixtures/kglm.model.tar.gz')
        # TODO: Really need to import a valid discriminator, not just the
        # generator twice.
        predictor = ClozePredictor.from_archive(archive, archive, 'cloze')
        predictor.predict_json(inputs)
        predictor.predict_json(inputs)
