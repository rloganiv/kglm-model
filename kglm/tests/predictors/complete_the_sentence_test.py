from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from kglm.data.dataset_readers import EnhancedWikitextKglmReader
from kglm.models.kglm import Kglm
from kglm.predictors.complete_the_sentence import CompleteTheSentencePredictor


class CompleteTheSentencePredictorTest(AllenNlpTestCase):
    def test_works(self):
        inputs = {
            "prefix": ["Benton", "Brindge", "is", "in"],
            "expected_tail": "Washington",
            "entity_id": "Q4890550",
            "entity_indices": [0, 2],
            "shortlist": ["Q4890550", "Q35657"]
        }
        model_archive = load_archive('kglm/tests/fixtures/kglm.model.tar.gz')
        predictor = CompleteTheSentencePredictor.from_archive(model_archive, 'complete-the-sentence')
        predictor.predict_json(inputs)
        predictor.predict_json(inputs)
