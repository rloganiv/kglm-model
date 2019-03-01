import json
from typing import Tuple

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
import numpy as np
from overrides import overrides

from kglm.data import SequentialArrayField


@Predictor.register('cloze')
class ClozePredictor(Predictor):
    def __init__(self, model: Model, sampler: Model, dataset_reader: DatasetReader):
        self._model = model
        self._sampler = sampler
        self._dataset_reader = dataset_reader

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        We need to break with the Predictor expectations slightly, instead of returning a single
        instance we return a conditioning instance (to warm up the model), and then a generative
        instance (e.g. the token to predict on).
        """
        ### Conditioning Instance ###

        # Manually add the start token
        tokens = ['@@START@@', *json_dict['prefix']]
        # Also need to offset
        start, end = json_dict['entity_indices']
        span = [start + 1, end + 1]
        # Repackage into the expected annotation format
        annotations = [{
            'id': json_dict['entity_id'],
            'relation': ['@@NEW@@'],
            'parent_id': [json_dict['entity_id']],
            'span': span
        }]
        data = {'tokens': [tokens], 'annotations': annotations}
        conditioning_instance = self._dataset_reader.text_to_instance(data)
        # Manually add the reset field here
        reset = SequentialArrayField(np.array(1), dtype=np.uint8)
        conditioning_instance.add_field('reset', reset)

        ### Generative Instance ###

        data = {'tokens': [[tokens[-1]]]}
        generative_instance = self._dataset_reader.text_to_instance(data)
        # Manually add the reset field here
        reset = SequentialArrayField(np.array(0), dtype=np.uint8)
        generative_instance.add_field('reset', reset)
        generative_instance.add_field('shortlist', conditioning_instance.fields['shortlist'])

        return conditioning_instance, generative_instance

    @overrides
    def predict_instance(self, instances: Tuple[Instance, Instance]) -> JsonDict:
        import pdb; pdb.set_trace()
        conditioning_instance, generative_instance = instances
        # Seed the model with the conditioning instance
        self._model.forward_on_instance(conditioning_instance)
        return self._model.forward_on_instance(generative_instance)
