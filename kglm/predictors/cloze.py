import json
import logging
from typing import Tuple

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset import Batch
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.nn import util
from allennlp.predictors import Predictor
import numpy as np
from overrides import overrides

from kglm.data import SequentialArrayField

logger = logging.getLogger(__name__)


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
        conditioning_instance, generative_instance = instances

        # TODO: Make this a parameter somewhere
        #  num_samples = 100
        #  best_logp = -float('inf')
        #  sample = None
        #  for _ in  range(num_samples):
        #      # Duplicate conditioning instance to generate samples
        #      cuda_device = self._sampler._get_prediction_device()
        #      dataset = Batch([conditioning_instance])
        #      dataset.index_instances(self._sampler.vocab)
        #      model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
        #      sampler_output = self._sampler.sample(**model_input)
        #      # Run model
        #      model_output = self._model(**sampler_output['sample'])
        #      # Compute importance. If it is the best then make this the
        #      # annotation used to generate the next word.
        #      importance = model_output['logp'] - sampler_output['logp']
        #      logger.debug('importance weight: %0.4f', importance)
        #      if importance > best_logp:
        #          logger.debug('best so far')
        #          best_logp = importance
        #          sample = sampler_output['sample']

        #  # Seed the model with the conditioning instance
        #  self._model(**sample)

        # Seed the model with the conditioning instance
        self._model.forward_on_instance(conditioning_instance)

        # Then generate
        return self._model.forward_on_instance(generative_instance)

    @classmethod
    def from_archive(cls, model_archive: Archive, sampler_archive: Archive, predictor_name: str = None) -> 'Predictor':
        """
        Instantiate a :class:`Predictor` from an :class:`~allennlp.models.archival.Archive`;
        that is, from the result of training a model. Optionally specify which `Predictor`
        subclass; otherwise, the default one for the model will be used.
        """
        # Duplicate the config so that the config inside the archive doesn't get consumed
        model_config = model_archive.config.duplicate()

        dataset_reader_params = model_config["dataset_reader"]
        dataset_reader = DatasetReader.from_params(dataset_reader_params)

        model = model_archive.model
        model.eval()

        sampler = sampler_archive.model
        sampler.eval()

        return Predictor.by_name(predictor_name)(model, sampler, dataset_reader)

