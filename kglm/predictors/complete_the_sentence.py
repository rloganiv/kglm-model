import json
import logging
from typing import Any, Dict, List, Tuple

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token
from allennlp.models import Model
from allennlp.models.archival import Archive, load_archive
from allennlp.nn import util
from allennlp.predictors import Predictor
import numpy as np
from overrides import overrides
import torch

from kglm.data import SequentialArrayField

logger = logging.getLogger(__name__)


@Predictor.register('complete-the-sentence')
class CompleteTheSentencePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
        logger.warning('CompleteTheSentencePredictor is meant to be used with '
                       '`kglm.run complete-the-sentence`, if you are using '
                       '`allennlp predict` then results may be incorrect.')
        self._model = model
        self._dataset_reader = dataset_reader

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        We need to break with the Predictor expectations slightly, instead of returning a single
        instance we return a conditioning instance (to warm up the model), and then a generative
        instance (e.g. the token to predict on).
        """
        ### Conditioning Instance ###
        # TODO: This is totally broken...

        # Manually add the start token
        tokens = ['@@START@@', *json_dict['prefix']]

        # Also need to offset
        # start, end = json_dict['entity_indices']
        # span = [start + 1, end + 1]

        # Repackage into the expected annotation format
        # annotations = [{
        #     'id': json_dict['entity_id'],
        #     'relation': ['@@NEW@@'],
        #     'parent_id': [json_dict['entity_id']],
        #     'span': span
        # }]
        # data = {'tokens': [tokens], 'annotations': annotations}
        # conditioning_instance = self._dataset_reader.text_to_instance(data)
        conditioning_instance = self._dataset_reader.text_to_instance(tokens[:-1])

        # Manually add the reset field here
        reset = SequentialArrayField(np.array(1), dtype=np.uint8)
        conditioning_instance.add_field('reset', reset)

        # Add the shortlist
        if 'shortlist' in json_dict:
            shortlist = json_dict['shortlist']
            field = TextField(
                [Token(x) for x in shortlist],
                token_indexers=self._dataset_reader._entity_indexers)
            conditioning_instance.fields['shortlist'] = field

        ### Generative Instance ###

        # data = {'tokens': [[tokens[-1]]]}
        # generative_instance = self._dataset_reader.text_to_instance(data)
        generative_instance = self._dataset_reader.text_to_instance([tokens[-1]])
        # Manually add the reset field here
        reset = SequentialArrayField(np.array(0), dtype=np.uint8)
        generative_instance.add_field('reset', reset)
        if 'shortlist' in json_dict:
            generative_instance.add_field('shortlist', conditioning_instance.fields['shortlist'])

        return conditioning_instance, generative_instance

    @overrides
    def predict_instance(self, instances: Tuple[Instance, Instance]) -> JsonDict:
        conditioning_instance, generative_instance = instances

        self._model.eval()

        with torch.no_grad():
            # TODO: Make this a parameter somewhere
            num_samples = 100

            # Duplicate instances (to sample in parallel)
            cuda_device = self._model._get_prediction_device()
            conditioning_batch = Batch([conditioning_instance] * num_samples)
            conditioning_batch.index_instances(self._model.vocab)
            conditioning_batch = util.move_to_device(conditioning_batch.as_tensor_dict(), cuda_device)

            generative_batch = Batch([generative_instance] * num_samples)
            generative_batch.index_instances(self._model.vocab)
            generative_batch = util.move_to_device(generative_batch.as_tensor_dict(), cuda_device)

            # Sample annotations and generate next token
            self._model._use_shortlist = True
            conditioning_output = self._model.sample(**conditioning_batch, emit_tokens=False)
            logger.debug('clears condition generation')
            # self._model(**conditioning_output)  # Shouldn't need to do this, but just in case
            # logger.debug('clears reconditioning')
            generative_output = self._model.sample(**generative_batch, emit_tokens=True)
            logger.debug('clears generation')
            del conditioning_batch, generative_batch

            aggregate_word_probs = self._aggregate_word_probs(generative_output)
            logger.debug('clears word probs')

            return aggregate_word_probs

    def _aggregate_word_probs(self, generative_output: Dict[str, Any]) -> Dict[str, List[Any]]:

        # Get vocab
        vocab = self._model.vocab
        vocab_size = vocab.get_vocab_size('tokens')

        # Get alias database
        alias_database = generative_output['metadata'][0]['alias_database']

        # Split into source and target vocab probs
        target_probs = generative_output['target_probs']
        source_vocab_probs = target_probs[:, :, :vocab_size]
        copy_vocab_probs = target_probs[:, :, vocab_size:]

        # Average source vocab prob is easy to compute
        source_vocab_avg = source_vocab_probs.mean(0).squeeze()
        prob_dict = dict()
        index_to_token_vocabulary = vocab.get_index_to_token_vocabulary('tokens')
        for i, prob in enumerate(source_vocab_avg):
            word = index_to_token_vocabulary[i]
            prob_dict[word] = prob.item()

        # Get alias indices
        alias_indices = generative_output['alias_indices']

        # The copy vocabs will be a little bit more difficult
        num_samples = target_probs.shape[0]
        raw_entity_ids = generative_output['raw_entity_ids']['raw_entity_ids']
        for alias_index, copy_probs, raw_entity_id in zip(alias_indices, copy_vocab_probs, raw_entity_ids):
            if raw_entity_id == 0:
                continue

            entity = vocab.get_token_from_index(raw_entity_id.item(), 'raw_entity_ids')
            try:
                id_map = alias_database._id_map_lookup[entity]
            except:
                logger.warning('Error could not find id map for entity "%s"', entity)
                continue
            reverse_id_map = {i: x for x, i in id_map.items()}
            for ind, prob in zip(alias_index, copy_probs.squeeze().tolist()):
                if ind == 0:
                    continue
                word = reverse_id_map[ind.item()]
                if word in prob_dict:
                    prob_dict[word] += prob / num_samples
                else:
                    prob_dict[word] = prob / num_samples

        # Lastly, convert the prob_dict to a ranked list of words
        prob_list = list(prob_dict.items())
        prob_list.sort(key=lambda x: x[1], reverse=True)

        words, probs = zip(*prob_list[:1000])
        return {'words': words, 'probs': probs}

