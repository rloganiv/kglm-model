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


@Predictor.register('cloze')
class ClozePredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader):
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

        # Add the shortlist
        if 'shortlist' in json_dict:
            shortlist = json_dict['shortlist']
            field = TextField(
                [Token(x) for x in shortlist],
                token_indexers=self._dataset_reader._entity_indexers)
            conditioning_instance.fields['shortlist'] = field

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

        with torch.no_grad():
            # TODO: Make this a parameter somewhere
            num_samples = 3

            # Duplicate instances (to sample in parallel)
            cuda_device = self._model._get_prediction_device()
            conditioning_batch = Batch([conditioning_instance] * num_samples)
            conditioning_batch.index_instances(self._model.vocab)
            conditioning_batch = util.move_to_device(conditioning_batch.as_tensor_dict(), cuda_device)

            generative_batch = Batch([generative_instance] * num_samples)
            generative_batch.index_instances(self._model.vocab)
            generative_batch = util.move_to_device(generative_batch.as_tensor_dict(), cuda_device)

            # Sample annotations and generate next token
            self._model.sample(**conditioning_batch, emit_tokens=False)
            generative_output = self._model.sample(**generative_batch, emit_tokens=True)
            del conditioning_batch, generative_batch

            aggregate_word_probs = self._aggregate_word_probs(generative_output)

            return aggregate_word_probs

    def _aggregate_word_probs(self, generative_output: Dict[str, Any]) -> Dict[str, List[Any]]:
        # Get vocab
        vocab = self._model.vocab
        vocab_size = vocab.get_vocab_size('tokens')

        # Get alias database
        metadata = generative_output['metadata']
        alias_database = metadata[0]['alias_database']

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
            prob_dict[word] = prob

        # The copy vocabs will be a little bit more difficult
        num_samples = target_probs.shape[0]
        raw_entity_ids = generative_output['raw_entity_ids']
        for copy_probs, raw_entity_id in zip(copy_vocab_probs, raw_entity_ids):
            if raw_entity_id == 0:
                continue

            alias_tokens, alias_inds = alias_database.lookup(raw_entity_id.unsqueeze(-1))
            alias_inds.view(-1)

            entity = vocab.get_token_from_index(raw_entity_id, 'raw_entity_ids')
            id_map = alias_database._id_map_lookup[entity]
            reverse_id_map = {i: x for x, i in id_map.items()}
            for ind, prob in zip(alias_inds, copy_probs):
                if ind == 0:
                    continue
                word = reverse_id_map[ind]
                prob_dict[word] += prob / num_samples

        # Lastly, convert the prob_dict to a ranked list of words
        prob_list = list(prob_dict.items())
        prob_list.sort(key=lambda x: x[1], reverse=True)

        words, probs = zip(*prob_list[:1000])
        return {'words': words, 'probs': probs}
