from typing import List

from allennlp.common.util import sanitize, JsonDict
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.predictors import Predictor
import numpy as np
import torch

from kglm.data.fields import SequentialArrayField
from kglm.data.dataset_readers.enhanced_wikitext import _flatten, _tokenize, normalize_entity_id

MAX_PARENTS = 10


@Predictor.register('kglm')
class KglmPredictor(Predictor):

    def predict_instance(self, instance) -> JsonDict:
        with torch.no_grad():
            try:
                outputs = self._model.forward_on_instance(instance)
            except RuntimeError:
                outputs = {'ERROR': 'Too big'}
            return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        with torch.no_grad():
            try:
                outputs = self._model.forward_on_instances(instances)
            except RuntimeError:
                outputs = {'ERROR': 'Too big'}
            return sanitize(outputs)

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        # pylint: disable=protected-access

        # Extract tokens and EOS offset
        tokens = [x + ['@@END@@'] for x in json_dict['tokens'][1:-1]]
        eos_offset = [[i] * len(x) for i, x in enumerate(tokens)]
        tokens = ['@@START@@'] + _flatten(tokens)
        eos_offset = [0] + _flatten(eos_offset)
        source = tokens[:-1]
        if self._dataset_reader._mode == 'generative':
            target = tokens[1:]
        else:
            target = None

        # Process annotations
        if 'annotations' not in json_dict:
            shortlist = None
            reverse_shortlist = None
            raw_entity_ids = None
            entity_ids = None
            relations = None
            parent_ids = None
            shortlist_inds = None
            mention_type = None
        else:
            # We maintain a "shortlist" of observed entities, that is used for baseline models
            # that only select entities from the set that appear in the document (as opposed to
            # the set of all possible entities).
            shortlist = [DEFAULT_PADDING_TOKEN]
            reverse_shortlist = {DEFAULT_PADDING_TOKEN: 0}
            raw_entity_ids = [DEFAULT_PADDING_TOKEN] * len(source)
            entity_ids = [DEFAULT_PADDING_TOKEN] * len(source)
            relations = [[DEFAULT_PADDING_TOKEN]] * len(source)
            parent_ids = [[DEFAULT_PADDING_TOKEN]] * len(source)
            shortlist_inds = np.zeros(shape=(len(source),))
            mention_type = np.zeros(shape=(len(source),))

            if self._dataset_reader._mode == "generative":
                alias_copy_inds = np.zeros(shape=(len(source),))
            else:
                alias_copy_inds = None

            # Process annotations
            for annotation in json_dict['annotations']:

                # Obtain the entity identifier for the annotated span
                raw_entity_id = annotation['id']
                raw_parent_id = annotation['parent_id']
                entity_id = normalize_entity_id(raw_entity_id)
                if entity_id is None:
                    continue
                parent_id = [normalize_entity_id(x) for x in raw_parent_id]
                assert len(parent_id) == len(raw_parent_id)
                relation = annotation['relation']
                new_entity = relation == ['@@NEW@@']

                # If neccessary, update the shortlist. Obtain the index of the entity identifier in
                # the shortlist.
                if entity_id not in reverse_shortlist:
                    reverse_shortlist[entity_id] = len(reverse_shortlist)
                    shortlist.append(entity_id)
                shortlist_ind = reverse_shortlist[entity_id]

                # Update the outputs
                # Offset is 0 in generative case, since each timestep is for predicting
                # attributes of the next token. In the discriminative case, each timestep
                # is for predicting attributes of the current token.
                mode_offset = -1 if self._dataset_reader._mode == "generative" else 0
                span = annotation['span']
                eos_offset_adjusted_span = tuple(i + eos_offset[i] for i in span)
                for i in range(*eos_offset_adjusted_span):
                    raw_entity_ids[i+mode_offset] = raw_entity_id
                    entity_ids[i+mode_offset] = entity_id
                    mention_type[i+mode_offset] = 3
                    if new_entity:
                        shortlist_inds[i+mode_offset] = shortlist_ind
                    else:
                        relations[i+mode_offset] = relation[:MAX_PARENTS]
                        parent_ids[i+mode_offset] = parent_id[:MAX_PARENTS]
                    if self._dataset_reader._mode == "generative":
                        alias_copy_inds[i+mode_offset] = self._dataset_reader._alias_database.token_to_uid(raw_entity_id, tokens[i])
                # Now put in proper mention type for first token
                start = annotation['span'][0]
                if new_entity:
                    mention_type[start+mode_offset] = 1
                else:
                    mention_type[start+mode_offset] = 2

        instance = self._dataset_reader.text_to_instance(
            source,
            target,
            shortlist,
            reverse_shortlist,
            raw_entity_ids,
            entity_ids,
            relations,
            parent_ids,
            shortlist_inds,
            mention_type,
            alias_copy_inds)
        reset = SequentialArrayField(np.array(1), dtype=np.uint8)
        instance.add_field('reset', reset)
        return instance
