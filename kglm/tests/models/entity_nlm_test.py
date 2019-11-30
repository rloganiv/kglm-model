# pylint: disable=protected-access,not-callable,unused-import

from allennlp.common.testing import ModelTestCase
import numpy as np
import torch

from kglm.training import LmTrainer
from kglm.common.testing import KglmModelTestCase
from kglm.data.dataset_readers.enhanced_wikitext import EnhancedWikitextEntityNlmReader
from kglm.models.entity_nlm import EntityNLM
from kglm.models.entity_disc import EntityNLMDiscriminator


class EntityNLMTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/entity_nlm.json",
                          "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file,
                                                  gradients_to_ignore=['_dummy_context_embedding'])

class EntityDiscTest(KglmModelTestCase):

    def setUp(self):
        super().setUp()
        self.set_up_model("kglm/tests/fixtures/training_config/entity_disc.json",
                          "kglm/tests/fixtures/enhanced-wikitext-test/train.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file,
                                                  gradients_to_ignore=['_dummy_context_embedding'])

    def test_annotation_logp(self):
        batch_size = 2

        # Need to reset the states
        reset = torch.ByteTensor([1] * batch_size)
        self.model.reset_states(reset)

        # Apply to random hidden state
        hidden = torch.randn(batch_size, self.model._embedding_dim)
        beam_states = [self.model._dynamic_embeddings.beam_state()]
        logp = self.model._annotation_logp(hidden, timestep=0, beam_states=beam_states)

        # Check that output has correct shape
        assert tuple(logp.shape) == (batch_size, 1, self.model.num_possible_annotations)

    def test_adjust_for_ongoing_mentions(self):
        batch_size = 2
        k = 3

        # Construct an example where the top-beam state for the second sequence in the batch is an ongoing mention
        logp = torch.zeros(batch_size, k, self.model.num_possible_annotations)
        output = {
            'entity_ids': torch.LongTensor([[0, 0, 0], [3, 0, 0]]),
            'mention_lengths': torch.LongTensor([[0, 0, 0], [2, 0, 0]])
        }

        # See that adjustment works
        logp = self.model._adjust_for_ongoing_mentions(logp, output)
        assert logp[0, 0, 0] == 0.0  # Should be unaffected
        assert logp[1, 0, 0] == -float('inf') # Should be affected

        # Only element with probability should have entity id == 3 and mention length == 2
        pred = logp[1, 0].argmax()
        assert self.model.entity_id_lookup[pred] == 3
        assert self.model.mention_length_lookup[pred] == 1

    def test_top_k_annotations(self):
        batch_size = 2
        k = 3

        # Check works correctly at start (e.g. if beam size is 1)
        logp = torch.randn(batch_size, 1, self.model.num_possible_annotations)
        annotations = self.model._top_k_annotations(logp, k)

        assert tuple(annotations['logp'].shape) == (batch_size, k)
        assert torch.allclose(annotations['backpointers'], torch.zeros(batch_size, k, dtype=torch.int64))

        # Check works correctly for other timesteps (e.g. previous beam size is k)
        logp = torch.randn(batch_size, k, self.model.num_possible_annotations)
        annotations = self.model._top_k_annotations(logp, k)

        assert tuple(annotations['logp'].shape) == (batch_size, k)

    def test_beam_search(self):
        batch_size = 2
        seq_len = 10
        k = 3
        vocab_size = self.model.vocab.get_vocab_size('tokens')
        source = {'tokens': torch.randint(vocab_size, size=(batch_size, seq_len))}
        reset = torch.ones(batch_size, dtype=torch.uint8)
        out = self.model.beam_search(source, reset, k)
