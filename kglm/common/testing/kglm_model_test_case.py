import copy
from typing import Any, Dict, Set, Union, Iterable

from numpy.testing import assert_allclose
import torch

from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data import DataIterator, DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model, load_archive


class KglmModelTestCase(ModelTestCase):
    """
    A subclass of :class:`~allennlp.common.testing.test_case.AllenNlpTestCase`
    that relaxes some assumptions with regard to the iterator.
    """
    def ensure_model_can_train_save_and_load(self,
                                             param_file: str,
                                             tolerance: float = 1e-4,
                                             cuda_device: int = -1,
                                             gradients_to_ignore: Set[str] = None,
                                             overrides: str = ""):
        """
        Parameters
        ----------
        param_file : ``str``
            Path to a training configuration file that we will use to train the model for this
            test.
        tolerance : ``float``, optional (default=1e-4)
            When comparing model predictions between the originally-trained model and the model
            after saving and loading, we will use this tolerance value (passed as ``rtol`` to
            ``numpy.testing.assert_allclose``).
        cuda_device : ``int``, optional (default=-1)
            The device to run the test on.
        gradients_to_ignore : ``Set[str]``, optional (default=None)
            This test runs a gradient check to make sure that we're actually computing gradients
            for all of the parameters in the model.  If you really want to ignore certain
            parameters when doing that check, you can pass their names here.  This is not
            recommended unless you're `really` sure you don't need to have non-zero gradients for
            those parameters (e.g., some of the beam search / state machine models have
            infrequently-used parameters that are hard to force the model to use in a small test).
        overrides : ``str``, optional (default = "")
            A JSON string that we will use to override values in the input parameter file.
        """
        save_dir = self.TEST_DIR / "save_and_load_test"
        archive_file = save_dir / "model.tar.gz"
        model = train_model_from_file(param_file, save_dir, overrides=overrides)
        loaded_model = load_archive(archive_file, cuda_device=cuda_device).model
        state_keys = model.state_dict().keys()
        loaded_state_keys = loaded_model.state_dict().keys()
        assert state_keys == loaded_state_keys
        # First we make sure that the state dict (the parameters) are the same for both models.
        for key in state_keys:
            assert_allclose(model.state_dict()[key].cpu().numpy(),
                            loaded_model.state_dict()[key].cpu().numpy(),
                            err_msg=key)
        params = Params.from_file(param_file)
        reader = DatasetReader.from_params(params['dataset_reader'])

        # Need to duplicate params because Iterator.from_params will consume.
        iterator_params = params['iterator']
        iterator_params2 = Params(copy.deepcopy(iterator_params.as_dict()))

        iterator = DataIterator.from_params(iterator_params)
        iterator2 = DataIterator.from_params(iterator_params2)

        # We'll check that even if we index the dataset with each model separately, we still get
        # the same result out.
        model_dataset = reader.read(params['validation_data_path'])
        iterator.index_with(model.vocab)
        model_batch = next(iterator(model_dataset, shuffle=False))
        if isinstance(model_batch, tuple):
            model_batch = model_batch[0]

        loaded_dataset = reader.read(params['validation_data_path'])
        iterator2.index_with(loaded_model.vocab)
        loaded_batch = next(iterator2(loaded_dataset, shuffle=False))
        if isinstance(loaded_batch, tuple):
            loaded_batch = loaded_batch[0]

        # Check gradients are None for non-trainable parameters and check that
        # trainable parameters receive some gradient if they are trainable.
        self.check_model_computes_gradients_correctly(model, model_batch, gradients_to_ignore)

        # The datasets themselves should be identical.
        assert model_batch.keys() == loaded_batch.keys()
        for key in model_batch.keys():
            self.assert_fields_equal(model_batch[key], loaded_batch[key], key, 1e-6)

        # Set eval mode, to turn off things like dropout, then get predictions.
        model.eval()
        loaded_model.eval()
        # Models with stateful RNNs need their states reset to have consistent
        # behavior after loading.
        for model_ in [model, loaded_model]:
            for module in model_.modules():
                if hasattr(module, 'stateful') and module.stateful:
                    module.reset_states()
        model_predictions = model(**model_batch)
        loaded_model_predictions = loaded_model(**loaded_batch)

        # Check loaded model's loss exists and we can compute gradients, for continuing training.
        loaded_model_loss = loaded_model_predictions["loss"]
        assert loaded_model_loss is not None
        loaded_model_loss.backward()

        # Both outputs should have the same keys and the values for these keys should be close.
        for key in model_predictions.keys():
            self.assert_fields_equal(model_predictions[key],
                                     loaded_model_predictions[key],
                                     name=key,
                                     tolerance=tolerance)

        return model, loaded_model
