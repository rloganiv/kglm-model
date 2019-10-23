import argparse
import json
import logging
import math
from typing import Any, Dict, Iterator

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment
from allennlp.common.checks import check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import BasicIterator, DataIterator
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.nn import util
import numpy as np
import torch

logger = logging.getLogger(__name__)


class BeamSum(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Upper bound the specified model perplexity using beam search'''
        subparser = parser.add_parser(name, description=description,
                                      help='Evaluate the specified module using importance sampling')

        subparser.add_argument('model_archive_file', type=str, help='path to an archived trained model')

        subparser.add_argument('sampler_archive_file', type=str,
                               help='path to an archived trained model for generating samples')

        subparser.add_argument('input_file', type=str, help='path to the file containing the evaluation data')

        subparser.add_argument('--output-file', type=str, help='path to output file')

        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        subparser.add_argument('--batch-size',
                               type=int,
                               default=None,
                               help='Batch size (default: whatever iterator was set to)')

        subparser.add_argument('--split-size',
                               type=int,
                               default=None,
                               help='Split size (default: whatever iterator was set to)')

        subparser.add_argument('-k', '--beam-width',
                               type=int,
                               default=None,
                               help='Beam width')

        return subparser


def logsumexp(prev: torch.FloatTensor,
              current: torch.FloatTensor,
              i: int,
              samples_per_batch: int):
    # NOTE: n is number of samples
    current_avg = current.view(samples_per_batch, -1).sum(dim=-1).logsumexp(dim=0) - np.log(samples_per_batch).item()
    if prev is None:
        return current_avg
    a = torch.max(prev, current_avg)
    sumexp = torch.exp(prev - a) * i / (i + 1) + torch.exp(current_avg - a) / (i + 1)
    return a + torch.log(sumexp)


def evaluate_perplexity(model: Model,
                        sampler: Model,
                        num_samples: int,
                        instances: Iterator[Instance],
                        data_iterator: DataIterator,
                        cuda_device: int,
                        beam_width: int) -> Dict[str, Any]:
    check_for_gpu(cuda_device)

    logger.info('Iterating over dataset')

    weight = None

    iterator = data_iterator(instances, num_epochs=1, shuffle=False)
    generator_tqdm = Tqdm.tqdm(iterator, total=0)

    model.eval()
    sampler.eval()
    sampler._state = None

    summand = None
    denom = None
    #summand = torch.tensor(0.0)
    # penalized_summand = torch.tensor(0.0)

    held_over_data = None

    for batch, _ in generator_tqdm:

        # We need sequence length to help compute perplexity
        n_tokens = util.get_text_field_mask(batch['source']).float().sum(dim=-1)
        if denom is None:
            denom = n_tokens
        else:
            denom += n_tokens

        summand = util.move_to_device(summand, cuda_device)
        batch = util.move_to_device(batch, cuda_device)

        # Draw a sample
        with torch.no_grad():
            sample = sampler.beam_search(batch['source'],
                                         batch['reset'],
                                         beam_width)

        # Evaluate on sample
        with torch.no_grad():
            model_output = model(**sample)

        model_logp = model_output['logp']

        print(torch.exp(-model_logp / n_tokens))

        if summand is None:
            summand = model_logp
        else:
            summand += model_logp

    ppl = torch.exp(-summand / denom)

    metrics = {
        'ppl': ppl
    }
    return metrics

def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load model from archive
    model_archive = load_archive(args.model_archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = model_archive.config
    prepare_environment(config)
    model = model_archive.model
    model.eval()

    # Load sampler
    sampler_archive = load_archive(args.sampler_archive_file, args.cuda_device, args.overrides, args.weights_file)
    sampler = sampler_archive.model
    sampler.eval()

    # Load the evaluation data. NOTE: We are using the model's reader!
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.input_file
    logger.info('Reading evaluation data from: %s', evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    # To avoid hairy issues with splitting, we opt to use a basic iterator so that we can
    # generate samples for entire sequences.
    iterator_params = config.pop('iterator', 'None')
    if args.batch_size is not None:
        iterator_params['batch_size'] = args.batch_size
    if args.split_size is not None:
        iterator_params['split_size'] = args.split_size
    iterator_params['truncate'] = False
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)
    metrics = evaluate_perplexity(model, sampler, args.num_samples, instances,
                                  iterator, args.cuda_device, args.beam_width)

    logger.info('Finished evaluating.')
    logger.info('Metrics:')
    for key, metric in metrics.items():
        logger.info('%s: %s', key, metric)

    return metrics
