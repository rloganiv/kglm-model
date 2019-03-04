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
import torch

logger = logging.getLogger(__name__)


class EvaluatePerplexity(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Evaluate the specified model perplexity using importance sampling'''
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

        subparser.add_argument('--batch-weight-key',
                               type=str,
                               default="",
                               help='If non-empty, name of metric used to weight the loss on a per-batch basis.')

        subparser.add_argument('--num-samples',
                               type=int,
                               default=100,
                               help='Number of importance samples to draw.')
        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_perplexity(model: Model,
                        sampler: Model,
                        num_samples: int,
                        instances: Iterator[Instance],
                        data_iterator: DataIterator,
                        cuda_device: int) -> Dict[str, Any]:
    check_for_gpu(cuda_device)

    logger.info('Iterating over dataset')

    summands = []

    with torch.no_grad():
        model.eval()
        sampler.eval()

        for i in range(num_samples):
            iterator = data_iterator(instances, num_epochs=1, shuffle=False)
            generator_tqdm = Tqdm.tqdm(iterator, total=0)

            summand = 0.0
            denom = 0

            for batch, _ in generator_tqdm:
                batch = util.move_to_device(batch, cuda_device)
                # We need sequence length to help compute perplexity
                n_tokens = util.get_text_field_mask(batch['source']).float().sum().item()
                denom += (n_tokens / 100)

                # Draw a sample
                sampler_output = sampler.sample(batch["source"], batch["reset"])
                sample_logp = sampler_output['logp']
                sample = sampler_output['sample']

                # Evaluate on sample
                model_output = model(**sample)
                model_logp = model_output['logp']
                summand += ((model_logp.sum() - sample_logp.sum()).item() / 100)

            summands.append(summand)
            t = torch.tensor(summands)
            t_sum = torch.logsumexp(t, dim=0)
            sum_logp = (t_sum - math.log(i+1)).item()
            ppl = math.exp(-sum_logp / denom)

            print('PPL: %f' % ppl)

    metrics = {'ppl': ppl}
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

    # Load the evaluation data
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
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)
    iterator.eval()
    metrics = evaluate_perplexity(model, sampler, args.num_samples,
                                  instances, iterator, args.cuda_device)

    logger.info('Finished evaluating.')
    logger.info('Metrics:')
    for key, metric in metrics.items():
        logger.info('%s: %s', key, metric)

    output_file = args.output_file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    return metrics
