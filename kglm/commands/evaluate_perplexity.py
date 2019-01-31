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
        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def evaluate_perplexity(model: Model,
                        sampler: Model,
                        instances: Iterator[Instance],
                        data_iterator: DataIterator,
                        cuda_device: int) -> Dict[str, Any]:
    check_for_gpu(cuda_device)

    num_samples = 100 # TODO: Make this something you can specify

    with torch.no_grad():
        model.eval()
        sampler.eval()

        iterator = data_iterator(instances, num_epochs=1, shuffle=False)
        logger.info('Iterating over dataset')
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

        total_cross_entropy = 0.0
        batch_count = 0

        for batch in generator_tqdm:
            batch_count += 1
            tokens = batch['tokens']
            tokens['tokens'] = tokens['tokens'].repeat(num_samples, 1)
            batch = util.move_to_device(batch, cuda_device)
            sequence_length = batch['tokens']['tokens'].shape[1]

            # Draw a sample
            sampler_output = sampler.sample(batch['tokens'])
            sample_logp = sampler_output['logp']
            sample = sampler_output['sample']
            sample['reset'] = True
            sample['tokens'] = batch['tokens']  # Add tokens to batch

            # logp of the model is the loss; we multiply by sequence length to go from token-level
            # to sequence-level probabilities.
            model_logp = model(**sample).get('logp')
            log_summands = model_logp - sample_logp

            # This is the log probability of the entire sentence
            logp = torch.logsumexp(log_summands, dim=0) - math.log(num_samples)

            # We care about per-token cross entropy
            per_word_cross_entropy = -logp / sequence_length
            print(math.exp(per_word_cross_entropy))
            total_cross_entropy += per_word_cross_entropy

        # Aggregate metrics
        avg_per_word_cross_entropy = total_cross_entropy / batch_count
        perplexity = torch.exp(avg_per_word_cross_entropy)

    metrics = {'perplexity': perplexity}
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
    iterator = BasicIterator(batch_size=1)
    iterator.index_with(model.vocab)
    metrics = evaluate_perplexity(model, sampler, instances, iterator, args.cuda_device)

    logger.info('Finished evaluating.')
    logger.info('Metrics:')
    for key, metric in metrics.items():
        logger.info('%s: %s', key, metric)

    output_file = args.output_file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    return metrics
