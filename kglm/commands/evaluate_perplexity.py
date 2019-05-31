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

        subparser.add_argument('--num-samples',
                               type=int,
                               default=100,
                               help='Number of importance samples to draw.')
        subparser.set_defaults(func=evaluate_from_args)

        return subparser


def _decode_sample(sample, vocab):
    for i in range(5):
        source_tokens = sample['target']['tokens'][i]
        source_tokens = [vocab.get_token_from_index(x, 'tokens') for x in
                        source_tokens.tolist()]
        mention_types = sample['mention_type'][i].tolist()
        entity_ids = sample['entity_ids']['entity_ids'][i]
        entity_ids = [vocab.get_token_from_index(x, 'entity_ids') for x in
                      entity_ids.tolist()]
        string = ' '.join('%s (%i, %s)' % x if x[1]!=0 else '%s' % x[0] for x
                          in zip(source_tokens, mention_types, entity_ids))
        logger.debug(string)


def evaluate_perplexity(model: Model,
                        sampler: Model,
                        num_samples: int,
                        instances: Iterator[Instance],
                        data_iterator: DataIterator,
                        cuda_device: int) -> Dict[str, Any]:
    check_for_gpu(cuda_device)

    logger.info('Iterating over dataset')

    with torch.no_grad():

        summands = []
        penalized_summands = []

        for i in range(num_samples):
            iterator = data_iterator(instances, num_epochs=1, shuffle=False)
            generator_tqdm = Tqdm.tqdm(iterator, total=0)

            model.eval()
            sampler.eval()

            summand = 0.0
            penalized_summand = 0.0
            denom = 0
            for batch, _ in generator_tqdm:

                batch = util.move_to_device(batch, cuda_device)

                # We need sequence length to help compute perplexity
                n_tokens = util.get_text_field_mask(batch['source']).float().sum().item()
                denom += n_tokens

                # Draw a sample
                sampler_output = sampler.sample(**batch)
                sample_logp = sampler_output['logp']
                sample = sampler_output['sample']

                _decode_sample(sample, sampler.vocab)

                # Evaluate on sample
                model_output = model(**sample)
                model_logp = model_output['logp']
                model_penalized_logp = model_output['penalized_logp']
                summand += (model_logp - sample_logp).item()
                penalized_summand += (model_penalized_logp - sample_logp).item()

                print('Temp PPL: %f' % math.exp(-summand / denom))
                print('Temp UPP: %f' % math.exp(-penalized_summand / denom))

            summands.append(summand)
            penalized_summands.append(penalized_summand)
            t = torch.tensor(summands)
            p = torch.tensor(penalized_summands)
            t_sum = torch.logsumexp(t, dim=0)
            p_sum = torch.logsumexp(p, dim=0)
            sum_logp = (t_sum - math.log(i+1)).item()
            sum_logp_penalized = (p_sum - math.log(i+1)).item()
            ppl = math.exp(-sum_logp / denom)
            upp = math.exp(-sum_logp_penalized / denom)

            print('PPL: %f' % ppl)
            print('UPP: %f' % upp)

    metrics = {'ppl': ppl, 'upp': upp}
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
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)
    iterator.eval()
    metrics = evaluate_perplexity(model, sampler, args.num_samples, instances, iterator, args.cuda_device)

    logger.info('Finished evaluating.')
    logger.info('Metrics:')
    for key, metric in metrics.items():
        logger.info('%s: %s', key, metric)

    output_file = args.output_file
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4)
    return metrics

