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

        subparser.add_argument('--out', type=str, help='prefix of output files')

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
                               default=1,
                               help='Batch size (default: whatever iterator was set to)')

        subparser.add_argument('--split-size',
                               type=int,
                               default=int(1e10),
                               help='Split size (default: whatever iterator was set to)')

        subparser.add_argument('--num-samples',
                               type=int,
                               default=10000,
                               help='Number of importance samples to draw.')

        subparser.add_argument('--samples-per-batch',
                               type=int,
                               default=1,
                               help='Number of importance samples to draw.')

        subparser.add_argument('--temperature',
                               type=float,
                               default=1.0)
        subparser.set_defaults(func=evaluate_from_args)

        subparser.add_argument('--offset',
                               action='store_true')
        subparser.set_defaults(func=evaluate_from_args)

        return subparser


# TODO: Make sure this still makes sense...
#  PRESERVED_FIELDS = {'source', 'reset'}
#  
#  
# def _offset(sample, held_over_data):
#     batch_size = sample['reset'].size(0)
#     new_sample = {'source': sample['source'],
#                   'reset': sample['reset']}
#     new_held_over_data = {}
#     for field in sample:
#         if field in PRESERVED_FIELDS:
#             continue
#         if held_over_data is None:
#             prefix = sample[field].new_zeros(batch_size)
#         else:
#             prefix = held_over_data[field]
#         new_sample[field] = torch.cat((prefix.unsqueeze(1), sample[field][:,:-1]), dim=1)
#         new_held_over_data[field] = sample[field][:,-1]
#     return new_sample, new_held_over_data

UNSPLIT_FIELDS = {'reset', 'metadata', 'shortlist'}
def split(batch, split_size: int):
    sequence_length = batch['source']['tokens'].shape[1]
    num_splits = sequence_length // split_size
    if not ((sequence_length % split_size) == 0):
        num_splits += 1
    else:
        logger.warning('Perfect split')

    def _chunk(x, start, stop):
        if isinstance(x, dict):
            return {k: v if k in UNSPLIT_FIELDS else _chunk(v, start, stop) for k, v in x.items()}
        if isinstance(x, torch.Tensor):
            return x[:, start:stop].contiguous()

    chunks = []
    for i in range(num_splits):
        chunk = _chunk(batch, i * split_size, (i + 1) * split_size)

        if i > 0:
            chunk['reset'] = torch.zeros_like(chunk['reset'])

        chunks.append(chunk)
    return chunks


def tile(t, amount):
    if isinstance(t, torch.Tensor):
        args = [1 for _ in t.shape]
        args[0] = amount
        return t.repeat(*args)
    elif isinstance(t, dict):
        return {k: tile(v, amount) for k, v in t.items()}
    elif isinstance(t, list):
        return [x for x in t for _ in range(amount)]


# def logsumexp(prev: torch.FloatTensor,
#               current: torch.FloatTensor,
#               i: int,
#               samples_per_batch: int):
#     # NOTE: n is number of samples
#     current_avg = current.view(samples_per_batch, -1).sum(dim=-1).logsumexp(dim=0) - np.log(samples_per_batch).item()
#     if prev is None:
#         return current_avg
#     a = torch.max(prev, current_avg)
#     sumexp = torch.exp(prev - a) * i / (i + 1) + torch.exp(current_avg - a) / (i + 1)
#     return a + torch.log(sumexp)


def evaluate_perplexity(model: Model,
                        sampler: Model,
                        num_samples: int,
                        instances: Iterator[Instance],
                        data_iterator: DataIterator,
                        cuda_device: int,
                        temperature: float = 1.0,
                        offset: bool = False,
                        samples_per_batch: int = 1,
                        split_size: int = int(1e10)) -> Dict[str, Any]:

    check_for_gpu(cuda_device)
    logger.info('Iterating over dataset')
    # weight = None

    model.eval()
    sampler.eval()

    iterator = data_iterator(instances, num_epochs=1, shuffle=False)
    generator_tqdm = Tqdm.tqdm(iterator, total=0)

    summand = 0.0
    denom = 0.0
    fp = []
    q = []
    all_weights = []

    for batch in generator_tqdm:

        batch_size = batch['reset'].shape[0]

        n_tokens = util.get_text_field_mask(batch['source']).float().sum()
        denom += n_tokens

        epoch_weights = []
        epoch_fp = []
        epoch_q = []

        batch = util.move_to_device(batch, cuda_device)

        # Tile if that's what we're doing
        if samples_per_batch > 1:
            batch = tile(batch, samples_per_batch)

        for i in range(num_samples // samples_per_batch):

            # summand = util.move_to_device(summand, cuda_device)
            # batch = util.move_to_device(batch, cuda_device)

            weights = None
            for j, chunk in enumerate(split(batch, split_size)):
                generator_tqdm.set_description(f"i={i} j={j}")

                chunk_tokens = util.get_text_field_mask(batch['source']).int().sum()
                if chunk_tokens == 0:
                    logger.debug('Zero chunk, skipping')
                    continue

                # Draw a sample
                with torch.no_grad():
                    sampler_output = sampler.sample(**chunk,
                                                    temperature=temperature,
                                                    offset=offset)
                sample_logp = sampler_output['logp']
                sample = sampler_output['sample']

                # if offset:
                #     sample, held_over_data = _offset(sample, held_over_data)

                with torch.no_grad():
                    model_output = model(**sample)

                model_logp = model_output['logp']
                split_weights = (model_logp - sample_logp).view(batch_size, samples_per_batch)

                if weights is None:
                    weights = split_weights
                else:
                    weights += split_weights
                # logger.debug(torch.exp(-split_weights/split_size))

            epoch_weights.append(weights) #.cpu())
            epoch_fp.append(model_logp.view(batch_size, samples_per_batch))# .cpu())
            epoch_q.append(sample_logp.view(batch_size, samples_per_batch))# .cpu())

        # Combine all the epoch weights
        combined_weights = torch.cat(epoch_weights, dim=1)
        combined_fp = torch.cat(epoch_fp, dim=1)
        combined_q = torch.cat(epoch_q, dim=1)
        all_weights.append(combined_weights)
        fp.append(combined_fp)
        q.append(combined_q)

        # Compute importance sampled logp of the sequences in the batch
        logp_hat = combined_weights.logsumexp(dim=1) - math.log(samples_per_batch)
        summand +=  logp_hat.sum()

        logger.info(f'PPL: {torch.exp(-summand / denom)}')

    # Create array of all the weights
    all_weights_array = torch.cat(all_weights, dim=0).cpu().numpy()
    fp_array = torch.cat(fp, dim=0).cpu().numpy()
    q_array = torch.cat(q, dim=0).cpu().numpy()

    # Compute perplexity
    ppl = torch.exp(-summand / denom)

    metrics = {
        'ppl': ppl,
        'weights': all_weights_array,
        'fp': fp_array,
        'q': q_array
    }
    return metrics

def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)
    logger.warning('This code will return improper results if sequences are split')

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

    iterator_params = config.pop('iterator', 'None')
    iterator_params['batch_size'] = args.batch_size
    # Make split size really large to prevent splits (otherwise we'd have to
    # deal with averaging the importance samples across splits ...
    # if args.split_size is not None:
        # iterator_params['split_size'] = args.split_size
    iterator_params['split_size'] = int(1e10)
    iterator_params['truncate'] = False  # TODO: Shouldn't need this anymore...
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)
    metrics = evaluate_perplexity(model, sampler, args.num_samples, instances,
                                  iterator, args.cuda_device, args.temperature,
                                  args.offset, args.samples_per_batch,
                                  args.split_size)

    logger.info('Finished evaluating.')

    if args.out:
        np.save(args.out + '_weights.npy', metrics['weights'])
        np.save(args.out + '_fp.npy', metrics['fp'])
        np.save(args.out + '_q.npy', metrics['q'])

    return metrics

