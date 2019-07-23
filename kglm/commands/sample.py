"""
CURRENTLY NOT WORKING.

Sample sentences from the langauge model.
"""
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

from kglm.data import AliasDatabase

logger = logging.getLogger(__name__)

class Sample(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        # pylint: disable=protected-access
        description = '''Generate samples from the model'''
        subparser = parser.add_parser(name, description=description,
                                      help='Generate samples from the model')

        subparser.add_argument('model_archive_file', type=str, help='path to an archived trained model')
        subparser.add_argument('alias_database', type=str, help='path to the alias database')

        subparser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('--output-file', type=str, help='path to output file')

        subparser.add_argument('--batch-size',
                               type=int,
                               default=100,
                               help='Number of samples to create.')

        subparser.add_argument('--length',
                               type=int,
                               default=100,
                               help='Length of generated sequence.')
        subparser.set_defaults(func=sample)

        return subparser

def sample(args: argparse.Namespace):
    model_archive = load_archive(args.model_archive_file,
                                 cuda_device=args.cuda_device)
    config = model_archive.config
    prepare_environment(config)
    model = model_archive.model
    model.eval()

    alias_database = AliasDatabase.load(args.alias_database)

    samples = model.sample(alias_database,
                           batch_size=args.batch_size,
                           length=args.length)


