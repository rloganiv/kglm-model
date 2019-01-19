import argparse
import logging
from typing import Any, Dict

from allennlp.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)


class Evaluate(Subcommand):
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = '''Evaluate the specified model using importance sampling'''
        subparser = parser.add_parser(name, description=description,
                                      help='Evaluate the specified module using importance sampling')
        subparser.set_defaults(func=evaluate_from_args)
        return subparser


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    raise NotImplementedError
