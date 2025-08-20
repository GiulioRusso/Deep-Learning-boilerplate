import argparse
import re

from net.initialization.parameters_ID import parameters_ID


def experiment_ID(parser: argparse.Namespace) -> str:
    """
    Concatenate experiment-initialization parameters

    :param parser: parser of parameters-parsing
    :return: experiment-initialization and experiment-initialization for resume
    """

    # ------------- #
    # PARAMETERS ID #
    # ------------- #
    # parameters initialization dictionary
    parameters_ID_dict = parameters_ID(parser=parser)

    # build experiment initialization
    experiment_ID = build_experiment_ID(parser=parser, params=parameters_ID_dict)

    return experiment_ID


def build_experiment_ID(parser: argparse.Namespace,
                        params: dict) -> str:
    """
    Build the experiment initialization string.

    :param parser: Parameter parsing.
    :param params: String initialization for parameters.
    :return: The experiment initialization string.
    """

    # TODO: Build the experiment ID as you need

    parts = [
        params['dataset'],
        params['augmentation'] if parser.do_augmentation else '',
        params['network_name'] if parser.network_name != 'default' else '',
        params['ep'] if parser.mode in ['train', 'test'] else params['ep_to_resume'],
        params['bs'],
        params['scheduler'],
        params['lr'],
        params['optimizer'],
        params['loss']
    ]

    # join the strings with '|' and collapse to '|' any more than one consecutive '|'
    raw_id = "|".join(parts)
    collapsed_id = re.sub(r'\|{2,}', '|', raw_id).rstrip('|')
    return collapsed_id

