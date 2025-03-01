import argparse
from typing import Tuple

from net.initialization.utility.parameters_ID import parameters_ID


def experimentID(parser: argparse.Namespace) -> Tuple[str, str]:
    """
    Concatenate experiment-ID parameters

    :param parser: parser of parameters-parsing
    :return: experiment-ID and experiment-ID for resume
    """

    # ------------- #
    # PARAMETERS ID #
    # ------------- #
    # parameters ID dictionary
    parameters_ID_dict = parameters_ID(parser=parser)

    # build experiment ID
    experiment_ID = parameters_ID_dict['dataset'] + "|" + parameters_ID_dict['split'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['GPU']
    experiment_resume_ID = parameters_ID_dict['dataset'] + "|" + parameters_ID_dict['split'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep_to_resume'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['GPU']

    return experiment_ID, experiment_resume_ID
