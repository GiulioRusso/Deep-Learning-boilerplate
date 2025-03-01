import argparse
import sys
from typing import Tuple

from net.initialization.utility.parameters_ID import parameters_ID
from net.utility.msg.msg_error import msg_error


def experimentID_fold(typeID: str,
                      parser: argparse.Namespace) -> Tuple[str, str]:
    """
    Get experiment-1-fold-ID and experiment-2-fold-ID

    :param typeID: type experiment-ID
    :param parser: parser of parameters-parsing
    :return: experiment-1-fold-ID,
             experiment-2-fold-ID
    """

    # ------------- #
    # PARAMETERS ID #
    # ------------- #
    # parameters ID dictionary
    parameters_ID_dict = parameters_ID(parser=parser)

    experiment_1_fold_ID = parameters_ID_dict['dataset'] + "|" + parameters_ID_dict['split_1_fold'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['GPU']
    experiment_2_fold_ID = parameters_ID_dict['dataset'] + "|" + parameters_ID_dict['split_2_fold'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['GPU']

    return experiment_1_fold_ID, experiment_2_fold_ID
