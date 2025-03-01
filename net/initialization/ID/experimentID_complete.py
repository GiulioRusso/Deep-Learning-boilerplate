import argparse

from net.initialization.utility.parameters_ID import parameters_ID


def experimentID_complete(parser: argparse.Namespace) -> str:
    """
    Concatenate experiment-ID complete

    :param parser: parser of parameters-parsing
    :return: experiment-ID complete
    """

    # ------------- #
    # PARAMETERS ID #
    # ------------- #
    # parameters ID dictionary
    parameters_ID_dict = parameters_ID(parser=parser)

    # build experiment complete ID
    experiment_complete_ID = parameters_ID_dict['dataset'] + "|" + parameters_ID_dict['norm'] + "|" + parameters_ID_dict['ep'] + "|" + parameters_ID_dict['lr'] + "|" + parameters_ID_dict['bs'] + "|" + parameters_ID_dict['backbone'] + "|" + parameters_ID_dict['GPU']

    return experiment_complete_ID
