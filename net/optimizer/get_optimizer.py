import argparse
from typing import Union, Iterator

from torch.nn import Parameter
from torch.optim import Adam, SGD


def get_optimizer(net_parameters: Iterator[Parameter],
                  parser: argparse.Namespace) -> Union[Adam, SGD]:
    """
    Get optimizer

    :param net_parameters: net parameters
    :param parser: parser of parameters-parsing
    :return: optimizer
    """

    # ---- #
    # ADAM #
    # ---- #
    if parser.optimizer == 'Adam':
        optimizer = Adam(params=net_parameters,
                         lr=parser.learning_rate)

    # --- #
    # SGD #
    # --- #
    elif parser.optimizer == 'SGD':
        optimizer = SGD(net_parameters,
                        lr=parser.learning_rate,
                        momentum=parser.lr_momentum)

    else:
        raise ValueError(f"Unknown optimizer in {__file__}. Got {parser.optimizer}.")

    return optimizer
