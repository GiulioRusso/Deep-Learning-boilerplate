import argparse
from typing import Union

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts

from net.parameters.parameters_choices import parameters_choices


def current_learning_rate(scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
                          optimizer: Union[Adam, SGD],
                          parser: argparse.Namespace) -> float:
    """
    Get current learning rate according to scheduler and optimizer type

    :param scheduler: scheduler
    :param optimizer: optimizer
    :param parser: parser of parameters-parsing
    :return: current learning rate
    """

    if parser.scheduler == 'ReduceLROnPlateau':
        learning_rate = my_get_last_lr(optimizer=optimizer)

    elif parser.scheduler == 'StepLR':
        learning_rate = scheduler.get_last_lr()[0]

    elif parser.scheduler == 'CosineAnnealing':
        learning_rate = scheduler.get_last_lr()[0]

    else:
        raise ValueError(f"Unknown normalization type in {__file__}. Choices are {parameters_choices['scheduler']}, but got {parser.scheduler} instead")

    return learning_rate


def my_get_last_lr(optimizer: Union[Adam, SGD]) -> float:
    """
    Get last Learning Rate
    :param optimizer: optimizer
    :return: last learning rate
    """

    for param_group in optimizer.param_groups:
        return param_group['lr']
