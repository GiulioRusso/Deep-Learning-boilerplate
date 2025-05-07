from typing import Union

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts


def load_best_model(net: torch.nn.Module,
                    metrics_type: str,
                    path: str):
    """
    Load best-model

    :param net: net
    :param metrics_type: metrics type
    :param path: path
    """

    # load model
    load_model = torch.load(path)

    # load state dict
    net.load_state_dict(load_model['net_state_dict'])


def load_resume_model(net: torch.nn.Module,
                      optimizer: Union[Adam, SGD],
                      scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
                      path: str):
    """
    Load resume-model

    :param net: net
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param path: path
    """

    # load model
    load_model = torch.load(path)

    # load state dict
    net.load_state_dict(load_model['net_state_dict'])

    # load optimizer state dict
    optimizer.load_state_dict(load_model['optimizer'])

    # load scheduler state dict
    scheduler.load_state_dict(load_model['scheduler'])

    # load rng state
    rng_state_resume = load_model['rng_state']

    # set resume seed
    torch.set_rng_state(rng_state_resume)

