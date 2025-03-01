from typing import Union

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts


def save_best_model(epoch: int,
                    net: torch.nn.Module,
                    metrics: dict,
                    metrics_type: str,
                    optimizer: Union[Adam, SGD],
                    scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
                    path: str):
    """
    Save best model

    :param epoch: num epoch
    :param net: net
    :param metrics: metrics
    :param metrics_type: metrics type
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param path: path to save model
    """

    # save model
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        metrics_type: max(metrics),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state()
    }, path)


def save_resume_model(epoch: int,
                      net: torch.nn.Module,
                      ROC_AUC: float,
                      optimizer: Union[Adam, SGD],
                      scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
                      path: str):
    """
    Save resume model

    :param epoch: num epoch
    :param net: net
    :param ROC_AUC: ROC AUC
    :param accuracy_top_n_5: accuracy top-n 5
    :param accuracy_top_n_10: accuracy top-n 10
    :param accuracy_top_n_20: accuracy top-n 20
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param path: path to save resume model
    """

    # save model
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'ROC AUC': ROC_AUC,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'rng_state': torch.get_rng_state()
    }, path)
