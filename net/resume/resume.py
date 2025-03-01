import argparse
from typing import Union, Tuple

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts

from net.model.utility.load_model import load_resume_model
from net.resume.metrics_resume import metrics_resume
from net.resume.metrics_train_resume import metrics_train_resume_csv
from net.resume.resume_models import resume_models


def resume(experiment_ID: str,
           net: torch.nn.Module,
           optimizer: Union[Adam, SGD],
           scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
           path: dict,
           parser: argparse.Namespace) -> Tuple[int, int]:
    """
    Resume Training

    :param experiment_ID: experiment ID
    :param net: net
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param parser: parser of parameters-parsing
    :return:
    """

    # train epochs range
    start_epoch_train = parser.epoch_to_resume + 1  # star train
    stop_epoch_train = start_epoch_train + (parser.epochs - parser.epoch_to_resume)  # stop train

    # ----------------- #
    # LOAD RESUME MODEL #
    # ----------------- #
    print("\n------------------"
          "\nLOAD RESUME MODEL:"
          "\n------------------")

    # load resume model
    load_resume_model(net=net,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      path=path['models']['resume_to_load'])

    # -------------- #
    # RESUME METRICS #
    # -------------- #
    print("\n---------------"
          "\nRESUME METRICS:"
          "\n---------------")
    # resume metrics performance
    metrics = metrics_resume(metrics_resume_path=path['metrics']['resume'])

    # resume metrics-train.csv
    metrics_train_resume_csv(metrics_path=path['metrics']['train'],
                             metrics=metrics)

    # ------------- #
    # RESUME MODELS #
    # ------------- #
    print("\n--------------"
          "\nRESUME MODELS:"
          "\n--------------")
    # resume best models
    resume_models(path_best_models=path['models']['best'],
                  path_resume_models=path['models']['resume_to_load'])

    return start_epoch_train, stop_epoch_train
