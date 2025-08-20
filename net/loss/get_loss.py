import argparse
import sys
from typing import Union

import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from net.loss.MyFocalLoss import MyFocalLoss
from net.loss.MySigmoidFocalLoss import MySigmoidFocalLoss


def get_loss(loss: str,
             parser: argparse.Namespace) -> Union[CrossEntropyLoss, BCEWithLogitsLoss, MySigmoidFocalLoss, MyFocalLoss]:
    """
    Get loss

    :param loss: loss name
    :param parser: parser of parameters-parsing
    :return: criterion (loss)
    """

    # TODO: Define the Loss function for your task

    # ------------------ #
    # CROSS ENTROPY LOSS #
    # ------------------ #
    if loss == 'CrossEntropyLoss':
        criterion = CrossEntropyLoss()

    # ------------------------------- #
    # BINARY CROSS ENTROPY (BCE) LOSS #
    # ------------------------------- #
    elif loss == 'BCEWithLogitsLoss':
        criterion = BCEWithLogitsLoss()

    # ------------------ #
    # SIGMOID FOCAL LOSS #
    # ------------------ #
    elif loss == 'SigmoidFocalLoss':
        criterion = MySigmoidFocalLoss(alpha=parser.alpha,
                                       gamma=parser.gamma)

    # ---------- #
    # FOCAL LOSS #
    # ---------- #
    elif loss == 'FocalLoss':

        criterion = MyFocalLoss(alpha=parser.alpha,
                                gamma=parser.gamma)

    else:
        raise ValueError(f"Unknown loss in {__file__}.Got {loss}.")

    return criterion
