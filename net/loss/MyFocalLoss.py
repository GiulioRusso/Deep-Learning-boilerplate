from typing import Any

import torch
import torch.nn.functional as F

from torch import nn


class MyFocalLoss(nn.Module):
    """
    My Focal Loss
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0):
        """
        __init__ method: run one when instantiating the object

        :param alpha: alpha parameter
        :param gamma: gamma parameters
        """

        super(MyFocalLoss, self).__init__()

        # alpha parameter (FocalLoss)
        self.alpha = alpha

        # gamma parameter (FocalLoss)
        self.gamma = gamma

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        forward method: directly call a method in the class when an instance name is called

        :param logits: logits
        :param targets: targets
        :return: focal loss
        """

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        alpha_t = self.alpha
        focal_loss = alpha_t * focal_loss

        loss = focal_loss.mean()

        return loss
