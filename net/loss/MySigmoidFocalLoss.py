import torch
import torch.nn.functional as F

from torch import nn


class MySigmoidFocalLoss(nn.Module):
    """
    My Sigmoid Focal Loss
    """

    def __init__(self,
                 alpha: float,
                 gamma: float):
        """
        __init__ method: run one when instantiating the object

        :param alpha: alpha parameter
        :param gamma: gamma parameters
        """

        super(MySigmoidFocalLoss, self).__init__()

        # alpha parameter (FocalLoss)
        self.alpha = alpha

        # gamma parameter (FocalLoss)
        self.gamma = gamma

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        forward method: directly call a method in the class when an instance name is called

        :param input: input
        :param target: target
        :return: sigmoid focal loss
        """

        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        # mean reduction loss
        loss = loss.mean()

        return loss
