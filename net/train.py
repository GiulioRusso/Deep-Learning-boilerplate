import argparse
import time
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from net.loss.MyFocalLoss import MyFocalLoss
from net.loss.MySigmoidFocalLoss import MySigmoidFocalLoss
from torch.utils.data import DataLoader

from net.metrics.utility.timer import timer


def train(num_epoch: int,
          epochs: int,
          net: torch.nn.Module,
          num_classes: int,
          dataloader: DataLoader,
          optimizer: Union[Adam, SGD],
          scheduler: Union[ReduceLROnPlateau, StepLR, CosineAnnealingWarmRestarts],
          criterion: Union[CrossEntropyLoss, BCEWithLogitsLoss, MySigmoidFocalLoss, MyFocalLoss],
          device: torch.device,
          parser: argparse.Namespace) -> float:
    """
    Training function

    :param num_epoch: num epoch
    :param epochs: epochs
    :param net: net
    :param dataloader: dataloader
    :param optimizer: optimizer
    :param scheduler: scheduler
    :param criterion: criterion (loss)
    :param device: device
    :param parser: parser of parameters-parsing

    :return: average epoch loss
    """

    # switch to train mode
    net.train()

    # reset performance measures
    epoch_loss_hist = []

    # for each batch in dataloader
    for num_batch, batch in enumerate(dataloader):

        # init batch time
        time_batch_start = time.time()

        # get data from dataloader
        images, annotations = batch['image'].float().to(device), batch['annotation'].to(device)

        # zero (init) the parameter gradients
        optimizer.zero_grad()

        # forward pass
        classifications = net(images)  # B x num_classes

        # ---- #
        # LOSS #
        # ---- #
        # evaluate loss
        if parser.loss == 'CrossEntropyLoss':  # logits: B x num_classes, targets: B

            loss = criterion(classifications.to(device), annotations)

        elif parser.loss == 'BCEWithLogitsLoss':  # logits: B x num_classes, targets: B x num_classes
            # reshape from B to B x num_classes
            annotations_one_hot = F.one_hot(annotations, num_classes).float().to(device)

            loss = criterion(classifications.to(device), annotations_one_hot)

        elif parser.loss == 'FocalLoss':  # logits: B x num_classes, targets: B

            loss = criterion(classifications.to(device), annotations)

        # append epoch loss
        epoch_loss_hist.append(float(loss))

        # loss gradient backpropagation
        loss.backward()

        # clip gradient
        if parser.clip_gradient:
            clip_grad_norm_(parameters=net.parameters(),
                            max_norm=parser.max_norm)

        # net parameters update
        optimizer.step()

        # batch time
        time_batch = time.time() - time_batch_start

        # batch time conversion
        batch_time = timer(time_elapsed=time_batch)

        print("Epoch: {}/{} |".format(num_epoch, epochs),
              "Batch: {}/{} |".format(num_batch + 1, len(dataloader)),
              "Loss: {:1.5f} |".format(float(loss)),
              "Time: {:.0f} s ".format(batch_time['seconds']))

        del loss
        del time_batch

    # step learning rate scheduler
    if parser.scheduler == 'ReduceLROnPlateau':
        scheduler.step(np.mean(epoch_loss_hist))
    elif parser.scheduler == 'StepLR':
        scheduler.step()

    # return avg epoch loss
    return sum(epoch_loss_hist) / len(dataloader)
