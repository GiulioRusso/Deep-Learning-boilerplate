import os
import time

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from net.classifications.classifications_validation import classifications_validation


def validation(num_epoch: int,
               epochs: int,
               net: torch.nn.Module,
               dataloader: DataLoader,
               classifications_path: str,
               device: torch.device):
    """
    Validation function

    :param num_epoch: number of epochs
    :param epochs: epochs
    :param net: net
    :param dataloader: dataloader
    :param classifications_path: classifications path
    :param device: device
    """

    # switch to test mode
    net.eval()

    # if classifications already exists: delete
    if os.path.isfile(classifications_path):
        os.remove(classifications_path)

    # do not accumulate gradients (faster)
    with torch.no_grad():
        # for each batch in dataloader
        for num_batch, batch in enumerate(dataloader):
            # init batch time
            time_batch_start = time.time()

            # get data from dataloader
            images, annotations, filenames = batch['image'].float().to(device), batch['annotation'].to(device), batch['filename'].to(device)

            # forward pass
            classifications = net(images)  # B x num_classes
            probabilities = softmax(classifications, dim=1)
            predictions = torch.argmax(probabilities, dim=1).to(device).float()  # from B x num_classes probabilities to B with predicted labels

            # save classifications.csv
            classifications_validation(filenames=filenames,
                                       predictions=predictions,
                                       scores=probabilities,
                                       annotations=annotations,
                                       classifications_path=classifications_path)

            # batch time
            time_batch = time.time() - time_batch_start

            # show
            print("Epoch: {}/{} |".format(num_epoch, epochs),
                  "Batch: {}/{} |".format(num_batch + 1, len(dataloader)),
                  "Time: {:.0f} s ".format(int(time_batch) % 60))
