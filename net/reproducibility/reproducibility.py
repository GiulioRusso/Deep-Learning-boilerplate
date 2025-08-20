import os
import random
import numpy as np
import torch
from lightning.pytorch import seed_everything


def reproducibility(seed: int):
    """
    Set seed for experiment reproducibility

    :param seed: seed
    """

    # validate input
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")

    # set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Lightning handles seeding
    seed_everything(seed, workers=True, verbose=False)

    # set cuDNN for deterministic behavior
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
