import os

from typing import List

import pandas as pd
import torch


def classifications_test(filenames: List[str],
                         predictions: torch.Tensor,
                         scores: torch.Tensor,
                         annotations: torch.Tensor,
                         classifications_path: str):
    """
    Compute classifications in validation and save in classifications.csv

    :param filenames: filenames
    :param predictions: classifications positive
    :param scores: classifications scores
    :param annotations: annotations
    :param classifications_path: classifications path
    """

    # -------------- #
    # CLASSIFICATION #
    # -------------- #
    # predictions
    predictions = predictions.tolist()

    # predictions
    scores = scores.tolist()

    # init
    annotations = annotations.tolist()

    # --------- #
    # DATA ROWS #
    # --------- #
    data_rows = [
        [filenames[i], predictions[i], scores[i], annotations[i]] for i in range(len(filenames))
    ]

    # define DataFrame
    df = pd.DataFrame(data_rows)

    # -------------------- #
    # SAVE CLASSIFICATIONS #
    # -------------------- #
    classifications_header = ["FILENAME", "PREDICTION", "SCORE", "ANNOTATION"]
    if not os.path.exists(classifications_path):
        df.to_csv(path_or_buf=classifications_path, mode='a', index=False, header=classifications_header, float_format='%g')  # write header
    else:
        df.to_csv(path_or_buf=classifications_path, mode='a', index=False, header=False, float_format='%g')  # write without header
