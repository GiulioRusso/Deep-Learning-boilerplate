import csv

import numpy as np


def save_coords(x: np.ndarray,
                y: np.ndarray,
                path: str):
    """
    Save coords

    :param x: coords x
    :param y: coords y
    :param coords_type: coords type
    :param path: path to save coords
    """

    # save coords
    with open(path, 'w') as file:
        # writer
        writer = csv.writer(file)

        # write header
        header = ["FPR", "TPR"]
        writer.writerow(header)

        # iterate row writer
        for row in range(len(x)):
            writer.writerow([x[row], y[row]])
