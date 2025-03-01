import sys

import numpy as np
import pandas as pd
from pandas import read_csv

from net.initialization.header.classifications import classifications_header


def classifications_concatenation(detections_path_1_fold: str,
                                  detections_path_2_fold: str,
                                  detections_concatenated_path: str):
    """
    Save classifications 1-fold and 2-fold concatenation

    :param detections_path_1_fold: classifications 1-fold path
    :param detections_path_2_fold:  classifications 2-fold path
    :param detections_concatenated_path: classifications concatenation path
    """

    # read classifications 1-fold
    classifications_1_fold = read_csv(filepath_or_buffer=detections_path_1_fold, usecols=classifications_header()).values
    print("classifications 1-fold reading: COMPLETE")

    # read classifications 2-fold
    classifications_2_fold = read_csv(filepath_or_buffer=detections_path_2_fold, usecols=classifications_header()).values
    print("classifications 2-fold reading: COMPLETE")

    # classifications complete
    classifications_complete = np.concatenate((classifications_1_fold, classifications_2_fold), axis=0)
    classifications_csv = pd.DataFrame(classifications_complete)
    classifications_csv.to_csv(detections_concatenated_path, mode='w', index=False, header=classifications_header(), float_format='%g')
    print("classifications complete saving: COMPLETE")
