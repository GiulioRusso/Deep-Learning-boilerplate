import csv
import os

from net.initialization.header.metrics import metrics_header
from net.metrics.utility.my_round_value import my_round_value
from net.metrics.utility.timer import timer


def metrics_test_csv(metrics_path: str,
                     metrics: dict):
    """
    Save metrics-test.csv

    :param metrics_path: metrics path
    :param metrics: metrics dictionary
    """

    # metrics round
    ROC_AUC = my_round_value(value=metrics['ROC_AUC'][-1], digits=3)

    # metrics timer conversion
    metrics_time_test = timer(time_elapsed=metrics['time']['test'][-1])
    metrics_time_metrics = timer(time_elapsed=metrics['time']['metrics'][-1])

    # check if file exists
    file_exists = os.path.isfile(metrics_path)

    # save metrics-train.csv
    with open(metrics_path, 'a') as file:
        # writer
        writer = csv.writer(file)

        if not file_exists:
            # write header
            header = metrics_header(metrics_type='test')
            writer.writerow(header)

        # write row
        writer.writerow([ROC_AUC,
                         "{} h {} m {} s".format(metrics_time_test['hours'], metrics_time_test['minutes'], metrics_time_test['seconds']),
                         "{} h {} m {} s".format(metrics_time_metrics['hours'], metrics_time_metrics['minutes'], metrics_time_metrics['seconds'])])
