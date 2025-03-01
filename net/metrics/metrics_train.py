import csv
import os

from net.metrics.utility.my_round_value import my_round_value
from net.metrics.utility.timer import timer


def metrics_train_csv(metrics_path: str,
                      metrics: dict):
    """
    Save metrics-train.csv

    :param metrics_path: metrics path
    :param metrics: metrics dictionary
    """

    # metrics round
    ticks = metrics['ticks'][-1]
    loss = my_round_value(value=metrics['loss'][-1], digits=3)
    ROC_AUC = my_round_value(value=metrics['ROC_AUC'][-1], digits=3)

    # metrics timer conversion
    metrics_time_train = timer(time_elapsed=metrics['time']['train'][-1])
    metrics_time_validation = timer(time_elapsed=metrics['time']['validation'][-1])
    metrics_time_metrics = timer(time_elapsed=metrics['time']['metrics'][-1])

    # check if file exists
    file_exists = os.path.isfile(metrics_path)

    # save metrics-train.csv
    with open(metrics_path, 'a') as file:
        # writer
        writer = csv.writer(file)

        if not file_exists:
            # write header
            header = ["EPOCH",
                      "LOSS",
                      "ROC AUC",
                      "TIME TRAIN",
                      "TIME VALIDATION",
                      "TIME METRICS"]
            writer.writerow(header)

        # write row
        writer.writerow([ticks,
                         loss,
                         ROC_AUC,
                         "{} h {} m {} s".format(metrics_time_train['hours'], metrics_time_train['minutes'], metrics_time_train['seconds']),
                         "{} h {} m {} s".format(metrics_time_validation['hours'], metrics_time_validation['minutes'], metrics_time_validation['seconds']),
                         "{} h {} m {} s".format(metrics_time_metrics['hours'], metrics_time_metrics['minutes'], metrics_time_metrics['seconds'])])
