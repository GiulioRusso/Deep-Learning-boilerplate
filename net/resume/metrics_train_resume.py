import csv

from net.metrics.utility.my_notation import scientific_notation


def metrics_train_resume_csv(metrics_path: str,
                             metrics: dict):
    """
    Resume metrics-train.csv

    :param metrics_path: metrics path
    :param metrics: metrics dictionary
    """

    with open(metrics_path, 'w') as file:
        writer = csv.writer(file)

        # write header
        header = ["EPOCH",
                  "LOSS",
                  "ROC AUC",
                  "TIME TRAIN",
                  "TIME VALIDATION",
                  "TIME METRICS"]
        writer.writerow(header)

        # iterate row writer
        for row in range(len(metrics['ticks'])):
            writer.writerow([metrics['ticks'][row],
                             metrics['loss']['loss'][row],
                             scientific_notation(number=metrics['learning_rate'][row]),
                             metrics['ROC_AUC'][row],
                             metrics['time']['train'][row],
                             metrics['time']['validation'][row],
                             metrics['time']['metrics'][row]])
