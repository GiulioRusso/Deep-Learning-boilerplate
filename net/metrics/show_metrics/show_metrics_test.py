from net.metrics.utility.my_round_value import my_round_value
from net.metrics.utility.timer import timer


def show_metrics_test(metrics: dict):
    """
    Show metrics-test

    :param metrics: metrics dictionary
    """

    # metrics round
    ROC_AUC = my_round_value(value=metrics['ROC_AUC'][-1], digits=3)

    # metrics timer conversion
    metrics_time_test = timer(time_elapsed=metrics['time']['test'][-1])
    metrics_time_metrics = timer(time_elapsed=metrics['time']['metrics'][-1])

    print("\n-------------"
          "\nMETRICS TEST:"
          "\n-------------"
          "\nROC AUC: {}".format(ROC_AUC))

    print("\n-----"
          "\nTIME:"
          "\n-----"
          "\nTEST: {} h {} m {} s".format(metrics_time_test['hours'], metrics_time_test['minutes'], metrics_time_test['seconds']),
          "\nMETRICS: {} h {} m {} s".format(metrics_time_metrics['hours'], metrics_time_metrics['minutes'], metrics_time_metrics['seconds']))
