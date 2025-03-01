from pandas import read_csv

from net.initialization.dict.metrics import metrics_dict


def metrics_resume(metrics_resume_path: str) -> dict:
    """
    Resume metrics dictionary

    :param metrics_resume_path: metrics path from resume experiment
    :return: metrics dictionary resumed
    """

    header_metrics = ["EPOCH",
                      "LOSS",
                      "ROC AUC",
                      "TIME TRAIN",
                      "TIME VALIDATION",
                      "TIME METRICS"]
    metrics = metrics_dict(metrics_type='train')

    metrics_resume_csv = read_csv(metrics_resume_path, usecols=header_metrics, float_precision='round_trip')

    metrics['ticks'] = metrics_resume_csv["EPOCH"].tolist()
    metrics['loss']['loss'] = metrics_resume_csv["LOSS"].tolist()
    metrics['learning_rate'] = metrics_resume_csv['LEARNING RATE'].tolist()
    metrics['ROC_AUC'] = metrics_resume_csv["ROC_AUC"].tolist()
    metrics['time']['train'] = metrics_resume_csv["TIME TRAIN"].tolist()
    metrics['time']['validation'] = metrics_resume_csv["TIME VALIDATION"].tolist()
    metrics['time']['metrics'] = metrics_resume_csv["TIME METRICS"].tolist()

    return metrics
