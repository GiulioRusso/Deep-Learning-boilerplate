import numpy as np
from sklearn.metrics import roc_auc_score


def ROC_AUC(classifications: np.ndarray) -> float:
    """
    Compute the Area Under the ROC curve

    :param classifications: classifications
    :return: accuracy score
    """

    predictions_score = classifications[:, 1]  # predictions score
    true_labels = classifications[:, 3]  # true labels (ground truth)

    try:
        ROC_AUC_value = roc_auc_score(y_true=true_labels.astype(int),
                                      y_score=predictions_score.astype(float))
    except ValueError:
        ROC_AUC_value = 0

    return ROC_AUC_value
