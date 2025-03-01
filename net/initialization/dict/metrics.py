from net.parameters.parameters_choices import parameters_choices


def metrics_dict(metrics_type: str) -> dict:
    """
    Get metrics dictionary according to type

    :param metrics_type: metrics type
    :return: metrics dictionary
    """

    if metrics_type == 'train':
        metrics = {
            'ticks': [],

            'loss': [],

            'learning_rate': [],

            'ROC_AUC': [],

            'time': {
                'train': [],
                'validation': [],
                'metrics': []
            }
        }

    elif metrics_type == 'test':
        metrics = {
            'ROC_AUC': [],

            'time': {
                'test': [],
                'metrics': []
            }
        }

    else:
        raise ValueError(f"Unknown metric type in {__file__}. Choices are {parameters_choices['optimizer']}, but got {metrics_type} instead")

    return metrics
