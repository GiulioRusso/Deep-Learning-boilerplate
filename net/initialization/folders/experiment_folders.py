import argparse


def experiment_folders_dict(parser: argparse.Namespace) -> dict:
    """
    Experiment folders dictionary

    :param parser: parser of parameters-parsing
    :return: folders dictionary
    """

    # TODO: fill the dict with your experiment info to track
    experiment_result_folders = {
        'log': 'log',

        'classifications': 'classifications',

        'metrics': 'metrics',

        'models': 'models',

        'output': 'output',

        'plots': 'plots',
    }

    return experiment_result_folders
