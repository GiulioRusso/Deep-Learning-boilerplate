import os


def experiment_complete_result_path_dict(experiment_path: str,
                                         experiment_complete_folders: dict) -> dict:
    """
    Concatenate experiment GravityNet complete result path

    :param experiment_path: experiment path
    :param experiment_complete_folders: experiment complete folders dictionary
    :return: experiment complete result path dictionary
    """

    # metrics test
    metrics_test_path = os.path.join(experiment_path, experiment_complete_folders['metrics_test'])

    # output
    output_path = os.path.join(experiment_path, experiment_complete_folders['output'])

    # plots test
    plots_test_path = os.path.join(experiment_path, experiment_complete_folders['plots_test'])

    experiment_result_path = {
        'metrics_test': metrics_test_path,
        'output': output_path,
        'plots_test': plots_test_path,
    }

    return experiment_result_path
