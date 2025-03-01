import os


def experiment_results_path_dict(experiment_path: str,
                                 experiment_folders: dict) -> dict:
    """
    Concatenate experiment result path

    :param experiment_path: experiment path
    :param experiment_folders: experiment folders dictionary
    :return: experiment complete result path dictionary
    """


    # log
    log_path = os.path.join(experiment_path, experiment_folders['log'])

    # classifications
    classifications_path = os.path.join(experiment_path, experiment_folders['classifications'])

    # metrics
    metrics_path = os.path.join(experiment_path, experiment_folders['metrics'])

    # models
    models_path = os.path.join(experiment_path, experiment_folders['models'])

    # output
    output_path = os.path.join(experiment_path, experiment_folders['output'])

    # plots
    plots_path = os.path.join(experiment_path, experiment_folders['plots'])

    experiment_result_path = {
        'log': log_path,

        'classifications': classifications_path,

        'models': models_path,

        'metrics': metrics_path,

        'output': output_path,

        'plots': plots_path,
    }

    return experiment_result_path
