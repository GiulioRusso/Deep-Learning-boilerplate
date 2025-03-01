import argparse
import os

from net.initialization.folders.default_folders import default_folders_dict
from net.initialization.folders.experiment_folders import experiment_folders_dict
from net.initialization.folders.dataset_folders import dataset_folders_dict
from net.initialization.path.experiment_results_path import experiment_results_path_dict
from net.initialization.utility.create_folder_and_subfolder import create_folder_and_subfolder


def initialization(network_name: str,
                   experiment_ID: str,
                   experiment_resume_ID: str,
                   parser: argparse.Namespace) -> dict:
    """
    Initialization of experiment results folder based on execution mode

    :param network_name: network name
    :param experiment_ID: experiment ID
    :param experiment_resume_ID: experiment resume ID
    :param parser: parser of parameters-parsing
    :return: path dictionary
    """

    # ------------ #
    # FOLDERS DICT #
    # ------------ #
    # default folders
    default_folders = default_folders_dict()

    # dataset folders
    dataset_folders = dataset_folders_dict()

    # experiment folder structure
    experiment_folders = experiment_folders_dict(parser=parser)

    # ------- #
    # DATASET #
    # ------- #

    # TODO: update the paths based on your data structure

    # images
    images_path = os.path.join(default_folders['datasets'], str(parser.dataset), dataset_folders["images"])
    # annotations
    annotations_path = os.path.join(default_folders['datasets'], str(parser.dataset), dataset_folders["annotations"])

    # TODO: evaluate list, split and statistics for your dataset and specify their path

    # lists
    lists_path = os.path.join(default_folders['datasets'], str(parser.dataset), dataset_folders['lists'])
    # data split
    split_filename = "split-{}-fold.csv".format(parser.split)
    split_path = os.path.join(default_folders['datasets'], dataset_folders["split"], split_filename)
    # statistics
    statistics_filename = "split-{}-statistics.csv".format(parser.split)
    statistics_path = os.path.join(default_folders['datasets'], parser.dataset, dataset_folders['statistics'], statistics_filename)

    path_dataset_dict = {
        'images': images_path,
        'annotations': annotations_path,

        'list': lists_path,
        'split': split_path,
        'statistics': statistics_path
    }

    # ---------- #
    # EXPERIMENT #
    # ---------- #
    # experiment path
    experiment_name = network_name + "|" + experiment_ID
    experiment_path = os.path.join(default_folders['experiments'], experiment_name)

    # experiment resume path
    experiment_resume_name = network_name + "|" + experiment_resume_ID
    experiment_resume_results_path = os.path.join(default_folders['experiments'], experiment_resume_name)

    # experiment subfolders structure
    experiment_results_path = experiment_results_path_dict(experiment_path=experiment_path,
                                                           experiment_folders=experiment_folders)

    # experiment resume path
    experiment_resume_results_path = experiment_results_path_dict(experiment_path=experiment_resume_results_path,
                                                                  experiment_folders=experiment_folders)

    # create experiment folder and subfolders
    if parser.mode in ['train', 'resume']:
        create_folder_and_subfolder(main_path=experiment_path,
                                    subfolder_path_dict=experiment_results_path)

    # ----------- #
    # RESULT PATH #
    # ----------- #

    # TODO: define your result filename and metrics based on your needs

    # classifications file
    classifications_validation_filename = "classifications-validation|" + experiment_ID + ".csv"
    classifications_validation_path = os.path.join(experiment_results_path['classifications'], classifications_validation_filename)

    classifications_test_filename = "classifications-test|" + experiment_ID + ".csv"
    classifications_test_path = os.path.join(experiment_results_path['classifications'], classifications_test_filename)

    # metrics test
    metrics_test_filename = "metrics-test|" + experiment_ID + ".csv"
    metrics_test_path = os.path.join(experiment_results_path['metrics'], metrics_test_filename)

    # metrics train and resume
    metrics_train_filename = "metrics-train|" + experiment_ID + ".csv"
    metrics_train_path = os.path.join(experiment_results_path['metrics'], metrics_train_filename)

    metrics_train_resume_filename = "metrics-train|" + experiment_resume_ID + ".csv"
    metrics_train_resume_path = os.path.join(experiment_resume_results_path['metrics'], metrics_train_resume_filename)

    # models best
    model_best_filename = network_name + "-best-model|" + experiment_ID + ".tar"
    model_best_path = os.path.join(experiment_results_path['models'], model_best_filename)

    # models resume
    model_resume_filename = network_name + "-best-model|" + experiment_ID + ".tar"
    model_resume_path = os.path.join(experiment_results_path['models'], model_resume_filename)

    model_resume_to_load_filename = network_name + "-resume-model|" + experiment_resume_ID + ".tar"
    model_resume_to_load_path = os.path.join(experiment_resume_results_path['models'], model_resume_to_load_filename)

    # plots train
    loss_filename = "Loss|" + experiment_ID + ".png"
    loss_path = os.path.join(experiment_results_path['plots'], loss_filename)

    # plots validation
    roc_auc_filename = "ROC-AUC|" + experiment_ID + ".png"
    roc_auc_path = os.path.join(experiment_results_path['plots'], roc_auc_filename)

    path = {
        'dataset': path_dataset_dict,

        'classifications': {
            'validation': classifications_validation_path,
            'test': classifications_test_path,
        },

        'metrics': {
            'train': metrics_train_path,
            'resume': metrics_train_resume_path,
            'test': metrics_test_path,
        },

        'models': {
            'best': model_best_path,
            'resume': model_resume_path,
            'resume_to_load': model_resume_to_load_path,
        },

        'plots': {
            'train': {
                'loss': loss_path,
            },
            'validation': {
                'ROC_AUC': roc_auc_path,
            }
        },

    }

    return path




