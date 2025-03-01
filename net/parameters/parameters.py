import argparse

from net.parameters.parameters_choices import parameters_choices
from net.parameters.parameters_default import parameters_default
from net.parameters.parameters_help import parameters_help


def parameters_parsing() -> argparse.Namespace:
    """
    Definition of parameters-parsing for each execution mode

    :return: parser of parameters parsing
    """

    # parser
    parser = argparse.ArgumentParser(description='Argument Parser')

    # -------------- #
    # EXECUTION MODE #
    # -------------- #
    parser.add_argument('--mode',
                        type=str,
                        choices=parameters_choices['mode'],
                        help=parameters_help['mode'])

    # ------- #
    # DATASET #
    # ------- #

    parser.add_argument('--dataset',
                        type=str,
                        default=parameters_default['dataset'],
                        help=parameters_help['dataset'])

    parser.add_argument('--num_classes',
                        type=str,
                        default=parameters_default['num_classes'],
                        help=parameters_help['num_classes'])

    parser.add_argument('--image_height',
                        type=int,
                        default=parameters_default['image_height'],
                        help=parameters_help['image_height'])

    parser.add_argument('--image_width',
                        type=int,
                        default=parameters_default['image_width'],
                        help=parameters_help['image_width'])

    parser.add_argument('--split',
                        type=str,
                        default=parameters_default['split'],
                        help=parameters_help['split'])

    parser.add_argument('--norm',
                        type=str,
                        default=parameters_default['norm'],
                        help=parameters_help['norm'])

    parser.add_argument('--do_dataset_augmentation',
                        action='store_true',
                        default=parameters_default['do_dataset_augmentation'],
                        help=parameters_help['do_dataset_augmentation'])

    # ------ #
    # DEVICE #
    # ------ #

    parser.add_argument('--GPU',
                        type=str,
                        default=parameters_default['GPU'],
                        choices=parameters_choices['GPU'],
                        help=parameters_help['GPU'])

    parser.add_argument('--num_threads',
                        type=int,
                        default=parameters_default['num_threads'],
                        help=parameters_help['num_threads'])

    # --------------- #
    # REPRODUCIBILITY #
    # --------------- #

    parser.add_argument('--seed',
                        type=int,
                        default=parameters_default['seed'],
                        help=parameters_help['seed'])

    parser.add_argument('--num_workers',
                        type=int,
                        default=parameters_default['num_workers'],
                        help=parameters_help['num_workers'])

    # ----------- #
    # DATA LOADER #
    # ----------- #

    parser.add_argument('--batch_size_test',
                        type=int,
                        default=parameters_default['batch_size_test'],
                        help=parameters_help['batch_size_test'])

    parser.add_argument('--batch_size_val',
                        type=int,
                        default=parameters_default['batch_size_val'],
                        help=parameters_help['batch_size_val'])

    parser.add_argument('--batch_size_train', '--bs',
                        type=int,
                        default=parameters_default['batch_size_train'],
                        help=parameters_help['batch_size_train'])

    # ------- #
    # NETWORK #
    # ------- #

    parser.add_argument('--network_name',
                        type=str,
                        default=parameters_default['network_name'],
                        help=parameters_help['network_name'])

    parser.add_argument('--backbone',
                        type=str,
                        default=parameters_default['backbone'],
                        choices=parameters_choices['backbone'],
                        help=parameters_help['backbone'])

    parser.add_argument('--pretrained',
                        action='store_true',
                        default=parameters_default['pretrained'],
                        help=parameters_help['pretrained'])

    # ---------------- #
    # HYPER-PARAMETERS #
    # ---------------- #
    parser.add_argument('--epochs', '--ep',
                        type=int,
                        default=parameters_default['epochs'],
                        help=parameters_help['epochs'])

    parser.add_argument('--epoch_to_resume', '--ep_to_resume',
                        type=int,
                        default=parameters_default['epoch_to_resume'],
                        help=parameters_help['epoch_to_resume'])

    parser.add_argument('--optimizer',
                        type=str,
                        default=parameters_default['optimizer'],
                        choices=parameters_choices['optimizer'],
                        help=parameters_help['optimizer'])

    parser.add_argument('--scheduler',
                        type=str,
                        default=parameters_default['scheduler'],
                        choices=parameters_choices['scheduler'],
                        help=parameters_help['scheduler'])

    parser.add_argument('--clip_gradient',
                        action='store_true',
                        default=parameters_default['clip_gradient'],
                        help=parameters_help['clip_gradient'])

    parser.add_argument('--learning_rate', '--lr',
                        type=float,
                        default=parameters_default['learning_rate'],
                        help=parameters_help['learning_rate'])

    parser.add_argument('--lr_momentum',
                        type=int,
                        default=parameters_default['lr_momentum'],
                        help=parameters_help['lr_momentum'])

    parser.add_argument('--lr_patience',
                        type=int,
                        default=parameters_default['lr_patience'],
                        help=parameters_help['lr_patience'])

    parser.add_argument('--lr_step_size',
                        type=int,
                        default=parameters_default['lr_step_size'],
                        help=parameters_help['lr_step_size'])

    parser.add_argument('--lr_gamma',
                        type=int,
                        default=parameters_default['lr_gamma'],
                        help=parameters_help['lr_gamma'])

    # ---- #
    # LOSS #
    # ---- #
    parser.add_argument('--alpha',
                           type=float,
                           default=parameters_default['alpha'],
                           help=parameters_help['alpha'])

    parser.add_argument('--gamma',
                           type=float,
                           default=parameters_default['gamma'],
                           help=parameters_help['gamma'])

    parser.add_argument('--lambda_factor',
                           type=int,
                           default=parameters_default['lambda'],
                           help=parameters_help['lambda'])

    # parser arguments
    parser = parser.parse_args()

    return parser