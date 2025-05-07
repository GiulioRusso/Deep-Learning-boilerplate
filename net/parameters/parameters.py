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
    parser_mode = parser.add_subparsers(title=parameters_help['mode'], dest='mode', metavar='mode')

    # execution mode
    parser_train = parser_mode.add_parser('train', help=parameters_help['train'])
    parser_resume = parser_mode.add_parser('resume', help=parameters_help['resume'])
    parser_test = parser_mode.add_parser('test', help=parameters_help['test'])

    # execution mode list
    execution_mode = [parser_train,
                      parser_resume,
                      parser_test]
    
    # for each subparser 'mode'
    for subparser in execution_mode:

        # ------- #
        # DATASET #
        # ------- #

        subparser.add_argument('--dataset',
                            type=str,
                            default=parameters_default['dataset'],
                            help=parameters_help['dataset'])

        subparser.add_argument('--num_classes',
                            type=str,
                            default=parameters_default['num_classes'],
                            help=parameters_help['num_classes'])

        subparser.add_argument('--image_height',
                            type=int,
                            default=parameters_default['image_height'],
                            help=parameters_help['image_height'])

        subparser.add_argument('--image_width',
                            type=int,
                            default=parameters_default['image_width'],
                            help=parameters_help['image_width'])

        subparser.add_argument('--split',
                            type=str,
                            default=parameters_default['split'],
                            help=parameters_help['split'])

        subparser.add_argument('--norm',
                            type=str,
                            default=parameters_default['norm'],
                            help=parameters_help['norm'])

        subparser.add_argument('--do_dataset_augmentation',
                            action='store_true',
                            default=parameters_default['do_dataset_augmentation'],
                            help=parameters_help['do_dataset_augmentation'])

        # ------ #
        # DEVICE #
        # ------ #

        subparser.add_argument('--GPU',
                            type=str,
                            default=parameters_default['GPU'],
                            choices=parameters_choices['GPU'],
                            help=parameters_help['GPU'])

        subparser.add_argument('--num_threads',
                            type=int,
                            default=parameters_default['num_threads'],
                            help=parameters_help['num_threads'])

        # --------------- #
        # REPRODUCIBILITY #
        # --------------- #

        subparser.add_argument('--seed',
                            type=int,
                            default=parameters_default['seed'],
                            help=parameters_help['seed'])

        subparser.add_argument('--num_workers',
                            type=int,
                            default=parameters_default['num_workers'],
                            help=parameters_help['num_workers'])

        # ----------- #
        # DATA LOADER #
        # ----------- #

        subparser.add_argument('--batch_size_test',
                            type=int,
                            default=parameters_default['batch_size_test'],
                            help=parameters_help['batch_size_test'])

        subparser.add_argument('--batch_size_val',
                            type=int,
                            default=parameters_default['batch_size_val'],
                            help=parameters_help['batch_size_val'])

        subparser.add_argument('--batch_size_train', '--bs',
                            type=int,
                            default=parameters_default['batch_size_train'],
                            help=parameters_help['batch_size_train'])

        # ------- #
        # NETWORK #
        # ------- #

        subparser.add_argument('--network_name',
                            type=str,
                            default=parameters_default['network_name'],
                            help=parameters_help['network_name'])

        subparser.add_argument('--backbone',
                            type=str,
                            default=parameters_default['backbone'],
                            choices=parameters_choices['backbone'],
                            help=parameters_help['backbone'])

        subparser.add_argument('--pretrained',
                            action='store_true',
                            default=parameters_default['pretrained'],
                            help=parameters_help['pretrained'])

        # ---------------- #
        # HYPER-PARAMETERS #
        # ---------------- #
        subparser.add_argument('--epochs', '--ep',
                            type=int,
                            default=parameters_default['epochs'],
                            help=parameters_help['epochs'])

        subparser.add_argument('--epoch_to_resume', '--ep_to_resume',
                            type=int,
                            default=parameters_default['epoch_to_resume'],
                            help=parameters_help['epoch_to_resume'])

        subparser.add_argument('--optimizer',
                            type=str,
                            default=parameters_default['optimizer'],
                            choices=parameters_choices['optimizer'],
                            help=parameters_help['optimizer'])

        subparser.add_argument('--scheduler',
                            type=str,
                            default=parameters_default['scheduler'],
                            choices=parameters_choices['scheduler'],
                            help=parameters_help['scheduler'])

        subparser.add_argument('--clip_gradient',
                            action='store_true',
                            default=parameters_default['clip_gradient'],
                            help=parameters_help['clip_gradient'])

        subparser.add_argument('--learning_rate', '--lr',
                            type=float,
                            default=parameters_default['learning_rate'],
                            help=parameters_help['learning_rate'])

        subparser.add_argument('--lr_momentum',
                            type=int,
                            default=parameters_default['lr_momentum'],
                            help=parameters_help['lr_momentum'])

        subparser.add_argument('--lr_patience',
                            type=int,
                            default=parameters_default['lr_patience'],
                            help=parameters_help['lr_patience'])

        subparser.add_argument('--lr_step_size',
                            type=int,
                            default=parameters_default['lr_step_size'],
                            help=parameters_help['lr_step_size'])

        subparser.add_argument('--lr_gamma',
                            type=int,
                            default=parameters_default['lr_gamma'],
                            help=parameters_help['lr_gamma'])

        # ---- #
        # LOSS #
        # ---- #
        subparser.add_argument('--alpha',
                            type=float,
                            default=parameters_default['alpha'],
                            help=parameters_help['alpha'])

        subparser.add_argument('--gamma',
                            type=float,
                            default=parameters_default['gamma'],
                            help=parameters_help['gamma'])

        subparser.add_argument('--lambda_factor',
                            type=int,
                            default=parameters_default['lambda'],
                            help=parameters_help['lambda'])

    # parser arguments
    parser = parser.parse_args()

    return parser