import os
import argparse

from net.initialization.utility.get_yaml import get_yaml


def parameters_parsing(parameters_path: str) -> argparse.Namespace:
    """
    Definition of parameters-parsing for each execution mode

    :param parameters_path: path to the parameters YAML configuation file
    :return: parser of parameters parsing
    """

    # parser
    parser = argparse.ArgumentParser(description='Argument Parser')

    # parameters
    parameters = get_yaml(yaml_path=parameters_path)

    # -------------- #
    # EXECUTION MODE #
    # -------------- #
    parser_mode = parser.add_subparsers(title=parameters['mode']['help'], dest='mode', metavar='mode')

    # execution mode
    parser_train = parser_mode.add_parser('train', help=parameters['train']['help'])
    parser_resume = parser_mode.add_parser('resume', help=parameters['resume']['help'])
    parser_test = parser_mode.add_parser('test', help=parameters['test']['help'])

    # execution mode list
    execution_mode = [parser_train,
                      parser_resume,
                      parser_test]

    # for each subparser 'mode'
    for subparser in execution_mode:
        # TODO: Add the necessary parameters for your application
        #  and add the corresponding information in the 'parameters.yaml'

        # ----- #
        # DEBUG #
        # ----- #
        subparser.add_argument('--verbose_prints',
                               action='store_true',
                               help=parameters['verbose_prints']['help'])

        subparser.add_argument('--verbose_plots',
                               action='store_true',
                               help=parameters['verbose_plots']['help'])

        subparser.add_argument('--debug',
                               action='store_true',
                               help=parameters['debug']['help'])

        # ------- #
        # UTILITY #
        # ------- #
        subparser.add_argument('--version',
                               type=str,
                               default=parameters['version']['default'],
                               help=parameters['version']['help'])

        # ------- #
        # DATASET #
        # ------- #
        subparser.add_argument('--dataset',
                               type=str,
                               default=parameters['dataset']['default'],
                               choices=parameters['dataset']['choices'],
                               help=parameters['dataset']['help'])

        subparser.add_argument('--split',
                               type=str,
                               default=parameters['split']['default'],
                               help=parameters['split']['help'])

        subparser.add_argument('--transform',
                               type=str,
                               default=parameters['transform']['default'],
                               help=parameters['transform']['help'])

        subparser.add_argument('--do_dataset_augmentation',
                               action='store_true',
                               default=parameters['do_dataset_augmentation']['default'],
                               help=parameters['do_dataset_augmentation']['help'])

        # > Task: Classification
        subparser.add_argument('--num_classes',
                               type=str,
                               default=parameters['num_classes']['default'],
                               help=parameters['num_classes']['help'])

        # > Optional: Computer Vision
        subparser.add_argument('--image_height',
                               type=int,
                               default=parameters['image_height']['default'],
                               help=parameters['image_height']['help'])

        subparser.add_argument('--image_width',
                               type=int,
                               default=parameters['image_width']['default'],
                               help=parameters['image_width']['help'])

        # ------ #
        # DEVICE #
        # ------ #
        subparser.add_argument('--GPU',
                               type=str,
                               default=parameters['GPU']['default'],
                               choices=parameters['GPU']['choices'],
                               help=parameters['GPU']['help'])

        subparser.add_argument('--num_threads',
                               type=int,
                               default=parameters['num_threads']['default'],
                               help=parameters['num_threads']['help'])

        subparser.add_argument('--num_workers',
                               type=int,
                               default=parameters['num_workers']['default'],
                               help=parameters['num_workers']['help'])

        # --------------- #
        # REPRODUCIBILITY #
        # --------------- #
        subparser.add_argument('--seed',
                               type=int,
                               default=parameters['seed']['default'],
                               help=parameters['seed']['help'])

        # ----------- #
        # DATA LOADER #
        # ----------- #
        subparser.add_argument('--batch_size_test',
                               type=int,
                               default=parameters['batch_size_test']['default'],
                               help=parameters['batch_size_test']['help'])

        subparser.add_argument('--batch_size_val',
                               type=int,
                               default=parameters['batch_size_val']['default'],
                               help=parameters['batch_size_val']['help'])

        subparser.add_argument('--batch_size_train', '--bs',
                               type=int,
                               default=parameters['batch_size_train']['default'],
                               help=parameters['batch_size_train']['help'])

        # ------- #
        # NETWORK #
        # ------- #
        subparser.add_argument('--network_name',
                               type=str,
                               default=parameters['network_name']['default'],
                               help=parameters['network_name']['help'])

        # > Optional: Backbone
        subparser.add_argument('--backbone',
                               type=str,
                               default=parameters['backbone']['default'],
                               choices=parameters['backbone']['choices'],
                               help=parameters['backbone']['help'])

        subparser.add_argument('--pretrained',
                               action='store_true',
                               default=parameters['pretrained']['default'],
                               help=parameters['pretrained']['help'])

        # ---------------- #
        # HYPER-PARAMETERS #
        # ---------------- #
        subparser.add_argument('--epochs', '--ep',
                               type=int,
                               default=parameters['epochs']['default'],
                               help=parameters['epochs']['help'])

        subparser.add_argument('--epoch_to_resume', '--ep_to_resume',
                               type=int,
                               default=parameters['epoch_to_resume']['default'],
                               help=parameters['epoch_to_resume']['help'])

        subparser.add_argument('--optimizer',
                               type=str,
                               default=parameters['optimizer']['default'],
                               choices=parameters['optimizer']['choices'],
                               help=parameters['optimizer']['help'])

        subparser.add_argument('--scheduler',
                               type=str,
                               default=parameters['scheduler']['default'],
                               choices=parameters['scheduler']['choices'],
                               help=parameters['scheduler']['help'])

        subparser.add_argument('--clip_gradient',
                               action='store_true',
                               default=parameters['clip_gradient']['default'],
                               help=parameters['clip_gradient']['help'])

        subparser.add_argument('--learning_rate', '--lr',
                               type=float,
                               default=parameters['learning_rate']['default'],
                               help=parameters['learning_rate']['help'])

        # > Optimizer: SGD
        subparser.add_argument('--lr_momentum',
                               type=int,
                               default=parameters['lr_momentum']['default'],
                               help=parameters['lr_momentum']['help'])

        # > Scheduler: ReduceLROnPlateau
        subparser.add_argument('--lr_patience',
                               type=int,
                               default=parameters['lr_patience']['default'],
                               help=parameters['lr_patience']['help'])

        # > Scheduler: StepLR
        subparser.add_argument('--lr_step_size',
                               type=int,
                               default=parameters['lr_step_size']['default'],
                               help=parameters['lr_step_size']['help'])

        # > Scheduler: StepLR
        subparser.add_argument('--lr_gamma',
                               type=int,
                               default=parameters['lr_gamma']['default'],
                               help=parameters['lr_gamma']['help'])

        # > Scheduler: CosineAnnealing
        subparser.add_argument('--lr_T0',
                               type=int,
                               default=parameters['lr_T0']['default'],
                               help=parameters['lr_T0']['help'])

        # ---- #
        # LOSS #
        # ---- #
        subparser.add_argument('--loss',
                               type=str,
                               default=parameters['loss']['default'],
                               choices=parameters['loss']['choices'],
                               help=parameters['loss']['help'])

        # > Loss: FocalLoss
        subparser.add_argument('--alpha',
                               type=float,
                               default=parameters['alpha']['default'],
                               help=parameters['alpha']['help'])

        subparser.add_argument('--gamma',
                               type=float,
                               default=parameters['gamma']['default'],
                               help=parameters['gamma']['help'])

        subparser.add_argument('--lambda',
                               type=int,
                               default=parameters['lambda']['default'],
                               help=parameters['lambda']['help'])

    # parser arguments
    parser = parser.parse_args()

    return parser
