import argparse

from net.metrics.utility.my_notation import scientific_notation


def parameters_ID(parser: argparse.Namespace) -> dict:
    """
    Get parameters ID

    :param parser: parser of parameters-parsing
    :return: parameters ID dictionary
    """

    dataset_ID = "dataset={}".format(parser.dataset)
    dataset_augmented_ID = "dataset={}-augmented".format(parser.dataset)
    augmented_ID = "do_dataset_augmentation={}".format(parser.do_dataset_augmentation)
    seed_ID = "seed={}".format(parser.seed)
    norm_ID = "norm={}".format(parser.norm)
    split_ID = "split={}".format(parser.split)
    split_1_fold_ID = "split=1-fold"
    split_2_fold_ID = "split=2-fold"
    ep_ID = "ep={}".format(parser.epochs)
    ep_resume_ID = "ep={}".format(parser.epoch_to_resume)
    optimizer_ID = "optimizer={}".format(parser.optimizer)
    scheduler_ID = "scheduler={}".format(parser.scheduler)
    clip_gradient_ID = "clip_gradient={}".format(parser.clip_gradient)
    lr_ID = "lr={}".format(scientific_notation(parser.learning_rate))
    lr_momentum_ID = "lr_momentum={}".format(parser.lr_momentum)
    lr_patience_ID = "lr_patience={}".format(parser.lr_patience)
    lr_step_size_ID = "lr_step_size={}".format(parser.lr_step_size)
    lr_gamma_ID = "lr_gamma={}".format(parser.lr_gamma)
    bs_ID = "bs={}".format(parser.batch_size_train)
    backbone_ID = "backbone={}".format(parser.backbone)
    pretrained_ID = "pretrained={}".format(parser.pretrained)
    lambda_ID = "lambda_factor={}".format(parser.lambda_factor)
    GPU_ID = "GPU={}".format(parser.GPU)

    parameters_ID_dict = {
        'dataset': dataset_ID,
        'dataset_augmented': dataset_augmented_ID,
        'augmented': augmented_ID,
        'seed': seed_ID,
        'norm': norm_ID,
        'split': split_ID,
        'split_1_fold': split_1_fold_ID,
        'split_2_fold': split_2_fold_ID,
        'ep': ep_ID,
        'ep_to_resume': ep_resume_ID,
        'optimizer': optimizer_ID,
        'scheduler': scheduler_ID,
        'clip_gradient': clip_gradient_ID,
        'lr': lr_ID,
        'lr_momentum': lr_momentum_ID,
        'lr_patience': lr_patience_ID,
        'lr_step_size': lr_step_size_ID,
        'lr_gamma': lr_gamma_ID,
        'bs': bs_ID,
        'backbone': backbone_ID,
        'pretrained': pretrained_ID,
        'lambda': lambda_ID,
        'GPU': GPU_ID,
    }

    return parameters_ID_dict
