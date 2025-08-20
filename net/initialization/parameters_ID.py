import argparse


def parameters_ID(parser: argparse.Namespace) -> dict:
    """
    Get parameters ID

    :param parser: parser of parameters-parsing
    :return: parameters ID dictionary
    """

    dataset_ID = "dataset={}".format(parser.dataset)
    network_ID = "net={}".format(parser.network_name)
    transforms_ID = "transform={}".format(parser.transform)
    ep_ID = "ep={}".format(parser.epoch_to_resume + parser.epochs)
    ep_resume_ID = "ep={}".format(parser.epoch_to_resume)
    optimizer_ID = "opt={}".format(parser.optimizer)
    scheduler_ID = "schdlr={}".format(parser.scheduler)
    lr_ID = "lr={}".format(parser.learning_rate)
    bs_ID = "bs={}".format(parser.batch_size_train)
    loss_ID = "loss={}".format(parser.loss)
    alpha_ID = "alpha={}".format(parser.alpha)
    beta_ID = "beta={}".format(parser.beta)
    lambda_ID = "lambda={}".format(getattr(parser, 'lambda')) # < safe access to do not overload Python 'lambda' keyword
    augmentation_ID = "augmented"

    parameters_ID_dict = {
        'dataset': dataset_ID,
        'network_name': network_ID,
        'transforms': transforms_ID,
        'ep': ep_ID,
        'ep_to_resume': ep_resume_ID,
        'optimizer': optimizer_ID,
        'scheduler': scheduler_ID,
        'lr': lr_ID,
        'bs': bs_ID,
        'loss': loss_ID,
        'alpha': alpha_ID,
        'beta': beta_ID,
        'lambda': lambda_ID,
        'augmentation': augmentation_ID
    }

    return parameters_ID_dict