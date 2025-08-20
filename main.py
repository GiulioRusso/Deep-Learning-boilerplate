import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from net.dataset.classes.MyDataset import MyDataset
from net.dataset.dataset_split import dataset_split
from net.dataset.dataset_transforms import dataset_transforms
from net.initialization.experiment_ID import experiment_ID
from net.initialization.utility.get_yaml import get_yaml
from net.model.LitModel import LitModel
from net.parameters.parameters import parameters_parsing
from net.reproducibility.reproducibility import reproducibility
from net.utility.get_latest_version import get_latest_version
from net.utility.logger import log_operation


def main():
    """
    My Deep Learning Project
    """

    print("| ============================ |\n"
          "|   MY DEEP LEARNING PROJECT   |\n"
          "| ============================ |\n")

    # ================== #
    # PARAMETERS-PARSING #
    # ================== #
    # command line parameter parsing
    parser = parameters_parsing(parameters_path=os.path.join("config.yaml", "parameters.yaml"))

    # ============== #
    # INITIALIZATION #
    # ============== #
    # experiment ID initialization
    exp_ID = experiment_ID(parser=parser)
    # experiment directories
    exp_dir = os.path.join("experiments", exp_ID)

    # path initialization
    path = get_yaml(yaml_path=os.path.join("config.yaml", "paths.yaml"))

    # ====== #
    # DEVICE #
    # ====== #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(parser.num_threads)

    # =============== #
    # REPRODUCIBILITY #
    # =============== #
    reproducibility(seed=parser.seed)

    # ============ #
    # LOAD DATASET #
    # ============ #
    # load dataset
    dataset = MyDataset(images_dir=path['dataset']['images'],
                        annotations_dir=path['dataset']['annotations'],
                        filename_list=path['dataset']['list'],
                        transforms=None)

    # ============= #
    # DATASET SPLIT #
    # ============= #
    # subset dataset according to data split
    dataset_train, dataset_val, dataset_test = dataset_split(data_split=path['dataset']['split'],
                                                             dataset=dataset)

    # ================== #
    # DATASET TRANSFORMS #
    # ================== #
    # original view
    train_transforms, val_transforms, test_transforms = dataset_transforms(parser=parser,
                                                                           normalization=parser.norm,
                                                                           statistics_path=os.path.join("config.yaml", "statistics.yaml"))

    # apply dataset transforms
    dataset_train.dataset.transforms = train_transforms
    dataset_val.dataset.transforms = val_transforms
    dataset_test.dataset.transforms = test_transforms

    # ============ #
    # DATA LOADERS #
    # ============ #
    # dataloader-train
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=parser.batch_size_train,
                                  shuffle=True,
                                  num_workers=parser.num_workers,
                                  pin_memory=True)

    # dataloader-val
    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=parser.batch_size_val,
                                shuffle=False,
                                num_workers=parser.num_workers,
                                pin_memory=True)

    # dataloader-test
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=parser.batch_size_test,
                                 shuffle=False,
                                 num_workers=parser.num_workers,
                                 pin_memory=True)

    if parser.mode == 'train':

        # ======== #
        # TRAINING #
        # ======== #
        # model
        model = LitModel(parser=parser)

        # logger: Lightning will create experiment/exp_ID/version_0
        logger = TensorBoardLogger(save_dir="experiments", name=exp_ID)

        # log
        log_operation(logger.log_dir, mode="train", parser=parser)

        # checkpoint callbacks come from the model (best + last)
        checkpoint_callbacks = model._get_checkpoint_callbacks()

        # trainer
        trainer = L.Trainer(
            max_epochs=parser.epochs,
            logger=logger,
            accelerator=device.type,
            devices=torch.cuda.device_count(),
            callbacks=checkpoint_callbacks,
        )

        # fit
        trainer.fit(model, dataloader_train, dataloader_val)

    elif parser.mode == 'resume':

        # ======== #
        # RESUMING #
        # ======== #
        # pick the specified version from this experiment
        version = get_latest_version(exp_dir=exp_dir) if parser.version == "latest" else parser.version

        # last checkpoint path
        last_ckpt = os.path.join(exp_dir, version, "checkpoints", "last.ckpt")
        if not os.path.isfile(last_ckpt):
            raise FileNotFoundError(f"Could not find last checkpoint at: {last_ckpt}")

        # restore model weights
        model = LitModel.load_from_checkpoint(checkpoint_path=last_ckpt, parser=parser)

        # continue logging into the SAME version folder
        logger = TensorBoardLogger(save_dir="experiments", name=exp_ID, version=version)

        # log
        log_operation(logger.log_dir, mode="resume", parser=parser, extra={"ckpt": last_ckpt})

        # checkpoint callbacks come from the model (best + last)
        checkpoint_callbacks = model._get_checkpoint_callbacks()

        # trainer
        trainer = L.Trainer(
            max_epochs=parser.epochs,
            logger=logger,
            accelerator=device.type,
            devices=torch.cuda.device_count(),
            callbacks=checkpoint_callbacks,
        )

        # resume training from last.ckpt so optimizer/scheduler states are restored
        trainer.fit(model, dataloader_train, dataloader_val, ckpt_path=last_ckpt)

    elif parser.mode == "test":

        # ======= #
        # TESTING #
        # ======= #
        # pick the latest version
        version = get_latest_version(exp_dir=exp_dir) if parser.version == "latest" else parser.version

        # load best checkpoint for testing
        best_ckpt = os.path.join(exp_dir, version, "checkpoints", "best.ckpt")
        if not os.path.isfile(best_ckpt):
            raise FileNotFoundError(f"Could not find best checkpoint at: {best_ckpt}")

        # restore model
        model = LitModel.load_from_checkpoint(checkpoint_path=best_ckpt, parser=parser)

        # log into the SAME version folder so test metrics land beside train/val
        logger = TensorBoardLogger(save_dir="experiments", name=exp_ID, version=version)

        # log
        log_operation(logger.log_dir, mode="test", parser=parser, extra={"ckpt": best_ckpt})

        # trainer
        trainer = L.Trainer(logger=logger, accelerator=device.type)

        # run test
        trainer.test(model, dataloaders=dataloader_test)

    else:
        raise ValueError(f"Unknown mode type in {__file__}. Got {parser.mode}")

    return 0


if __name__ == "__main__":
    main()