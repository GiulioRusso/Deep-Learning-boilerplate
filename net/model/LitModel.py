import os
import argparse
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import nn, Tensor
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from net.loss.get_loss import get_loss
from net.optimizer.get_optimizer import get_optimizer
from net.scheduler.get_scheduler import get_scheduler


class LitModel(L.LightningModule):
    """
    Minimal Lightning model with:
      - simple MLP,
      - custom weight initialization,
      - loss from external factory,
      - optimizer/scheduler from external factories,
      - explicit train/val/test steps,
      - epoch start/end hooks,
      - centralized logging helpers.
    """

    def __init__(
        self,
        parser: argparse.Namespace,
        input_dim: int = 10,
    ) -> None:
        """
        Initialize the Lightning Model.

        :param parser: parsed CLI namespace with hyperparameters (optimizer, scheduler, lr, etc.).
        :param input_dim: number of input features (flatten images in the DataModule or dataset).
        """
        super().__init__()
        self.parser = parser

        # TODO: Define the necessary parameters for your architecture

        # core hparams (read from parser; fall back to reasonable defaults)
        self.input_dim: int = int(input_dim)
        self.num_classes: int = int(getattr(parser, "num_classes", 1))
        self.learning_rate: float = float(getattr(parser, "learning_rate", 1e-4))

        # TODO: Build the layers of your architecture

        # build a tiny MLP
        layers: List[nn.Module] = []
        in_features = self.input_dim
        for h in (128, 64):
            layers += [nn.Linear(in_features, h), nn.ReLU(inplace=True)]
            in_features = h
        out_features = 1 if self.num_classes == 1 else self.num_classes
        layers += [nn.Linear(in_features, out_features)]
        self.net = nn.Sequential(*layers)

        # initialize weights
        self._initialize_weights()

        # loss from external factory
        self.criterion: nn.Module = get_loss(loss=self.parser.loss, parser=self.parser)

        # batch-size hint for correct metric reduction
        self._last_batch_size: Optional[int] = None

    # --------------------- #
    # WEIGHT INITIALIZATION #
    # --------------------- #
    def _initialize_weights(self) -> None:
        """
        Initialize module weights.

        :return:
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------- #
    # FORWARD  #
    # -------- #
    # TODO: Build the forward based on your model logic
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        :param x: input tensor of shape (B, input_dim).
        :return: logits of shape (B, 1) for binary or (B, num_classes) for multi-class.
        """
        return self.net(x)

    # --------------------- #
    # OPTIMIZER / SCHEDULER #
    # --------------------- #
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure the optimizer and the scheduler using external configuration.

        :return: Configured optimizer and scheduler.
        """
        # define optimizer
        optimizer = get_optimizer(net_parameters=self.parameters(), parser=self.parser)

        # get scheduler using your utility function
        scheduler = get_scheduler(optimizer=optimizer, parser=self.parser)

        # handle ReduceLROnPlateau which needs a monitored metric
        if self.parser.scheduler == "ReduceLROnPlateau":
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "validation/loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }

        # for other schedulers like StepLR, CosineAnnealing
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    # --------------- #
    # LOGGING HELPERS #
    # --------------- #
    def _log_loss(self, split: str, loss: Tensor, *, on_step: bool, on_epoch: bool, prog_bar: bool = False) -> None:
        """
        Log a loss scalar.

        :param split: 'train' | 'val' | 'test'
        :param loss: loss tensor
        :param on_step: log on step
        :param on_epoch: log on epoch
        :param prog_bar: show in progress bar
        :return:
        """
        self.log(f"{split}/loss", loss, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=self._last_batch_size)

    def _log_acc(self, split: str, acc: Tensor, *, on_step: bool, on_epoch: bool, prog_bar: bool = False) -> None:
        """
        Log an accuracy scalar.

        :param split: 'train' | 'val' | 'test'
        :param acc: accuracy tensor
        :param on_step: log on step
        :param on_epoch: log on epoch
        :param prog_bar: show in progress bar
        :return:
        """
        self.log(f"{split}/acc", acc, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=self._last_batch_size)

    def _log_lr(self) -> None:
        """
        Log the current learning rate (first param group).

        :return:
        """
        opt = self.optimizers(use_pl_optimizer=False)
        if isinstance(opt, list) and len(opt) > 0:
            opt = opt[0]
        if opt is None:
            return
        lr_val = opt.param_groups[0].get("lr", None)
        if lr_val is not None:
            self.log("lr", torch.tensor(lr_val, device=self.device), on_step=False, on_epoch=True, prog_bar=True)

    # TODO: Implement the train, validation and test steps based on your task

    # -------- #
    # TRAINING #
    # -------- #
    def on_train_epoch_start(self) -> None:
        """
        Called at the start of the training epoch.

        :return:
        """
        self._log_lr()

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """
        Compute training loss and metrics for a single-class (binary) problem and log them.

        :param batch: sample dict with keys {'filename', 'image', 'annotation'}
        :param batch_idx: batch index
        :return: training loss tensor (used for backward)
        """
        # unpack sample
        x: Tensor = batch["image"]
        y: Tensor = batch["annotation"]

        # set batch size for proper metric reduction
        self._last_batch_size = int(x.shape[0]) if hasattr(x, "shape") else None

        # forward
        logits: Tensor = self.forward(x)

        # loss (expects logits) and accuracy with 0.5 threshold
        y_bin: Tensor = y.float().view_as(logits)
        loss: Tensor = self.criterion(logits, y_bin)
        with torch.no_grad():
            preds: Tensor = (logits.sigmoid() >= 0.5).float()
            acc: Tensor = (preds == y_bin.round()).float().mean()

        # logging
        self._log_loss("train", loss.detach(), on_step=True, on_epoch=True, prog_bar=True)
        self._log_acc("train", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """
        Called at the end of the training epoch.

        :return:
        """
        pass

    # ---------- #
    # VALIDATION #
    # ---------- #
    def on_validation_epoch_start(self) -> None:
        """
        Called at the start of the validation epoch.

        :return:
        """
        pass

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """
        Compute validation loss and metrics for a single-class (binary) problem and log them.

        :param batch: sample dict with keys {'filename', 'image', 'annotation'}
        :param batch_idx: batch index
        :return:
        """
        # unpack sample
        x: Tensor = batch["image"]
        y: Tensor = batch["annotation"]

        # set batch size for proper metric reduction
        self._last_batch_size = int(x.shape[0]) if hasattr(x, "shape") else None

        # forward
        logits: Tensor = self.forward(x)

        # loss (expects logits) and accuracy with 0.5 threshold
        y_bin: Tensor = y.float().view_as(logits)
        loss: Tensor = self.criterion(logits, y_bin)

        preds: Tensor = (logits.sigmoid() >= 0.5).float()
        acc: Tensor = (preds == y_bin.round()).float().mean()

        # logging
        self._log_loss("val", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self._log_acc("val", acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch.

        :return:
        """
        pass

    # --------- #
    # TEST STEP #
    # --------- #
    def on_test_epoch_start(self) -> None:
        """
        Called at the start of the test epoch.

        :return:
        """
        pass

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """
        Compute test loss and metrics for a single-class (binary) problem and log them.

        :param batch: sample dict with keys {'filename', 'image', 'annotation'}
        :param batch_idx: batch index
        :return:
        """
        # unpack sample
        x: Tensor = batch["image"]
        y: Tensor = batch["annotation"]

        # set batch size for proper metric reduction
        self._last_batch_size = int(x.shape[0]) if hasattr(x, "shape") else None

        # forward
        logits: Tensor = self.forward(x)

        # loss (expects logits) and accuracy with 0.5 threshold
        y_bin: Tensor = y.float().view_as(logits)
        loss: Tensor = self.criterion(logits, y_bin)
        with torch.no_grad():
            preds: Tensor = (logits.sigmoid() >= 0.5).float()
            acc: Tensor = (preds == y_bin.round()).float().mean()

        # logging
        self._log_loss("test", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self._log_acc("test", acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of the test epoch.

        :return:
        """
        pass

    # ------- #
    # UTILITY #
    # ------- #
    # TODO: Chose which metrics you need to monitor in order to save the best model
    def _get_checkpoint_callback(monitor: str = "val/loss") -> List[ModelCheckpoint]:
        """
        Returns two ModelCheckpoint callbacks:
        - One for the best model based on the monitored metric
        - One that always saves the last epoch model
        """

        # save best model
        best_ckpt = ModelCheckpoint(
            monitor=monitor,
            save_top_k=1,
            mode="min",
            filename="best"
        )

        # save last epoch model
        last_ckpt = ModelCheckpoint(
            save_last=True,
            filename="last"
        )

        return [best_ckpt, last_ckpt]
