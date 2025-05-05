import os
import os.path as op
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split

path = op.abspath(op.join(op.dirname(__file__), ".."))
if path not in sys.path:
    sys.path.append(path)
del sys, path
from deeptaskgen.callbacks.callbacks import (
    FinalLayerFreeze,
    LogGradients,
    LogParameters,
    RCLossMarginTune,
    SaveLastModel,
)
from deeptaskgen.datasets.taskgen_dataset import TaskGenDataset
from deeptaskgen.losses.loss_metric import RCLossAnneal
from deeptaskgen.utils.parser import default_parser


def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if op.exists(args.working_dir):
        raise FileExistsError(f"{args.working_dir} already exists!")
    else:
        os.makedirs(args.working_dir)

    """Load Data"""
    train_ids = np.genfromtxt(args.train_list, dtype=int, delimiter=",")
    if train_ids[0] == -1:
        train_ids = np.genfromtxt(args.train_list, dtype=str, delimiter=",")
    if args.val_list is not None:
        val_ids = np.genfromtxt(args.val_list, dtype=int, delimiter=",")
        if val_ids[0] == -1:
            val_ids = np.genfromtxt(args.val_list, dtype=str, delimiter=",")
    else:
        train_ids, val_ids = train_test_split(
            train_ids, test_size=args.val_percent, random_state=args.seed
        )

    unmask = False
    if args.mask is not None:
        unmask = True

    train_set = TaskGenDataset(
        train_ids,
        args.rest_dir,
        args.task_dir,
        num_samples=args.n_samples_per_subj,
        mask=args.mask,
        unmask=unmask,
        random_state=args.seed,
        precision=args.precision,
    )

    train_loader = args._dataloader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        drop_last=False,
    )

    if args.run_validation:
        val_set = TaskGenDataset(
            val_ids,
            args.rest_dir,
            args.task_dir,
            num_samples=args.n_samples_per_subj,
            mask=args.mask,
            unmask=unmask,
            precision=args.precision,
        )

        val_loader = args._dataloader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            drop_last=False,
        )

    """Init Model"""
    # Loads model from checkpoint if specified
    if args.checkpoint_file is not None:
        if not op.exists(args.checkpoint_file):
            raise FileNotFoundError(f"{args.checkpoint_file} does not exist!")
        model = args.architecture.load_from_checkpoint(
            args.checkpoint_file,
            in_chans=args.n_channels,
            out_chans=args.n_out_channels,
            fdim=args.fdim,
            activation=args.activation,
            optimizer=args.optimizer,
            up_mode=args.upsampling_mode,
            loss_fn=args.loss,
            add_loss=args.add_loss,
            max_level=args.max_depth,
            n_conv=args.n_conv_layers,
            batch_norm=args.batch_norm,
            lr_scheduler=args.lr_scheduler,
            lr=args.lr,
        )
    else:
        model = args.architecture(
            in_chans=args.n_channels,
            out_chans=args.n_out_channels,
            fdim=args.fdim,
            activation=args.activation,
            optimizer=args.optimizer,
            up_mode=args.upsampling_mode,
            loss_fn=args.loss,
            add_loss=args.add_loss,
            max_level=args.max_depth,
            n_conv=args.n_conv_layers,
            batch_norm=args.batch_norm,
            lr_scheduler=args.lr_scheduler,
            lr=args.lr,
        )

    """Checkpoint"""
    checkpoint_callback_loss = ModelCheckpoint(
        monitor="val/loss",
        dirpath=args.working_dir,
        filename="best_loss",
        save_top_k=1,
        mode="min",
        # save_last=True,
    )
    checkpoint_callback_corr = ModelCheckpoint(
        monitor="val/corr",
        dirpath=args.working_dir,
        filename="best_corr",
        save_top_k=1,
        mode="max",
    )
    checkpoint_callback_diag_index = ModelCheckpoint(
        monitor="val/diag_index_norm",
        dirpath=args.working_dir,
        filename="best_diag_index_norm",
        save_top_k=1,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [
        checkpoint_callback_loss,
        checkpoint_callback_corr,
        checkpoint_callback_diag_index,
        lr_monitor,
        SaveLastModel(),
    ]

    if args.checkpoint_interval is not None:
        checkpoint_callback_interval = ModelCheckpoint(
            dirpath=op.join(args.working_dir, "checkpoints"),
            filename="{epoch}",
            save_top_k=-1,
            every_n_epochs=args.checkpoint_interval,
        )
        callbacks.append(checkpoint_callback_interval)

    # Logger
    if args.logger == "tensorboard":
        logger = TensorBoardLogger(
            args.working_dir, name="logs", version=args.ver, default_hp_metric=False
        )
        callbacks.extend([LogGradients(), LogParameters()])
    elif args.logger == "wandb":
        logger = WandbLogger(
            name=args.ver,
            project="deeptaskgen",
            config=args._hparams,
            save_dir=args.working_dir,
        )
        logger.watch(model, log="all", log_freq=50)
    logger.log_hyperparams(args._hparams)

    """Train Model"""
    ## TODO: Add early stopping!
    if isinstance(args.loss, RCLossAnneal):
        callbacks.append(RCLossMarginTune())

    # Freeze final layer (i.e., finetune only backbone)
    if args.freeze_final_layer:
        callbacks.append(FinalLayerFreeze())

    ## TODO: Add multiple GPU support!
    trainer = pl.Trainer(
        max_epochs=args.n_epochs,
        accelerator=args.device,
        default_root_dir=args.working_dir,
        callbacks=callbacks,
        logger=logger,
        precision=args.precision,
    )
    if args.run_validation:
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
    else:
        trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    train(default_parser())
