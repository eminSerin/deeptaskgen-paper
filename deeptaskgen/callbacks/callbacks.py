import os.path as op

from pytorch_lightning.callbacks import BaseFinetuning, Callback


class RCLossMarginTune(Callback):
    """Callback to tune the within and between margin of the RCLossAnneal loss function."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.loss_fn.update_margins(trainer.current_epoch)


class SaveLastModel(Callback):
    """Callback to save the model at the end of training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_end(self, trainer, pl_module):
        trainer.save_checkpoint(op.join(trainer.default_root_dir, "last.ckpt"))


class LogGradients(Callback):
    """Callback to log the gradients of the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        for tag, value in pl_module.named_parameters():
            if value.grad is not None:
                pl_module.logger.experiment.add_histogram(
                    f"{tag}/grad",
                    value.grad.data.cpu(),
                    global_step=trainer.global_step,
                )


class LogParameters(Callback):
    """Callback to log the parameters of the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_end(self, trainer, pl_module):
        for tag, value in pl_module.named_parameters():
            pl_module.logger.experiment.add_histogram(
                f"{tag}/weight", value.data.cpu(), global_step=trainer.global_step
            )


class FinalLayerFreeze(BaseFinetuning):
    """Callback to freeze the final layer of the model."""

    def __init__(self) -> None:
        super().__init__()

    def freeze_before_training(self, pl_module) -> None:
        self.freeze(pl_module, pl_module.out_block[1])

    def finetune_function(self, pl_module, epoch, optimizer, *args, **kwargs) -> None:
        pass
