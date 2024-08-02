import os.path as op
import sys

import pytorch_lightning as pl
import torch

sys.path.append(op.abspath(op.join(__file__, "../../../../deeptaskgen")))
from deeptaskgen.models.unet import UNet3DMinimal  # type: ignore

ABS_PATH = op.abspath(op.join(__file__, "../../../.."))

# Model trained on HCP-YA with 47 task contrasts maps.
REF_MODEL = torch.load(
    op.join(
        ABS_PATH, "experiments/training/results/unetminimal_100_0.001/best_r2.ckpt"
    ),
    map_location="cpu",
)

# Finetuned models
CHECKPOINT_MAP = {
    "emotion-faces-shapes": op.realpath(
        op.join(
            ABS_PATH,
            "experiments/transfer_learning/hcp_development/results/finetuned_50_0.001_emotion-faces-shapes/best_r2.ckpt",
        )
    ),
    "gambling-reward": op.realpath(
        op.join(
            ABS_PATH,
            "experiments/transfer_learning/hcp_development/results/finetuned_50_0.001_gambling-reward/best_r2.ckpt",
        )
    ),
}

for cont in CHECKPOINT_MAP:
    """Init Model"""
    finetuned_model = UNet3DMinimal.load_from_checkpoint(
        CHECKPOINT_MAP[cont],
        in_chans=50,
        out_chans=1,
        fdim=64,
        activation="relu_inplace",
        optimizer="adam",
        up_mode="trilinear",
        loss_fn="mse",
        max_level=1,
        n_conv=1,
    ).to("cpu")
    # Replace the last layer with the output layer from the reference model.
    finetuned_model.out_block[1].weight.data, finetuned_model.out_block[1].bias.data = (
        REF_MODEL["state_dict"]["out_block.1.weight"],
        REF_MODEL["state_dict"]["out_block.1.bias"],
    )
    # Save model
    trainer = pl.Trainer(default_root_dir=op.dirname(CHECKPOINT_MAP[cont]))
    trainer.strategy.connect(finetuned_model)
    trainer.save_checkpoint(
        CHECKPOINT_MAP[cont].replace("best_r2.ckpt", "best_r2_all.ckpt")
    )
