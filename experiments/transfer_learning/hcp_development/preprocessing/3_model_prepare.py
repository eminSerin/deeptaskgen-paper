import copy
import os.path as op
import sys

import pytorch_lightning as pl
import torch

sys.path.append(op.abspath(op.join(__file__, "../../../../../deeptaskgen")))
from deeptaskgen.models.unet import UNet3DMinimal  # type: ignore

# Model trained on HCP-YA with 47 task contrast maps.
ABS_PATH = op.abspath(op.join(__file__, "../../../../.."))
REF_MODEL = op.join(
    ABS_PATH, "experiments/training/results/unetminimal_100_0.001/allset/last.ckpt"
)
CONT_MAP = {"emotion-faces-shapes": 11, "gambling-reward": 45}


"""Init Model"""
base_model = UNet3DMinimal.load_from_checkpoint(
    REF_MODEL,
    in_chans=50,
    out_chans=47,
    fdim=64,
    activation="relu_inplace",
    optimizer="adam",
    up_mode="trilinear",
    loss_fn="mse",
    max_level=1,
    n_conv=1,
).to("cpu")
for cont in CONT_MAP:
    pretrained_model = copy.deepcopy(base_model)
    base_out_weight = (
        pretrained_model.out_block[1]
        .weight.detach()
        .cpu()[CONT_MAP[cont], :, :, :, :]
        .unsqueeze(0)
    )
    base_out_bias = (
        pretrained_model.out_block[1].bias.detach().cpu()[CONT_MAP[cont]].unsqueeze(0)
    )
    # Generate a output layer with 1 channel.
    pretrained_model.out_block[1] = torch.nn.Conv3d(
        64,
        1,
        kernel_size=1,
        stride=1,
    )
    # Replace the last layer with the output layer from the reference model.
    (
        pretrained_model.out_block[1].weight.data,
        pretrained_model.out_block[1].bias.data,
    ) = (
        base_out_weight,
        base_out_bias,
    )
    # Save model
    out_mdl_path = op.realpath(
        op.join(ABS_PATH, "experiments/transfer_learning/hcp_development")
    )
    trainer = pl.Trainer(default_root_dir=out_mdl_path)
    trainer.strategy.connect(pretrained_model)
    trainer.save_checkpoint(op.join(out_mdl_path, f"hcp-ya_{cont}.ckpt"))
