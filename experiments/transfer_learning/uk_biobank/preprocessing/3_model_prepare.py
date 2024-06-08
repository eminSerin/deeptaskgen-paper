import os.path as op
import sys
from logging import config

import pytorch_lightning as pl
import torch

sys.path.append("../../../..")
from deeptaskgen.deeptaskgen.models.unet import UNet3DMinimal

# Model trained on HCP-YA with 47 task contrast maps.
REF_MODEL = torch.load(
    "experiments/training/results/unetminimal_100_0.001/best_r2.ckpt",
    map_location="cpu",
)

"""Init Model"""
pretrained_model = UNet3DMinimal.load_from_checkpoint(
    REF_MODEL,
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
# Remove all output channels except EMOTION FACES-SHAPES.
base_out_weight = (
    pretrained_model.out_block[1].weight.detach().cpu()[11, :, :, :, :].unsqueeze(0)
)
base_out_bias = pretrained_model.out_block[1].bias.detach().cpu()[11].unsqueeze(0)
# Generate a output layer with 1 channel.
pretrained_model.out_block[1] = torch.nn.Conv3d(
    64,
    1,
    kernel_size=1,
    stride=1,
)
# Replace the last layer with the output layer from the reference model.
pretrained_model.out_block[1].weight.data, pretrained_model.out_block[1].bias.data = (
    base_out_weight,
    base_out_bias,
)
# Save model
out_mdl_path = op.realpath("experiments/transfer_learning/uk_biobank/preprocessing")
trainer = pl.Trainer(default_root_dir=out_mdl_path)
trainer.strategy.connect(pretrained_model)
trainer.save_checkpoint(op.join(out_mdl_path, "hcp-ya_emotion-faces-shapes.ckpt"))
