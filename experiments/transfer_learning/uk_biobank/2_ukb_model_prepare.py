import os.path as op
import sys

import pytorch_lightning as pl
import torch

sys.path.append(op.abspath(op.join(__file__, "../../../../deeptaskgen")))
from deeptaskgen.models.unet import AttentionUNet3D  # type: ignore

ABS_PATH = op.abspath(op.join(__file__, "../../../.."))

# Finetuned model.
CHECKPOINT = op.realpath(
    op.join(
        ABS_PATH,
        "experiments/transfer_learning/uk_biobank/results/finetuned_50_0.001/best_corr.ckpt",
    )
)

# Model trained on HCP-YA with 47 task contrast maps.
REF_MODEL = torch.load(
    op.join(
        ABS_PATH,
        "experiments/training/results/attentionunet_100_0.001_gm/best_corr.ckpt",
    ),
    map_location="cpu",
)

"""Init Model"""
finetuned_model = AttentionUNet3D.load_from_checkpoint(
    CHECKPOINT,
    in_chans=50,
    out_chans=1,
    fdim=64,
    activation="relu_inplace",
    optimizer="adam",
    up_mode="trilinear",
    max_level=1,
    n_conv=1,
).to("cpu")
# Replace the last layer with the output layer from the reference model.
finetuned_model.out_block[1].weight.data, finetuned_model.out_block[1].bias.data = (
    REF_MODEL["state_dict"]["out_block.1.weight"],
    REF_MODEL["state_dict"]["out_block.1.bias"],
)
# Save model
trainer = pl.Trainer(default_root_dir=op.dirname(CHECKPOINT))
trainer.strategy.connect(finetuned_model)
trainer.save_checkpoint(CHECKPOINT.replace("best_corr.ckpt", "best_corr_all.ckpt"))
