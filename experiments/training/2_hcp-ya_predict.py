import os.path as op
import subprocess
import sys

sys.path.append("../../../..")


# Prediction related args.
EPOCHS = 100
LR = 1e-3
REST_DIR = op.realpath("experiments/training/data/rest")
PREDICT_FUNC = op.realpath("deeptaskgen/deeptaskgen/predict.py")
WORKING_DIR = op.realpath(f"experiments/training/results/unetminimal_{EPOCHS}_{LR}")
PRED_DIR = op.join(WORKING_DIR, "pred")

# Trained model.
CHECKPOINT = op.realpath(op.join(WORKING_DIR, "best_r2.ckpt"))

args = [
    f"--rest_dir={REST_DIR}",
    f"--working_dir={PRED_DIR}",
    f"--checkpoint_file={CHECKPOINT}",
    f"--test_list={op.realpath('experiments/training/data/hcp_test_ids.txt')}",
    "--n_samples_per_subj=1",
    "--n_out_channels=47",
    "--batch_norm=True",
    "--lr_scheduler=True",
    "--architecture=unetminimal",
    "--conv_dim=3",
    "--max_depth=1",
    "--n_conv_layers=1",
    "--fdim=64",
]
args = " ".join(args)

# Train the model.
cmd = f"{sys.executable} {PREDICT_FUNC} {args}"
subprocess.call(cmd, shell=True)
