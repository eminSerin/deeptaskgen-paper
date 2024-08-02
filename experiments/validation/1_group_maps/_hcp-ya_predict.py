# Predict task contrast maps for all the HCP-D dataset.

import os.path as op
import subprocess
import sys

sys.path.append(op.abspath(op.join(__file__, "../../../..")))
ABS_PATH = sys.path[-1]

# Prediction related args.
EPOCHS = 100
LR = 1e-3
REST_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/rest"))
PREDICT_FUNC = op.realpath(op.join(ABS_PATH, "deeptaskgen/deeptaskgen/predict.py"))
WORKING_DIR = op.realpath(
    op.join(ABS_PATH, f"experiments/training/results/unetminimal_{EPOCHS}_{LR}")
)
PRED_DIR = op.join(WORKING_DIR, "pred")

# Trained model.
CHECKPOINT = op.realpath(op.join(ABS_PATH, WORKING_DIR, "best_r2.ckpt"))

args = [
    f"--rest_dir={op.realpath(op.join(ABS_PATH, REST_DIR))}",
    f"--working_dir={PRED_DIR}",
    f"--checkpoint_file={op.realpath(op.join(ABS_PATH, CHECKPOINT))}",
    f"--test_list={op.realpath(op.join(ABS_PATH, 'experiments/training/data/hcp_all_ids.txt'))}",
    "--n_samples_per_subj=8",
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
