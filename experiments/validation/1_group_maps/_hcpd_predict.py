# Predict task contrast maps for all the HCP-D dataset.

import os.path as op
import subprocess
import sys

sys.path.append(op.abspath(op.join(__file__, "../../../..")))
ABS_PATH = sys.path[-1]

# Prediction related args.
EPOCHS = 50
LR = 1e-3
REST_DIR = op.join(ABS_PATH, "experiments/transfer_learning/hcp_development/data/rest")
TEST_LIST = op.join(
    ABS_PATH, "experiments/transfer_learning/hcp_development/data/hcpd_all_ids.txt"
)
PREDICT_FUNC = op.join(ABS_PATH, "deeptaskgen/deeptaskgen/predict.py")

# Finetuned model after replacing the last layer with output layer from HCP-YA for 47 task contrasts.
CHECKPOINT_MAP = {
    "emotion-faces-shapes": op.join(
        ABS_PATH,
        "experiments/transfer_learning/hcp_development/results/finetuned_50_0.001_emotion-faces-shapes/best_r2_all.ckpt",
    ),
}

# Rest of the code...

for cont in CHECKPOINT_MAP:
    pred_dir = op.join(op.dirname(CHECKPOINT_MAP[cont]), "pred")
    args = [
        f"--rest_dir={REST_DIR}",
        f"--working_dir={pred_dir}",
        f"--checkpoint_file={CHECKPOINT_MAP[cont]}",
        f"--test_list={TEST_LIST}",
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