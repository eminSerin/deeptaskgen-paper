import os.path as op
import subprocess
import sys

sys.path.append(op.abspath(op.join(__file__, "../../../..")))
ABS_PATH = sys.path[-1]

# Prediction related args.
TEST_LIST = op.realpath(
    op.join(
        ABS_PATH, "experiments/transfer_learning/hcp_development/data/hcpd_test_ids.txt"
    )
)
REST_DIR = op.realpath(
    op.join(ABS_PATH, "experiments/transfer_learning/hcp_development/data/rest")
)
PRED_DIR = op.realpath(
    op.join(
        ABS_PATH,
        "experiments/transfer_learning/hcp_development/results/nofinetune/pred",
    )
)
PREDICT_FUNC = op.realpath(op.join(ABS_PATH, "deeptaskgen/deeptaskgen/predict.py"))

# No-Finetuned model only trained on HCP-YA for 47 task contrasts.
CHECKPOINT = op.realpath(
    op.join(ABS_PATH, "experiments/training/results/unetminimal_100_0.001/best_r2.ckpt")
)

args = [
    f"--rest_dir={REST_DIR}",
    f"--working_dir={PRED_DIR}",
    f"--checkpoint_file={CHECKPOINT}",
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
