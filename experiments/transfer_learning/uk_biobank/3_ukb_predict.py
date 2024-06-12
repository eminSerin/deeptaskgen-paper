import os.path as op
import subprocess
import sys

sys.path.append(op.abspath(op.join(__file__, "../../../..")))
ABS_PATH = sys.path[-1]

# Prediction related args.
EPOCHS = 50
LR = 1e-3
REST_DIR = op.realpath(
    op.join(ABS_PATH, "experiments/transfer_learning/uk_biobank/data/rest")
)
PREDICT_FUNC = op.realpath(op.join(ABS_PATH, "deeptaskgen/deeptaskgen/predict.py"))
WORKING_DIR = op.realpath(
    op.join(
        ABS_PATH,
        f"experiments/transfer_learning/uk_biobank/results/finetuned_{EPOCHS}_{LR}",
    )
)
PRED_DIR = op.join(WORKING_DIR, "pred")

# Finetuned model after replacing the last layer with output layer from HCP-YA for 47 task contrasts.
CHECKPOINT = op.realpath(
    op.join(
        ABS_PATH,
        "experiments/transfer_learning/uk_biobank/results/finetuned_50_0.001/best_r2_all.ckpt",
    )
)

args = [
    f"--rest_dir={REST_DIR}",
    f"--working_dir={PRED_DIR}",
    f"--checkpoint_file={CHECKPOINT}",
    f"--test_list={op.realpath(op.join(ABS_PATH, 'experiments/transfer_learning/uk_biobank/data/ukb_test_ids.txt'))}",
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
