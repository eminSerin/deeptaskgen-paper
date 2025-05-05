import os.path as op
import subprocess
import sys

sys.path.append(op.abspath(op.join(__file__, "../../..")))
ABS_PATH = sys.path[-1]

# Prediction related args.
EPOCHS = 100
LR = 1e-3
REST_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/rest"))
TASK_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/task"))
TRAIN_FUNC = op.join(ABS_PATH, "deeptaskgen/deeptaskgen/train.py")
WORKING_DIR = op.realpath(
    op.join(ABS_PATH, f"experiments/training/results/attentionunet_{EPOCHS}_{LR}_gm")
)
LOSS_MASK = op.realpath(
    op.join(ABS_PATH, "experiments/utils/templates/MNI_2mm_GM_mask_crop.nii")
)
# Subjects!
TRAIN_LIST = op.realpath(
    op.join(ABS_PATH, "experiments/training/data/hcp_train_ids.txt")
)
VAL_LIST = op.realpath(op.join(ABS_PATH, "experiments/training/data/hcp_val_ids.txt"))

# Prediction arguments.
args = [
    f"--rest_dir={REST_DIR}",
    f"--task_dir={TASK_DIR}",
    f"--working_dir={WORKING_DIR}",
    "--loss=crr",
    f"--train_list={TRAIN_LIST}",
    f"--val_list={VAL_LIST}",
    "--n_samples_per_subj=8",
    "--n_workers=10",
    "--architecture=attentionunet",
    "--n_out_channels=47",
    "--batch_size=10",
    f"--n_epochs={EPOCHS}",
    "--max_depth=1",
    "--n_conv_layers=1",
    "--fdim=64",
    f"--lr={LR}",
    f"--loss_mask={LOSS_MASK}",
    "--alpha=0.25",
]
args = " ".join(args)

# Train the model.
cmd = f"{sys.executable} {TRAIN_FUNC} {args}"
subprocess.call(cmd, shell=True)
