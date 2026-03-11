import os.path as op
import subprocess
import sys

sys.path.append(op.abspath(op.join(__file__, "../../../..")))
ABS_PATH = sys.path[-1]

# Prediction related args.
REST_DIR = op.join(ABS_PATH, "experiments/transfer_learning/hcp_development/data/rest")
TASK_DIR = op.join(
    ABS_PATH, "experiments/transfer_learning/hcp_development/data/task_reward"
)
TRAIN_FUNC = op.join(ABS_PATH, "deeptaskgen/deeptaskgen/train.py")
WORKING_DIR = op.join(
    ABS_PATH,
    "experiments/transfer_learning/hcp_development/results/finetuned_50_0.001_gambling-reward",
)
# Model trained on HCP-YA (gambling-reward).
CHECKPOINT = op.join(
    ABS_PATH,
    "experiments/transfer_learning/hcp_development/hcp-ya_gambling-reward.ckpt",
)
LOSS_MASK = op.realpath(
    op.join(ABS_PATH, "experiments/utils/templates/MNI_2mm_GM_mask_crop.nii")
)

# Subjects!
TRAIN_LIST = op.join(
    ABS_PATH, "experiments/transfer_learning/hcp_development/data/hcpd_train_ids.txt"
)
VAL_LIST = op.join(
    ABS_PATH, "experiments/transfer_learning/hcp_development/data/hcpd_val_ids.txt"
)
# Prediction arguments.
args = [
    f"--rest_dir={REST_DIR}",
    f"--task_dir={TASK_DIR}",
    f"--working_dir={WORKING_DIR}",
    "--loss=crr",
    "--freeze_final_layer=True",
    f"--train_list={TRAIN_LIST}",
    f"--val_list={VAL_LIST}",
    f"--checkpoint_file={CHECKPOINT}",
    "--n_samples_per_subj=1",
    "--n_workers=10",
    "--architecture=attentionunet",
    "--n_out_channels=1",
    "--batch_size=10",
    "--n_epochs=50",
    "--max_depth=1",
    "--n_conv_layers=1",
    "--fdim=64",
    "--lr=0.001",
    f"--loss_mask={LOSS_MASK}",
    "--alpha=0.25",
]
args = " ".join(args)

# Train the model.
cmd = f"{sys.executable} {TRAIN_FUNC} {args}"
subprocess.call(cmd, shell=True)
