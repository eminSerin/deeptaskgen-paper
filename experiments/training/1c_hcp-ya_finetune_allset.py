# Train pre-trained DeepTaskGen model on whole HCP-YA dataset for 20 more epochs.
# This will prepare the model for transfer learning.
# This step is not strickly neaded, but it is recommended as the sample size is small for high dimensional input data.
import os.path as op
import subprocess
import sys

sys.path.append(op.abspath(op.join(__file__, "../../..")))
ABS_PATH = sys.path[-1]

# Prediction related args.
EPOCHS = 20
N_JOBS = 10
BATCH_SIZE = 10
REST_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/rest"))
TASK_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/task"))
TRAIN_FUNC = op.join(ABS_PATH, "deeptaskgen/deeptaskgen/train.py")
WORKING_DIR = op.realpath(
    op.join(ABS_PATH, "experiments/training/results/attentionunet_100_0.001_gm")
)
LOSS_MASK = op.realpath(
    op.join(ABS_PATH, "experiments/utils/templates/MNI_2mm_GM_mask_crop.nii")
)

# Subjects!
TRAIN_LIST = op.realpath(op.join(ABS_PATH, "experiments/training/data/hcp_all_ids.txt"))

# Prediction arguments.
args = [
    f"--rest_dir={REST_DIR}",
    f"--task_dir={TASK_DIR}",
    f"--working_dir={op.join(WORKING_DIR, 'allset')}",
    f"--checkpoint_file={op.join(WORKING_DIR, 'best_corr.ckpt')}",
    "--loss=mse",
    f"--train_list={TRAIN_LIST}",
    "--run_validation=False",
    "--n_samples_per_subj=8",
    f"--n_workers={N_JOBS}",
    "--architecture=attentionunet",
    "--n_out_channels=47",
    f"--batch_size={BATCH_SIZE}",
    f"--n_epochs={EPOCHS}",
    "--lr_scheduler=False",
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
