import os.path as op
import subprocess
import sys

sys.path.append(op.abspath(op.join(__file__, "../../../..")))
ABS_PATH = sys.path[-1]

# Prediction related args.
REST_DIR = op.realpath(
    op.join(ABS_PATH, "experiments/transfer_learning/uk_biobank/data/rest")
)
TEST_LIST = op.realpath(
    op.join(ABS_PATH, "experiments/transfer_learning/uk_biobank/data/ukb_test_ids.txt")
)
PREDICT_FUNC = op.realpath(op.join(ABS_PATH, "deeptaskgen/deeptaskgen/predict.py"))
WORKING_DIR = op.realpath(
    op.join(
        ABS_PATH,
        "experiments/transfer_learning/uk_biobank/results/finetuned_50_0.001",
    )
)
PRED_DIR = op.join(WORKING_DIR, "pred")
PRED_MASK = op.realpath(
    op.join(ABS_PATH, "experiments/utils/templates/MNI_2mm_GM_mask_crop.nii")
)

# Finetuned model after replacing the last layer with output layer from HCP-YA for 47 task contrasts.
CHECKPOINT = op.realpath(
    op.join(
        ABS_PATH,
        "experiments/transfer_learning/uk_biobank/results/finetuned_50_0.001/best_corr_all.ckpt",
    )
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
    "--architecture=attentionunet",
    "--conv_dim=3",
    "--max_depth=1",
    "--n_conv_layers=1",
    "--fdim=64",
    f"--pred_mask={PRED_MASK}",
]
args = " ".join(args)

# Train the model.
cmd = f"{sys.executable} {PREDICT_FUNC} {args}"
subprocess.call(cmd, shell=True)
