import os.path as op
import sys
import warnings

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from utils.utils import compute_corr_coeff

warnings.filterwarnings("ignore")

ABS_PATH = sys.path[-1]
MNI_CROP_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii")
TEST_SUBJ = op.join(
    ABS_PATH, "transfer_learning/hcp_development/data/hcpd_test_ids.txt"
)
TASK_MAP = {
    "emotion_faces-shapes": {
        "idx": 11,
        "actual": op.join(
            ABS_PATH, "transfer_learning/hcp_development/data/task_faces-shapes"
        ),
        "pred": {
            "finetune": op.join(
                ABS_PATH,
                "transfer_learning/hcp_development/results/finetuned_50_0.001_gambling-reward/pred",
            ),
            "nofinetune": op.join(
                ABS_PATH,
                "transfer_learning/hcp_development/results/nofinetune/pred",
            ),
            "tavor": op.join(
                ABS_PATH,
                "transfer_learning/hcp_development/results/tavor/pred",
            ),
        },
    },
    "gambling_reward": {
        "idx": 45,
        "actual": op.join(
            ABS_PATH, "transfer_learning/hcp_development/data/task_reward"
        ),
        "pred": {
            "finetune": op.join(
                ABS_PATH,
                "transfer_learning/hcp_development/results/finetuned_50_0.001_emotion-faces-shapes/pred",
            ),
            "nofinetune": op.join(
                ABS_PATH,
                "transfer_learning/hcp_development/results/nofinetune/pred",
            ),
            "tavor": op.join(
                ABS_PATH,
                "transfer_learning/hcp_development/results/tavor/pred",
            ),
        },
    },
}

EXT_MAP = {
    "actual": "_joint_MNI_task_contrasts.npy",
    "pred": "_pred.npy",
}

N_JOBS = 16


# Define helper functions
def compute_subj_contrast_corr(pred, ref):
    """Computes correlation between predicted and true task contrasts"""
    n_subj = ref.shape[0]
    return compute_corr_coeff(
        ref[:, :, :].reshape(n_subj, -1),
        pred[:, :, :].reshape(n_subj, -1),
    )


def load_contrasts(
    sub_ids, contrast_dir, contrast_ext, contrast_idx=None, n_jobs=1, dtype=np.float32
):
    """Loads contrast files"""

    def load_input(input):
        if contrast_idx is not None:
            return np.load(input).astype(dtype)[:, contrast_idx, :, :, :]
        return np.load(input).astype(dtype)

    return np.asarray(
        Parallel(n_jobs=n_jobs)(
            delayed(load_input)(op.join(contrast_dir, f"{sub}{contrast_ext}"))
            for sub in tqdm(sub_ids)
        ),
        dtype=dtype,
    )


def compute_corr(pred_tasks, actual_tasks, task_name, contrast_idx):
    # Load test IDs
    subj_ids = np.genfromtxt(
        TEST_SUBJ,
        dtype="<U13",
    )
    # Brain mask indices
    mask_idx = np.nonzero(nib.load(MNI_CROP_MASK).get_fdata())

    n_subj = len(subj_ids)
    # Load test contrasts (i.e., y-true)
    print("Loading actual task contrasts...")
    test_contrasts = load_contrasts(
        subj_ids, actual_tasks, EXT_MAP["actual"], n_jobs=N_JOBS
    )[..., mask_idx[0], mask_idx[1], mask_idx[2]]
    print(f"Actual contrasts shape: {test_contrasts.shape}")

    # If predict_tasks is a file, then consider it as average.
    print("Loading contrasts...")
    pred_cont = load_contrasts(
        subj_ids, pred_tasks, EXT_MAP["pred"], contrast_idx=contrast_idx, n_jobs=N_JOBS
    )[..., mask_idx[0], mask_idx[1], mask_idx[2]]
    print(f"Predicted contrasts shape: {pred_cont.shape}")
    print("Computing correlation matrices...")
    corr = compute_corr_coeff(
        test_contrasts[:, :, :].reshape(n_subj, -1),
        pred_cont[:, :, :].reshape(n_subj, -1),
    )
    np.save(op.join(op.dirname(pred_tasks), f"corr_scores_{task_name}.npy"), corr)


if __name__ == "__main__":
    for task in TASK_MAP:
        for pred_task in TASK_MAP[task]["pred"]:
            print(f"Computing correlation for {task} - {pred_task}...")
            compute_corr(
                TASK_MAP[task]["pred"][pred_task],
                TASK_MAP[task]["actual"],
                task,
                TASK_MAP[task]["idx"],
            )
            print("Done!")
