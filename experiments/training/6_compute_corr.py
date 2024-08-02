import os.path as op
import sys
import warnings

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from experiments.utils.utils import compute_corr_coeff

warnings.filterwarnings("ignore")

ABS_PATH = sys.path[-1]
MNI_CROP_MASK = op.join(
    ABS_PATH, "experiments/utils/templates/MNI_2mm_brain_mask_crop.nii"
)
ACTUAL_PATH = op.join(ABS_PATH, "experiments/training/data/task")
TEST_SUBJ = op.join(ABS_PATH, "experiments/training/data/hcp_test_ids.txt")
TASK_MAP = {
    "deeptaskgen": op.join(
        ABS_PATH, "experiments/training/results/unetminimal_100_0.001/pred"
    ),
    "tavor": op.join(ABS_PATH, "experiments/training/results/tavor/pred"),
    "retest": op.join(ABS_PATH, "experiments/training/data/task_retest"),
    "group_avg": op.join(ABS_PATH, "experiments/training/results/avg_train_task.npy"),
}
EXT_MAP = {
    "actual": "_joint_MNI_task_contrasts.npy",
    "pred": "_pred.npy",
}
N_JOBS = 16


# Define helper functions
def compute_subj_contrast_corr(pred, ref, n_jobs=1):
    """Computes correlation between predicted and true task contrasts"""
    n_subj = ref.shape[0]
    n_contrast = ref.shape[1]
    return Parallel(n_jobs=n_jobs)(
        delayed(compute_corr_coeff)(
            ref[:, i, :].reshape(n_subj, -1),
            pred[:, i, :].reshape(n_subj, -1),
        )
        for i in range(n_contrast)
    )


def load_contrasts(sub_ids, contrast_dir, contrast_ext, n_jobs=1, dtype=np.float32):
    """Loads contrast files"""

    def load_input(input):
        if input.endswith(".npy"):
            return np.load(input).astype(dtype)
        elif input.endswith((".nii.gz", ".nii")):
            return np.expand_dims(nib.load(input).get_fdata().astype(dtype), axis=0)

    return np.asarray(
        Parallel(n_jobs=n_jobs)(
            delayed(load_input)(op.join(contrast_dir, f"{sub}{contrast_ext}"))
            for sub in tqdm(sub_ids)
        ),
        dtype=dtype,
    )


def load_pred_contrasts(*args, **kwargs):
    """Load predicted task contrast maps."""
    data = load_contrasts(*args, **kwargs)
    if data.ndim == 5:
        return data
    elif data.ndim == 6:
        return np.mean(data, 1)
    else:
        raise ValueError("Invalid data shape!")


def compute_corr(pred_tasks, actual_tasks, subj_path, retest=False):
    # Load test IDs
    subj_ids = np.genfromtxt(
        subj_path,
        dtype="<U13",
    )
    # Brain mask indices
    mask_idx = np.nonzero(nib.load(MNI_CROP_MASK).get_fdata())

    # Load test contrasts (i.e., y-true)
    print("Loading actual task contrasts...")
    test_contrasts = load_contrasts(
        subj_ids, actual_tasks, EXT_MAP["actual"], n_jobs=N_JOBS
    )[..., mask_idx[0], mask_idx[1], mask_idx[2]]
    print(f"Actual contrasts shape: {test_contrasts.shape}")

    # If predict_tasks is a file, then consider it as average.
    print("Loading contrasts...")
    if op.isdir(pred_tasks):
        if not retest:
            pred_cont = load_pred_contrasts(
                subj_ids, pred_tasks, EXT_MAP["pred"], n_jobs=N_JOBS
            )[..., mask_idx[0], mask_idx[1], mask_idx[2]]
        else:
            pred_cont = load_contrasts(
                subj_ids, pred_tasks, EXT_MAP["actual"], n_jobs=N_JOBS
            )[..., mask_idx[0], mask_idx[1], mask_idx[2]]
    elif op.isfile(pred_tasks):
        # If group average.
        pred_cont = np.tile(np.load(pred_tasks), (subj_ids.shape[0], 1, 1, 1, 1))[
            ..., mask_idx[0], mask_idx[1], mask_idx[2]
        ]
    else:
        raise ValueError(
            "Please provide either average task contrast or path for predicted contrast maps!"
        )
    print(f"Predicted contrasts shape: {pred_cont.shape}")
    print("Computing correlation matrices...")
    corr = compute_subj_contrast_corr(
        pred_cont,
        test_contrasts,
        n_jobs=N_JOBS,
    )
    np.save(op.join(op.dirname(pred_tasks), "corr_scores.npy"), corr)


if __name__ == "__main__":
    for task, task_path in TASK_MAP.items():
        print(f"Task: {task}")
        if not op.exists(op.join(op.dirname(task_path), "corr_scores.npy")):
            if task == "group_avg":
                compute_corr(task_path, ACTUAL_PATH, TEST_SUBJ)
            elif task == "retest":
                compute_corr(
                    task_path,
                    ACTUAL_PATH,
                    TEST_SUBJ,
                    retest=True,
                )
            else:
                compute_corr(task_path, ACTUAL_PATH, TEST_SUBJ)
