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
    ABS_PATH, "experiments/utils/templates/MNI_2mm_GM_mask_crop.nii"
)
MASK_IDX = np.nonzero(nib.load(MNI_CROP_MASK).get_fdata())
ACTUAL_PATH = op.join(ABS_PATH, "experiments/training/data/task")
TEST_SUBJ = op.join(ABS_PATH, "experiments/training/data/hcp_test_ids.txt")
TASK_MAP = {
    "deeptaskgen": op.join(
        ABS_PATH, "experiments/training/results/attentionunet_100_0.001_gm/pred"
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


if __name__ == "__main__":
    print("Loading actual task contrasts...")
    actual_maps = load_contrasts(
        TEST_SUBJ, ACTUAL_PATH, "_joint_MNI_task_contrasts.npy", n_jobs=N_JOBS
    )[..., MASK_IDX[0], MASK_IDX[1], MASK_IDX[2]]
    print(f"Actual contrasts shape: {actual_maps.shape}")
    for task, task_path in TASK_MAP.items():
        print(f"Task: {task}")
        if not op.exists(op.join(op.dirname(task_path), f"corr_scores_{task}.npy")):
            if task == "group_avg":
                pred_cont = np.tile(
                    np.load(task_path), (TEST_SUBJ.shape[0], 1, 1, 1, 1)
                )[..., MASK_IDX[0], MASK_IDX[1], MASK_IDX[2]]
            elif task == "retest":
                pred_cont = load_contrasts(
                    TEST_SUBJ, task_path, "_joint_MNI_task_contrasts.npy", n_jobs=N_JOBS
                )[..., MASK_IDX[0], MASK_IDX[1], MASK_IDX[2]]
            else:
                pred_cont = load_pred_contrasts(
                    TEST_SUBJ, task_path, "_pred.npy", n_jobs=N_JOBS
                )[..., MASK_IDX[0], MASK_IDX[1], MASK_IDX[2]]
            print(f"Predicted contrasts shape: {pred_cont.shape}")
            print("Computing correlation matrices...")
            corr = compute_subj_contrast_corr(
                pred_cont,
                actual_maps,
                n_jobs=N_JOBS,
            )
            np.save(op.join(op.dirname(task_path), f"corr_scores_{task}.npy"), corr)
