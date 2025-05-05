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
MASK = op.join(ABS_PATH, "experiments/utils/templates/MNI_2mm_GM_mask_crop.nii")
MASK_IDX = np.nonzero(nib.load(MASK).get_fdata())
TEST_SUBJ = np.genfromtxt(
    op.join(ABS_PATH, "experiments/transfer_learning/uk_biobank/data/ukb_test_ids.txt"),
    dtype=str,
)
N_SUBJ = len(TEST_SUBJ)
TASK_MAP = {
    "emotion_faces-shapes": {
        "idx": 11,
        "actual": "experiments/transfer_learning/uk_biobank/data/task_faces",
        "pred": op.join(
            ABS_PATH,
            "experiments/transfer_learning/uk_biobank/results/finetuned_50_0.001/pred",
        ),
    }
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


def threshold_image(img, thresh=0.05):
    """
    Threshold image by keeping only the top percentage of voxels by absolute value

    Parameters:
    -----------
    img : numpy.ndarray
        Image array
    thresh : float, default=0.05
        Percentage of voxels to keep (0-1)

    Returns:
    --------
    img_thresh : numpy.ndarray
        Binary mask where True indicates voxels above threshold.
    """
    # Get the number of voxels in the last dimension
    n_voxels = img.shape[-1]
    # Calculate the number of voxels to keep
    k_voxel = int(thresh * n_voxels)

    # Calculate threshold values for each position in all dimensions except the last
    voxel_thresholds = np.sort(np.abs(img), axis=-1)[..., -k_voxel]

    # Create binary mask of values above threshold
    return np.abs(img) > voxel_thresholds[..., np.newaxis]


def dice_score(pred, actual):
    """
    Calculates the Dice score between two binary masks.

    It works with 3D or 4D arrays. If given array is 4D,
    it will compute Dice score for each 3D volume along the first dimension.

    Parameters:
    -----------
    pred : numpy.ndarray
        Predicted binary mask.
    actual : numpy.ndarray
        Ground truth binary mask.

    Returns:
    --------
    float or numpy.ndarray
        Dice score. If input is 3D, returns a float. If input is 4D,
        returns a numpy array of Dice scores for each volume.
    """
    if pred.ndim != actual.ndim:
        raise ValueError(
            f"Input 'pred' and 'actual' must have the same number of dimensions, but got {pred.ndim}D and {actual.ndim}D."
        )
    if pred.ndim not in [1, 2]:
        raise ValueError(f"Input 'pred' must be 1D or 2D, but got {pred.ndim}D.")
    if pred.ndim == 1:
        intersection = np.sum(pred * actual)
        union = np.sum(pred) + np.sum(actual)
        if union == 0:
            return (
                1.0 if intersection == 0 else 0.0
            )  # Handle empty mask case to avoid division by zero
        return 2.0 * intersection / union
    else:
        intersection = np.sum(pred * actual, axis=(-1))
        union = np.sum(pred, axis=(-1)) + np.sum(actual, axis=(-1))
        dice_values = np.where(
            union == 0,
            np.where(
                intersection == 0, 1.0, 0.0
            ),  # Handle empty mask case for each volume
            2.0 * intersection / union,
        )
        return dice_values


def dice_auc(
    pred, actual, thresh=0.05, max_thresh=0.51, step=0.05, n_jobs=1, verbose=False
):
    """
    Compute the Dice AUC between two binary masks.

    Parameters:
    -----------
    pred : numpy.ndarray
        Predicted binary mask.
    actual : numpy.ndarray
        Ground truth binary mask.
    thresh : float, default=0.05
        Initial threshold value.
    max_thresh : float, default=0.51
        Maximum threshold value (exclusive).
    step : float, default=0.05
        Step size for threshold values.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all processors.
    verbose: bool, default=False
        If True, print progress messages.

    Returns:
    --------
    float
        Dice AUC.
    """
    if verbose:
        print("Computing Dice scores...")
    thresholds = np.arange(thresh, max_thresh, step)

    def compute_dice_for_threshold(th):
        thresh_pred_th = threshold_image(pred, th)
        thresh_actual_th = threshold_image(actual, th)
        return dice_score(thresh_pred_th, thresh_actual_th)

    # Compute dice scores in parallel
    dices = Parallel(n_jobs=n_jobs)(
        delayed(compute_dice_for_threshold)(th) for th in thresholds
    )

    # Calculate AUC using trapezoidal rule
    return np.trapz(dices, dx=step, axis=0)


if __name__ == "__main__":
    for task in TASK_MAP:
        print("Loading actual task contrasts...")
        test_contrasts = load_contrasts(
            TEST_SUBJ,
            TASK_MAP[task]["actual"],
            EXT_MAP["actual"],
            n_jobs=N_JOBS,
        )[..., MASK_IDX[0], MASK_IDX[1], MASK_IDX[2]]
        print("Loading precited task contrasts...")
        pred_cont = load_contrasts(
            TEST_SUBJ,
            TASK_MAP[task]["pred"],
            EXT_MAP["pred"],
            n_jobs=N_JOBS,
            contrast_idx=TASK_MAP[task]["idx"],
        )[..., MASK_IDX[0], MASK_IDX[1], MASK_IDX[2]]
        print(f"Computing correlation for {task}...")
        corr = compute_corr_coeff(
            test_contrasts.reshape(N_SUBJ, -1),
            pred_cont.reshape(N_SUBJ, -1),
        )
        np.save(
            op.join(
                op.dirname(TASK_MAP[task]["pred"]),
                f"corr_scores_{task}.npy",
            ),
            corr,
        )
        print(f"Computing Dice AUC for {task}...")
        dice = dice_auc(
            pred_cont.reshape(N_SUBJ, -1),
            test_contrasts.reshape(N_SUBJ, -1),
            n_jobs=N_JOBS,
        )
        np.save(
            op.join(
                op.dirname(TASK_MAP[task]["pred"]),
                f"dice_auc_{task}.npy",
            ),
            dice,
        )
