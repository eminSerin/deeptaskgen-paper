import os.path as op
import sys
import warnings

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))

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
RESULTS_PATH = op.join(ABS_PATH, "experiments/training/results")
N_JOBS = 16

# Dice Parameters
MIN_THRESH = 0.05
MAX_THRESH = 0.51
STEP = 0.05


# Define helper functions
def process_single_subject(
    idx,
    actual_tasks,
    pred_tasks,
    group_average=False,
    retest=False,
):
    actual = np.load(op.join(actual_tasks, f"{idx}{EXT_MAP['actual']}"))[
        ..., MASK_IDX[0], MASK_IDX[1], MASK_IDX[2]
    ]
    if retest:
        pred = np.load(op.join(pred_tasks, f"{idx}{EXT_MAP['actual']}"))
    elif group_average:
        pred = np.load(TASK_MAP["group_avg"])
    else:
        pred = np.load(op.join(pred_tasks, f"{idx}{EXT_MAP['pred']}")).mean(axis=0)
    pred = pred[..., MASK_IDX[0], MASK_IDX[1], MASK_IDX[2]]
    return dice_auc(pred, actual, MIN_THRESH, MAX_THRESH, STEP)


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
    return np.trapezoid(dices, dx=step, axis=0)


if __name__ == "__main__":
    for task, task_path in TASK_MAP.items():
        print(f"Task: {task}")
        if task == "group_avg":
            out_file = op.join(RESULTS_PATH, "dice-auc_scores_group_avg.npy")
            if op.exists(out_file):
                print("Group average already computed!")
                continue
            dice_aucs = Parallel(
                n_jobs=N_JOBS,
            )(
                delayed(process_single_subject)(
                    subj_id,
                    ACTUAL_PATH,
                    task_path,
                    group_average=True,
                )
                for subj_id in tqdm(TEST_SUBJ)
            )
            np.save(out_file, dice_aucs)
        elif task == "retest":
            out_file = op.join(RESULTS_PATH, "dice-auc_scores_retest.npy")
            if op.exists(out_file):
                print("Retest already computed!")
                continue
            dice_aucs = Parallel(
                n_jobs=N_JOBS,
            )(
                delayed(process_single_subject)(
                    subj_id,
                    ACTUAL_PATH,
                    task_path,
                    retest=True,
                )
                for subj_id in tqdm(TEST_SUBJ)
            )
            np.save(out_file, dice_aucs)
        else:
            out_file = op.join(op.dirname(task_path), f"dice-auc_scores_{task}.npy")
            if op.exists(out_file):
                print(f"{task} already computed!")
                continue
            dice_aucs = Parallel(
                n_jobs=N_JOBS,
            )(
                delayed(process_single_subject)(
                    subj_id,
                    ACTUAL_PATH,
                    task_path,
                )
                for subj_id in tqdm(TEST_SUBJ)
            )
            np.save(out_file, dice_aucs)
