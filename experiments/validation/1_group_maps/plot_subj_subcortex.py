import os
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.datasets import load_mni152_template
from nilearn.image import math_img
from nilearn.masking import apply_mask, unmask
from nilearn.plotting import plot_stat_map
from tqdm import tqdm

BASE_PATH = (
    "/home/serine/projects/task-generation/deeptaskgen/deeptaskgen-research/experiments"
)
MASK = nib.load(
    op.join(BASE_PATH, "utils/templates/MNI_2mm_subcortical_mask_crop.nii.gz")
)
SUBJ_LIST = np.genfromtxt(
    op.join(BASE_PATH, "training/data/hcp_test_ids.txt"), dtype=str
)
FIG_DIR = op.join(BASE_PATH, "training/subject_plots_subcortex/")
AUC_DB = pd.read_csv(
    op.join(BASE_PATH, "visualization_div/results_data/dice_auc_hcp_subcortex.csv")
)

# Constants
CONTRAST_MAP = {
    "EMOTION FACES-SHAPES": 11,
    "GAMBLING REWARD": 45,
    "LANGUAGE MATH-STORY": 2,
    "MOTOR AVG": 37,
    "RELATIONAL REL": 4,
    "SOCIAL TOM-RANDOM": 8,
    "WM 2BK-0BK": 22,
}

MDL_NAME_MAP = {
    "retest": "Retest",
    "deeptaskgen": "DeepTaskGen",
    "group_avg": "Average",
    "tavor": "Linear Regression",
}


def load_nifti(path):
    """Helper function to load NIfTI file"""
    return nib.load(path)


def create_nifti_from_array(arr):
    """Helper function to create NIfTI image from array"""
    return nib.Nifti1Image(arr, MASK.affine, MASK.header)


def plot_brain_volumes_grid(
    brain_img,
    thresh,
    fig_ax=None,
    cut_coords=(-2, -6, -23),
    cmap="cold_hot",
):
    """
    Plot a 2x1 grid of volumetric brain views (top: no threshold, bottom: thresholded)
    for a given Nifti image using orthogonal slices.

    Parameters
    ----------
    brain_img : Nifti1Image
        The brain activation image to plot.
    title : str, optional
        The figure title.
    thresh : float or None, optional
        The threshold for display. If None, uses 50th percentile of positive values.
    fig_ax : matplotlib.figure.Figure or None, optional
        The figure or axes to plot onto. If None, a new figure is created.
    cut_coords : tuple of float, optional
        The MNI coordinates (x, y, z) for the center of the slices.
    cmap : str or matplotlib.colors.Colormap, optional
        The colormap to use.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The resulting figure, or None if plotting onto a fig_ax.
    """
    # Fetch ICBM template
    bg_img = load_mni152_template()

    # Determine if we need to create a new figure or use the provided fig_ax
    if fig_ax is None:
        fig, axes = plt.subplots(2, 1, figsize=(5, 6))
    else:
        axes = fig_ax.subplots(2, 1)

    # Plot 1: No threshold (or minimal threshold for display)
    plot_stat_map(
        brain_img,
        bg_img=bg_img,
        axes=axes[0],
        display_mode="ortho",
        cut_coords=cut_coords,
        threshold=0,  # Use a very small threshold instead of None to show all values
        cmap=cmap,
        colorbar=False,
        annotate=False,
        draw_cross=False,
        black_bg=False,
    )

    # Plot 2: With threshold
    plot_stat_map(
        brain_img,
        bg_img=bg_img,
        axes=axes[1],
        display_mode="ortho",
        cut_coords=cut_coords,
        threshold=thresh,
        cmap=cmap,
        colorbar=False,
        annotate=False,
        draw_cross=False,
        black_bg=False,
    )


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


def calculate_thresholds(data_map, mask, percentile=97.5):
    """Calculate threshold values for each data type"""
    return {
        key: np.percentile(np.abs(apply_mask(data, mask)), percentile)
        for key, data in data_map.items()
    }


def create_threshold_masks(data_map, mask, thresh_map):
    """Create binary masks based on threshold values"""
    return {
        key: apply_mask(data_map[key], mask) > thresh
        for key, thresh in thresh_map.items()
    }


def plot_subj(subj_id):
    data_map = {}
    for contrast, cont_idx in CONTRAST_MAP.items():
        contrast_dir = contrast.replace(" ", "_").lower()
        # Load actual data
        data_map["actual"] = unmask(
            apply_mask(
                load_nifti(
                    f"{BASE_PATH}/training/data/contrast_z_maps/{contrast_dir}/{subj_id}.nii.gz"
                ),
                MASK,
            ),
            MASK,
        )
        # Load and process DeepTaskGen prediction
        pred = load_nifti(
            f"{BASE_PATH}/training/results-experiment/attentionunet_100_0.001_triplet_0.25_gm/contrast_z_maps/{contrast_dir}/{subj_id}.nii.gz"
        )
        data_map["deeptaskgen"] = unmask(apply_mask(pred, MASK), MASK)

        # Load group average
        group_avg_path = f"{BASE_PATH}/training/data/contrast_z_maps/{contrast_dir}/group_mean.nii.gz"
        data_map["group_avg"] = unmask(
            apply_mask(load_nifti(group_avg_path), MASK), MASK
        )

        # Load and process TAVOR prediction
        tavor_arr = np.load(
            f"{BASE_PATH}/training/results/hcp-ya_tavor/pred/{subj_id}_pred.npy"
        ).mean(axis=0)[cont_idx]
        data_map["tavor"] = unmask(
            apply_mask(create_nifti_from_array(tavor_arr), MASK), MASK
        )

        # Load retest data
        retest_arr = np.load(
            f"{BASE_PATH}/training/data/task_retest/{subj_id}_joint_MNI_task_contrasts.npy"
        )[cont_idx]
        data_map["retest"] = unmask(
            apply_mask(create_nifti_from_array(retest_arr), MASK), MASK
        )

        thresh_map = calculate_thresholds(data_map, MASK, 75)
        thresh_img_map = create_threshold_masks(data_map, MASK, thresh_map)
        fig = plt.figure(figsize=(5, 4))
        for mdl in data_map:
            if mdl == "actual":
                out_fig = op.join(
                    FIG_DIR, contrast_dir, f"{subj_id}_{contrast_dir}_actual.png"
                )
            else:
                dice = dice_score(thresh_img_map[mdl], thresh_img_map["actual"])
                auc = AUC_DB[
                    (AUC_DB["Subject"] == int(subj_id))
                    & (AUC_DB["Contrast"] == contrast)
                    & (AUC_DB["Method"] == MDL_NAME_MAP[mdl])
                ]["Dice AUC"].values[0]
                out_fig = op.join(
                    FIG_DIR,
                    contrast_dir,
                    f"{subj_id}_{contrast_dir}_{mdl}_{auc:.3f}_{dice:.3f}.png",
                )
            os.makedirs(op.dirname(out_fig), exist_ok=True)
            plot_brain_volumes_grid(
                data_map[mdl],
                thresh=thresh_map[mdl],
                fig_ax=fig,
            )
            fig.savefig(out_fig, dpi=500, bbox_inches="tight")


Parallel(n_jobs=20)(delayed(plot_subj)(subj_id) for subj_id in tqdm(SUBJ_LIST))
