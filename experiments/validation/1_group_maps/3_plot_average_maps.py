import os
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
from joblib import Parallel, delayed
from nilearn.plotting import plot_stat_map
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

ABS_PATH = op.abspath(op.join(__file__, "../../.."))

MASK = nib.load(op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii"))

FIG_DIR = op.join(ABS_PATH, "validation/1_group_maps/figures")

CONTRASTS = (
    "EMOTION FACES-SHAPES",
    "GAMBLING REWARD",
    "LANGUAGE MATH-STORY",
    "MOTOR AVG",
    "RELATIONAL REL",
    "SOCIAL TOM-RANDOM",
    "WM 2BK-0BK",
)

TASK_COORDS = {
    "WM": (-38, -16, 38),
    "GAMBLING": (14, 6, 0),
    "MOTOR": (8, -16, -20),
    "LANGUAGE": (-52, -2, -8),
    "TOM": (50, -56, 18),
    "RELATIONAL": (40, 40, 16),
    "EMOTION": (21, -3, -15),
    "SOCIAL": (50, -56, 18),
}
DIRECTORY_MAP = {
    "hcp_actual": op.join(ABS_PATH, "training/data/task/contrast_z_maps"),
    "hcp_pred": op.join(
        ABS_PATH, "training/results/unetminimal_100_0.001/contrast_z_maps"
    ),
    "hcpd_pred": op.join(
        ABS_PATH,
        "transfer_learning/hcp_development/results/finetuned_50_0.001_emotion-faces-shapes/contrast_z_maps",
    ),
    "ukb_pred": op.join(
        ABS_PATH,
        "transfer_learning/ukb/results/finetuned_50_0.001_emotion-faces-shapes/contrast_z_maps",
    ),
}
OUTPUT_EXT = "png"
DPI = 1000
N_JOBS = 8

# Create folders for each contrast
for cont in CONTRASTS:
    os.makedirs(op.join(FIG_DIR, cont.replace(" ", "_").lower()), exist_ok=True)


def scale_img(img, range=(0, 1)):
    im_shape = img.shape
    return nib.Nifti1Image(
        MinMaxScaler(feature_range=range)
        .fit_transform(img.get_fdata().reshape(-1, 1))
        .reshape(im_shape),
        affine=img.affine,
        header=img.header,
    )


def plot_save_fig(brain_map, coords, output, vmax=None, thresh=None, cmap=None):
    if cmap is None:
        cmap = "cold_hot"
    fig = plt.figure(dpi=DPI)
    disp = plot_stat_map(
        brain_map,
        threshold=thresh,
        display_mode="z",
        cut_coords=[coords[2]],
        black_bg=False,
        colorbar=True,
        draw_cross=False,
        vmax=vmax,
        cmap=cmap,
        annotate=False,
        figure=fig,
    )
    disp.savefig(output, dpi=DPI)
    disp.close()


def plot_contrast(cont, data_path, dataset):
    task = cont.split(" ")[0]
    cont = cont.replace(" ", "_").lower()
    fig_file_base = op.join(FIG_DIR, cont, f"{dataset}_")

    # Plot group mean
    bmap = nib.load(op.join(data_path, cont, "group_mean.nii.gz"))
    plot_save_fig(
        bmap,
        TASK_COORDS[task],
        fig_file_base + f"group_mean.{OUTPUT_EXT}",
        vmax=10,
        thresh=1,
    )


def main():
    for dataset, data_path in DIRECTORY_MAP.items():
        print(f"Processing {dataset}...")
        prog_bar = tqdm(CONTRASTS)
        Parallel(n_jobs=N_JOBS)(
            delayed(plot_contrast)(cont, data_path, dataset) for cont in prog_bar
        )


if __name__ == "__main__":
    main()
