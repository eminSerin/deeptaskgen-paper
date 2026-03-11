import os
import os.path as op
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from utils.utils import get_contrasts

ABS_PATH = sys.path[-1]
MASK = nib.load(op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii"))
FIG_DIR = op.join(ABS_PATH, "validation/2_umaps/figures")

CONTRASTS = np.array(
    [f"{contrast[0]} {contrast[2]}" for contrast in np.array(get_contrasts())]
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

PALETTE = {
    "Language": "#42944A",
    "Relational": "#AE3033",
    "Social": "#FBDF4F",
    "Emotion": "#283F94",
    "Wm": "#0d0d0d",
    "Motor": "#514075",
    "Gambling": "#8AAFB8",
}

DATASETS = [
    "hcp_actual",
    "hcp_pred",
    "hcpd_pred",
    "ukb_pred",
]


def plot_umaps(dataset):
    umap_df = pd.read_csv(
        op.join(ABS_PATH, f"validation/2_umaps/umaps/{dataset}_umap.csv")
    )
    _, ax = plt.subplots(figsize=(5, 5), dpi=1000)

    # Create a scatter plot of the UMAP embeddings
    sns.scatterplot(
        data=umap_df,
        x="UMAP1",
        y="UMAP2",
        hue="Tasks",
        ax=ax,
        palette=PALETTE,
    )
    # plt.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    plt.legend().set_visible(True)

    # Remove axis ticks and values
    ax.set_xticks([])
    ax.set_yticks([])

    # Despine
    sns.despine(offset=10, trim=True)

    # Show the plot
    plt.savefig(
        op.join(
            FIG_DIR,
            f"{dataset}_umap.pdf",
        ),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    for dataset in DATASETS:
        plot_umaps(dataset)
