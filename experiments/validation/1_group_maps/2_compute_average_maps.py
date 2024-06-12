import os.path as op
import sys

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from utils.utils import get_contrasts

ABS_PATH = sys.path[-1]
MASK = nib.load(op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii"))
CONTRASTS = np.array(
    [f"{contrast[0]} {contrast[2]}" for contrast in np.array(get_contrasts())]
)
N_JOBS = 4
# Keys represent: Actual HCP-YA, Predicted HCP-YA, Predicted HCP-D, and Predicted UKB.
DIRECTORY_MAP = {
    op.join(ABS_PATH, "training/data/task/contrast_z_maps"): op.join(
        ABS_PATH, "training/data/hcp_all_ids.txt"
    ),
    op.join(
        ABS_PATH, "training/results/unetminimal_100_0.001/contrast_z_maps"
    ): op.join(ABS_PATH, "training/data/hcp_all_ids.txt"),
    op.join(
        ABS_PATH,
        "experiments/transfer_learning/hcp_development/results/finetuned_50_0.001_emotion-faces-shapes/contrast_z_maps",
    ): op.join(
        ABS_PATH, "experiments/transfer_learning/hcp_development/data/hcpd_all_ids.txt"
    ),
    op.join(
        ABS_PATH,
        "experiments/transfer_learning/ukb/results/finetuned_50_0.001_emotion-faces-shapes/contrast_z_maps",
    ): op.join(ABS_PATH, "experiments/transfer_learning/ukb/data/ukb_test_ids.txt"),
}


def compute_average(task_dir, subj_ids):
    subj_list = np.genfromtxt(subj_ids, dtype=str)

    def process_contrast(cont):
        nib.Nifti1Image(
            np.mean(
                [
                    nib.load(op.join(task_dir, cont, f"{subj}.nii.gz"))
                    for subj in subj_list
                ],
                axis=0,
            ),
            affine=MASK.affine,
            header=MASK.header,
        ).to_filename(op.join(task_dir, cont, "group_mean.nii.gz"))

    # Use joblib to parallelize the loop
    Parallel(n_jobs=N_JOBS)(
        delayed(process_contrast)(cont.replace(" ", "_").lower()) for cont in CONTRASTS
    )


if __name__ == "__main__":
    for task_dir, subj_ids in tqdm(DIRECTORY_MAP.items()):
        compute_average(task_dir, subj_ids)
