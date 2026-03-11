import os.path as op

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

ABS_PATH = op.abspath(op.join(__file__, "../../.."))

MASK = nib.load(op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii"))

CONTRASTS = (
    "EMOTION FACES-SHAPES",
    "GAMBLING REWARD",
    "LANGUAGE MATH-STORY",
    "MOTOR AVG",
    "RELATIONAL REL",
    "SOCIAL TOM-RANDOM",
    "WM 2BK-0BK",
)

N_JOBS = 4

# Keys represent: Actual HCP-YA, Predicted HCP-YA, Predicted HCP-D, and Predicted UKB.
DIRECTORY_MAP = {
    op.join(ABS_PATH, "training/data/contrast_z_maps"): op.join(
        ABS_PATH, "training/data/hcp_all_ids.txt"
    ),
    op.join(
        ABS_PATH, "training/results/unetminimal_100_0.001/contrast_z_maps"
    ): op.join(ABS_PATH, "training/data/hcp_all_ids.txt"),
    op.join(
        ABS_PATH,
        "transfer_learning/hcp_development/results/finetuned_50_0.001_emotion-faces-shapes/contrast_z_maps",
    ): op.join(ABS_PATH, "transfer_learning/hcp_development/data/hcpd_all_ids.txt"),
    op.join(
        ABS_PATH,
        "transfer_learning/uk_biobank/results/finetuned_50_0.001/contrast_z_maps",
    ): op.join(ABS_PATH, "transfer_learning/uk_biobank/data/ukb_test_ids.txt"),
}


def compute_average(task_dir, subj_ids):
    subj_list = np.genfromtxt(subj_ids, dtype=str)

    def process_contrast(cont):
        nib.Nifti1Image(
            np.mean(
                [
                    nib.load(op.join(task_dir, cont, f"{subj}.nii.gz")).get_fdata()
                    for subj in subj_list
                ],
                axis=0,
            ),
            affine=MASK.affine,
            header=MASK.header,
        ).to_filename(op.join(task_dir, cont, "group_mean.nii.gz"))

    # Use joblib to parallelize the loop
    Parallel(n_jobs=N_JOBS)(
        delayed(process_contrast)(cont.replace(" ", "_").lower())
        for cont in tqdm(CONTRASTS)
    )


if __name__ == "__main__":
    for task_dir, subj_ids in DIRECTORY_MAP.items():
        print(f"Computing average maps for {task_dir}")
        compute_average(task_dir, subj_ids)
