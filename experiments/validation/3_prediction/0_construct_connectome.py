# Description: This script computes the vectorized connectivity matrices for the UK Biobank dataset.
# Size of connectivity matrices: 50 x 50, while the output vectorized connectivity matrices are of size 1225 x 1.
import os
import os.path as op
import sys

import nibabel as nib
import numpy as np
from nilearn.masking import apply_mask
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from utils.dual_regression import _extract_timeseries
from utils.utils import crop_img_w_ref

ABS_PATH = sys.path[-1]

HCP_ICA = op.join(ABS_PATH, "utils/templates/melodic_IC_MNI_2mm.nii.gz")
MNI_BRAIN_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask.nii")
MNI_CROP_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii")
INPUT_DIR = op.join(ABS_PATH, "transfer_learning/uk_biobank/data/rest")
OUT_DIR = op.join(ABS_PATH, "transfer_learning/uk_biobank/data/connectome")
SUBJs = np.loadtxt(
    op.join(ABS_PATH, "transfer_learning/uk_biobank/data/ukb_test_ids.txt"), dtype=str
)


def extract_ts(img, mask, ica):
    """Extract timeseries from input image."""
    return np.transpose(_extract_timeseries(apply_mask(img, mask), ica))


def compute_connectivity(
    ts,
    out_file,
    vectorize=True,
):
    """Compute connectivity."""
    print("Computing connectivity...")
    conn_mat = np.corrcoef(ts)
    if vectorize:
        conn_mat = conn_mat[np.triu_indices(conn_mat.shape[0], k=1)]
    np.savetxt(out_file, conn_mat, delimiter=",")
    return conn_mat


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pbar = tqdm(SUBJs, desc="Computing vectorized connectivity matrices")
    for subj in pbar:
        subj_dir = op.join(
            INPUT_DIR,
            subj,
            "fMRI",
            "rfMRI.ica",
            "filtered_func_data_clean_standard.nii.gz",
        )
        out_file = op.join(OUT_DIR, f"{subj}.txt")
        if not op.exists(out_file):
            ica_data = apply_mask(
                HCP_ICA,
                MNI_BRAIN_MASK,
            )
            img = crop_img_w_ref(
                nib.load(subj_dir),
                MNI_CROP_MASK,
            )
            compute_connectivity(
                extract_ts(img, MNI_CROP_MASK, ica_data),
                out_file,
            )
    print("COMPLETED!")


if __name__ == "__main__":
    main()
