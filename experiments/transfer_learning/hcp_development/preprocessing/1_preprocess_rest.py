import os
import os.path as op
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.masking import apply_mask
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../../..")))
from utils.dual_regression import _extract_timeseries
from utils.utils import compute_corr_coeff, crop_img_w_ref

ABS_PATH = sys.path[-1]
HCP_ICA = op.join(ABS_PATH, "utils/templates/melodic_IC_MNI_2mm.nii.gz")
MNI_BRAIN_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask.nii")
MNI_CROP_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii")
RAW_DIR = op.join(ABS_PATH, "transfer_learning/hcp_development/data/raw")
OUT_DIR = op.join(ABS_PATH, "transfer_learning/hcp_development/data/rest")
SUBJ_IDS = np.genfromtxt(
    op.join(ABS_PATH, "transfer_learning/hcp_development/data/hcpd_all_ids.txt"),
    dtype=str,
)
N_CORES = 1


def extract_ts(input, mask, ica, output):
    """Extract timeseries from input image."""
    print("Extracting timeseries...")
    if not op.exists(output):
        ts = _extract_timeseries(
            apply_mask(input, mask),
            ica,
        )
        ts = pd.DataFrame(ts)
        ts.to_csv(output, header=None, index=False)
        return ts
    else:
        return pd.read_csv(output, header=None)


def compute_v2r_conn(
    img,
    ts,
    out_file,
):
    """Compute voxel-to-ROI connectivity."""
    print("Computing voxel-to-ROI connectivity...")
    # Silence numpy warnings.
    np.seterr(divide="ignore", invalid="ignore")

    # Organize inputs
    if not isinstance(img, np.ndarray):
        img = img.get_fdata()
    if isinstance(ts, pd.DataFrame):
        ts = ts.to_numpy()

    img_dim = img.shape[:-1]

    # Compute correlation coefficient and save
    np.save(
        out_file,
        np.transpose(
            np.nan_to_num(
                compute_corr_coeff(img.reshape(-1, img.shape[-1]), ts.T)
            ).reshape((*img_dim, -1)),
            (3, 0, 1, 2),
        ),
    )


def main():
    def process_subject(subj_id):
        if not op.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        out_file = op.join(OUT_DIR, f"{subj_id}_sample0_rsfc.npy")
        if not op.exists(out_file):
            input_dir = op.join(
                RAW_DIR,
                f"{subj_id}_V1_MR/MNINonLinear/Results/rfMRI_REST/rfMRI_REST_hp0_clean.nii.gz",
            )
            ica_data = apply_mask(HCP_ICA, MNI_BRAIN_MASK)
            img = crop_img_w_ref(nib.load(input_dir), MNI_CROP_MASK)
            ts_file = input_dir.replace(".nii.gz", "_ts.csv")
            ts = extract_ts(img, MNI_CROP_MASK, ica_data, ts_file)
            compute_v2r_conn(img, ts, out_file=out_file)

    print("Processing subjects...")
    _ = Parallel(n_jobs=N_CORES)(
        delayed(process_subject)(subj_id) for subj_id in tqdm(SUBJ_IDS)
    )
    print("COMPLETED!")


if __name__ == "__main__":
    main()
