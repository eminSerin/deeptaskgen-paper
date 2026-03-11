import os
import os.path as op
import sys
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.masking import apply_mask
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from utils.dual_regression import _extract_timeseries
from utils.utils import compute_corr_coeff, crop_img_w_ref

ABS_PATH = sys.path[-1]
HCP_ICA = op.join(ABS_PATH, "utils/templates/melodic_IC_MNI_2mm.nii.gz")
MNI_BRAIN_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask.nii")
MNI_CROP_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii")
RAW_DIR = op.join(ABS_PATH, "training/data/raw")
OUT_DIR = op.join(ABS_PATH, "training/data/rest")
SUBJ_IDS = np.genfromtxt(op.join(ABS_PATH, "training/data/hcp_all_ids.txt"), dtype=str)
N_CORES = 1
NUM_SAMPLES = 8
TOTAL_TP = 4800


def concat_rest(imgs):
    """Concatenate resting-state images."""
    print("Concatenating images...")
    return nib.concat_images(imgs, axis=3)


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


def compute_v2r_conn(img, ts, out_file, num_samples=1):
    """Compute voxel-to-ROI connectivity."""
    print("Computing voxel-to-ROI connectivity...")
    # Silence numpy warnings.
    np.seterr(divide="ignore", invalid="ignore")

    sample_length = TOTAL_TP // num_samples

    # Organize inputs
    if not isinstance(img, np.ndarray):
        img = img.get_fdata()
    if isinstance(ts, pd.DataFrame):
        ts = ts.to_numpy()

    img_dim = img.shape[:-1]

    # Compute correlation coefficient and save
    for i in tqdm(range(num_samples)):
        rsfc_file = out_file + f"_sample{i}_rsfc.npy"
        if not op.exists(rsfc_file):
            ts_idx = np.arange(i * sample_length, (i + 1) * sample_length)

            np.save(
                rsfc_file,
                np.transpose(
                    np.nan_to_num(
                        compute_corr_coeff(
                            img[..., ts_idx].reshape((-1, len(ts_idx))),
                            ts.T[:, ts_idx],
                        ),
                    ).reshape((*img_dim, -1)),
                    (3, 0, 1, 2),
                ),
            )


def main():
    def process_subject(subj_id):
        """Preprocess resting-state data."""
        os.makedirs(OUT_DIR, exist_ok=True)
        img_dirs = sorted(
            glob(
                f"{RAW_DIR}/{subj_id}/MNINonLinear/Results/rfMRI_*/*_hp2000_clean.nii.gz"
            )
        )
        out_file = op.join(OUT_DIR, f"{subj_id}")
        if not op.exists(out_file + "_sample_0_rsfc.npy"):
            ica_data = apply_mask(HCP_ICA, MNI_BRAIN_MASK)
            img = crop_img_w_ref(concat_rest(img_dirs), MNI_CROP_MASK)
            ts_file = out_file + "_ts.csv"
            ts = extract_ts(img, MNI_CROP_MASK, ica_data, ts_file)
            compute_v2r_conn(img, ts, out_file=out_file, num_samples=NUM_SAMPLES)

    print("Processing subjects...")
    _ = Parallel(n_jobs=N_CORES)(
        delayed(process_subject)(subj_id) for subj_id in tqdm(SUBJ_IDS)
    )
    print("COMPLETED!")


if __name__ == "__main__":
    main()
