# This script warps the preprocessed data to MNI space
# It requires FSL to be installed!
import os.path as op
import subprocess

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Constants
ABS_PATH = op.abspath(op.join(__file__, "../../../.."))
FSL_PATH = ""  #! Write the path to FSL here
MNI_TEMPLATE = f"{FSL_PATH}/data/standard/MNI152_T1_2mm_brain"
RAW_DIR = op.join(ABS_PATH, "transfer_learning/uk_biobank/data/raw")
SUBJ_IDS = np.genfromtxt(
    op.join(ABS_PATH, "transfer_learning/uk_biobank/data/ukb_all_ids_.txt"), dtype=str
)
N_CORES = 1


def warp2standard(img_path):
    # Warp native space to MNI
    print("Warping into native space...")
    if not op.exists(img_path):
        raise FileNotFoundError(img_path)
    out_path = img_path.replace(".nii.gz", "_standard.nii.gz")
    if not op.exists(out_path):
        cmd = [
            "applywarp -i",
            img_path,
            "-r",
            MNI_TEMPLATE + ".nii.gz",
            "-m",
            MNI_TEMPLATE + "_mask.nii.gz",
            "-w",
            op.join(op.dirname(img_path), "reg", "example_func2standard_warp.nii.gz"),
            "-o",
            out_path,
        ]
        print(" ".join(cmd))
        subprocess.call(" ".join(cmd), shell=True, check=True)
    print("COMPLETED!")


if __name__ == "__main__":
    _ = Parallel(n_jobs=N_CORES)(
        delayed(warp2standard)(
            op.join(
                RAW_DIR, subj, "fMRI", "rfMRI.ica", "filtered_func_data_clean.nii.gz"
            )
        )
        for subj in tqdm(SUBJ_IDS)
    )
