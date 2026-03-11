import os
import os.path as op
import sys

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.appendsys.path.append(op.abspath(op.join(__file__, "../../../..")))
from utils.utils import crop_img_w_ref  # noqa: E402

ABS_PATH = sys.path[-1]
MNI_CROP_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii")
RAW_DIR = op.join(ABS_PATH, "transfer_learning/uk_biobank/data/raw")
OUT_DIR = op.join(ABS_PATH, "transfer_learning/uk_biobank/data/task_faces")
SUBJ_IDS = np.genfromtxt(
    op.join(ABS_PATH, "transfer_learning/ukb/data/ukb_all_ids.txt"), dtype=str
)
COPE_LIST = ["zstat5"]
N_CORES = 4


def join_task_contrasts(subj_id):
    if not op.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    out_file = op.join(OUT_DIR, f"{subj_id}_joint_MNI_task_contrasts.npy")
    if not op.exists(out_file):
        task_list = []
        for cope in COPE_LIST:
            cope_dir = op.join(RAW_DIR, subj_id, "fMRI", "tfMRI.feat", "stats")
            input = op.join(cope_dir, f"{cope}_standard.nii.gz")
            task_list.append(crop_img_w_ref(nib.load(input), MNI_CROP_MASK).get_fdata())
        np.save(
            out_file,
            np.stack(task_list, 0),
        )
    print("COMPLETED!")


if __name__ == "__main__":
    _ = Parallel(n_jobs=N_CORES)(
        delayed(join_task_contrasts)(subj_id) for subj_id in tqdm(SUBJ_IDS)
    )
