import os
import os.path as op
import sys

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from utils.utils import crop_img_w_ref

ABS_PATH = sys.path[-1]

MNI_CROP_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii")
DATA_DIR = op.realpath(op.join(ABS_PATH, "transfer_learning/hcp_development/data/raw"))
CONTRASTS = {
    "EMOTION": "faces-shapes",
    "GAMBLING": "reward",
}
SUBJ_IDS = np.genfromtxt(
    op.join(ABS_PATH, "transfer_learning/hcp_development/data/hcpd_all_ids.txt")
)
N_JOBS = 8


def main(subj_id):
    for task in CONTRASTS:
        cont = CONTRASTS[task]
        input = op.join(
            DATA_DIR,
            f"{subj_id}_V1_MR",
            "MNINonLinear",
            "Results",
            f"tfMRI_{task}_PA",
            "contrasts",
            f"tfMRI_{task}_PA_hp0_clean_contrast-{cont}_z_statmap.nii.gz",
        )
        out_dir = f"transfer_learning/uk_biobank/data/task_{cont}"
        os.makedirs(out_dir, exist_ok=True)

        out_file = op.join(out_dir, f"{subj_id}_joint_MNI_task_contrasts.npy")
        if not op.exists(out_file):
            task_list = []
            task_list.append(crop_img_w_ref(nib.load(input), MNI_CROP_MASK).get_fdata())
            np.save(
                out_file,
                np.stack(task_list, 0),
            )


if __name__ == "__main__":
    _ = Parallel(n_jobs=N_JOBS)(delayed(main)(subj_id) for subj_id in SUBJ_IDS)
    main()
