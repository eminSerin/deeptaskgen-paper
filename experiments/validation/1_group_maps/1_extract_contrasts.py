import os
import os.path as op
import sys

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append("../../..")
from utils.utils import get_contrasts

MASK = nib.load("utils/templates/MNI_2mm_brain_mask_crop.nii")
CONTRASTS = np.array(
    [f"{contrast[0]} {contrast[2]}" for contrast in np.array(get_contrasts())]
)
# Keys represents: Actual HCP-YA, Predicted HCP-YA, Predicted HCP-D, and Predicted UKB.
DIRECTORIES_MAP = {
    "training/data/task": {
        "subj_ids": "training/data/hcp_all_ids.txt",
        "pred": False,
        "multi_set": False,
    },
    "training/results/unetminimal_100_0.001/pred": {
        "subj_ids": "training/data/hcp_all_ids.txt",
        "pred": True,
        "multi_set": True,
    },
    "experiments/transfer_learning/hcp_development/results/finetuned_50_0.001_emotion-faces-shapes/pred": {
        "subj_ids": "experiments/transfer_learning/hcp_development/data/hcpd_all_ids.txt",
        "pred": True,
        "multi_set": False,
    },
    "experiments/transfer_learning/ukb/results/finetuned_50_0.001_emotion-faces-shapes/pred": {
        "subj_ids": "experiments/transfer_learning/ukb/data/ukb_test_ids.txt",
        "pred": True,
        "multi_set": False,
    },
}
N_JOBS = 4


def load_task_img(task_dir, subj, pred=False, multi_set=False):
    ext = "_pred.npy" if pred else "_joint_MNI_task_contrasts.npy"
    if multi_set:
        return np.load(op.join(task_dir, subj + ext)).mean(axis=0).squeeze()
    else:
        return np.load(op.join(task_dir, subj + ext))


def extract_task_contrast(
    task_dir,
    subj_ids,
    pred=False,
    multi_set=False,
    n_jobs=1,
):
    out_dir = op.join(op.dirname(task_dir), "contrast_z_maps")
    subj_list = np.genfromtxt(subj_ids, dtype=str)

    def process_subject(subj):
        img = load_task_img(task_dir, subj, pred=pred, multi_set=multi_set)
        for c_idx, cont in enumerate(CONTRASTS):
            cont_dir = op.join((out_dir), cont.replace(" ", "_").lower())
            cont_file = op.join(cont_dir, subj + ".nii.gz")
            if not op.exists(cont_file):
                if not op.exists(cont_dir):
                    os.makedirs(cont_dir)
                nib.nifti1.Nifti1Image(
                    img[c_idx, :, :, :],
                    affine=MASK.affine,
                    header=MASK.header,
                ).to_filename(cont_file)

    img_list = Parallel(n_jobs=n_jobs)(
        delayed(process_subject)(subj) for subj in tqdm(subj_list)
    )
    return img_list


if __name__ == "__main__":
    for task_dir, params in tqdm(DIRECTORIES_MAP.items()):
        extract_task_contrast(
            task_dir,
            params["subj_ids"],
            pred=params["pred"],
            multi_set=params["multi_set"],
            n_jobs=N_JOBS,
        )
