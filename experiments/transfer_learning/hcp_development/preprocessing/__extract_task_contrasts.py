import json
import os
import os.path as op
import re
import sys
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
ABS_PATH = sys.path[-1]

TASK_INFO = dict(
    guessing=dict(
        contrast=[
            "feedbackLowLose + feedbackHighLose",
            "feedbackLowWin + feedbackHighWin",
            "feedbackLowLose + feedbackHighLose - feedbackLowWin - feedbackHighWin",
        ],
        name=["punish", "reward", "punish-reward"],
    ),
    emotion=dict(
        contrast=["faces", "shapes", "faces - shapes"],
        name=["faces", "shapes", "faces-shapes"],
    ),
)
DATA_DIR = op.realpath(
    op.join(ABS_PATH, "experiments/transfer_learning/hcp_development/data/raw")
)


def fit_firstlevel(file, task, events, confound, mask):
    """Fits first level model for a given task."""
    hdr = nib.load(file).header
    tr, nvol = hdr["pixdim"][4], hdr["dim"][4]
    frame_times = np.linspace(0.5 * tr, tr * nvol, num=nvol, endpoint=False)

    dm = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model="glover",
        high_pass=0.01,
        add_regs=confound,
        drift_model=None,
    )

    task_glm = FirstLevelModel(
        standardize=False, high_pass=None, mask_img=mask, smoothing_fwhm=6, n_jobs=-1
    )
    task_glm = task_glm.fit(file, design_matrices=dm)
    cont_dir = op.join(op.dirname(file), "contrasts")
    if not op.exists(cont_dir):
        os.makedirs(cont_dir)
    for cont, cont_name in zip(TASK_INFO[task]["contrast"], TASK_INFO[task]["name"]):
        cont_file = op.join(
            cont_dir, op.basename(f).replace(".nii.gz", f"_contrast-{cont_name}_")
        )
        task_glm.compute_contrast(cont, output_type="z_score").to_filename(
            cont_file + "z_statmap.nii.gz"
        )
        task_glm.compute_contrast(cont, output_type="effect_size").to_filename(
            cont_file + "effect_statmap.nii.gz"
        )


def try_fit_firstlevel(file, task):
    """Runs fit_firstlevel and catches exceptions."""
    events = pd.read_csv(file.replace("_hp0_clean.nii.gz", "_events.tsv"), sep="\t")
    movement = pd.DataFrame(
        np.genfromtxt(op.join(op.dirname(file), "Movement_Regressors_hp0_clean.txt"))
    )
    mask = nib.load(op.join(op.dirname(file), "brainmask_fs.2.nii.gz"))
    try:
        fit_firstlevel(file, task, events, movement, mask)
    except Exception as e:
        return {"file": file, "task": task, "error": e}


if __name__ == "__main__":
    files = sorted(
        glob(
            op.join(
                DATA_DIR,
                "*",
                "MNINonLinear",
                "Results",
                "tfMRI_*",
                "tfMRI_*_hp0_clean.nii.gz",
            )
        )
    )
    err = []
    pbar = tqdm(files)
    for f in pbar:
        task = re.search(r"tfMRI_(\w+)_", f).group(1).lower()
        subj = re.search(r"HCD\d+", f).group(0)
        pbar.set_description(f"Subject {subj}, Task: {task}")
        err.append(try_fit_firstlevel(f, task))

    err = [e for e in err if e is not None]
    with open(op.join(DATA_DIR, "task_error.json"), "w") as f:
        json.dump(err, f, indent=4)
