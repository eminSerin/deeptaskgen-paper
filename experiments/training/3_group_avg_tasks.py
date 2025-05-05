import os.path as op
import sys
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))

ABS_PATH = sys.path[-1]
TASK_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/task"))
WORKING_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/results/"))
TRAIN_LIST = op.realpath(
    op.join(ABS_PATH, "experiments/training/data/hcp_train_val_ids.txt")
)

if __name__ == "__main__":
    """Compute the average task for the training set."""
    subj = np.genfromtxt(
        TRAIN_LIST,
        dtype=str,
    )
    n_subj = len(subj)
    task_files = [glob(op.join(TASK_DIR, f"{s}*.npy"))[0] for s in subj]
    for i, file in enumerate(tqdm(task_files)):
        if i == 0:
            avg_task = np.load(file) / n_subj
        else:
            avg_task += np.load(file) / n_subj
    np.save(
        op.join(WORKING_DIR, "avg_train_task.npy"),
        avg_task,
    )
