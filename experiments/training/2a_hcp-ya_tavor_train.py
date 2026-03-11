import os
import os.path as op
import pickle
import sys
from datetime import datetime

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
ABS_PATH = sys.path[-1]
REST_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/rest"))
TASK_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/task"))
WORKING_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/results/tavor"))
os.makedirs(WORKING_DIR, exist_ok=True)
PARCEL_IMG = nib.load(  # 50 Labels from Group-ICA
    op.join(
        ABS_PATH,
        "utils/templates/melodic_IC_MNI_dlabel_crop.nii",
    )
)
PARCEL_MASK = PARCEL_IMG.get_fdata()
PARCEL_IDX = np.setdiff1d(np.unique(PARCEL_MASK), [0])  # Remove 0 label (background)
N_PARCELS = len(PARCEL_IDX)  # 50 parcels
TRAIN_LIST = op.realpath(
    op.join(ABS_PATH, "experiments/training/data/hcp_train_val_ids.txt")
)
N_JOBS = 16
N_CONTRASTS = 47
N_SAMPLES = 8
N_ROIS = N_PARCELS
SEED = 42

np.random.seed(SEED)


def _standardize(X):
    if X.shape[1] != N_ROIS:
        raise ValueError(f"X must have {N_ROIS} features, got {X.shape[1]}")
    return StandardScaler().fit_transform(X)


def fit_parcel(X, y, standardize=True):
    if standardize:
        X = _standardize(X)
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add constant regressor
    return np.linalg.pinv(X) @ y


def train_subject(subject, n_samples=8):
    if n_samples > 1:
        sample_idx = np.random.randint(0, n_samples)
    else:
        sample_idx = 0
    rest = np.load(op.join(REST_DIR, f"{subject}_sample{sample_idx}_rsfc.npy"))
    task = np.load(op.join(TASK_DIR, f"{subject}_joint_MNI_task_contrasts.npy"))
    coefs = np.zeros((N_CONTRASTS, N_PARCELS, N_ROIS + 1))
    for c in range(N_CONTRASTS):
        for p, p_idx in enumerate(PARCEL_IDX):
            coefs[c, p, :] = fit_parcel(
                rest[..., PARCEL_MASK == p_idx].T,
                task[c][PARCEL_MASK == p_idx],
            )
    return coefs


if __name__ == "__main__":
    print("Training Tavor model for HCP-YA dataset...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(train_subject)(subject, n_samples=N_SAMPLES)
        for subject in tqdm(TRAIN_LIST)
    )
    coefs = np.array(results)
    print("Training complete! Saving model...")
    with open(op.join(WORKING_DIR, "tavor_hcp-ya_model.pkl"), "wb") as f:
        pickle.dump(
            {
                "coefs": coefs,
                "args": {
                    "n_samples": N_SAMPLES,
                    "n_contrasts": N_CONTRASTS,
                    "n_rois": N_ROIS,
                    "n_parcels": N_PARCELS,
                    "ref_img": PARCEL_IMG,
                    "seed": SEED,
                    "rest_dir": REST_DIR,
                    "task_dir": TASK_DIR,
                    "working_dir": WORKING_DIR,
                    "subjects": TRAIN_LIST,
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            },
            f,
        )
    print("Done!")
