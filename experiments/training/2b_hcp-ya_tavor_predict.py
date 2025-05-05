import os
import os.path as op
import pickle
import sys

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from experiments.utils import tavor

sys.modules["tavor"] = tavor

sys.path.append(op.abspath(op.join(__file__, "../../..")))
ABS_PATH = sys.path[-1]
REST_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/rest"))
TASK_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/data/task"))
WORKING_DIR = op.realpath(op.join(ABS_PATH, "experiments/training/results/tavor"))
PRED_DIR = op.join(WORKING_DIR, "pred")
os.makedirs(PRED_DIR, exist_ok=True)
PARCEL_IMG = nib.load(  # 50 Labels from Group-ICA
    op.join(
        ABS_PATH,
        "utils/templates/melodic_IC_MNI_dlabel_crop.nii",
    )
)
PARCEL_MASK = PARCEL_IMG.get_fdata()
PARCEL_IDX = np.setdiff1d(np.unique(PARCEL_MASK), [0])  # Remove 0 label (background)
N_PARCELS = len(PARCEL_IDX)  # 50 parcels
TEST_LIST = op.realpath(op.join(ABS_PATH, "experiments/training/data/hcp_test_ids.txt"))
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


def predict_parcel(coef_, X, standardize=True):
    if standardize:
        X = _standardize(X)
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add constant regressor
    return np.dot(X, coef_)


def predict_subject(coefs, subject, n_samples=8, overwrite=False):
    out_file = op.join(PRED_DIR, f"{subject}_pred.npy")
    if op.exists(out_file) and not overwrite:
        print(f"Skipping {subject} because it already exists")
        return
    y_pred = np.zeros((N_SAMPLES, N_CONTRASTS) + PARCEL_MASK.shape)
    for s in range(N_SAMPLES):
        rest = np.load(op.join(REST_DIR, f"{subject}_sample{s}_rsfc.npy"))
        for c in range(N_CONTRASTS):
            for p, p_idx in enumerate(PARCEL_IDX):
                y_pred[s, c, PARCEL_MASK == p_idx] = predict_parcel(
                    coefs[c, p, :], rest[..., PARCEL_MASK == p_idx].T
                )
    np.save(
        op.join(PRED_DIR, f"{subject}_pred.npy"),
        y_pred,
    )


if __name__ == "__main__":
    print("Predicting test set from HCP-YA dataset...")
    with open(op.join(WORKING_DIR, "tavor_hcp-ya_model.pkl"), "rb") as f:
        model = pickle.load(f)
    coefs = model["coefs"].mean(axis=0)  # Average over training subjects
    Parallel(n_jobs=N_JOBS)(
        delayed(predict_subject)(coefs, subject, n_samples=N_SAMPLES, overwrite=False)
        for subject in tqdm(TEST_LIST)
    )
