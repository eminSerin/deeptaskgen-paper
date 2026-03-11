import os.path as op

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from nilearn.masking import apply_mask
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class DataLoader:
    def __init__(self, data_path, mask, rest=False, n_jobs=1, dtype=np.float32):
        self.data_path = data_path
        self.mask = mask
        if isinstance(mask, str):
            self.mask = nib.load(mask)
        self.rest = rest
        self.n_jobs = n_jobs
        self.dtype = dtype

    def load_data(self, subj_ids):
        _load_data = self._load_rest_data if self.rest else self._load_data
        print("Contrast maps are being loaded...")
        data = Parallel(n_jobs=self.n_jobs)(
            delayed(_load_data)(subj_id) for subj_id in subj_ids
        )
        return np.array(data, dtype=self.dtype)

    def _load_data(self, subj_id):
        return apply_mask(
            op.join(self.data_path, f"{subj_id}.nii.gz"), self.mask
        ).reshape(-1)

    def _load_rest_data(self, subj_id):
        return apply_mask(
            nib.Nifti1Image(
                np.load(
                    op.join(self.data_path, f"{subj_id}_sample0_rsfc.npy")
                ).transpose(1, 2, 3, 0),
                affine=self.mask.affine,
            ),
            self.mask,
        ).reshape(-1)


# Helper functions
def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def compute_metrics(y_true, y_pred, estimator_type="classifier"):
    if estimator_type == "classifier":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
    elif estimator_type == "regressor":
        return {
            "corr": correlation_score(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
        }
