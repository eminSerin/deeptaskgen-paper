# Pipelining with Nested CV
import gc
import os
import os.path as op
import pickle
import sys
from time import time

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

sys.path.append("../../../..")
from utils.utils import DataLoader

MASK = nib.load("utils/templates/MNI_2mm_brain_mask_crop.nii")

CONFIG = {
    "kfold": 10,
    "inner_cv": 5,
    "n_jobs": 5,
    "seed": 42,
    "n_holdout": 10,
}
EST_MAP = {
    "regression": Ridge(solver="lsqr"),
    "classification": RidgeClassifier(solver="lsqr"),
}
TARGET_MAP = {
    "sex": "classification",
    "fluid": "regression",
    "overall_health": "regression",
    "strength": "regression",
}

TARGET_PATH = "validation/3_prediction/3.1_brain_age_prediction/targets/"
WORK_DIR = "validation/3_prediction/3.1_brain_age_prediction/results"
CONTRASTS = (
    "LANGUAGE MATH-STORY",
    "RELATIONAL REL",
    "SOCIAL TOM-RANDOM",
    "EMOTION FACES-SHAPES",
    "WM 2BK-0BK",
    "MOTOR AVG",
    "GAMBLING REWARD",
)
DATA_MAP = {
    "ukb_actual": "experiments/transfer_learning/uk_biobank/data/",
    "ukb_predict": "experiments/transfer_learning/uk_biobank/results/finetuned_50_0.001/",
}


def alpha_heuristic(X, estimator):
    # Set alpha parameter based on features and samples
    n_samples, n_features = X.shape
    alpha = n_features / n_samples
    estimator.set_params(alpha=alpha)
    return estimator


def run_cv_fold(train_idx, test_idx, X, y, estimator):
    # Set alpha parameter based on features and samples
    estimator = alpha_heuristic(X[train_idx], estimator)
    estimator.fit(X[train_idx], y[train_idx])
    y_pred = estimator.predict(X[test_idx]).astype(np.float32)
    gc.collect()
    return y[test_idx], y_pred


def custom_cv(X, y, estimator):
    print("Running Nested Cross Validation...")
    cv = KFold(n_splits=CONFIG["kfold"], shuffle=True, random_state=CONFIG["seed"])

    y_true, y_pred = zip(
        *Parallel(n_jobs=CONFIG["n_jobs"])(
            delayed(run_cv_fold)(train_idx, test_idx, X, y, estimator, CONFIG)
            for i, (train_idx, test_idx) in tqdm(enumerate(cv.split(X)))
        )
    )

    perf_dict = {}
    estimator = alpha_heuristic(X, estimator)
    estimator.fit(X, y)
    perf_dict["y_true"] = np.hstack(y_true)
    perf_dict["y_pred"] = np.hstack(y_pred)
    perf_dict["model"] = estimator
    perf_dict["y_true_all"] = y
    perf_dict["y_pred_all"] = estimator.predict(X).astype(np.float32)
    return perf_dict


def main(dataset, contrast, target_path, index_path, target_name):
    # Prepare data.
    target_db = pd.read_csv(target_path)
    with open(index_path, "rb") as f:
        holdout_fold = pickle.load(f)
    work_dir = op.join(WORK_DIR, dataset)
    os.makedirs(work_dir, exist_ok=True)
    y = target_db["target_name"].values.astype(np.float32)
    X = DataLoader(
        data_path=op.join(DATA_MAP[dataset], "contrast_z_maps", contrast),
        mask=MASK,
        rest=False,
    ).load_data(target_db["subject"].values)

    # Run Repeated Cross-Validation Scheme
    estimator = EST_MAP[TARGET_MAP[target_name]]
    print("Running Holdout Nested Cross Validation...")
    pred_holdout_list = []
    for i, fold in enumerate(holdout_fold):
        train_idx, holdout_idx = fold["train_idx"], fold["test_idx"]
        t0 = time()
        y_train, y_holdout = y[train_idx], y[holdout_idx]
        perf_dict = custom_cv(X[train_idx], y_train, estimator, CONFIG)
        perf_dict["y_true_holdout"] = y_holdout
        perf_dict["y_pred_holdout"] = (
            perf_dict["model"].predict(X[holdout_idx]).astype(np.float32)
        )
        perf_dict["y_train_idx"] = train_idx
        perf_dict["y_holdout_idx"] = holdout_idx
        pred_holdout_list.append(perf_dict)
        print(
            f"Holdout Fold {i+1}/{CONFIG['n_holdout']} done in {((time()-t0)/60):.2f} minutes."
        )
        gc.collect()

    # Save results to a pickle file
    out_file = op.join(work_dir, f"{dataset}_{target_name}_{contrast}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(pred_holdout_list, f)


if __name__ == "__main__":
    for target_name in TARGET_PATH:
        target_path = op.join(TARGET_PATH, f"ukb_{target_name}.csv")
        index_path = op.join(TARGET_PATH, f"ukb_{target_name}_holdout.pkl")
        for dataset in DATA_MAP:
            if dataset == "ukb_actual":
                for contrast in ["EMOTION FACES-SHAPES"]:
                    print(
                        f"Target: {target_name}, Dataset: {dataset}, Contrast: {contrast}"
                    )
                    main(
                        dataset,
                        contrast.replace(" ", "_").lower(),
                        target_path,
                        index_path,
                        target_name,
                    )
            else:
                for contrast in CONTRASTS:
                    print(
                        f"Target: {target_name}, Dataset: {dataset}, Contrast: {contrast}"
                    )
                    main(
                        dataset,
                        contrast.replace(" ", "_").lower(),
                        target_path,
                        index_path,
                        target_name,
                    )
