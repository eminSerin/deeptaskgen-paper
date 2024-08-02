# Pipelining with Nested CV
import gc
import os
import os.path as op
import pickle
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from validation.utils.utils import DataLoader, compute_metrics

ABS_PATH = sys.path[-1]
MASK = nib.load(op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii"))

CONFIG = {
    "kfold": 5,
    "seed": 42,
    "n_perm": 1000,
    "n_jobs": 5,
}

EST_MAP = {
    "regression": Ridge(solver="lsqr"),
    "classification": RidgeClassifier(solver="lsqr"),
}

TARGET_MAP = {
    "age": "regression",
    "sex": "classification",
    "fluid": "regression",
    "overall_health": "regression",
    "strength": "regression",
    "depression": "classification",
    "neuroticism": "regression",
    "PHQ": "regression",
    "alcohol_freq": "regression",
    "RDS": "regression",
    "GAD": "regression",
}

TARGET_PATH = op.join(
    ABS_PATH, "validation/3_prediction/__ukb_phenotypes/ukb_all_targets.csv"
)

WORK_DIR = op.join(ABS_PATH, "validation/3_prediction/results")

CONTRASTS_MAP = {
    "ukb_actual": (
        "REST",
        "EMOTION FACES-SHAPES",
    ),
    "ukb_pred": (
        "EMOTION FACES-SHAPES",
        "GAMBLING REWARD",
        "MOTOR AVG",
        "LANGUAGE MATH-STORY",
        "RELATIONAL REL",
        "SOCIAL TOM-RANDOM",
        "WM 2BK-0BK",
    ),
}

DATA_MAP = {
    "ukb_actual": op.join(
        ABS_PATH, "transfer_learning/uk_biobank/data/contrast_z_maps"
    ),
    "ukb_pred": op.join(
        ABS_PATH,
        "transfer_learning/uk_biobank/results/finetuned_50_0.001/contrast_z_maps",
    ),
}


def load_data(input_path, target_db, contrast):
    if contrast == "rest":
        input_path = op.join(op.dirname(input_path), "connectome")
        darray = []
        for subj in target_db["Subject"].values:
            darray.append(np.genfromtxt(op.join(input_path, f"{subj}.txt")))
        return np.array(darray)
    return DataLoader(data_path=op.join(input_path, contrast), mask=MASK).load_data(
        target_db["Subject"].values
    )


def alpha_heuristic(X, estimator):
    # Set alpha parameter based on features and samples
    n_samples, n_features = X.shape
    alpha = n_features / n_samples
    estimator.set_params(alpha=alpha)
    return estimator


def cv_iter(train_idx, test_idx, X, y, estimator):
    # Set the alpha parameter, fit the model, and predict
    estimator_ = alpha_heuristic(
        X[train_idx], estimator
    )  # Heuristic for alpha based on features and samples
    estimator_.fit(X[train_idx], y[train_idx])
    y_pred = estimator_.predict(X[test_idx]).astype(np.float32)
    # Clean up memory
    gc.collect()
    return y[test_idx], y_pred


def outer_cv(X, y, estimator, CONFIG):
    cv = KFold(n_splits=CONFIG["kfold"], shuffle=True, random_state=CONFIG["seed"])
    y_true, y_pred = zip(
        *Parallel(n_jobs=CONFIG["n_jobs"])(
            delayed(cv_iter)(train_idx, test_idx, X, y, estimator)
            for train_idx, test_idx in cv.split(X)
        )
    )
    estimator = alpha_heuristic(X, estimator)
    estimator.fit(X, y)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "model": estimator,
        "scores": compute_metrics(
            np.hstack(y_true),
            np.hstack(y_pred),
            estimator._estimator_type,
        ),
    }


def main(dataset, contrast, target_name):
    # Random seed and estimator
    est = EST_MAP[TARGET_MAP[target_name]]
    np.random.seed(CONFIG["seed"])

    # Prepare data.
    target_db = pd.read_csv(TARGET_PATH)
    work_dir = op.join(WORK_DIR, dataset)
    os.makedirs(work_dir, exist_ok=True)
    y = target_db[target_name].dropna()
    X = load_data(DATA_MAP[dataset], target_db, contrast)
    X = X[y.index]
    y = y.values

    # Output file
    out_file = os.path.join(
        work_dir,
        f"{dataset}_{target_name}_{contrast}.pkl",
    )
    perm_checkpoint = op.exists(out_file)

    # Permutation checkpoint
    if perm_checkpoint:
        with open(out_file, "rb") as f:
            perm_res_list = pickle.load(f)
        print(f"Loaded {len(perm_res_list)} permutation results.")
        CONFIG["n_perm"] = CONFIG["n_perm"] + 1 - len(perm_res_list)

    # Run analysis with cross-validation.
    if not perm_checkpoint:
        perm_res_list = []
        print("Running true analysis...")
        perm_res_list.append(outer_cv(X, y, est, CONFIG))
        if TARGET_MAP[target_name] == "regression":
            print(
                f"Correlation: {perm_res_list[0]['scores']['corr']:.3f}, "
                f"Rsquared: {perm_res_list[0]['scores']['r2']:.3f}"
            )
        else:
            print(
                f"F1: {perm_res_list[0]['scores']['f1']:.3f}, "
                f"Balanced Accuracy: {perm_res_list[0]['scores']['balanced_accuracy']:.3f}"
            )
        print("Running permutation analysis...")
    else:
        print("Running the remaining permutations...")
    for p in tqdm(range(CONFIG["n_perm"])):
        y_perm = np.random.permutation(y)
        perm_res_list.append(outer_cv(X, y_perm, est, CONFIG))

        # Save results to a pickle file every 20 permutations.
        if p % 20 == 0:
            print("Saving checkpoint...")
            with open(out_file, "wb") as f:
                pickle.dump(perm_res_list, f)
    # Save final results to a pickle file.
    with open(out_file, "wb") as f:
        pickle.dump(perm_res_list, f)
    print("Done!")


if __name__ == "__main__":
    for target_name in TARGET_MAP:
        for dataset in CONTRASTS_MAP:
            for contrast in CONTRASTS_MAP[dataset]:
                print(
                    f"Dataset: {dataset}, Target: {target_name}, Contrast: {contrast}"
                )
                main(dataset, contrast.replace(" ", "_").lower(), target_name)
