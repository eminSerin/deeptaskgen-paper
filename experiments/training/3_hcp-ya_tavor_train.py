import os
import os.path as op
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed

sys.path.append("../../../..")
from utils.tavor import DataLoader, parallel_fit_wrapper

REST_DIR = op.realpath("experiments/training/data/rest")
TASK_DIR = op.realpath("experiments/training/data/task")
WORKING_DIR = op.realpath(f"experiments/training/results/tavor")
TRAIN_LIST = op.realpath("experiments/training/data/hcp_train_ids.txt")
N_JOBS = 16
N_CONTRASTS = 47

if __name__ == "__main__":
    train_ids = np.genfromtxt(TRAIN_LIST, dtype=str)
    X_train_loader = DataLoader(
        REST_DIR,
        batch_size=1,
        ids=train_ids,
        sample=8,
    )
    cont_models = Parallel(n_jobs=N_JOBS)(
        delayed(parallel_fit_wrapper)(
            X_train_loader,
            DataLoader(
                TASK_DIR,
                batch_size=1,
                ids=train_ids,
                idx=i,
            ),
            n_jobs=1,
        )
        for i in range(N_CONTRASTS)
    )
    os.makedirs(WORKING_DIR, exist_ok=True)
    with open("tavor_hcp_model.pkl", "wb") as f:
        pickle.dump(cont_models, f)
