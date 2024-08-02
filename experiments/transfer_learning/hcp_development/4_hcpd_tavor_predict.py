import os
import os.path as op
import pickle
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../../..")))
from experiments.utils import tavor

sys.modules["tavor"] = tavor

ABS_PATH = sys.path[-1]

REST_DIR = op.realpath(
    op.join(ABS_PATH, "experiments/transfer_learning/hcp_development/data/rest")
)
WORKING_DIR = op.realpath(
    op.join(ABS_PATH, "experiments/transfer_learning/hcp_development/results/tavor/")
)
PREDS_DIR = op.join(WORKING_DIR, "pred")
os.makedirs(PREDS_DIR, exist_ok=True)
TEST_LIST = op.realpath(
    op.join(
        ABS_PATH, "experiments/transfer_learning/hcp_development/data/hcpd_test_ids.txt"
    )
)
N_JOBS = 16
N_CONTRASTS = 47
N_SAMPLE = 1

if __name__ == "__main__":
    with open(
        op.join(
            ABS_PATH, "experiments/training/results/tavor/tavor_hcp_model.pkl"
        ),
        "rb",
    ) as f:
        models = pickle.load(f)
    subj_ids = np.genfromtxt(TEST_LIST, dtype=str)
    for subj in tqdm(subj_ids):
        pred_file = op.join(PREDS_DIR, f"{subj}_pred.npy")
        if not os.path.exists(pred_file):
            pred_list = []
            for sample in range(N_SAMPLE):
                rest = np.load(op.join(REST_DIR, f"{subj}_sample{sample}_rsfc.npy"))
                dshape = rest.shape[1:]
                rest = np.expand_dims(rest.reshape(rest.shape[0], -1), axis=0)
                tmp = []
                for est in models:
                    tmp.append(est.predict(rest).reshape(dshape))
                pred_list.append(np.stack(tmp, axis=0))
            pred_list = np.stack(pred_list, axis=0)
            np.save(pred_file, pred_list)
