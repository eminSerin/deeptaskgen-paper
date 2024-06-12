import os
import os.path as op
import pickle
import sys

import pandas as pd
from sklearn.model_selection import KFold

sys.path.append(op.abspath(op.join(__file__, "../../../..")))

VARIABLES = ["sex", "fluid", "overall_health", "strength"]
ABS_PATH = sys.path[-1]
TARGET_DIR = op.join(
    ABS_PATH, "experiments/validation/3_prediction/3.2_additional_variables/targets/"
)
os.makedirs(TARGET_DIR, exist_ok=True)

if __name__ == "__main__":
    # Load target df.
    # Already prepared with experiments/validation/3_prediction/3.1.brain_age_prediction/1_prepare_target.py
    phenotypes = pd.read_csv(
        op.join(
            ABS_PATH, "validation/3_prediction/__ukb_phenotypes/ukb_all_targets.csv"
        )
    )

    for var in VARIABLES:
        # Save target df
        df = phenotypes[["subject", var]].copy()
        df[var] = df[df[var].notna()]
        df[var] = df[var].dropna()
        df.to_csv(op.join(TARGET_DIR, f"ukb_{var}.csv"), index=False)

        # Holdout set
        kFold = KFold(n_splits=10, shuffle=True, random_state=42)
        holdout_list = []
        for train_idx, test_idx in kFold.split(df):
            holdout_list.append({"train_idx": train_idx, "test_idx": test_idx})
        with open(op.join(TARGET_DIR, f"ukb_{var}_holdout.pkl"), "wb") as f:
            pickle.dump(holdout_list, f)
