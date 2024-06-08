import os
import os.path as op
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.append("../../../..")

TARGET_DIR = "experiments/validation/3_prediction/3.1_brain_age_prediction/targets/"
os.makedirs(TARGET_DIR)

if __name__ == "__main__":
    # UK Biobank
    beh = pd.read_csv(
        "experiments/transfer_learning/ukb/data/ukb_test_ids.txt",
        header=None,
        names=["subject"],
    )

    # Age
    beh = beh.merge(
        pd.read_csv(
            "validation/3_prediction/__ukb_phenotypes/age_ICV.csv",  # Edit here based on your own path to age file.
        )[["eid", "age"]].rename(columns={"eid": "subject"}),
        on="subject",
    )

    # Sex
    beh = beh.merge(
        pd.read_csv(
            "/validation/3_prediction/__ukb_phenotypes/01_basic_demographics.csv"
        )[["eid", "31-0.0"]].rename(columns={"31-0.0": "sex", "eid": "subject"}),
        on="subject",
    )
    # Replace sex with M and F
    beh["sex"] = beh["sex"].replace({0: "F", 1: "M"})

    # Fluid Intelligence
    chunksize = 10000
    beh = beh.merge(
        pd.concat(
            pd.read_csv(
                "/validation/3_prediction/__ukb_phenotypes/32_cognitive_phenotypes.csv",
                usecols=["eid", "20016-2.0"],
                chunksize=chunksize,
            )
        ).rename(columns={"eid": "subject", "20016-2.0": "fluid"}),
        on="subject",
    )

    # Overall Health
    beh = beh.merge(
        pd.read_csv("/validation/3_prediction/__ukb_phenotypes/50_health_outcomes.csv")[
            ["eid", "2178-2.0"]
        ].rename(columns={"2178-2.0": "overall_health", "eid": "subject"}),
        on="subject",
    )

    # Grip Stregth
    beh = beh.merge(
        pd.concat(
            pd.read_csv(
                "/validation/3_prediction/__ukb_phenotypes/20_physical_general.csv",
                usecols=["eid", "46-2.0", "47-2.0", "1707-2.0"],
                chunksize=chunksize,
            )
        ).rename(
            columns={
                "eid": "subject",
                "46-2.0": "grip_left",
                "47-2.0": "grip_right",
                "1707-2.0": "handedness",
            }
        ),
        on="subject",
    )
    conditions = [
        beh["handedness"] == 1,
        beh["handedness"] == 2,
    ]
    choices = [
        beh["grip_right"],
        beh["grip_left"],
    ]
    beh["strength"] = np.select(conditions, choices, default=np.nan)
    beh.to_csv("validation/3_prediction/__ukb_phenotypes/ukb_all_targets.csv")

    # Prepare target for brain-age prediction
    age = beh[beh["age"].notna()]
    age.to_csv(op.join(TARGET_DIR, "ukb_age.csv"), index=False)

    # Holdout set
    kFold = KFold(n_splits=10, shuffle=True, random_state=42)
    holdout_list = []
    for train_idx, test_idx in kFold.split(beh):
        holdout_list.append({"train_idx": train_idx, "test_idx": test_idx})
    with open(op.join(TARGET_DIR, "ukb_age_holdout.pkl"), "wb") as f:
        pickle.dump(holdout_list, f)
