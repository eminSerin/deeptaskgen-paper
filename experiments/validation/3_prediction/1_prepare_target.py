import os.path as op

import numpy as np
import pandas as pd

ABS_PATH = op.abspath(op.join(__file__, "../../.."))

if __name__ == "__main__":
    # UK Biobank
    print("Reading IDs...")
    beh = pd.read_csv(
        op.join(ABS_PATH, "transfer_learning/uk_biobank/data/ukb_predict_ids.txt"),
        header=None,
        names=["subject"],
    )

    # Age
    print("Adding age...")
    beh = beh.merge(
        pd.read_csv(
            op.join(
                ABS_PATH,
                "validation/3_prediction/__ukb_phenotypes/99_miscellaneous.csv",
            ),  # Edit here based on your own path to age file.
        )[["eid", "21003-2.0"]].rename(columns={"eid": "subject", "21003-2.0": "age"}),
        on="subject",
    )

    # Sex
    print("Adding sex...")
    beh = beh.merge(
        pd.read_csv(
            op.join(
                ABS_PATH,
                "validation/3_prediction/__ukb_phenotypes/01_basic_demographics.csv",
            )
        )[["eid", "31-0.0"]].rename(columns={"31-0.0": "sex", "eid": "subject"}),
        on="subject",
    )
    # Replace sex with M and F
    beh["sex"] = beh["sex"].replace({0: "F", 1: "M"})

    # Fluid Intelligence
    print("Adding fluid intelligence...")
    chunksize = 10000
    beh = beh.merge(
        pd.concat(
            pd.read_csv(
                op.join(
                    ABS_PATH,
                    "validation/3_prediction/__ukb_phenotypes/32_cognitive_phenotypes.csv",
                ),
                usecols=["eid", "20016-2.0"],
                chunksize=chunksize,
            )
        ).rename(columns={"eid": "subject", "20016-2.0": "fluid"}),
        on="subject",
    )

    # Overall Health
    print("Adding overall health...")
    chunksize = 10000
    beh = beh.merge(
        pd.concat(
            pd.read_csv(
                op.join(
                    ABS_PATH,
                    "validation/3_prediction/__ukb_phenotypes/50_health_outcomes.csv",
                ),
                usecols=[
                    "eid",
                    "2178-2.0",
                    "20002-0.1261",
                    "2443-2.0",
                    "20002-0.1286",
                    "20002-0.1263",
                    "20002-0.1065",
                ],
                chunksize=chunksize,
            )
        ).rename(
            columns={
                "eid": "subject",
                "2178-2.0": "overall_health",
                "20002-0.1261": "multiple_sclerosis",
                "2443-2.0": "diabetes",
                "20002-0.1286": "depression",
                "20002-0.1263": "dementia",
                "20002-0.1065": "hypertension",
            }
        ),
        on="subject",
    )

    # Grip Stregth
    print("Adding grip strength...")
    beh = beh.merge(
        pd.concat(
            pd.read_csv(
                op.join(
                    ABS_PATH,
                    "validation/3_prediction/__ukb_phenotypes/20_physical_measures_general.csv",
                ),
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
    beh = beh.drop(columns=["grip_left", "grip_right", "handedness"])

    # Alcohol Frequency
    print("Adding alcohol frequency...")
    beh = beh.merge(
        pd.read_csv(
            op.join(
                ABS_PATH,
                "validation/3_prediction/__ukb_phenotypes/13_lifestyle_and_environment_alcohol.csv",
            ),
            usecols=["eid", "1558-2.0", "1588-2.0"],
        ).rename(
            columns={
                "eid": "subject",
                "1558-2.0": "alcohol_freq",
                "1588-2.0": "beer_freq",
            }
        ),
        on="subject",
    )

    # Base database for other clinical variables
    self_report = beh[["subject"]].merge(
        pd.read_csv(
            op.join(
                ABS_PATH,
                "validation/3_prediction/__ukb_phenotypes/51_mental_health_self_report.csv",
            ),
        ).rename(columns={"eid": "subject"}),
        on="subject",
    )

    # Neuroticism
    print("Adding neuroticism...")
    beh = beh.merge(
        self_report[["subject", "20127-0.0"]].rename(
            columns={"20127-0.0": "neuroticism"}
        ),
        on="subject",
        how="left",
    )

    # PHQ-9
    print("Adding PHQ-9...")
    tmp_df = (
        self_report[
            [
                "subject",
                "20507-0.0",
                "20508-0.0",
                "20510-0.0",
                "20511-0.0",
                "20513-0.0",
                "20514-0.0",
                "20517-0.0",
                "20518-0.0",
                "20519-0.0",
            ]
        ]
        .replace(-818.0, np.nan)
        .dropna()
    )
    tmp_df["PHQ"] = tmp_df.iloc[:, 1:].sum(axis=1)
    beh = beh.merge(tmp_df[["subject", "PHQ"]], on="subject", how="left")

    # RDS-4
    print("Adding RDS-4...")
    tmp_df = (
        self_report[
            [
                "subject",
                "2050-2.0",
                "2060-2.0",
                "2070-2.0",
                "2080-2.0",
            ]
        ]
        .replace(-818.0, np.nan)
        .dropna()
    )
    tmp_df["RDS"] = tmp_df.iloc[:, 1:].sum(axis=1)
    beh = beh.merge(tmp_df[["subject", "RDS"]], on="subject", how="left")

    # GAD-7
    print("Adding GAD-7...")
    tmp_df = (
        self_report[
            [
                "subject",
                "20505-0.0",
                "20506-0.0",
                "20509-0.0",
                "20512-0.0",
                "20515-0.0",
                "20516-0.0",
                "20520-0.0",
            ]
        ]
        .replace(-818.0, np.nan)
        .dropna()
    )
    tmp_df["GAD"] = tmp_df.iloc[:, 1:].sum(axis=1)
    beh = beh.merge(tmp_df[["subject", "GAD"]], on="subject", how="left")

    # Saving
    print("Saving final prediction database...")
    beh.to_csv(
        op.join(
            ABS_PATH, "validation/3_prediction/__ukb_phenotypes/ukb_all_targets.csv"
        ),
        index=False,
    )
