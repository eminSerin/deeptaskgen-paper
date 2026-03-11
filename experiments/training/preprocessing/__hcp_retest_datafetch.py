# Import libraries
import os
import os.path as op

import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm

# Constants
TASKS = ["LANGUAGE", "RELATIONAL", "SOCIAL", "EMOTION", "WM", "MOTOR", "GAMBLING"]
COPEIDS = {
    "LANGUAGE": (1, 2, 3),
    "RELATIONAL": (1, 2, 3),
    "SOCIAL": (1, 2, 6),
    "EMOTION": (1, 2, 3),
    "WM": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22),
    "MOTOR": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
    "GAMBLING": (1, 2, 3),
}
ABS_PATH = op.abspath(op.join(__file__, "../../.."))
SUBJ_IDS = np.genfromtxt(op.join(ABS_PATH, "training/data/hcp_test_ids.txt"), dtype=str)
DATASET = "HCP_Retest"
TASK_EXT = "zstat1.dtseries.nii"

# Define main directories
RAW_DIR = op.join(ABS_PATH, "training/data/raw_retest")
os.makedirs(RAW_DIR, exist_ok=True)
ERR_DIR = op.join(RAW_DIR, "errors")
os.makedirs(ERR_DIR, exist_ok=True)


def download_file(
    hcp_s3, bucket, dataset, subj, sub_dir, file_name, out_dir, task, error_list
):
    out_file = op.join(out_dir, file_name)
    if not op.isfile(out_file):
        try:
            hcp_s3.download_file(
                Bucket=bucket,
                Key=op.join(dataset, sub_dir) + file_name,
                Filename=out_file,
            )
        except Exception as e:
            error_list.append(
                {"Subject": subj, "Task": task, "File": file_name, "Error": e}
            )


def fetch_data(subj):
    error_list = []  # List of files with download issues.
    hcp_s3 = boto3.client("s3")

    # Download Task data
    for task in tqdm(TASKS, desc="Downloading task data"):
        for ci in COPEIDS[task]:
            sub_dir = (
                f"{subj}/MNINonLinear/Results/tfMRI_{task}/"
                + f"tfMRI_{task}_hp200_s2_level2_MSMAll.feat/"
                + f"GrayordinatesStats/cope{ci}.feat/"
            )
            out_dir = op.join(RAW_DIR, sub_dir)
            os.makedirs(out_dir, exist_ok=True)
            download_file(
                hcp_s3,
                "hcp-openaccess",
                DATASET,
                subj,
                sub_dir,
                TASK_EXT,
                out_dir,
                task,
                error_list,
            )

    pd.DataFrame(error_list).to_json(op.join(ERR_DIR, f"{subj}.json"))
    print(f"Finished {subj}")


if __name__ == "__main__":
    for subj in tqdm(SUBJ_IDS, desc="Downloading data"):
        fetch_data(subj)
