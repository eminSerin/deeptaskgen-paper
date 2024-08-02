import gc
import os
import os.path as op

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.masking import apply_mask
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from umap import UMAP

ABS_PATH = op.abspath(op.join(__file__, "../../.."))

MASK = nib.load(op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii"))

UMAP_DIR = op.join(ABS_PATH, "validation/2_umaps/umaps")

CONTRASTS = (
    "EMOTION FACES-SHAPES",
    "GAMBLING REWARD",
    "LANGUAGE MATH-STORY",
    "MOTOR AVG",
    "RELATIONAL REL",
    "SOCIAL TOM-RANDOM",
    "WM 2BK-0BK",
)

DIRECTORY_MAP = {
    "hcp_actual": op.join(ABS_PATH, "training/data/contrast_z_maps"),
    "hcp_pred": op.join(
        ABS_PATH, "training/results/unetminimal_100_0.001/contrast_z_maps"
    ),
    "hcpd_pred": op.join(
        ABS_PATH,
        "transfer_learning/hcp_development/results/finetuned_50_0.001_emotion-faces-shapes/contrast_z_maps",
    ),
    "ukb_pred": op.join(
        ABS_PATH,
        "transfer_learning/ukb/results/finetuned_50_0.001/contrast_z_maps",
    ),
}

SUBJ_IDS = {
    "hcp_actual": op.join(ABS_PATH, "training/data/hcp_all_ids.txt"),
    "hcp_pred": op.join(ABS_PATH, "training/data/hcp_all_ids.txt"),
    "hcpd_pred": op.join(
        ABS_PATH, "transfer_learning/hcp_development/data/hcpd_all_ids.txt"
    ),
    "ukb_pred": op.join(ABS_PATH, "transfer_learning/ukb/data/ukb_test_ids.txt"),
}
N_JOBS = 4
RAND_SEED = 1


def vectorize_img(img_list, mask_img, scale=False):
    data = apply_mask(img_list, mask_img)
    if scale:
        data = MinMaxScaler().fit_transform(data)
    return data


def compute_umaps(dataset):
    data_dir = DIRECTORY_MAP[dataset]
    subj_list = np.genfromtxt(SUBJ_IDS[dataset], dtype=str)
    out_file = op.join(UMAP_DIR, f"{dataset}_umap.csv")
    print("Vectorizing contrast maps...")
    vectorized_maps = Parallel(n_jobs=N_JOBS)(
        delayed(vectorize_img)(
            [
                nib.load(
                    op.join(
                        data_dir,
                        cont.replace(" ", "_").lower(),
                        f"{subj}.nii.gz",
                    )
                )
                for cont in CONTRASTS
            ],
            MASK,
        )
        for subj in tqdm(subj_list)
    )
    vectorized_maps = np.concatenate(vectorized_maps, axis=0)

    print("Fitting UMAP...")
    vectorized_maps = MinMaxScaler().fit_transform(vectorized_maps)
    umap = UMAP(
        n_components=2,
        n_neighbors=15,
        metric="euclidean",
        transform_seed=RAND_SEED,
        random_state=RAND_SEED,
        verbose=True,
    )
    low_dim_data = umap.fit_transform(vectorized_maps)

    # Save UMAPs to a .csv file.
    tasks = [cont.split(" ")[0].capitalize() for cont in CONTRASTS]
    umap_df = pd.DataFrame(low_dim_data, columns=["UMAP1", "UMAP2"])
    umap_df["Tasks"] = tasks * len(subj_list)
    umap_df["Contrasts"] = list(CONTRASTS) * len(subj_list)
    umap_df.to_csv(out_file, index=False)
    del umap
    gc.collect()


if __name__ == "__main__":
    os.makedirs(UMAP_DIR, exist_ok=True)
    for dataset in tqdm(DIRECTORY_MAP.keys()):
        compute_umaps(dataset)
