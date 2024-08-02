import os
import os.path as op
import shutil
import subprocess
import sys
from glob import glob

import h5py
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from nilearn.image import crop_img, resample_to_img
from tqdm import tqdm

sys.path.append(op.abspath(op.join(__file__, "../../..")))
from utils.utils import get_contrasts

# Define paths.
ABS_PATH = sys.path[-1]
MNI_CROP_MASK = op.join(ABS_PATH, "utils/templates/MNI_2mm_brain_mask_crop.nii")
RAW_DIR = op.join(ABS_PATH, "training/data/raw")
OUT_DIR = op.realpath(op.join(ABS_PATH, "training/data/task"))

SUBJ_IDS = np.genfromtxt(op.join(ABS_PATH, "training/data/hcp_all_ids.txt"), dtype=str)
N_JOBS = 1

# Define path to FS spaces.
fsaverage = {
    "L": op.realpath(
        op.join(
            ABS_PATH,
            "utils/templates/resample_fsaverage/fsaverage_std_sphere.L.164k_fsavg_L.surf.gii",
        )
    ),
    "R": op.realpath(
        op.join(
            ABS_PATH,
            "utils/templates/resample_fsaverage/fsaverage_std_sphere.R.164k_fsavg_R.surf.gii",
        )
    ),
}
fsLR = {
    "L": op.realpath(
        op.join(
            ABS_PATH,
            "utils/templates/resample_fsaverage/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii",
        )
    ),
    "R": op.realpath(
        op.join(
            ABS_PATH,
            "utils/templates/resample_fsaverage/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii",
        )
    ),
}
fsaverage_MNI = nib.load(
    op.join(
        op.join(
            ABS_PATH, "utils/regfusion/templates/FSL_MNI152_FS4.5.0_cortex_estimate.nii"
        )
    )
)


def cifti_separate(input, output):
    cmd = f"wb_command -cifti-separate {input} COLUMN -volume-all {output + '/vol.nii.gz'} -metric CORTEX_LEFT {output + '/hemi_L_LR.func.gii'} -metric CORTEX_RIGHT {output + '/hemi_R_LR.func.gii'}"
    subprocess.run(cmd, check=True, shell=True)


def fslr2fsaverage(file, input_space, target_space):
    """Convert fsLR to fsaverage."""
    if not op.exists(file):
        raise FileNotFoundError(f"{file} does not exist!")
    cmd = f"wb_command -metric-resample {file} {input_space} {target_space} BARYCENTRIC {file.replace('LR', 'fsaverage')}"
    subprocess.run(cmd, check=True, shell=True)


def combine_projected_cortex_subcortex(
    proj_img, subcortex, ref_img, out_file=None, rm_file=False
):
    if out_file is None:
        out_file = op.join(op.dirname(op.dirname(proj_img)), "zstat1.nii.gz")

    if not op.exists(out_file):
        img_data = np.array(h5py.File(proj_img, "r")["proj_img"]).transpose(1, 2, 0)
        # sub_cortex = nib.load(op.join(op.dirname(proj_img), "vol.nii.gz"))
        subcortex = nib.load(subcortex)
        img = resample_to_img(
            nib.Nifti1Image(img_data, ref_img.affine),
            subcortex,
        )
        nib.Nifti1Image(
            img.get_fdata() + subcortex.get_fdata(), subcortex.affine
        ).to_filename(out_file)
        if rm_file:
            os.remove(proj_img)
            os.remove(subcortex)


def cifti2MNI(subj_dir, rm_file=True):
    """Convert cifti to fsaverage."""
    files = sorted(
        glob(
            op.join(
                subj_dir,
                "MNINonLinear",
                "Results",
                "tfMRI_*",
                "*",
                "GrayordinatesStats",
                "cope*",
                "zstat1.dtseries.nii",
            )
        )
    )
    for file in tqdm(files):
        if not op.exists(file.replace(".dtseries.nii", ".nii.gz")):
            tmp_dir = file + ".tmp"
            # Separate cifti into volume and surface files
            os.makedirs(tmp_dir, exist_ok=True)
            print(f"Separating {file}...")
            cifti_separate(file, tmp_dir)
            # Convert fsLR to fsaverage
            print(f"Converting {file} to fsaverage...")
            for hemi in ["L", "R"]:
                fslr2fsaverage(
                    op.join(tmp_dir, f"hemi_{hemi}_LR.func.gii"),
                    fsLR[hemi],
                    fsaverage[hemi],
                )
            # Project fsaverage to MNI
            print(f"Projecting {file} to MNI...")
            subprocess.run(
                f'cd {op.realpath(op.join(ABS_PATH, "utils/"))} && matlab -nodisplay -nosplash -nodesktop -r "fsaverage2vol(\\"{tmp_dir}\\"); quit"',
                check=True,
                shell=True,
            )
            # Combine projected cortex and subcortex
            print("Combining projected cortex and subcortex...")
            combine_projected_cortex_subcortex(
                op.join(tmp_dir, "proj_img.mat"),
                op.join(tmp_dir, "vol.nii.gz"),
                fsaverage_MNI,
            )
            # Remove tmp dir
            if rm_file:
                shutil.rmtree(tmp_dir)


def crop_img_w_ref(input, ref=None):
    """Crop image to reference image."""
    print("Cropping image...")
    cropped = crop_img(input)
    if ref is not None:
        cropped = resample_to_img(cropped, ref, interpolation="nearest")
    return cropped


def join_task_contrasts(subj_dir, out_dir=None):
    sub_task_data = []
    contrasts = get_contrasts()
    if out_dir is None:
        out_dir = OUT_DIR
    subj_id = op.basename(subj_dir)
    sub_task_data_file = op.join(out_dir, f"{subj_id}_joint_MNI_task_contrasts.npy")
    if not op.exists(sub_task_data_file):
        print(f"Joining task contrasts for {subj_id}...")
        try:
            for item in contrasts:
                task, ci, _ = item
                task_file = op.join(
                    subj_dir,
                    "MNINonLinear",
                    "Results",
                    f"tfMRI_{task}",
                    f"tfMRI_{task}_hp200_s2_level2_MSMAll.feat",
                    "GrayordinatesStats",
                    f"cope{ci}.feat",
                    "zstat1.nii.gz",
                )

                # Crop and resample
                sub_task_data.append(
                    crop_img_w_ref(task_file, MNI_CROP_MASK).get_fdata()
                )

            if sub_task_data:
                np.save(sub_task_data_file, np.transpose(sub_task_data, (1, 2, 3, 0)))
        except Exception as e:
            print(f"Error processing {subj_id}: {e}")


def main():
    def process_subject(subj_id):
        os.makedirs(OUT_DIR, exist_ok=True)
        out_file = op.join(OUT_DIR, f"{subj_id}_joint_MNI_task_contrasts.npy")
        if not op.exists(out_file):
            cifti2MNI(op.join(RAW_DIR, subj_id))
            join_task_contrasts(op.join(RAW_DIR, subj_id))
            print("COMPLETED!")
        else:
            print(f"{out_file} already exists!")

    _ = Parallel(n_jobs=N_JOBS)(
        delayed(process_subject)(subj_id) for subj_id in SUBJ_IDS
    )


if __name__ == "__main__":
    main()
