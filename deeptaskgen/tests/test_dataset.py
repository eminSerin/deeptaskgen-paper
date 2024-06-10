"""Test the integrity of dataset."""

import os.path as op
from glob import glob

import click
import numpy as np
from joblib import Parallel, delayed
from nilearn.masking import unmask
from tqdm import tqdm


def test_dataset(img, mask=None):
    try:
        img = np.load(img)
        if mask is not None:
            im_shape = img.shape
            if im_shape[0] > im_shape[1]:
                img = img.T
            unmask(img, mask)
    except Exception as e:
        print(f"Error in {img}: {e}")
        return img


@click.command()
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--mask", type=click.Path(exists=True), required=False, default=None)
@click.option("--n_jobs", type=int, required=False, default=1)
def main(data_dir, mask=None, n_jobs=1):
    files = glob(op.join(data_dir, "*.npy"))
    error_files = Parallel(n_jobs=n_jobs)(
        delayed(test_dataset)(f, mask) for f in tqdm(files, desc="Testing dataset...")
    )
    np.savetxt(
        op.join(op.dirname(data_dir), f"error_files_{op.basename(data_dir)}.txt"),
        error_files,
        fmt="%s",
    )


if __name__ == "__main__":
    main()
