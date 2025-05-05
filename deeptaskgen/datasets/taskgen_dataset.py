import os.path as op

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import _unmask_timeseries


##TODO: Add docstring!
def load_timeseries(
    rest_file,
    task_file=None,
    device="cpu",
    unmask=False,
    mask=None,
    crop=False,
    dtype=torch.float32,
):
    # Rest
    if not op.exists(rest_file):
        raise FileExistsError(f"{rest_file} does not exist!")
    # Task
    if task_file is not None:
        if not op.exists(task_file):
            raise FileExistsError(f"{task_file} does not exist!")

    if unmask:
        rest_img = (
            torch.from_numpy(_unmask_timeseries(np.load(rest_file).T, mask, crop))
            .type(dtype)
            .to(device)
        )
        if task_file is not None:
            task_img = (
                torch.from_numpy(_unmask_timeseries(np.load(task_file), mask, crop))
                .type(dtype)
                .to(device)
            )
            return rest_img, task_img
        return rest_img
    else:
        rest_img = torch.from_numpy(np.load(rest_file)).type(dtype).to(device)
        if task_file is not None:
            task_img = torch.from_numpy(np.load(task_file)).type(dtype).to(device)
            return rest_img, task_img
        return rest_img


class TaskGenDataset(Dataset):
    """Dataset class for task generation experiment.

    This class can be used for CNN models.

    Parameters
    ----------
    subj_ids : np.ndarray
        List of subject IDs.
    rest_dir : str
        Path to directory containing resting state timeseries.
    task_dir : str
        Path to directory containing task timeseries.
    num_samples : int, optional
        Number of samples per subject, by default 8
    device : str, optional
        Device to load data to, by default "cpu"
    unmask : bool, optional
        Whether or not to unmask timeseries, by default False.
    mask : nib.Nifti1Image, optional
        Mask array, by default None
    crop : bool, optional
        Crops the image to get rid of unnecessary blank spaces around the borders of brain, by default False.
    precision : str, optional
        Precision to use, by default "32"

    Returns
    ----------
    torch.utils.data.Dataset:
        Dataset object for task generation experiment.
    """

    def __init__(
        self,
        subj_ids,
        rest_dir,
        task_dir=None,
        num_samples=8,
        device="cpu",
        unmask=False,
        mask=None,
        crop=False,
        precision="32",
    ) -> None:
        super().__init__()
        self.subj_ids = subj_ids
        self.rest_dir = rest_dir
        self.task_dir = task_dir
        self.num_samples = num_samples
        self.device = device
        self.unmask = unmask
        self.mask = mask
        self.crop = crop
        self.precision = precision

        if precision == "16":
            self._dtype = torch.float16
        elif precision == "bf16":
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float32

        if unmask and (mask is None):
            raise ValueError("Mask must be provided to unmask timeseries!")

    def __getitem__(self, idx):
        subj = self.subj_ids[idx]
        sample_id = np.random.randint(0, self.num_samples)

        # Rest
        rest_file = op.join(self.rest_dir, f"{subj}_sample{sample_id}_rsfc.npy")

        # Task
        if self.task_dir is not None:
            task_file = op.join(self.task_dir, f"{subj}_joint_MNI_task_contrasts.npy")

        return load_timeseries(
            rest_file,
            task_file,
            self.device,
            self.unmask,
            self.mask,
            self.crop,
            self._dtype,
        )

    def __len__(self):
        return len(self.subj_ids)
