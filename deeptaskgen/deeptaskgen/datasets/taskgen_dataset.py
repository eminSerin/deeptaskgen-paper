import os.path as op

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import _unmask_timeseries


##TODO: Add docstring!
def load_timeseries(
    rest_file,
    task_file=None,
    device="cpu",
    unmask=False,
    mask=None,
    crop=False,
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
            .type(torch.float)
            .to(device)
        )
        if task_file is not None:
            task_img = (
                torch.from_numpy(_unmask_timeseries(np.load(task_file), mask, crop))
                .type(torch.float)
                .to(device)
            )
            return rest_img, task_img
        return rest_img
    else:
        rest_img = torch.from_numpy(np.load(rest_file)).type(torch.float).to(device)
        if task_file is not None:
            task_img = torch.from_numpy(np.load(task_file)).type(torch.float).to(device)
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
            rest_file, task_file, self.device, self.unmask, self.mask, self.crop
        )

    def __len__(self):
        return len(self.subj_ids)


class TaskGenDatasetLinear(TaskGenDataset):
    """Dataset class for task generation experiment using simple linear regression method presented in Tavor et al., 2016."""

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
    ) -> None:
        super().__init__(
            subj_ids, rest_dir, task_dir, num_samples, device, unmask, mask, crop
        )

    def __getitem__(self, idx):
        if self.task_dir is not None:
            rest_data, task_data = super().__getitem__(idx)
            return rest_data.flatten(1), task_data.flatten(1)
        else:
            rest_data = super().__getitem__(idx)
            return rest_data.flatten(1)


class AverageDataLoader(DataLoader):
    """DataLoader that averages over the samples in a batch.

    Parameters
    ----------
    PyTorch DataLoader arguments

    Returns
    ----------
    torch.utils.data.DataLoader:
        DataLoader object that averages over the samples in a batch.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

    def __iter__(self):
        for batch in super().__iter__():
            for i in range(len(batch)):
                batch[i] = torch.mean(batch[i], dim=0).unsqueeze(dim=0)
            yield batch


class ResidualizedDataloader(DataLoader):
    """DataLoader that residualizes the samples in a batch.

    Residualization is done by subtracting the mean of the
    batch from each sample in the batch.

    Parameters
    ----------
    PyTorch DataLoader arguments

    Returns
    ----------
    torch.utils.data.DataLoader:
        DataLoader object that residualizes the samples in a batch.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

    def __iter__(self):
        for batch in super().__iter__():
            for i in range(len(batch)):
                batch[i] = batch[i] - torch.mean(batch[i], dim=0).unsqueeze(dim=0)
            yield batch
