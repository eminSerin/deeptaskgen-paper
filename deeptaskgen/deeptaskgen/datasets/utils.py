import os.path as op

import nibabel as nib
import numpy as np
import torch
from nilearn.image import crop_img
from nilearn.masking import unmask


def _load_data(data):
    """Loads data from file, or returns data if it is already a torch.Tensor.

    Parameters
    ----------
    data : torch.Tensor, np.ndarray, str
        Data to load.

    Returns
    -------
    torch.Tensor
        Loaded data.

    Raises
    ------
    FileExistsError
        If data is a string, and the file does not exist.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, str):
        if not op.exists(data):
            raise FileExistsError(f"{data} does not exist!")
        if data.endswith((".nii.gz", ".nii")):
            return torch.from_numpy(nib.load(data).get_fdata())
        elif data.endswith(".npy"):
            return torch.from_numpy(np.load(data))


def _unmask_timeseries(timeseries, mask, crop=False):
    """Unmask timeseries, and reconstructs volumetric images, and optionally crop image.

    Parameters
    ----------
    timeseries : np.ndarray
        Extracted ROI timeseries (ts x voxels).
    mask : nib.Nifti1Image
        Mask image in NIFTI format.
    crop : bool, optional
        Crops the image to get rid of unnecessary blank spaces around the borders of brain, by default False.

    Returns
    -------
    4D np.ndarray
        Reconstructed volumetric images (ts x x_dim x y_dim x z_dim)
    """
    img = unmask(timeseries, mask)
    if crop:
        img = crop_img(img)
    return img.get_fdata().transpose(3, 0, 1, 2)


## TODO: Add docstring!
class MaskTensor:
    def __init__(self, mask) -> None:
        self._input_shape = None
        mask = _load_data(mask)
        if not mask.dim() == 3:
            raise ValueError("Mask must be a 3D tensor")
        self._mask_idx = torch.nonzero(mask, as_tuple=True)

    def apply_mask(self, input):
        """Applies mask to input."""
        if not isinstance(input, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        if not input.dim() >= 3:
            raise ValueError("Input must be at least 3D tensor")
        self._input_shape = input.shape
        return input[..., self._mask_idx[0], self._mask_idx[1], self._mask_idx[2]]

    def unmask(self, input):
        """Unmasks input."""
        if not isinstance(input, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        unmasked = torch.zeros(self._input_shape, dtype=input.dtype)
        unmasked[..., self._mask_idx[0], self._mask_idx[1], self._mask_idx[2]] = input
        return unmasked
