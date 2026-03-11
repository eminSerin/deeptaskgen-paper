from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import MaskTensor
from torchmetrics import functional as FM

##TODO: Add docstrings!

"""Functional"""


def r2_score(input, target):
    """Wrapper for torchmetrics.functional.r2_score"""
    return FM.r2_score(input.flatten(), target.flatten())


def r2_loss(input, target):
    """Loss function for R2 score."""
    return 1 - r2_score(input, target)


def corrcoef(input, target):
    """Wrapper for torchmetrics.functional.pearson_corrcoef"""
    return FM.pearson_corrcoef(input.flatten(), target.flatten())


def corrcoef_loss(input, target):
    """Loss function for Pearson correlation coefficient."""
    return 1 - corrcoef(input, target)


def corrcoef_subj(input, target, eps=1e-8):
    """Compute the correlation coefficient between sets of subjects.
    2D Input matrix must be of shape (n_subjects, n_voxels).
    """
    if input.ndim != 2 or target.ndim != 2:
        input = input.reshape(input.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

    # Center the rows
    A_mA = input - input.mean(dim=1, keepdim=True)
    B_mB = target - target.mean(dim=1, keepdim=True)

    # Dot products for numerator
    numerator = torch.einsum("ij,kj->ik", A_mA, B_mB)

    # Rowwise sum of squares for denominator
    ssA = torch.einsum("ij,ij->i", A_mA, A_mA)
    ssB = torch.einsum("ij,ij->i", B_mB, B_mB)
    denominator = torch.sqrt(torch.clamp(ssA[:, None] * ssB[None, :], min=eps))

    return numerator / denominator


def diag_index(input, target, eps=1e-8, normalize=False):
    """
    Compute the diagonal index of a correlation matrix.
    The diagonal index is the difference between the mean of the diagonal elements and the mean of the off-diagonal elements.
    """
    corr_mat = corrcoef_subj(input, target, eps)
    n_subj, _ = corr_mat.shape
    diag_elements = torch.diagonal(corr_mat).mean()
    mask = ~torch.eye(
        n_subj, dtype=torch.bool, device=corr_mat.device
    )  # (n_subj, n_subj)
    off_diag_elements = corr_mat[mask].mean()
    diag_index = diag_elements - off_diag_elements
    if normalize:
        return diag_index * diag_elements
    return diag_index


def relative_diag_index(input, target, eps=1e-8):
    """Computes the relative diagonal index as a percentage of off-diagonal mean."""
    corr_mat = corrcoef_subj(input, target, eps)
    n_subj, _ = corr_mat.shape
    diag_elements = torch.diagonal(corr_mat).mean()
    mask = ~torch.eye(
        n_subj, dtype=torch.bool, device=corr_mat.device
    )  # (n_subj, n_subj)
    off_diag_elements = corr_mat[mask].mean()
    diag_index = diag_elements - off_diag_elements
    return diag_index * 100 / off_diag_elements


def contrastive_loss(input, target):
    """Adaptation of the contrastive loss from Ngo et al., 2022.
    It can be used with more than 2 subjects."""
    n_samples = input.shape[0]
    if n_samples < 2:
        raise ValueError(
            "Not enough samples to compute contrastive loss, there must be at least 2 samples"
        )
    contrastive_loss = 0
    count = 0
    for i, j in combinations(range(n_samples), 2):
        contrastive_loss += FM.mean_squared_error(input[i], target[j])
        count += 1
    contrastive_loss /= count
    return contrastive_loss


def dice(input, target):
    raise NotImplementedError


def dice_auc(input, target):
    raise NotImplementedError


def rc_loss(input, target, within_margin=0, between_margin=0):
    """Construction Reconstruction Loss (RC Loss) as described in [1].

    Parameters
    ----------
    input : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Target values.
    within_margin : int, optional
        Same subject (reconstructive) margin, by default 0.
    between_margin : int, optional
        Between subject (contrastive) margin, by default 0.

    Returns
    -------
    torch.float
        RC Loss.

    References:
    -----------
    [1] Ngo, Gia H., et al. "Predicting individual task contrasts from resting‐state functional connectivity using a surface‐based convolutional network." NeuroImage 248 (2022): 118849.
    """
    if input.shape[0] < 2:
        raise ValueError(
            "Input and target must have at least 2 samples! Otherwise it cannot compute contrastive loss."
        )
    recon_loss = FM.mean_squared_error(input, target)
    contrast_loss = contrastive_loss(input, target)
    return torch.clamp(recon_loss - within_margin, min=0.0) + torch.clamp(
        recon_loss - contrast_loss + between_margin, min=0.0
    )


"""OOP"""


class BaseLoss(nn.Module):
    def __init__(self, mask=None, loss_fn=None):
        super().__init__()
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)
        self.loss_fn = loss_fn

    def forward(self, input, target):
        if self.mask is not None:
            return self.loss_fn(
                self.mask.apply_mask(input), self.mask.apply_mask(target)
            )
        return self.loss_fn(input, target)


class MSELoss(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=nn.functional.mse_loss)


class MAELoss(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=nn.functional.l1_loss)


class MSLELoss(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=FM.mean_squared_log_error)


class PearsonCorr(nn.Module):
    def __init__(self, loss=False, mask=None):
        super().__init__()
        self.loss = loss
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)

    def forward(self, input, target):
        if self.mask is not None:
            input = self.mask.apply_mask(input)
            target = self.mask.apply_mask(target)
        if self.loss:
            return 1 - corrcoef(input, target)
        return corrcoef(input, target)


class ContrastiveLoss(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=contrastive_loss)


class DiagIndex(BaseLoss):
    def __init__(self, mask=None, normalize=False):
        super().__init__(mask=mask, loss_fn=diag_index)
        self.normalize = normalize

    def forward(self, input, target):
        return self.loss_fn(input, target, normalize=self.normalize)


class RelativeDiagIndex(BaseLoss):
    def __init__(self, mask=None):
        super().__init__(mask=mask, loss_fn=relative_diag_index)


class R2(nn.Module):
    def __init__(self, loss=False, mask=None):
        super().__init__()
        self.loss = loss
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)

    def forward(self, input, target):
        if self.mask is not None:
            input = self.mask.apply_mask(input)
            target = self.mask.apply_mask(target)
        if self.loss:
            return r2_loss(input, target)
        return r2_score(input, target)


class RCLossAnneal(nn.Module):
    """Reconstruction and Contrastive Loss with Annealing.

    Parameters
    ----------
    epoch : int, optional
        The current epoch, by default 0
    init_within_margin : int, optional
        Initial same subject (reconstructive) margin, by default 4
    init_between_margin : int, optional
        Initial between subject (contrastive) margin, by default 5
    min_within_margin : int, optional
        Minimum same subject (reconstructive) margin, by default 1
    max_between_margin : int, optional
        Maximum between subject (contrastive) margin, by default 10
    margin_anneal_step : int, optional
        The number of epochs should be done before margin annealing happens, by default 10
    mask : torch.Tensor, optional
        Mask tensor, by default None.

    Returns
    ----------
    torch.float:
        RC Loss between target and input.
    """

    def __init__(
        self,
        epoch=0,
        init_within_margin=4.0,
        init_between_margin=5,
        min_within_margin=1.0,
        max_between_margin=10,
        margin_anneal_step=10,
        mask=None,
    ):
        super().__init__()
        self.init_within_margin = init_within_margin
        self.init_between_margin = init_between_margin
        self.min_within_margin = min_within_margin
        self.max_between_margin = max_between_margin
        self.margin_anneal_step = margin_anneal_step
        self.within_margin = init_within_margin
        self.between_margin = init_between_margin
        self.mask = mask
        if mask is not None:
            self.mask = MaskTensor(mask)
        self.update_margins(epoch)

    def update_margins(self, epoch):
        if (epoch % self.margin_anneal_step == 0) & (epoch > 0):
            steps = epoch // self.margin_anneal_step

            self.within_margin = torch.max(
                torch.tensor(self.within_margin * 0.5**steps),
                torch.tensor(self.min_within_margin),
            )
            self.between_margin = torch.min(
                torch.tensor(self.between_margin * 2.0**steps),
                torch.tensor(self.max_between_margin),
            )

    def _rc_loss(self, input, target, within_margin, between_margin):
        """Reconstruction and Contrastive Loss.

        See deeptaskgen.loss.rc_loss for more details.
        """
        self.recon_loss = FM.mean_squared_error(input, target)
        self.contrast_loss = contrastive_loss(input, target)
        return torch.clamp(self.recon_loss - within_margin, min=0.0) + torch.clamp(
            self.recon_loss - self.contrast_loss + between_margin, min=0.0
        )

    def forward(self, input, target):
        if self.mask is not None:
            return self._rc_loss(
                self.mask.apply_mask(input),
                self.mask.apply_mask(target),
                within_margin=self.within_margin,
                between_margin=self.between_margin,
            )
        return self._rc_loss(
            input,
            target,
            within_margin=self.within_margin,
            between_margin=self.between_margin,
        )


class CRRLoss(nn.Module):
    """Contrast-Regularized Reconstruction Loss.

    Parameters
    ----------
    margin : float, optional
        The margin value for the reconstruction loss, by default 1.0
    alpha : float, optional
        The weight for the contrastive loss, by default 0.10
    mask : torch.Tensor, optional
        Mask tensor, by default None.

    Returns
    ----------
    torch.float:
        CRRLoss between target and input.
    """

    def __init__(self, margin=1.0, alpha=0.10, mask=None):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.recon_loss = None
        self.contrast_loss = None
        self.triplet_loss = None
        self.mask = None
        if mask is not None:
            self.mask = MaskTensor(mask)

    def _triplet_loss(self, input, target):
        """Triplet Loss."""
        recon_loss = F.mse_loss(input, target)
        contrast_loss = contrastive_loss(input, target)
        return (
            recon_loss,
            contrast_loss,
            torch.clamp(recon_loss - contrast_loss + self.margin, min=0),
        )

    def forward(self, input, target):
        if self.mask is not None:
            input = self.mask.apply_mask(input)
            target = self.mask.apply_mask(target)
        self.recon_loss, self.contrast_loss, self.triplet_loss = self._triplet_loss(
            input, target
        )
        return self.recon_loss + self.alpha * self.triplet_loss
