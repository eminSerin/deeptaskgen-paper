import pytorch_lightning as pl
from torch import nn


class Tavor(pl.LightningModule):
    """Very simple PyTorch implementation of "Task-free MRI predicts individual differences in brain activity during task performance" (Tavor et al., 2016).

    Arguments
    ----------
    in_features : int
        Number of input features.
    out_features : int, optional
        Number of output features, by default None.
        If None, out_features = in_features.

    References
    ----------
    Tavor, Ido, et al. "Task-free MRI predicts individual differences in brain activity during task performance." Science 352.6282 (2016): 216-220.
    """

    def __init__(
        self,
        in_features,
        out_features=None,
    ) -> None:
        super().__init__()

        self.in_fetures = in_features
        if out_features is None:
            out_features = in_features
        self.out_features = out_features
        self.fn = nn.Sequential(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.fn(x)
