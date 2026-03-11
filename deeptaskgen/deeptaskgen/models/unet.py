"""_summary_
"""

import torch.nn.functional as F
from torch import nn, optim

from .base_model import BaseModel
from .utils import _interpolate, _nConv, _skip_concat, call_layer


class _BaseUnet(BaseModel):
    """Implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation[1].

    ![U-Net Architecture](./_imgs/u-net.png)

    Attributes
    ----------
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    max_level : int, optional
        Maximum level of hidden downstream and upstreams blocks, excluding input, output and bottleneck blocks, by default 3.
    fdim : int, optional
        Initial number of features, by default 64.
        In each level, the number of features is doubled.
    up_mode : str, optional
        Upscaling mode, by default "nearest-exact".
        Upscaling is used if the input and output shapes are different.
    **args, **kwargs: dict
        Arguments for the BaseModel class.
        See deeptaskgen.models.base_model.BaseModel for more details.

    References
    ----------
    [1] - Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        max_level=3,
        dims=3,
        fdim=64,
        n_conv=3,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu_inplace",
        up_mode="trilinear",
        loss_fn=F.mse_loss,
        optimizer=optim.Adam,
        lr=0.001,
        batch_norm=True,
        lr_scheduler=True,
        **kwargs,
    ) -> None:
        super().__init__(
            in_chans,
            out_chans,
            max_level,
            fdim,
            n_conv,
            kernel_size,
            padding,
            stride,
            activation,
            up_mode,
            loss_fn,
            optimizer,
            lr,
            batch_norm=batch_norm,
            lr_scheduler=lr_scheduler,
            **kwargs,
        )
        self.dims = dims
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = call_layer("MaxPool", dims)(kernel_size=2, stride=2)
        self.upscale = nn.ModuleList()

        # Input block
        self.in_block = _nConv(
            self.in_chans,
            self.fdim,
            n_conv=self.n_conv,
            dims=dims,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.activation,
            batch_norm=batch_norm,
        )

        # Down
        in_dim = self._features[0]
        for feat in self._features[1:]:
            self.downs.append(
                _nConv(
                    in_dim,
                    feat,
                    n_conv=self.n_conv,
                    dims=dims,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    activation=self.activation,
                    batch_norm=self.batch_norm,
                )
            )
            in_dim = feat

        # Bottleneck
        self.bottleneck = _nConv(
            feat,
            feat * 2,
            n_conv=self.n_conv,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dims=dims,
            activation=self.activation,
            batch_norm=self.batch_norm,
        )

        # Upscale blocks
        for feat in reversed(self._features):
            self.upscale.append(
                call_layer("ConvTranspose", dims)(
                    feat * 2, feat, kernel_size=self.kernel_size, stride=2
                )
            )

        # Ups
        for feat in reversed(self._features[1:]):
            self.ups.append(
                _nConv(
                    feat * 2,
                    feat,
                    n_conv=self.n_conv,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    dims=dims,
                    activation=self.activation,
                    batch_norm=self.batch_norm,
                )
            )

        # Final block
        self.out_block = nn.Sequential(
            _nConv(
                feat,
                self.fdim,
                n_conv=self.n_conv,
                dims=dims,
                kernel_size=self.kernel_size,
                padding=self.padding,
                activation=self.activation,
                batch_norm=self.batch_norm,
            ),
            call_layer("Conv", dims)(self.fdim, self.out_chans, kernel_size=1),
        )

    def forward(self, x):
        in_shape = x.shape
        skip_connections = []
        x = self.in_block(x)
        skip_connections.append(x)
        for down in self.downs:
            x = down(self.pool(x))
            skip_connections.append(x)
        x = self.bottleneck(self.pool(x))
        skip_connections = skip_connections[::-1]

        for up, scale, skip in zip(self.ups, self.upscale, skip_connections):
            x = up(_skip_concat(scale(x), skip, mode=self.up_mode))

        x = self.out_block(
            _skip_concat(self.upscale[-1](x), skip_connections[-1], mode=self.up_mode)
        )
        if x.shape[2:] != in_shape[2:]:
            return _interpolate(x, in_shape[2:], mode=self.up_mode)
        return x


class UNet1D(_BaseUnet):
    def __init__(
        self,
        *args,
        dims=1,
        up_mode="linear",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            dims=dims,
            up_mode=up_mode,
            **kwargs,
        )


class UNet3D(_BaseUnet):
    def __init__(
        self,
        *args,
        dims=3,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            dims=dims,
            **kwargs,
        )


"""Minimal UNet implementation for 2D and 3D images.
It is a simplified version of the UNet implementation, which 
uses nn.Upsample instead of nn.ConvTranspose2d/3d. It has less
parameters."""


class _UNetMinimal(_BaseUnet):
    def __init__(
        self,
        in_chans,
        out_chans,
        max_level=3,
        dims=3,
        fdim=64,
        n_conv=3,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu_inplace",
        up_mode="trilinear",
        loss_fn=F.mse_loss,
        optimizer=optim.Adam,
        lr=0.001,
        **kwargs,
    ) -> None:
        super().__init__(
            in_chans,
            out_chans,
            max_level,
            dims,
            fdim,
            n_conv,
            kernel_size,
            padding,
            stride,
            activation,
            up_mode,
            loss_fn,
            optimizer,
            lr,
            **kwargs,
        )
        # Upscale blocks
        self.upscale = nn.ModuleList()
        for _ in reversed(self._features):
            self.upscale.append(
                nn.Upsample(scale_factor=2, mode=self.up_mode, align_corners=True)
            )

        # Ups
        self.ups = nn.ModuleList()
        for feat in reversed(self._features[1:]):
            self.ups.append(
                nn.Sequential(
                    _nConv(
                        feat * 2 + feat,
                        feat,
                        n_conv=self.n_conv,
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                        dims=dims,
                        activation=self.activation,
                    ),
                )
            )

        self.out_block[0] = _nConv(
            feat + self.fdim,
            self.fdim,
            n_conv=self.n_conv,
            dims=dims,
            kernel_size=self.kernel_size,
            padding=self.padding,
            activation=self.activation,
        )


class UNet2DMinimal(_UNetMinimal):
    def __init__(
        self,
        *args,
        dims=2,
        up_mode="bilinear",
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            dims=dims,
            up_mode=up_mode,
            **kwargs,
        )


class UNet3DMinimal(_UNetMinimal):
    def __init__(
        self,
        *args,
        dims=3,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            dims=dims,
            **kwargs,
        )
