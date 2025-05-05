"""U-Net model implementation with attention gates for biomedical image segmentation.

This module contains:
- AttentionGate: Implements additive attention mechanism between skip connections and upsampled features
- _BaseUnet: Base U-Net architecture with configurable depth, dimensions and upsampling
- UNet3D: Specialized 3D implementations of U-Net
- UNet3DMinimal: Minimal variants of 3D U-Nets
- AttentionUNet3D: 3D U-Net variant with attention gates
- AttentionUNet3DMinimal: Minimal 3D U-Net variant with attention gates

The implementation follows the original U-Net architecture with improvements including:
- Configurable depth and feature dimensions
- Optional attention gates
- Flexible upsampling methods
- Minimal variants with simplified architecture
"""

import torch.nn.functional as F
from torch import nn, optim

from .base_model import BaseModel
from .utils import _interpolate, _nConv, _skip_concat, call_layer


class AttentionGate(nn.Module):
    """Attention Gate module for Attention U-Net.

    Implements additive attention mechanism between skip connections and upsampled features.
    """

    def __init__(self, F_g, F_l, F_int, dims=3, up_mode="trilinear"):
        """
        Args:
            F_g: Input feature size from the upsampling path
            F_l: Input feature size from the skip connection
            F_int: Intermediate feature size
            dims: Number of dimensions (2D or 3D)
        """
        super(AttentionGate, self).__init__()
        self.W_g = call_layer("Conv", dims)(
            F_g, F_int, kernel_size=1, stride=1, padding=0
        )
        self.W_x = call_layer("Conv", dims)(
            F_l, F_int, kernel_size=1, stride=1, padding=0
        )
        self.psi = nn.Sequential(
            call_layer("Conv", dims)(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)
        self.up_mode = up_mode

    def forward(self, g, x):
        # g: gating signal from the upsampling path (coarser scale)
        # x: skip connection features (finer scale)

        g = self.W_g(g)
        x = self.W_x(x)

        # Ensure g1 and x1 have the same spatial dimensions for addition
        if g.shape[2:] != x.shape[2:]:
            g = _interpolate(g, size=x.shape[2:], mode=self.up_mode)

        psi = self.relu(g + x)
        psi = self.psi(psi)

        return x * psi


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
        attention=False,
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
        self.attention = attention

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

        # Attention gates
        if self.attention:
            self.attention_gates = nn.ModuleList()
            for feat in reversed(self._features):
                self.attention_gates.append(
                    AttentionGate(feat, feat, feat, dims=dims, up_mode=self.up_mode)
                )

        # Final block
        self.out_block = nn.Sequential(
            _nConv(
                self.fdim * 2,
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

        if self.attention:
            for i, (up, scale, skip, attn) in enumerate(
                zip(
                    self.ups,
                    self.upscale,
                    skip_connections[:-1],
                    self.attention_gates[:-1],
                )
            ):
                x = scale(x)
                x = up(_skip_concat(x, attn(x, skip), mode=self.up_mode))
        else:
            for up, scale, skip in zip(self.ups, self.upscale, skip_connections):
                x = up(_skip_concat(scale(x), skip, mode=self.up_mode))

        if self.attention:
            x = self.upscale[-1](x)
            x = self.out_block(
                _skip_concat(
                    x,
                    self.attention_gates[-1](x, skip_connections[-1]),
                    mode=self.up_mode,
                )
            )
        else:
            x = self.out_block(
                _skip_concat(
                    self.upscale[-1](x), skip_connections[-1], mode=self.up_mode
                )
            )
        if x.shape[2:] != in_shape[2:]:
            return _interpolate(x, in_shape[2:], mode=self.up_mode)
        return x


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
        batch_norm=True,
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
            batch_norm=batch_norm,
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
                        batch_norm=self.batch_norm,
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
            batch_norm=self.batch_norm,
        )

        # Attention gates
        if self.attention:
            self.attention_gates = nn.ModuleList()
            for feat in reversed(self._features):
                self.attention_gates.append(
                    AttentionGate(feat * 2, feat, feat, dims=dims, up_mode=self.up_mode)
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


class AttentionUNet3D(_BaseUnet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, dims=3, attention=True, **kwargs)


class AttentionUNet3DMinimal(_UNetMinimal):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, dims=3, attention=True, **kwargs)
