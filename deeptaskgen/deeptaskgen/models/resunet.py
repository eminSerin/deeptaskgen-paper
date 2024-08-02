import torch.nn.functional as F
from torch import nn, optim

from .base_model import BaseModel
from .utils import _BaseLayer, _interpolate, _skip_add, _skip_concat, call_layer


class ResUnit(_BaseLayer):
    """Residual unit.

    Parameters
    ----------
        See `deeptaskgen.models.utils._BaseLayer` for more information.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        n_conv=2,
        dims=3,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu_inplace",
        up_mode="trilinear",
        batch_norm=True,
        lr_scheduler=True,
    ) -> None:
        super().__init__(
            in_chans,
            out_chans,
            n_conv,
            kernel_size,
            padding,
            stride,
            activation,
            up_mode,
            batch_norm=batch_norm,
            lr_scheduler=lr_scheduler,
        )
        if n_conv is None:
            self.n_conv = 2
        layers = nn.ModuleList()
        in_ch = self.in_chans
        for _ in range(self.n_conv):
            if batch_norm:
                layers.append(call_layer("BatchNorm", dims)(in_ch))
            layers.append(self._activation_fn)
            layers.append(
                call_layer("Conv", dims)(
                    in_ch,
                    self.out_chans,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                )
            )
            in_ch = self.out_chans
        self.conv = nn.Sequential(*layers)

        # Skip connection
        self.skip = nn.Sequential()
        if self.in_chans != self.out_chans:
            self.skip = nn.Sequential(
                (
                    call_layer("Conv", dims)(
                        self.in_chans,
                        self.out_chans,
                        kernel_size=1,
                        padding=0,
                        stride=1,
                    )
                )
            )

    def forward(self, x):
        return _skip_add(self.conv(x), self.skip(x), mode=self.up_mode)


class _BaseResUNet(BaseModel):
    """ResUNet architecture.

    Deep-learning based segmentation model for 3D medical images, combining
    deep residual networks and U-net architecture [1].

    Parameters
    ----------
    See models.base_model.BaseModel for more details.


    References
    ----------
    [1] - Zhang, Zhengxin, Qingjie Liu, and Yunhong Wang. "Road extraction by deep  residual u-net." IEEE Geoscience and Remote Sensing Letters 15.5 (2018): 749-753.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        max_level=3,
        dims=3,
        fdim=64,
        n_conv=2,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu",
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
            **kwargs,
        )
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.upscale = nn.ModuleList()
        self.pool = call_layer("MaxPool", dims)(kernel_size=2, stride=2)

        # Input block
        self.in_block = ResUnit(
            self.in_chans,
            self.fdim,
            kernel_size=kernel_size,
            dims=dims,
            padding=self.padding,
            stride=self.stride,
            activation=self.activation,
            up_mode=self.up_mode,
            n_conv=self.n_conv,
            batch_norm=self.batch_norm,
        )

        # Downs
        in_dim = self._features[0]
        for feat in self._features[1:]:
            self.downs.append(
                ResUnit(
                    in_dim,
                    feat,
                    kernel_size=kernel_size,
                    dims=dims,
                    padding=self.padding,
                    stride=self.stride,
                    activation=self.activation,
                    up_mode=self.up_mode,
                    n_conv=self.n_conv,
                    batch_norm=self.batch_norm,
                )
            )
            in_dim = feat

        # Bottleneck
        self.bottleneck = ResUnit(
            feat,
            feat * 2,
            kernel_size=kernel_size,
            dims=dims,
            padding=self.padding,
            stride=self.stride,
            activation=self.activation,
            up_mode=self.up_mode,
            n_conv=self.n_conv,
            batch_norm=self.batch_norm,
        )

        # Upscale blocks
        for feat in reversed(self._features):
            self.upscale.append(
                call_layer("ConvTranspose", dims)(
                    feat * 2,
                    feat,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.padding,
                )
            )

        # Ups
        for feat in reversed(self._features[1:]):
            self.ups.append(
                ResUnit(
                    feat * 2,
                    feat,
                    n_conv=self.n_conv,
                    dims=dims,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                    activation=self.activation,
                    up_mode=self.up_mode,
                    batch_norm=self.batch_norm,
                )
            )

        self.out_block = nn.Sequential(
            ResUnit(
                feat,
                fdim,
                n_conv=self.n_conv,
                dims=dims,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                activation=self.activation,
                up_mode=self.up_mode,
                batch_norm=self.batch_norm,
            ),
            call_layer("Conv", dims)(fdim, self.out_chans, kernel_size=1),
        )

    def forward(self, x):
        in_shape = x.shape
        # Downward path
        skip_connections = []
        x = self.in_block(x)
        skip_connections.append(x)
        for down in self.downs:
            x = down(self.pool(x))
            skip_connections.append(x)
        x = self.bottleneck(self.pool(x))
        skip_connections = skip_connections[::-1]

        # Upward path
        for up, skip, scale in zip(self.ups, skip_connections, self.upscale):
            x = up(_skip_concat(scale(x), skip, mode=self.up_mode))

        x = self.out_block(
            _skip_concat(self.upscale[-1](x), skip_connections[-1], mode=self.up_mode)
        )
        if x.shape[2:] != in_shape[2:]:
            return _interpolate(x, in_shape[2:], mode=self.up_mode)
        return x


class ResUNet1D(_BaseResUNet):
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


class ResUNet3D(_BaseResUNet):
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
