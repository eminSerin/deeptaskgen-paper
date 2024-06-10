import torch
import torch.nn.functional as F
from torch import nn, optim

from .base_model import BaseModel
from .utils import (
    _activation_fn,
    _BaseLayer,
    _nConv,
    _skip_add,
    _skip_concat,
    call_layer,
)


class InputLayer(_BaseLayer):
    """Input layer."""

    def __init__(
        self,
        *args,
        n_conv=1,
        dims=3,
        padding=2,
        **kwargs,
    ) -> None:
        super().__init__(n_conv=n_conv, padding=padding, *args, **kwargs)
        self.conv = _nConv(
            self.in_chans,
            self.out_chans,
            n_conv=n_conv,
            dims=dims,
            kernel_size=5,
            padding=2,
            activation=self.activation,
        )

    def forward(self, x):
        out = self.conv(x)
        if (self.in_chans < self.out_chans) and (self.out_chans % self.in_chans == 0):
            return self._activation_fn(
                torch.add(out, x.repeat(1, self.out_chans // self.in_chans, 1, 1, 1))
            )
        return out


class Down(_BaseLayer):
    """Downsampling layer"""

    def __init__(self, *args, padding=2, dims=3, **kwargs) -> None:
        super().__init__(*args, padding=padding, **kwargs)
        self.down_conv = _nConv(
            self.in_chans,
            self.out_chans,
            n_conv=1,
            kernel_size=2,
            stride=2,
            dims=dims,
            activation=self.activation,
        )
        self.conv = _nConv(
            in_chans=self.out_chans,
            out_chans=self.out_chans,
            n_conv=self.n_conv,
            dims=dims,
            activation=self.activation,
            kernel_size=5,
            padding=self.padding,
        )

    def forward(self, x):
        x = self._activation_fn(self.down_conv(x))
        return self._activation_fn(_skip_add(self.conv(x), x, mode=self.up_mode))


class Up(_BaseLayer):
    """Upsampling layer"""

    def __init__(self, *args, padding=2, dims=3, **kwargs) -> None:
        super().__init__(*args, padding=padding, **kwargs)
        self.up_conv = call_layer("ConvTranspose", dims)(
            self.in_chans, self.out_chans // 2, kernel_size=2, stride=2
        )
        self.__activation_fn_up_conv = _activation_fn(
            self.activation, self.out_chans // 2
        )
        self.conv = _nConv(
            in_chans=self.out_chans,
            out_chans=self.out_chans,
            n_conv=self.n_conv,
            dims=dims,
            activation=self.activation,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

    def forward(self, x, skip):
        x = _skip_concat(
            self.__activation_fn_up_conv(self.up_conv(x)), skip, mode=self.up_mode
        )
        return self._activation_fn(_skip_add(self.conv(x), x, mode=self.up_mode))


class OutputLayer(_BaseLayer):
    "Output layer."

    def __init__(self, *args, kernel_size=1, dims=3, **kwargs) -> None:
        super().__init__(*args, kernel_size=kernel_size, **kwargs)
        self.conv = call_layer("Conv", dims)(
            self.in_chans,
            self.out_chans,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

    def forward(self, x):
        return self.conv(x)


class _BaseVNet(BaseModel):
    """Implementation of VNet: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation

    ![V-Net Architecture](./_imgs/V-net.png)

    Arguments:
    ----------
    max_level : int
        Maximum level of the network.
    fdim : int
        Number of feature maps in the first layer.
        This number will be doubled at each level.
    kernel_size : int
        Kernel size of the convolutional layers, by default 5.
    padding : int
        Padding of the convolutional layers, by default 2.
    **args, **kwargs: dict
        Arguments for the BaseLayer class.
        See `deeptaskgen.models.base_model.BaseModel` for more details.

    Reference:
    ----------
    Milletari, Fausto, Nassir Navab, and Seyed-Ahmad Ahmadi.
        "V-net: Fully convolutional neural networks for volumetric
        medical image segmentation." 2016 fourth international conference
        on 3D vision (3DV). IEEE, 2016.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        max_level=2,
        dims=3,
        fdim=64,
        n_conv=None,
        kernel_size=5,
        padding=2,
        stride=1,
        activation="prelu",
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
        self._n_convs = [2 if i == 0 else 3 for i in range(max_level - 1)]

        # Construct layers
        self.input_layer = InputLayer(
            self.in_chans,
            out_chans=fdim,
            dims=dims,
            activation=self.activation,
            padding=self.padding,
        )
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        up_features = self._features[::-1]
        up_n_convs = self._n_convs[::-1]
        for i in range(max_level - 1):
            self.downs.append(
                Down(
                    self._features[i],
                    out_chans=self._features[i] * 2,
                    n_conv=self._n_convs[i],
                    dims=dims,
                    activation=self.activation,
                    up_mode=self.up_mode,
                    padding=self.padding,
                    kernel_size=self.kernel_size,
                )
            )
            if i == 0:
                up_dim = up_features[i]
            else:
                up_dim = up_features[i - 1]
            self.ups.append(
                Up(
                    in_chans=up_dim,
                    out_chans=up_features[i],
                    n_conv=up_n_convs[i],
                    dims=dims,
                    activation=self.activation,
                    up_mode=self.up_mode,
                    padding=self.padding,
                    kernel_size=self.kernel_size,
                )
            )
        self.out_layer = OutputLayer(
            up_features[i],
            out_chans=self.out_chans,
            dims=dims,
            padding=self.padding,
            kernel_size=self.kernel_size,
        )

    def forward(self, x):
        """Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch_size, in_chans, *3D spatial_dims)

        Returns
        -------
        torch.Tensor
            Segmented images of shape (batch_size, out_chans, *3D spatial_dims)
        """
        skip_connections = []
        x = self.input_layer(x)
        skip_connections.append(x)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
        skip_connections.pop(-1)
        skip_connections = skip_connections[::-1]
        for up, skip in zip(self.ups, skip_connections):
            x = up(x, skip)
        return self.out_layer(x)


class VNet3D(_BaseVNet):
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


class VNet1D(_BaseVNet):
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
