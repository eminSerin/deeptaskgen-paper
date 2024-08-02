import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def call_layer(layer, dims):
    """Calls a layer with the specified dimensions

    Parameters
    ----------
    layer : str
        Layer name to call. Check `torch.nn` for more details.
    dims : int
        Dimensions of the layer.

    Returns
    -------
    torch.nn
        torch.nn layer

    Raises
    ------
    ValueError
        When dims is not 1, 2 or 3.
    """
    if not dims in [1, 2, 3]:
        raise ValueError("Dimensions must be 1, 2 or 3")
    return getattr(nn, f"{layer}{dims}d")


class _BaseLayer(pl.LightningModule):
    """Base layer object for all layers used in
    the U-Net and V-Net architectures.

    Arguments:
    ----------
    in_chans : int
        Number of input channels.
    out_chans : int
        Number of output channels.
    n_conv : int
        Number of convolutional layers, by default None.
    kernel_size : int
        Convolutional kernel size, by default 3.
    padding : int
        Convolutional padding, by default 1.
    stride : int
        Convolutional stride, by default 1.
    activation : str, optional
        Activation function.
        See `deeptaskgen.models.utils._activation_fn`
        for more details, by default "relu".
    up_mode : str, optional
        Upscaling mode, by default "trilinear".
        Upscaling is used if the input and target
        shapes are different.

    """

    def __init__(
        self,
        in_chans,
        out_chans,
        n_conv=None,
        kernel_size=3,
        padding=1,
        stride=1,
        activation="relu",
        up_mode="trilinear",
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.n_conv = n_conv
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self._activation_fn = _activation_fn(activation, out_chans)
        self.up_mode = up_mode


class _nConv(_BaseLayer):
    """Performs n numbers of convolution
    required in hourglass neural
    network architectures.
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        n_conv=None,
        dims=3,
        kernel_size=3,
        padding=1,
        stride=1,
        batch_norm=True,
        activation="relu_inplace",
        up_mode="trilinear",
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
        )
        layers = []

        in_ch = self.in_chans
        for _ in range(n_conv):
            layers.append(
                call_layer("Conv", dims)(
                    in_ch,
                    self.out_chans,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                )
            )
            if batch_norm:
                layers.append(call_layer("BatchNorm", dims)(self.out_chans))
            layers.append(self._activation_fn)
            in_ch = self.out_chans

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


def _skip_concat(x, skip, mode="trilinear"):
    """Concatenates the skip connection

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    skip : torch.Tensor
        Skip connection tensor.
    mode : str, optional
        Interpolation method, by default "trilinear"
        See `torch.nn.functional.interpolate` for more
        information.

    Returns
    -------
    torch.Tensor
        Output tensor.
    """
    if x.shape != skip.shape:
        return torch.cat([_interpolate(x, skip.shape[2:], mode=mode), skip], dim=1)
    return torch.cat([x, skip], dim=1)


def _skip_add(x, skip, mode="trilinear"):
    """Addes the skip connection

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    skip : torch.Tensor
        Skip connection tensor.
    mode : str, optional
        Interpolation method, by default "trilinear"
        See `torch.nn.functional.interpolate` for more
        information.

    Returns
    -------
    torch.Tensor
        Output tensor.
    """
    if x.shape != skip.shape:
        return torch.add(_interpolate(x, skip.shape[2:], mode=mode), skip)
    return torch.add(x, skip)


def _interpolate(x, size, mode="trilinear", align_corners=None):
    """_summary_

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    size : tuple
        Tuple of target size.
    mode : str, optional
        Interpolation method, by default "trilinear".
        See torch.nn.functional.interpolate for more details.
    align_corners : bool or None, optional
        Whether to align corners, by default None.
        It only works for mode "bilinear" and "trilinear".

    Returns
    -------
    torch.Tensor
        Interpolated tensor.
    """
    if mode not in ["nearest", "nearest-exact"]:
        align_corners = None
    else:
        align_corners = True
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


def _activation_fn(activation="relu_inplace", n_channels=None):
    """It returns the activation function.

    Parameters
    ----------
    activation : str
        Type of activation function.
        It currently supports "relu", "leakyrelu",
        "prelu", "elu", "tanh", "sigmoid",
        by default "relu". For ReLU, LeakyReLU,
        and ELU, it returns the inplace version
        if you add "_inplace" to the activation
        function name. For example, "relu_inplace".
    n_channels : int, optional
        Number of parameters for PReLU activation
        function, by default None. This is only
        for PReLU activation function. If None,
        it is set to 1.
    Returns
    -------
    nn.Module
        Activation function.
    """
    if activation == "relu":
        return nn.ReLU(inplace=False)
    elif activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    elif activation == "leakyrelu":
        return nn.LeakyReLU(inplace=False)
    elif activation == "leakyrelu_inplace":
        activation = nn.LeakyReLU(inplace=True)
    elif activation == "elu":
        return nn.ELU(inplace=False)
    elif activation == "elu_inplace":
        return nn.ELU(inplace=True)
    elif activation == "selu":
        return nn.SELU(inplace=False)
    elif activation == "selu_inplace":
        return nn.SELU(inplace=True)
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "prelu":
        if n_channels is None:
            n_channels = 1
        return nn.PReLU(num_parameters=n_channels, init=0.25)
    else:
        raise NotImplementedError


def _reparameterize(mu, logvar):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the latent distribution Q(z|X).
    logvar : torch.Tensor
        Log-variance of the latent distribution Q(z|X).

    Returns
    -------
    torch.Tensor
        Sampled latent vector.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def xavier_init(model):
    """Xavier initialization for the weights."""
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            bound = torch.sqrt(6) / torch.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)


def gaussian_noise(tensor, std=0.1):
    """
    Add Gaussian noise to a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor.
    std : float, optional
        The standard deviation of the Gaussian noise (default is 0.1).

    Returns
    -------
    torch.Tensor
        The tensor with added Gaussian noise.

    """
    return tensor + torch.randn_like(tensor) * std
