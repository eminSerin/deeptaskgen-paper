from torchinfo import summary


def tensor_size(T):
    """Prints the size of given tensor in
    GBs.

    Parameters
    ----------
    T : torch.tensor
        Input tensor.
    """
    T_size = T.element_size() * T.nelement()
    print(f"Tensor size in GB: {T_size / 1024**3:.2f}")


def model_summary(model, input_shape):
    """Prints model summary including
    total number of paramters and estimated
    ram usage.

    Parameters
    ----------
    model : nn.Module
        PyTorch neural network model
        object.
    input_shape : tuple
        Tuple of input shape.
    """
    return summary(model, input_shape)
