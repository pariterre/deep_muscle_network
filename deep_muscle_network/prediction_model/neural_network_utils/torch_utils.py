import torch


def get_torch_device() -> torch.device:
    """
    Get the device to use for the computations.

    Returns
    -------
    torch.device
        The device to use.
    """
    # TODO : Test this function
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
