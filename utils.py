import random

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """
    Set the random seed.

    Parameters:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_all_rng_states() -> dict:
    """
    Get the RNG states for all available devices.

    Returns:
        Dictionary with the states.
    """
    states = {"cpu": torch.get_rng_state()}

    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state()

    if torch.backends.mps.is_available():
        states["mps"] = torch.mps.get_rng_state()

    return states


def set_all_rng_states(states: dict) -> None:
    """
    Set the RNG states for the devices in the states dictionary.

    Parameters:
        states: States dictionary.
    """
    torch.set_rng_state(states["cpu"])

    if "cuda" in states:
        torch.cuda.set_rng_state(states["cuda"])

    if "mps" in states:
        torch.mps.set_rng_state(states["mps"])


def count_parameters(mod: nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Parameters:
        mod: The model.

    Returns:
        The parameter count.
    """
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


def find_device(force_cpu: bool) -> torch.device:
    """
    Find the best the device for the execution.

    Parameters:
        force_cpu: If true, the device will be forced to CPU.

    Returns:
        The device.
    """
    if force_cpu:
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def numpy_to_tensor(image: NDArray) -> torch.Tensor:
    """
    Convert a 3-channel numpy matrix to normalized tensor.

    Parameters:
        image: Numpy image.

    Returns:
        Tensor.
    """
    return torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
