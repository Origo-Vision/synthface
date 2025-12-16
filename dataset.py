from __future__ import annotations

import glob
import pathlib

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, v2

import utils

def augmentations() -> Compose:
    return Compose(
        [
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2)),
            v2.GaussianNoise(sigma=0.03),
            v2.RandomPosterize(bits=6),
        ]
    )


class SynthDataset(Dataset):
    def __init__(
        self: SynthDataset,
        datadir: pathlib.Path,
        augmentations: Compose = Compose([]),
    ) -> None:
        super().__init__()

        self.augmentations = augmentations

        self.images = list(map(pathlib.Path, sorted(glob.glob(str(datadir / "*.png")))))
        self.masks = list(map(pathlib.Path, sorted(glob.glob(str(datadir / "*.npy")))))

    def __len__(self: SynthDataset) -> int:
        return len(self.images)

    def __getitem__(
        self: SynthDataset, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv.imread(str(self.images[index]), cv.IMREAD_COLOR_RGB)
        image = utils.numpy_to_tensor(image)

        mask = np.load(self.masks[index])
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return self.augmentations(image), mask
