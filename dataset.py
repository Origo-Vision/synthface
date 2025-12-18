from __future__ import annotations

import glob
import pathlib

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, v2


def appearance() -> Compose:
    return Compose(
        [
            v2.ColorJitter(brightness=0.15, contrast=0.3, saturation=0.1, hue=0.1),
            v2.RandomGrayscale(p=0.05),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2)),
            v2.RandomPosterize(bits=6),
            v2.JPEG(quality=(40, 100)),
        ]
    )


def warping() -> Compose:
    return Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomResizedCrop(size=512, scale=(0.6, 1.0)),
        ]
    )


class SynthDataset(Dataset):
    def __init__(
        self: SynthDataset,
        datadir: pathlib.Path,
        appearance: Compose = appearance(),
        warping: Compose = warping(),
    ) -> None:
        super().__init__()

        self.appearance = appearance
        self.warping = warping

        self.images = list(map(pathlib.Path, sorted(glob.glob(str(datadir / "*.png")))))
        self.masks = list(map(pathlib.Path, sorted(glob.glob(str(datadir / "*.npy")))))

    def __len__(self: SynthDataset) -> int:
        return len(self.images)

    def __getitem__(
        self: SynthDataset, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv.imread(str(self.images[index]), cv.IMREAD_COLOR_RGB)
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.uint8)

        # Apply appearance transformations on uint8.
        image = self.appearance(image)
        image = image.to(torch.float32) / 255.0

        # Apply slight noise on float32.
        image = v2.functional.gaussian_noise(image, sigma=0.01)

        mask = np.load(self.masks[index])
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Apply warping on both the image and the mask.
        rng_state = torch.get_rng_state()
        image = self.warping(image)

        torch.set_rng_state(rng_state)
        mask = self.warping(mask)

        return image, mask
