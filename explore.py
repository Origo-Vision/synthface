import argparse
import pathlib

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from dataset import SynthDataset, augmentations


def main(options: argparse.Namespace) -> None:
    data = SynthDataset(datadir=options.datadir, augmentations=augmentations())

    if options.imageindex >= len(data):
        print("Requested image index is too big")
        return

    image, mask = data[options.imageindex]

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze(0), cmap="gray")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imageindex", type=int, help="Image index")
    parser.add_argument(
        "--datadir", type=pathlib.Path, required=True, help="Data directory"
    )
    options = parser.parse_args()
    main(options)
