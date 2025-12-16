import argparse
import pathlib

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from dataset import SynthDataset, augmentations
import model


def main(options: argparse.Namespace) -> None:
    net = model.load(options.model)
    data = SynthDataset(datadir=options.datadir, augmentations=augmentations())

    if options.imageindex >= len(data):
        print("Requested image index is too big")
        return

    image, mask = data[options.imageindex]

    pred_mask = net(image.unsqueeze(0)).detach().numpy()
    pred_mask = pred_mask > 0.5

    pred_mask = pred_mask[0, 0, :, :]
    privacy = image.permute(1, 2, 0).numpy().copy()
    privacy[pred_mask] = (0.0, 1.0, 0.0)

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 4, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Image")

    plt.subplot(1, 4, 2)
    plt.imshow(mask.squeeze(0), cmap="gray")
    plt.axis("off")
    plt.title("GT Segmentation")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask, cmap="gray")
    plt.axis("off")
    plt.title("Pred Segmentation")

    plt.subplot(1, 4, 4)
    plt.imshow(privacy, cmap="gray")
    plt.axis("off")
    plt.title("Applied Pred Privacy")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("imageindex", type=int, help="Image index")
    parser.add_argument(
        "--datadir", type=pathlib.Path, required=True, help="Data directory"
    )
    parser.add_argument("--model", type=pathlib.Path, required=True, help="Model")
    options = parser.parse_args()
    main(options)
