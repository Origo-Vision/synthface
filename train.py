import argparse
import pathlib
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SynthDataset, augmentations
import model
import utils


def main(options: argparse.Namespace) -> None:
    device = utils.find_device(options.force_cpu)
    print(f"Device={device}")

    train_dataset = SynthDataset(
        datadir=options.datadir_train, augmentations=augmentations()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=options.batch_size, shuffle=True
    )

    valid_dataset = SynthDataset(
        datadir=options.datadir_valid, augmentations=augmentations()
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=options.batch_size, shuffle=False
    )

    net = model.empty().to(device)
    print(f"Number of network parameters={utils.count_parameters(net)}")

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-03)
    loss_fn = torch.nn.BCELoss()

    min_loss = 100.0

    for epoch in range(options.epochs):
        print(f"epoch {epoch+1:4d}/{options.epochs:4d}")

        print("Training ...")
        net.train()

        num_train_batches = len(train_loader)
        accum_train_loss = 0.0
        for batch, data in enumerate(train_loader):
            print(f"\r  batch {batch+1:4d}/{num_train_batches:4d} ... ", end="")

            Xb, Yb = data
            Xb = Xb.to(device)
            Yb = Yb.to(device)

            Ypred = net(Xb)

            loss = loss_fn(Ypred, Yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_train_loss += loss.item()

        avg_train_loss = accum_train_loss / num_train_batches
        print(f"\r  avg train loss={avg_train_loss:.7f}")

        print("Validation ...")
        net.eval()

        num_valid_batches = len(valid_loader)
        accum_valid_loss = 0.0

        with torch.no_grad():
            for batch, data in enumerate(valid_loader):
                print(f"\r  batch {batch+1:4d}/{num_valid_batches:4d} ... ", end="")

                Xb, Yb = data
                Xb = Xb.to(device)
                Yb = Yb.to(device)

                Ypred = net(Xb)

                loss = loss_fn(Ypred, Yb)

                accum_valid_loss += loss.item()

        avg_valid_loss = accum_valid_loss / num_valid_batches
        print(f"\r  avg valid loss={avg_valid_loss:.7f}")

        if avg_valid_loss < min_loss:
            model.save(net, pathlib.Path("save.pth"))
            min_loss = avg_valid_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=1598, help="Random seed number")
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force inference on the CPU"
    )
    parser.add_argument(
        "--datadir-train",
        type=pathlib.Path,
        required=True,
        help="Training dataset directory",
    )
    parser.add_argument(
        "--datadir-valid",
        type=pathlib.Path,
        required=True,
        help="Validation dataset directory",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="The batch size")
    parser.add_argument("--epochs", type=int, default=30, help="The number of epochs")
    options = parser.parse_args()

    utils.set_seed(options.seed)
    main(options)
