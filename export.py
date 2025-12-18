import argparse
import pathlib
from zipfile import ZipFile

import cv2 as cv
import numpy as np


def main(options: argparse.Namespace) -> None:
    if options.datadir_train == options.datadir_valid:
        print("Error train and valid must not be same directories")
        return

    if options.datadir_train.exists() and not options.datadir_train.is_dir():
        print("Error: The --datadir-train exists, but is no directory")
        return

    if options.datadir_valid.exists() and not options.datadir_valid.is_dir():
        print("Error: The --datadir-valid exists, but is no directory")
        return

    if not options.datadir_train.exists():
        options.datadir_train.mkdir(parents=True)

    if not options.datadir_valid.exists():
        options.datadir_valid.mkdir(parents=True)

    count = 1
    with ZipFile(options.zipfile, "r") as zip:
        for name in zip.namelist():
            path = pathlib.Path(name)
            if path.stem.isdigit():
                datadir = (
                    options.datadir_valid
                    if (count % 10 == 0)
                    else options.datadir_train
                )

                zip.extract(name, path=str(datadir))

                seg_name = path.stem + "_seg.png"
                zip.extract(seg_name, "/tmp")
                tmp_file = pathlib.Path("/tmp") / seg_name

                seg_image = cv.imread(str(tmp_file), cv.IMREAD_GRAYSCALE)
                mask = np.logical_and(seg_image > 0, seg_image < 255)

                np.save(datadir / (path.stem + ".npy"), mask.astype(np.float32))
                tmp_file.unlink()

                print(f"\rCopy item #{count}", end="")

                count += 1

                if count == options.max_count:
                    break

        print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("zipfile", type=pathlib.Path, help="The zip-file to export")
    parser.add_argument(
        "--datadir-train",
        type=pathlib.Path,
        required=True,
        help="Destination training data directory",
    )
    parser.add_argument(
        "--datadir-valid",
        type=pathlib.Path,
        required=True,
        help="Destination validation data directory",
    )
    parser.add_argument(
        "--max-count", type=int, default=1e9, help="Max number of images to copy"
    )
    options = parser.parse_args()
    main(options)
