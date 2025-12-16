import argparse
import time
import pathlib

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
import torch

import model
import utils


def crop_scale(image: NDArray, size: int = 512) -> NDArray:
    h, w = image.shape[:2]
    if h == size and w == size:
        return image

    dim = min(h, w)
    y = (h - dim) // 2
    x = (w - dim) // 2

    image = image[y : y + dim, x : x + dim, :]

    return cv.resize(image, dsize=(size, size))


def main(options: argparse.Namespace) -> None:
    device = utils.find_device(options.force_cpu)
    print(f"Device={device}")

    net = model.load(options.model).to(device)
    print(f"Model loaded with {utils.count_parameters(net)} parameters")
    net.eval()

    video = cv.VideoCapture(options.camera)
    if not video.isOpened():
        return

    cv.namedWindow("frame")
    with torch.no_grad():
        while True:
            frame_start = time.monotonic()

            available, bgr = video.read()
            if not available:
                print("Stream is ended")
                return

            bgr = crop_scale(bgr)
            rgb = utils.numpy_to_tensor(cv.cvtColor(bgr, cv.COLOR_BGR2RGB)).to(device).unsqueeze(0)

            inference_start = time.monotonic()
            face_mask = net(rgb)
            inference_duration = time.monotonic() - inference_start

            face_mask = face_mask.squeeze(0, 1).detach().cpu().numpy()
            masked = bgr.copy()
            masked[face_mask > 0.5] = (0, 255, 0)

            cv.imshow("frame", np.hstack((bgr, masked)))
            cv.setWindowTitle("frame", f"Inference time={inference_duration*1000:.1f}ms")

            frame_duration = round((time.monotonic() - frame_start) * 1000)
            delay = 30 - frame_duration if frame_duration < 30 else 1

            key = cv.waitKey(delay)
            if key > 0 and (key == 27 or chr(key) == "q"):
                print("Quit")
                break

    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("camera", type=int, help="Camera id")
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force inference on the CPU"
    )
    parser.add_argument("--model", type=pathlib.Path, required=True, help="Model")
    options = parser.parse_args()

    main(options)
