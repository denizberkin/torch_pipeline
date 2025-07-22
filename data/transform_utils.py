from typing import Tuple

import cv2
import numpy as np


def ratio_preserved_resize(shape: Tuple[int, int]) -> np.ndarray:
    """
    Returns resizer function based on provided shape to be transformed 
    ### Args:
        - shape (Tuple[int, int]): Target shape for resizing
    ### Returns:
        - function with signature (image: np.ndarray) -> np.ndarray that resizes to the `shape`
    """
    def resizer(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        num_channels = image.shape[2] if image.ndim == 3 else 1

        if w / shape[1] < h / shape[0]:
            image = np.resize(image, (int(shape[1] * w / h), shape[0], num_channels))
            image = np.concatenate([image, np.zeros((shape[0] - image.shape[0], shape[1], num_channels), dtype=image.dtype)], axis=0)
        else:
            image = np.resize(image, (shape[1], int(shape[0] * h / w), num_channels))
            image = np.concatenate([image, np.zeros((shape[0], shape[1] - image.shape[1], num_channels), dtype=image.dtype)], axis=1)
        return image
    return resizer


def to_float():
    def convert(image: np.ndarray):
        return image.astype(np.float32) / 255.0
    return convert


def img2roi(image: np.ndarray):
    """
    Extract roi from image, remove background as much as possible
    ### Args:
        - image (np.ndarray): Input image
    ### Returns:
        - roi (np.ndarray): Region of interest
        - coords (list): Coordinates of the ROI [x0, x1, y0, y1]
    """
    if image.max() <= 1:
        image = np.array(image * 255, dtype=np.uint8)
    # Binarize the image
    bin_img = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)[1]

    # Make contours around the binarized image, keep only the largest contour
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)

    # Find ROI from largest contour
    ys = contour.squeeze()[:, 0]
    xs = contour.squeeze()[:, 1]
    coords = [np.min(xs), np.max(xs), np.min(ys), np.max(ys)]
    roi =  image[coords[0]: coords[1], coords[2]: coords[3]]
    return roi, coords