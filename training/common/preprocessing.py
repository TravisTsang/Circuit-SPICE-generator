"""Image preprocessing shared between training and inference.

CRITICAL: these constants must stay in sync with
``statics_ocv/segmentation.py::TraceSegmenter._preprocess``. If training uses a
different normalization than inference, the U-Net will silently underperform at
runtime even though training metrics look fine.

    BGR image  -> RGB
    resize to (IMAGE_SIZE, IMAGE_SIZE)
    scale to [0, 1]
    normalize with ImageNet mean/std
    CHW float tensor
"""

from __future__ import annotations

import cv2
import numpy as np

# Must match statics_ocv.config.ModelConfig.image_size and segmentation._preprocess.
IMAGE_SIZE = 768
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Scale to [0,1] and apply ImageNet normalization. Input/output are HxWx3."""

    scaled = image_rgb.astype(np.float32) / 255.0
    return (scaled - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_bgr(image_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Mirror of inference preprocessing, returning an HxWx3 normalized array.

    The training Dataset converts this to a CHW tensor; we keep it as an array
    here so augmentation can run before normalization.
    """

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
    return normalize_rgb(resized)
