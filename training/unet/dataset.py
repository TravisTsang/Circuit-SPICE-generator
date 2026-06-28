"""Dataset for U-Net wire-trace segmentation.

Expects the layout documented in the README::

    datasets/trace_segmentation/
        images/{train,val}/<name>.png
        masks/{train,val}/<name>.png   # single channel, wire=255, background=0

Image and mask filenames must match (extension may differ). Masks are read as
grayscale and binarized at 127.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from training.common.preprocessing import IMAGE_SIZE, normalize_rgb

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _find_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """Match each image to a mask with the same stem (any image extension)."""

    mask_by_stem: dict[str, Path] = {}
    for ext in _IMAGE_EXTS:
        for m in masks_dir.glob(f"*{ext}"):
            mask_by_stem[m.stem] = m

    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for ext in _IMAGE_EXTS:
        for img in images_dir.glob(f"*{ext}"):
            mask = mask_by_stem.get(img.stem)
            if mask is None:
                missing += 1
                continue
            pairs.append((img, mask))

    if missing:
        print(f"[dataset] WARNING: {missing} images in {images_dir} have no matching mask")
    return sorted(pairs)


class TraceSegmentationDataset(Dataset):
    """Image/mask pairs with light, label-preserving augmentation.

    Augmentation uses only numpy/OpenCV so no extra dependency is required. For
    stronger augmentation, swap in albumentations inside ``_augment``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        size: int = IMAGE_SIZE,
        augment: bool | None = None,
    ) -> None:
        root = Path(root)
        self.images_dir = root / "images" / split
        self.masks_dir = root / "masks" / split
        self.size = size
        self.augment = (split == "train") if augment is None else augment

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        self.pairs = _find_pairs(self.images_dir, self.masks_dir)
        if not self.pairs:
            raise RuntimeError(
                f"No image/mask pairs found under {root} (split={split}). "
                "Generate synthetic data or ingest a real dataset first."
            )
        print(f"[dataset] {split}: {len(self.pairs)} pairs from {self.images_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, image_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Horizontal / vertical flips (wires are direction-agnostic).
        if random.random() < 0.5:
            image_rgb, mask = image_rgb[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            image_rgb, mask = image_rgb[::-1, :], mask[::-1, :]

        # 90-degree rotations.
        k = random.randint(0, 3)
        if k:
            image_rgb = np.rot90(image_rgb, k)
            mask = np.rot90(mask, k)

        # Brightness / contrast jitter on the image only.
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)   # contrast
            beta = random.uniform(-25, 25)     # brightness
            image_rgb = np.clip(image_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Mild gaussian noise to mimic scan/photo grain.
        if random.random() < 0.3:
            noise = np.random.normal(0, 8, image_rgb.shape).astype(np.float32)
            image_rgb = np.clip(image_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(image_rgb), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        image_rgb = cv2.resize(image_rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image_rgb, mask = self._augment(image_rgb, mask)

        image = normalize_rgb(image_rgb).transpose(2, 0, 1)
        mask_bin = (mask >= 127).astype(np.float32)[None, :, :]  # 1xHxW

        return torch.from_numpy(image).float(), torch.from_numpy(mask_bin).float()
