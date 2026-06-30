"""U-Net based conductive trace segmentation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .config import AppConfig

LOGGER = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Two convolution blocks used by the U-Net encoder and decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Compact U-Net for binary wire-trace segmentation.

    The class supports common state-dict checkpoints. If your training code used
    a different channel schedule, export the model as TorchScript and this module
    will load it without depending on this architecture definition.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = torch.nn.functional.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


@dataclass(slots=True)
class SegmentationResult:
    """Outputs produced by the trace segmenter."""

    probability_map: np.ndarray
    binary_mask: np.ndarray


class TraceSegmenter:
    """Load a trained U-Net and produce clean wire masks from schematic images."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = torch.device(config.models.device)
        self.model = self._load_model(config.models.unet_path)
        self.model.to(self.device)
        self.model.eval()

    def segment_image(self, image_path: str | Path) -> SegmentationResult:
        """Run segmentation for an image path and return probability and mask arrays."""

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"OpenCV could not read image: {path}")

        return self.segment_array(image_bgr)

    def segment_array(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Run segmentation for an already-loaded BGR image."""

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image array with shape HxWx3.")

        original_h, original_w = image_bgr.shape[:2]
        tensor = self._preprocess(image_bgr)

        with torch.no_grad():
            logits = self.model(tensor.to(self.device))
            probs = torch.sigmoid(logits)

        probability = probs.squeeze().detach().cpu().numpy().astype(np.float32)
        probability = cv2.resize(
            probability,
            (original_w, original_h),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = (probability >= self.config.models.segmentation_threshold).astype(np.uint8)
        mask = self._postprocess_mask(mask)
        LOGGER.info(
            "Segmented trace mask: %d foreground pixels at threshold %.2f",
            int(mask.sum()),
            self.config.models.segmentation_threshold,
        )
        return SegmentationResult(probability_map=probability, binary_mask=mask)

    def save_debug_outputs(
        self,
        result: SegmentationResult,
        output_dir: str | Path,
        stem: str,
    ) -> None:
        """Save probability and binary-mask images for inspection."""

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        probability_u8 = np.clip(result.probability_map * 255, 0, 255).astype(np.uint8)
        mask_u8 = (result.binary_mask * 255).astype(np.uint8)
        cv2.imwrite(str(out / f"{stem}_trace_probability.png"), probability_u8)
        cv2.imwrite(str(out / f"{stem}_trace_mask.png"), mask_u8)

    def _load_model(self, path: Path) -> nn.Module:
        """Load a TorchScript model or a regular U-Net state-dict checkpoint."""

        if not path.exists():
            raise FileNotFoundError(
                f"U-Net weights not found: {path}. Train or copy the model there, "
                "or override models.unet_path in a YAML config."
            )

        LOGGER.info("Loading U-Net trace segmenter from %s", path)
        try:
            scripted = torch.jit.load(str(path), map_location=self.device)
            LOGGER.info("Loaded TorchScript U-Net model.")
            return scripted
        except Exception as script_error:
            LOGGER.debug("TorchScript load failed, trying state dict: %s", script_error)

        checkpoint = torch.load(str(path), map_location=self.device)
        model_kwargs = {}
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            model_kwargs = checkpoint.get("model_kwargs", {})
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

        if not isinstance(state_dict, dict):
            raise ValueError(
                "Unsupported U-Net checkpoint format. Expected TorchScript or a state dict."
            )

        cleaned_state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        model = UNet(**model_kwargs)
        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            LOGGER.warning("U-Net checkpoint missing keys: %s", ", ".join(missing[:8]))
        if unexpected:
            LOGGER.warning("U-Net checkpoint has unexpected keys: %s", ", ".join(unexpected[:8]))
        return model

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Resize, normalize, and tensorize the BGR image."""

        size = self.config.models.image_size
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.float()

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Remove isolated noise and close small gaps after thresholding."""

        kernel_size = max(1, int(self.config.graph.wire_close_kernel_px))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        return (opened > 0).astype(np.uint8)
