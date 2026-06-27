"""Central configuration for schematic OCR and SPICE export.

The defaults are deliberately conservative. They are intended to work for
noisy scans and hand-drawn images after the U-Net has removed most background
noise, while still surfacing warnings when topology cannot be inferred safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ModelConfig:
    """Paths and runtime settings for model inference."""

    unet_path: Path = ROOT_DIR / "models" / "unet_trace_segmentation.pt"
    yolo_path: Path = ROOT_DIR / "models" / "yolov8_components.pt"
    device: str = "cpu"
    image_size: int = 768
    segmentation_threshold: float = 0.50
    yolo_confidence: float = 0.35
    yolo_iou: float = 0.45
    easyocr_languages: tuple[str, ...] = ("en",)
    easyocr_gpu: bool = False


@dataclass(slots=True)
class GraphConfig:
    """Pixel-level tolerances for wire skeletonization and graph recovery."""

    min_wire_pixels: int = 8
    endpoint_gap_warning_px: float = 16.0
    terminal_search_radius_px: int = 22
    terminal_inner_margin_px: int = 10
    component_bbox_padding_px: int = 4
    max_text_association_distance_px: float = 180.0
    wire_close_kernel_px: int = 3


@dataclass(slots=True)
class SpiceConfig:
    """Component naming, terminal counts, and default SPICE values."""

    ground_names: tuple[str, ...] = ("ground", "gnd", "earth")
    class_to_prefix: dict[str, str] = field(
        default_factory=lambda: {
            "resistor": "R",
            "r": "R",
            "capacitor": "C",
            "cap": "C",
            "c": "C",
            "inductor": "L",
            "l": "L",
            "diode": "D",
            "d": "D",
            "led": "D",
            "voltage_source": "V",
            "vsource": "V",
            "voltage": "V",
            "battery": "V",
            "current_source": "I",
            "isource": "I",
            "current": "I",
            "ground": "0",
            "gnd": "0",
            "switch": "S",
            "bjt": "Q",
            "npn": "Q",
            "pnp": "Q",
            "mosfet": "M",
            "nmos": "M",
            "pmos": "M",
            "opamp": "X",
        }
    )
    terminal_counts: dict[str, int] = field(
        default_factory=lambda: {
            "R": 2,
            "C": 2,
            "L": 2,
            "D": 2,
            "V": 2,
            "I": 2,
            "S": 4,
            "Q": 3,
            "M": 3,
            "X": 5,
            "0": 1,
        }
    )
    default_values: dict[str, str] = field(
        default_factory=lambda: {
            "R": "1k",
            "C": "1u",
            "L": "1m",
            "D": "DDEFAULT",
            "V": "DC 5",
            "I": "DC 1m",
            "S": "SWDEFAULT",
            "Q": "QDEFAULT",
            "M": "MDEFAULT",
            "X": "OPAMP_DEFAULT",
        }
    )
    model_statements: dict[str, str] = field(
        default_factory=lambda: {
            "DDEFAULT": ".model DDEFAULT D",
            "SWDEFAULT": ".model SWDEFAULT SW(Ron=1 Roff=1Meg Vt=2 Vh=0.2)",
            "QDEFAULT": ".model QDEFAULT NPN",
            "MDEFAULT": ".model MDEFAULT NMOS",
        }
    )


@dataclass(slots=True)
class AppConfig:
    """Application-wide settings."""

    models: ModelConfig = field(default_factory=ModelConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    spice: SpiceConfig = field(default_factory=SpiceConfig)
    output_dir: Path = ROOT_DIR / "data" / "output"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path | None) -> "AppConfig":
        """Load a partial YAML configuration and merge it into defaults."""

        config = cls()
        if path is None:
            return config

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.
            raise ImportError(
                "PyYAML is required to load YAML configuration files. "
                "Install dependencies from requirements.txt."
            ) from exc

        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        if not isinstance(raw, dict):
            raise ValueError("Configuration YAML must contain a mapping at top level.")

        _merge_dataclass(config, raw)
        return config

    def normalize_component_class(self, class_name: str) -> str:
        """Return a lower-case class key suitable for lookup tables."""

        return class_name.strip().lower().replace("-", "_").replace(" ", "_")

    def spice_prefix_for(self, class_name: str) -> str:
        """Return the configured SPICE prefix for a detector class."""

        key = self.normalize_component_class(class_name)
        return self.spice.class_to_prefix.get(key, key[:1].upper() or "X")

    def terminal_count_for_prefix(self, prefix: str) -> int:
        """Return the expected number of terminals for a SPICE prefix."""

        return self.spice.terminal_counts.get(prefix.upper(), 2)

    def is_ground_class(self, class_name: str) -> bool:
        """True if a detector class should be treated as a ground symbol."""

        key = self.normalize_component_class(class_name)
        return key in self.spice.ground_names or self.spice_prefix_for(key) == "0"


def _merge_dataclass(target: Any, source: dict[str, Any]) -> None:
    """Recursively merge a dictionary into nested dataclass instances."""

    for key, value in source.items():
        if not hasattr(target, key):
            raise ValueError(f"Unknown configuration key: {key}")

        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
            continue

        if isinstance(current, Path):
            value = Path(value)
        elif isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)

        setattr(target, key, value)
