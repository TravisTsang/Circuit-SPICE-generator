"""YOLOv8 component detection and EasyOCR text association."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import AppConfig

LOGGER = logging.getLogger(__name__)

REFERENCE_RE = re.compile(r"^(?P<prefix>[RCLVDIQMSX])\s*(?P<number>\d+)$", re.IGNORECASE)
NUMERIC_VALUE_RE = re.compile(
    r"(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<suffix>meg|ohm|Ω|[munpfkKMGVAFH]?)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class BoundingBox:
    """Axis-aligned image-space bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def expanded(self, pixels: float) -> "BoundingBox":
        return BoundingBox(
            self.x1 - pixels,
            self.y1 - pixels,
            self.x2 + pixels,
            self.y2 + pixels,
        )

    def distance_to_point(self, point: tuple[float, float]) -> float:
        """Euclidean distance from a point to this rectangle, zero if inside."""

        px, py = point
        dx = max(self.x1 - px, 0.0, px - self.x2)
        dy = max(self.y1 - py, 0.0, py - self.y2)
        return math.hypot(dx, dy)


@dataclass(slots=True)
class OCRText:
    """Text span detected by EasyOCR."""

    text: str
    bbox: BoundingBox
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center


@dataclass(slots=True)
class ComponentDetection:
    """Detected component plus OCR-derived metadata."""

    uid: str
    class_name: str
    spice_prefix: str
    bbox: BoundingBox
    confidence: float
    label: str | None = None
    value: str | None = None
    ocr_texts: list[OCRText] = field(default_factory=list)
    terminal_nets: list[str] = field(default_factory=list)
    terminal_points: list[tuple[int, int] | None] = field(default_factory=list)

    @property
    def is_ground(self) -> bool:
        return self.spice_prefix == "0"

    @property
    def element_name(self) -> str:
        return self.label or self.uid


@dataclass(slots=True)
class DetectionResult:
    """All detector outputs required by later stages."""

    components: list[ComponentDetection]
    texts: list[OCRText]


class ComponentDetector:
    """Detect schematic symbols and associate nearby OCR text."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.yolo_model = self._load_yolo(config.models.yolo_path)
        try:
            import easyocr
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.
            raise ImportError(
                "easyocr is required. Install dependencies from requirements.txt."
            ) from exc
        self._cv2 = self._load_cv2()
        self.ocr_reader = easyocr.Reader(
            list(config.models.easyocr_languages),
            gpu=config.models.easyocr_gpu,
            verbose=False,
        )

    def detect(self, image_path: str | Path) -> DetectionResult:
        """Run YOLOv8 and EasyOCR on an input image."""

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        image_bgr = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"OpenCV could not read image: {path}")

        components = self._detect_components(image_bgr)
        texts = self._detect_text(image_bgr)
        self._associate_text(components, texts)
        LOGGER.info("Detected %d components and %d OCR spans.", len(components), len(texts))
        return DetectionResult(components=components, texts=texts)

    def _load_yolo(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"YOLOv8 component weights not found: {path}. Train or copy the model there, "
                "or override models.yolo_path in a YAML config."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.
            raise ImportError(
                "ultralytics is required for YOLOv8 detection. "
                "Install dependencies from requirements.txt."
            ) from exc
        LOGGER.info("Loading YOLOv8 component detector from %s", path)
        return YOLO(str(path))

    @staticmethod
    def _load_cv2():
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency is in requirements.
            raise ImportError(
                "opencv-python is required for image loading. "
                "Install dependencies from requirements.txt."
            ) from exc
        return cv2

    def _detect_components(self, image_bgr) -> list[ComponentDetection]:
        """Run YOLO and normalize detections into component records."""

        result_batch = self.yolo_model.predict(
            source=image_bgr,
            conf=self.config.models.yolo_confidence,
            iou=self.config.models.yolo_iou,
            verbose=False,
        )
        if not result_batch:
            return []

        result = result_batch[0]
        names = result.names
        counters: dict[str, int] = {}
        components: list[ComponentDetection] = []

        if result.boxes is None:
            return components

        for box in result.boxes:
            xyxy = box.xyxy.detach().cpu().numpy().reshape(-1)
            if xyxy.size != 4:
                LOGGER.warning("Skipping malformed YOLO box with shape %s", xyxy.shape)
                continue

            cls_id = int(box.cls.detach().cpu().item())
            confidence = float(box.conf.detach().cpu().item())
            class_name = str(names.get(cls_id, cls_id))
            prefix = self.config.spice_prefix_for(class_name)
            counter_key = prefix if prefix != "0" else "GND"
            counters[counter_key] = counters.get(counter_key, 0) + 1
            uid = f"{counter_key}{counters[counter_key]}" if prefix != "0" else f"GND{counters[counter_key]}"

            components.append(
                ComponentDetection(
                    uid=uid,
                    class_name=self.config.normalize_component_class(class_name),
                    spice_prefix=prefix,
                    bbox=BoundingBox(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    confidence=confidence,
                )
            )

        return components

    def _detect_text(self, image_bgr) -> list[OCRText]:
        """Run EasyOCR and convert quadrilateral boxes to axis-aligned boxes."""

        image_rgb = self._cv2.cvtColor(image_bgr, self._cv2.COLOR_BGR2RGB)
        raw_results = self.ocr_reader.readtext(image_rgb, detail=1, paragraph=False)
        texts: list[OCRText] = []
        for quad, text, confidence in raw_results:
            cleaned = clean_ocr_text(text)
            if not cleaned:
                continue
            xs = [float(point[0]) for point in quad]
            ys = [float(point[1]) for point in quad]
            texts.append(
                OCRText(
                    text=cleaned,
                    bbox=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
                    confidence=float(confidence),
                )
            )
        return texts

    def _associate_text(
        self,
        components: list[ComponentDetection],
        texts: list[OCRText],
    ) -> None:
        """Attach nearby reference designators and values to components."""

        if not components or not texts:
            self._fill_missing_component_names(components)
            return

        used_reference_text_ids: set[int] = set()
        max_distance = self.config.graph.max_text_association_distance_px

        for component in sorted(components, key=lambda c: c.bbox.area, reverse=True):
            nearby = sorted(
                (
                    (self._association_score(component, text), idx, text)
                    for idx, text in enumerate(texts)
                    if component.bbox.distance_to_point(text.center) <= max_distance
                ),
                key=lambda item: item[0],
            )

            for _, idx, text in nearby[:4]:
                component.ocr_texts.append(text)
                parsed_reference = parse_reference_designator(text.text)
                if (
                    parsed_reference
                    and parsed_reference[0] == component.spice_prefix
                    and idx not in used_reference_text_ids
                ):
                    component.label = f"{parsed_reference[0]}{parsed_reference[1]}"
                    used_reference_text_ids.add(idx)
                    continue

                parsed_value = normalize_value_for_spice(text.text, component.spice_prefix)
                if parsed_value and component.value is None:
                    component.value = parsed_value

        self._fill_missing_component_names(components)

    def _fill_missing_component_names(self, components: Iterable[ComponentDetection]) -> None:
        """Ensure non-ground components have stable SPICE element names."""

        used = {
            component.label.upper()
            for component in components
            if component.label and not component.is_ground
        }
        counters: dict[str, int] = {}
        for component in components:
            if component.is_ground:
                continue
            if component.label:
                continue
            prefix = component.spice_prefix.upper()
            number = counters.get(prefix, 0) + 1
            while f"{prefix}{number}" in used:
                number += 1
            counters[prefix] = number
            component.label = f"{prefix}{number}"
            used.add(component.label.upper())

    @staticmethod
    def _association_score(component: ComponentDetection, text: OCRText) -> float:
        """Rank text spans by proximity to a component center and rectangle edge."""

        cx, cy = component.bbox.center
        tx, ty = text.center
        center_distance = math.hypot(cx - tx, cy - ty)
        edge_distance = component.bbox.distance_to_point(text.center)
        return edge_distance * 0.75 + center_distance * 0.25


def clean_ocr_text(text: str) -> str:
    """Normalize common OCR artifacts while preserving SPICE-relevant symbols."""

    cleaned = text.strip()
    cleaned = cleaned.replace("O", "0") if re.search(r"\d[O]|[O]\d", cleaned) else cleaned
    cleaned = cleaned.replace("Ω", "Ω")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned


def parse_reference_designator(text: str) -> tuple[str, int] | None:
    """Parse OCR text such as R1 or C23 into a SPICE prefix and number."""

    match = REFERENCE_RE.match(clean_ocr_text(text))
    if not match:
        return None
    return match.group("prefix").upper(), int(match.group("number"))


def normalize_value_for_spice(text: str, prefix: str) -> str | None:
    """Convert OCR values into SPICE-friendly values.

    SPICE accepts engineering suffixes such as ``k``, ``u``, and ``meg``. This
    function preserves those forms, strips unit words where safe, and handles
    simple source annotations such as ``5V`` or ``10mA``.
    """

    cleaned = clean_ocr_text(text)
    if parse_reference_designator(cleaned):
        return None

    source_prefixes = {"V", "I"}
    match = NUMERIC_VALUE_RE.search(cleaned)
    if not match:
        if prefix in {"D", "Q", "M", "X"} and re.match(r"^[A-Za-z][\w.-]+$", cleaned):
            return cleaned.upper()
        return None

    number = match.group("number")
    suffix = (match.group("suffix") or "").lower()
    suffix_map = {
        "ω": "",
        "ohm": "",
        "v": "",
        "a": "",
        "f": "",
        "h": "",
        "k": "k",
        "m": "m",
        "u": "u",
        "n": "n",
        "p": "p",
        "meg": "meg",
        "g": "G",
    }
    spice_suffix = suffix_map.get(suffix, suffix)
    value = f"{number}{spice_suffix}"
    if prefix.upper() in source_prefixes and not cleaned.upper().startswith(("DC", "AC", "SIN", "PULSE")):
        return f"DC {value}"
    return value
