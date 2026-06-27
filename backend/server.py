"""FastAPI web server for schematic-image inference.

This module is intentionally application-only: it loads already-trained models,
receives uploaded images, runs inference/evaluation, and returns structured
metadata to the React frontend. It contains no training loops, optimizers,
losses, or training dataset wrappers.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from statics_ocv.config import AppConfig, ROOT_DIR

if TYPE_CHECKING:
    from statics_ocv.detection import DetectionResult
    from statics_ocv.graph_builder import CircuitTopology
    from statics_ocv.netlist_exporter import NetlistResult
    from statics_ocv.segmentation import SegmentationResult

LOGGER = logging.getLogger(__name__)

CircuitDomain = Literal["hand-drawn", "digital"]

UPLOAD_DIR = ROOT_DIR / "data" / "uploads"
OUTPUT_DIR = ROOT_DIR / "data" / "output" / "web"

DOMAIN_ALIASES: dict[str, CircuitDomain] = {
    "hand-drawn": "hand-drawn",
    "hand_drawn": "hand-drawn",
    "handdrawn": "hand-drawn",
    "hand": "hand-drawn",
    "digital": "digital",
    "cad": "digital",
}


@dataclass(frozen=True, slots=True)
class DomainWeights:
    """U-Net and YOLO weight paths for one inference domain."""

    unet_path: Path
    yolo_path: Path


@dataclass(slots=True)
class InferenceBundle:
    """Cached model objects for one domain."""

    config: AppConfig
    segmenter: object
    detector: object
    graph_builder: object
    exporter: object
    lock: threading.Lock
    loaded_at: float


class DomainModelRegistry:
    """Cache domain-specific models and avoid repeated heavyweight loading."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[CircuitDomain, InferenceBundle] = {}
        self._weights = {
            "hand-drawn": DomainWeights(
                unet_path=Path(
                    os.getenv(
                        "HAND_DRAWN_UNET_WEIGHTS",
                        ROOT_DIR / "models" / "hand_drawn_unet_trace_segmentation.pt",
                    )
                ),
                yolo_path=Path(
                    os.getenv(
                        "HAND_DRAWN_YOLO_WEIGHTS",
                        ROOT_DIR / "models" / "hand_drawn_yolov8_components.pt",
                    )
                ),
            ),
            "digital": DomainWeights(
                unet_path=Path(
                    os.getenv(
                        "DIGITAL_UNET_WEIGHTS",
                        ROOT_DIR / "models" / "digital_unet_trace_segmentation.pt",
                    )
                ),
                yolo_path=Path(
                    os.getenv(
                        "DIGITAL_YOLO_WEIGHTS",
                        ROOT_DIR / "models" / "digital_yolov8_components.pt",
                    )
                ),
            ),
        }

    def get(self, domain: CircuitDomain) -> InferenceBundle:
        """Return cached model bundle, loading it once per domain if needed."""

        with self._lock:
            if domain in self._cache:
                return self._cache[domain]

            weights = self._weights[domain]
            missing = [
                str(path)
                for path in (weights.unet_path, weights.yolo_path)
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    "Missing model weight file(s) for "
                    f"{domain}: {', '.join(missing)}"
                )

            cfg = AppConfig()
            cfg.models.unet_path = weights.unet_path
            cfg.models.yolo_path = weights.yolo_path
            cfg.models.device = os.getenv("MODEL_DEVICE", cfg.models.device)
            cfg.output_dir = OUTPUT_DIR

            from statics_ocv.detection import ComponentDetector
            from statics_ocv.graph_builder import CircuitGraphBuilder
            from statics_ocv.netlist_exporter import NetlistExporter
            from statics_ocv.segmentation import TraceSegmenter

            LOGGER.info("Loading %s inference bundle.", domain)
            bundle = InferenceBundle(
                config=cfg,
                segmenter=TraceSegmenter(cfg),
                detector=ComponentDetector(cfg),
                graph_builder=CircuitGraphBuilder(cfg),
                exporter=NetlistExporter(cfg),
                lock=threading.Lock(),
                loaded_at=time.time(),
            )
            self._cache[domain] = bundle
            return bundle

    def status(self) -> dict[str, object]:
        """Return model path and cache status for health checks."""

        with self._lock:
            return {
                domain: {
                    "loaded": domain in self._cache,
                    "unet_path": str(weights.unet_path),
                    "yolo_path": str(weights.yolo_path),
                    "unet_exists": weights.unet_path.exists(),
                    "yolo_exists": weights.yolo_path.exists(),
                    "loaded_at": self._cache[domain].loaded_at
                    if domain in self._cache
                    else None,
                }
                for domain, weights in self._weights.items()
            }


app = FastAPI(title="Circuits OCR API", version="0.1.0")
registry = DomainModelRegistry()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    """Simple readiness endpoint for the frontend and local checks."""

    return {"ok": True, "models": registry.status()}


@app.post("/api/process")
async def process_circuit(
    image: UploadFile = File(...),
    domain: str = Form(...),
) -> JSONResponse:
    """Receive an image, run domain-specific inference, and return metadata."""

    normalized_domain = normalize_domain(domain)
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image file.")

    saved_path = save_upload(image)
    try:
        bundle = registry.get(normalized_domain)
        response = run_inference(saved_path, normalized_domain, bundle)
        return JSONResponse(response)
    except FileNotFoundError as exc:
        LOGGER.exception("Model weights are missing.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Inference failed.")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        saved_path.unlink(missing_ok=True)


def normalize_domain(domain: str) -> CircuitDomain:
    """Normalize frontend domain values into backend domain keys."""

    key = domain.strip().lower().replace(" ", "_")
    try:
        return DOMAIN_ALIASES[key]
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail="domain must be one of: hand-drawn, digital",
        ) from exc


def save_upload(upload: UploadFile) -> Path:
    """Persist an uploaded image temporarily under the workspace."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        suffix = ".png"

    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return path


def run_inference(
    image_path: Path,
    domain: CircuitDomain,
    bundle: InferenceBundle,
) -> dict[str, object]:
    """Run the cached inference pipeline and serialize the result."""

    import cv2

    start = time.perf_counter()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not decode uploaded image: {image_path.name}")
    height, width = image_bgr.shape[:2]

    with bundle.lock:
        segmentation = bundle.segmenter.segment_array(image_bgr)
        detections = bundle.detector.detect(image_path)
        topology = bundle.graph_builder.build(
            segmentation.binary_mask,
            detections.components,
        )
        netlist = bundle.exporter.export(topology, title=image_path.stem)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return serialize_response(
        image_path=image_path,
        domain=domain,
        image_width=width,
        image_height=height,
        elapsed_ms=elapsed_ms,
        segmentation=segmentation,
        detections=detections,
        topology=topology,
        netlist=netlist,
        bundle=bundle,
    )


def serialize_response(
    image_path: Path,
    domain: CircuitDomain,
    image_width: int,
    image_height: int,
    elapsed_ms: float,
    segmentation: "SegmentationResult",
    detections: "DetectionResult",
    topology: "CircuitTopology",
    netlist: "NetlistResult",
    bundle: InferenceBundle,
) -> dict[str, object]:
    """Convert pipeline dataclasses into frontend-friendly JSON."""

    return {
        "domain": domain,
        "filename": image_path.name,
        "image": {
            "width": image_width,
            "height": image_height,
        },
        "timing": {
            "elapsedMs": elapsed_ms,
            "modelsLoadedAt": bundle.loaded_at,
        },
        "weights": {
            "unet": str(bundle.config.models.unet_path),
            "yolo": str(bundle.config.models.yolo_path),
        },
        "segmentation": {
            "foregroundPixels": int(segmentation.binary_mask.sum()),
            "threshold": bundle.config.models.segmentation_threshold,
        },
        "components": [serialize_component(component) for component in topology.components],
        "ocr": [serialize_text(text) for text in detections.texts],
        "nets": [
            {"label": int(label), "name": name}
            for label, name in sorted(topology.net_names.items())
        ],
        "warnings": netlist.warnings,
        "netlist": netlist.text,
    }


def serialize_component(component) -> dict[str, object]:
    """Serialize one detected component with final graph assignments."""

    return {
        "name": component.element_name,
        "uid": component.uid,
        "className": component.class_name,
        "spicePrefix": component.spice_prefix,
        "confidence": round(component.confidence, 4),
        "value": component.value,
        "bbox": {
            "x1": component.bbox.x1,
            "y1": component.bbox.y1,
            "x2": component.bbox.x2,
            "y2": component.bbox.y2,
            "width": component.bbox.width,
            "height": component.bbox.height,
        },
        "terminalNets": component.terminal_nets,
        "terminalPoints": component.terminal_points,
        "ocrTexts": [serialize_text(text) for text in component.ocr_texts],
    }


def serialize_text(text) -> dict[str, object]:
    """Serialize an OCR text span."""

    return {
        "text": text.text,
        "confidence": round(text.confidence, 4),
        "bbox": {
            "x1": text.bbox.x1,
            "y1": text.bbox.y1,
            "x2": text.bbox.x2,
            "y2": text.bbox.y2,
            "width": text.bbox.width,
            "height": text.bbox.height,
        },
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
