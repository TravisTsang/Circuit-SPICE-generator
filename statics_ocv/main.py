"""Command-line orchestration for image-to-SPICE conversion."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import AppConfig

LOGGER = logging.getLogger(__name__)


def run_pipeline(
    image_path: str | Path,
    output_path: str | Path | None = None,
    config: AppConfig | None = None,
    dump_intermediates: bool = False,
) -> Path:
    """Run the complete image-to-netlist pipeline.

    Returns the path to the generated netlist.
    """

    cfg = config or AppConfig()
    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Input image not found: {image}")

    out_path = Path(output_path) if output_path else cfg.output_dir / f"{image.stem}.cir"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from .detection import ComponentDetector
    from .graph_builder import CircuitGraphBuilder
    from .netlist_exporter import NetlistExporter
    from .segmentation import TraceSegmenter

    LOGGER.info("Starting schematic OCR pipeline for %s", image)
    segmenter = TraceSegmenter(cfg)
    segmentation = segmenter.segment_image(image)

    detector = ComponentDetector(cfg)
    detections = detector.detect(image)

    builder = CircuitGraphBuilder(cfg)
    topology = builder.build(segmentation.binary_mask, detections.components)

    exporter = NetlistExporter(cfg)
    result = exporter.write(topology, out_path, title=image.stem)

    if dump_intermediates:
        _dump_intermediates(
            cfg=cfg,
            image_path=image,
            segmentation_mask=segmentation.binary_mask,
            topology=topology,
            netlist_warnings=result.warnings,
            output_dir=out_path.parent,
        )
        segmenter.save_debug_outputs(segmentation, out_path.parent, image.stem)

    for warning in result.warnings:
        LOGGER.warning(warning)

    LOGGER.info("Pipeline complete: %s", out_path)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert noisy schematic images into SPICE netlists.",
    )
    parser.add_argument("image", type=Path, help="Path to schematic image.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .cir/.net path. Defaults to data/output/<image>.cir.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config overriding paths and thresholds.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Override torch device for U-Net inference.",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=None,
        help="Override YOLOv8 component model path.",
    )
    parser.add_argument(
        "--unet-model",
        type=Path,
        default=None,
        help="Override U-Net trace segmentation model path.",
    )
    parser.add_argument(
        "--dump-intermediates",
        action="store_true",
        help="Write debug masks, skeletons, and topology JSON beside the netlist.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Override logging level.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cfg = AppConfig.from_yaml(args.config)
        if args.device:
            cfg.models.device = args.device
        if args.yolo_model:
            cfg.models.yolo_path = args.yolo_model
        if args.unet_model:
            cfg.models.unet_path = args.unet_model
        if args.log_level:
            cfg.log_level = args.log_level

        configure_logging(cfg.log_level)
        output = run_pipeline(
            image_path=args.image,
            output_path=args.output,
            config=cfg,
            dump_intermediates=args.dump_intermediates,
        )
        print(output)
        return 0
    except Exception as exc:
        configure_logging("ERROR")
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _dump_intermediates(
    cfg: AppConfig,
    image_path: Path,
    segmentation_mask,
    topology,
    netlist_warnings: list[str],
    output_dir: Path,
) -> None:
    """Write debug artifacts for visual and topology inspection."""

    import cv2
    import numpy as np

    stem = image_path.stem
    skeleton_path = output_dir / f"{stem}_skeleton.png"
    labels_path = output_dir / f"{stem}_net_labels.npy"
    topology_path = output_dir / f"{stem}_topology.json"

    if topology.skeleton is not None:
        cv2.imwrite(str(skeleton_path), (topology.skeleton * 255).astype(np.uint8))
    if topology.net_label_image is not None:
        np.save(labels_path, topology.net_label_image)

    serializable = {
        "image": str(image_path),
        "mask_foreground_pixels": int(segmentation_mask.sum()),
        "net_names": {str(key): value for key, value in topology.net_names.items()},
        "components": [
            {
                "name": component.element_name,
                "uid": component.uid,
                "class_name": component.class_name,
                "prefix": component.spice_prefix,
                "value": component.value,
                "confidence": component.confidence,
                "bbox": {
                    "x1": component.bbox.x1,
                    "y1": component.bbox.y1,
                    "x2": component.bbox.x2,
                    "y2": component.bbox.y2,
                },
                "terminal_nets": component.terminal_nets,
                "terminal_points": component.terminal_points,
                "ocr_texts": [text.text for text in component.ocr_texts],
            }
            for component in topology.components
        ],
        "warnings": netlist_warnings,
        "settings": {
            "terminal_search_radius_px": cfg.graph.terminal_search_radius_px,
            "endpoint_gap_warning_px": cfg.graph.endpoint_gap_warning_px,
        },
    }
    topology_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
