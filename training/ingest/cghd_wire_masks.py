"""Derive *approximate* wire-only masks from CGHD stroke segmentation maps.

CGHD ships binary segmentation maps of ALL pen strokes (wires + component bodies
+ text), but the U-Net needs WIRES ONLY. This script approximates a wire mask by
taking the stroke map and erasing the interior of every non-wire bounding box
(components, text, junctions, etc.) from the matching VOC annotation.

This is a heuristic with known limitations:
  * Axis-aligned boxes erase wire pixels that pass *through* a component box.
  * Component leads just outside the box survive and look like wires.
So treat these masks as weak labels. For clean U-Net targets, prefer the
synthetic generator (training/synthetic/generate.py), and use these CGHD masks
as additional real-domain data if they help.

Run from the repo root (after downloading CGHD)::

    python -m training.ingest.cghd_wire_masks --src datasets/cghd_raw --val-frac 0.15
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from training.ingest.cghd_to_yolo import _drafter_of, find_image, parse_voc

REPO_ROOT = Path(__file__).resolve().parents[2]

# CGHD classes whose boxes should be erased from the stroke map to leave wires.
# Everything that is NOT a wire/junction/crossover is a non-wire structure.
WIRE_LIKE = {"junction", "crossover", "terminal"}  # keep these as "wire" pixels


def stroke_map_for(xml_path: Path) -> Path | None:
    stem = xml_path.stem
    seg_dir = xml_path.parent.parent / "segmentation"
    for ext in (".jpg", ".jpeg", ".png"):
        cand = seg_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Approximate wire-only masks from CGHD.")
    p.add_argument("--src", type=Path, default=REPO_ROOT / "datasets" / "cghd_raw")
    p.add_argument("--dst", type=Path, default=REPO_ROOT / "datasets" / "trace_segmentation")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--invert", action="store_true",
                   help="Set if CGHD strokes are black-on-white (so strokes become 255).")
    args = p.parse_args(argv)

    xml_files = sorted(args.src.rglob("annotations/*.xml")) or sorted(args.src.rglob("*.xml"))
    if not xml_files:
        print(f"No VOC annotations under {args.src}. Download CGHD first.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    drafters = sorted({_drafter_of(x) for x in xml_files})
    rng.shuffle(drafters)
    val_drafters = set(drafters[: max(1, int(len(drafters) * args.val_frac))])

    written = 0
    for xml_path in xml_files:
        seg_path = stroke_map_for(xml_path)
        if seg_path is None:
            continue  # only a subset of CGHD has segmentation maps
        img_path = find_image(xml_path)
        if img_path is None:
            continue

        try:
            w, h, objs = parse_voc(xml_path)
        except Exception:  # noqa: BLE001 - skip malformed
            continue

        stroke = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if stroke is None:
            continue
        stroke = cv2.resize(stroke, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (stroke < 127).astype(np.uint8) * 255 if args.invert else (stroke >= 127).astype(np.uint8) * 255

        # Erase the interior of every non-wire-like component/text box.
        for name, xmin, ymin, xmax, ymax in objs:
            base = name.split(".")[0]
            if name in WIRE_LIKE or base in WIRE_LIKE:
                continue
            x1, y1 = max(0, int(xmin)), max(0, int(ymin))
            x2, y2 = min(w, int(xmax)), min(h, int(ymax))
            mask[y1:y2, x1:x2] = 0

        split = "val" if _drafter_of(xml_path) in val_drafters else "train"
        out_img = args.dst / "images" / split
        out_msk = args.dst / "masks" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)

        stem = f"cghd_{img_path.stem}"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out_img / f"{stem}.png"), img)
        cv2.imwrite(str(out_msk / f"{stem}.png"), mask)
        written += 1

    print(f"[cghd-mask] wrote {written} approximate wire masks to {args.dst}")
    if written == 0:
        print("[cghd-mask] NOTE: only part of CGHD has segmentation maps (~284). "
              "If 0, those maps may be absent from your download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
