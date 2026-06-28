"""Convert the CGHD hand-drawn circuit dataset into YOLO detection labels.

CGHD: "A Public Ground-Truth Dataset for Handwritten Circuit Diagram Images"
(DFKI/cghd). ~3.2k images, PASCAL VOC bounding boxes, ~50 fine-grained classes.

This collapses CGHD's fine-grained taxonomy into the 11 classes the inference
app uses (see statics_ocv/config.py and training/yolo/data.yaml). CGHD classes
not present in CLASS_MAP (text, junction, logic gates, ICs, etc.) are skipped;
their counts are printed so you can extend the mapping if needed.

Download CGHD first (e.g. `git clone https://github.com/DFKI/cghd` or from
Zenodo/HuggingFace) into datasets/cghd_raw/, then run from the repo root::

    python -m training.ingest.cghd_to_yolo --src datasets/cghd_raw --val-frac 0.15

Splitting is done per-drafter so the same hand never appears in both train and
val (prevents optimistic metrics).
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# CGHD fine-grained class -> our coarse class (training/yolo/data.yaml names).
CLASS_MAP = {
    "gnd": "ground",
    "vss": "ground",
    "voltage.dc": "voltage_source",
    "voltage.ac": "voltage_source",
    "voltage.battery": "voltage_source",
    "resistor": "resistor",
    "resistor.adjustable": "resistor",
    "resistor.photo": "resistor",
    "varistor": "resistor",
    "capacitor.unpolarized": "capacitor",
    "capacitor.polarized": "capacitor",
    "inductor": "inductor",
    "diode": "diode",
    "diode.light_emitting": "diode",
    "diode.zener": "diode",
    "diode.thyrector": "diode",
    "transistor.bjt": "bjt",
    "transistor.fet": "mosfet",
    "operational_amplifier": "opamp",
    "operational_amplifier.schmitt_trigger": "opamp",
    "switch": "switch",
}

NAME_TO_ID = {
    "resistor": 0, "capacitor": 1, "inductor": 2, "diode": 3,
    "voltage_source": 4, "current_source": 5, "ground": 6,
    "switch": 7, "bjt": 8, "mosfet": 9, "opamp": 10,
}


def _drafter_of(xml_path: Path) -> str:
    """Best-effort drafter id from the path (…/drafter_5/annotations/…)."""
    for part in xml_path.parts:
        if part.lower().startswith("drafter"):
            return part
    return "unknown"


def parse_voc(xml_path: Path):
    """Return (width, height, [(class_name, xmin, ymin, xmax, ymax), ...])."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.findtext("width")))
    h = int(float(size.findtext("height")))
    objs = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        objs.append((name, xmin, ymin, xmax, ymax))
    return w, h, objs


def find_image(xml_path: Path) -> Path | None:
    """Locate the image matching an annotation file."""
    stem = xml_path.stem
    images_dir = xml_path.parent.parent / "images"
    for ext in (".jpg", ".jpeg", ".png"):
        cand = images_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    # Fallback: search the drafter folder.
    for ext in (".jpg", ".jpeg", ".png"):
        hits = list(xml_path.parent.parent.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert CGHD VOC annotations to YOLO labels.")
    p.add_argument("--src", type=Path, default=REPO_ROOT / "datasets" / "cghd_raw",
                   help="Root of the downloaded CGHD dataset (contains drafter_* folders).")
    p.add_argument("--dst", type=Path, default=REPO_ROOT / "datasets" / "component_detection")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    xml_files = sorted(args.src.rglob("annotations/*.xml")) or sorted(args.src.rglob("*.xml"))
    if not xml_files:
        print(f"No VOC .xml annotations found under {args.src}. Download CGHD there first.",
              file=sys.stderr)
        return 1
    print(f"[cghd] found {len(xml_files)} annotation files")

    # Assign drafters to train/val so a hand never spans both splits.
    rng = random.Random(args.seed)
    drafters = sorted({_drafter_of(x) for x in xml_files})
    rng.shuffle(drafters)
    n_val = max(1, int(len(drafters) * args.val_frac))
    val_drafters = set(drafters[:n_val])
    print(f"[cghd] {len(drafters)} drafters; {n_val} held out for val")

    mapped, skipped = Counter(), Counter()
    written = 0
    for xml_path in xml_files:
        try:
            w, h, objs = parse_voc(xml_path)
        except ET.ParseError as exc:
            print(f"[cghd] skip unparseable {xml_path.name}: {exc}", file=sys.stderr)
            continue
        img_path = find_image(xml_path)
        if img_path is None:
            print(f"[cghd] no image for {xml_path.name}", file=sys.stderr)
            continue

        lines = []
        for name, xmin, ymin, xmax, ymax in objs:
            coarse = CLASS_MAP.get(name)
            if coarse is None:
                skipped[name] += 1
                continue
            mapped[coarse] += 1
            xc = (xmin + xmax) / 2 / w
            yc = (ymin + ymax) / 2 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            lines.append(f"{NAME_TO_ID[coarse]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue  # nothing of interest in this image

        split = "val" if _drafter_of(xml_path) in val_drafters else "train"
        out_img_dir = args.dst / "images" / split
        out_lbl_dir = args.dst / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        stem = f"cghd_{img_path.stem}"
        shutil.copy2(img_path, out_img_dir / f"{stem}{img_path.suffix}")
        (out_lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1

    print(f"\n[cghd] wrote {written} labeled images")
    print("[cghd] mapped boxes per class:")
    for cls, n in mapped.most_common():
        print(f"         {cls:16s} {n}")
    if skipped:
        print("[cghd] skipped CGHD classes (extend CLASS_MAP to include):")
        for cls, n in skipped.most_common(20):
            print(f"         {cls:32s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
