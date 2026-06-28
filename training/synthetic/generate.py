"""Generate synthetic schematic images with perfectly aligned labels.

For each sample this writes:
  * an RGB image (white background, black strokes)
  * a wire-only binary mask (wire=255) -> U-Net target
  * a YOLO label file (component boxes) -> detector target

Output goes into both dataset trees so one render trains both models::

    datasets/trace_segmentation/{images,masks}/<split>/synth_*.png
    datasets/component_detection/{images,labels}/<split>/synth_*.{png,txt}

Usage (from the repo root)::

    python -m training.synthetic.generate --n 500 --domain digital
    python -m training.synthetic.generate --n 500 --domain hand_drawn

The class ids match training/yolo/data.yaml and statics_ocv/config.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from training.synthetic import glyphs

REPO_ROOT = Path(__file__).resolve().parents[2]

# Must match training/yolo/data.yaml names.
CLASS_IDS = {
    "resistor": 0, "capacitor": 1, "inductor": 2, "diode": 3,
    "voltage_source": 4, "current_source": 5, "ground": 6,
    "switch": 7, "bjt": 8, "mosfet": 9, "opamp": 10,
}
PLACEABLE = ["resistor", "capacitor", "inductor", "diode", "voltage_source", "current_source"]


def _draw_wire(img, mask, a, b, thickness, domain):
    """Draw a wire on both the image (black) and the mask (white)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if domain == "hand_drawn":
        # Break the segment into a slightly jittered polyline.
        n = 4
        prev = a
        for i in range(1, n + 1):
            t = i / n
            pt = a + (b - a) * t
            if i < n:
                pt = pt + np.random.normal(0, 1.2, 2)
            cv2.line(img, tuple(prev.astype(int)), tuple(pt.astype(int)), glyphs.BLACK, thickness, cv2.LINE_AA)
            cv2.line(mask, tuple(prev.astype(int)), tuple(pt.astype(int)), 255, thickness, cv2.LINE_AA)
            prev = pt
    else:
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), glyphs.BLACK, thickness, cv2.LINE_AA)
        cv2.line(mask, tuple(a.astype(int)), tuple(b.astype(int)), 255, thickness, cv2.LINE_AA)


def _post_process(img, domain):
    """Domain-specific image degradation. The mask is never degraded."""
    if domain == "hand_drawn":
        # Paper-like texture + noise + slight blur to mimic photos of sketches.
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.6:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        tint = np.random.randint(235, 256, (1, 1, 3), dtype=np.uint8)
        img = np.where(img > 200, tint, img).astype(np.uint8)
    else:
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def generate_one(size: int, domain: str, rng: random.Random):
    """Render one schematic; return (image_bgr, wire_mask, yolo_lines)."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    labels: list[tuple[int, float, float, float, float]] = []

    scale = max(1, size // 256)
    thickness = 2 * scale

    margin = int(size * 0.12)
    cols = rng.randint(2, 4)
    rows = rng.randint(2, 4)
    xs = np.linspace(margin, size - margin, cols + 1)
    ys = np.linspace(margin, size - margin, rows + 1)
    nodes = {(i, j): np.array([xs[i], ys[j]]) for i in range(cols + 1) for j in range(rows + 1)}

    # Candidate edges between adjacent grid nodes.
    edges = []
    for i in range(cols + 1):
        for j in range(rows + 1):
            if i + 1 <= cols:
                edges.append(((i, j), (i + 1, j)))
            if j + 1 <= rows:
                edges.append(((i, j), (i, j + 1)))
    rng.shuffle(edges)

    used = edges[: max(3, int(len(edges) * rng.uniform(0.45, 0.8)))]
    component_count = 0
    for (na, nb) in used:
        a, b = nodes[na], nodes[nb]
        place = rng.random() < 0.55 and component_count < 8
        if not place:
            _draw_wire(img, mask, a, b, thickness, domain)
            continue

        cls = rng.choice(PLACEABLE)
        u = (b - a) / (np.linalg.norm(b - a) + 1e-6)
        length = np.linalg.norm(b - a)
        glyph_len = min(length * 0.5, 60 * scale)
        mid = (a + b) / 2
        t1 = mid - u * (glyph_len / 2)
        t2 = mid + u * (glyph_len / 2)

        # Wires from nodes to the component terminals (on image + mask).
        _draw_wire(img, mask, a, t1, thickness, domain)
        _draw_wire(img, mask, t2, b, thickness, domain)

        # Glyph body (image only).
        x1, y1, x2, y2 = glyphs.TWO_TERMINAL[cls](img, t1, t2, thickness, scale)
        labels.append(_to_yolo(CLASS_IDS[cls], x1, y1, x2, y2, size))
        component_count += 1

    # Attach a ground symbol to a random bottom node.
    if rng.random() < 0.85:
        gi = rng.randint(0, cols)
        gnode = nodes[(gi, rows)]
        direction = np.array([0.0, 1.0])
        (gx1, gy1, gx2, gy2), stub_top = glyphs.ground(img, gnode, direction, thickness, scale)
        _draw_wire(img, mask, gnode, stub_top, thickness, domain)
        labels.append(_to_yolo(CLASS_IDS["ground"], gx1, gy1, gx2, gy2, size))

    img = _post_process(img, domain)
    return img, mask, labels


def _to_yolo(cls_id, x1, y1, x2, y2, size):
    x1, x2 = sorted((max(0, x1), min(size, x2)))
    y1, y2 = sorted((max(0, y1), min(size, y2)))
    xc = (x1 + x2) / 2 / size
    yc = (y1 + y2) / 2 / size
    w = (x2 - x1) / size
    h = (y2 - y1) / size
    return (cls_id, xc, yc, w, h)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate synthetic schematic training data.")
    p.add_argument("--n", type=int, default=300, help="Number of images to generate.")
    p.add_argument("--domain", choices=["digital", "hand_drawn"], default="digital")
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seg-root", type=Path, default=REPO_ROOT / "datasets" / "trace_segmentation")
    p.add_argument("--det-root", type=Path, default=REPO_ROOT / "datasets" / "component_detection")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    for i in range(args.n):
        split = "val" if rng.random() < args.val_frac else "train"
        img, mask, labels = generate_one(args.size, args.domain, rng)
        stem = f"synth_{args.domain}_{i:05d}"

        seg_img = args.seg_root / "images" / split
        seg_msk = args.seg_root / "masks" / split
        det_img = args.det_root / "images" / split
        det_lbl = args.det_root / "labels" / split
        for d in (seg_img, seg_msk, det_img, det_lbl):
            d.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(seg_img / f"{stem}.png"), img)
        cv2.imwrite(str(seg_msk / f"{stem}.png"), mask)
        cv2.imwrite(str(det_img / f"{stem}.png"), img)
        with (det_lbl / f"{stem}.txt").open("w") as fh:
            for cls_id, xc, yc, w, h in labels:
                fh.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        if (i + 1) % 50 == 0:
            print(f"[synth] {i + 1}/{args.n} ({args.domain})")

    print(f"[synth] done: {args.n} images for domain '{args.domain}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
