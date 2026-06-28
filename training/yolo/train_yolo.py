"""Train the YOLOv8 component detector via Ultralytics.

Thin wrapper that trains and then copies the best weights to the path the web
backend expects. The eleven class names are defined in ``data.yaml`` and must
match ``statics_ocv/config.py``.

Usage (from the repo root)::

    python -m training.yolo.train_yolo \
        --data training/yolo/data.yaml \
        --model yolov8s.pt --imgsz 768 --epochs 100 --batch 8 \
        --out models/digital_yolov8_components.pt

The class list is the same for both domains; train one detector on hand-drawn
data and another on digital/CAD data, writing to the two domain-specific paths.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_ssl_certs() -> None:
    """Point urllib/requests at certifi so base-weight downloads don't fail.

    Fresh Python installs on macOS often can't verify GitHub's TLS cert, which
    breaks the automatic ``yolov8*.pt`` download. Using certifi's bundle fixes it.
    """
    try:
        import certifi
    except ImportError:
        return
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def _resolve_data_yaml(data_path: Path) -> Path:
    """Return a data.yaml whose ``path`` is absolute.

    Ultralytics resolves a relative ``path:`` against its global ``datasets_dir``
    (``~/datasets``), not the yaml's own location, which silently points training
    at the wrong directory. We rewrite ``path`` to an absolute directory so it
    works no matter where it is run from.
    """
    import yaml

    spec = yaml.safe_load(data_path.read_text())
    raw = Path(spec.get("path", "."))
    abs_path = raw if raw.is_absolute() else (data_path.parent / raw).resolve()
    if abs_path == raw:
        return data_path  # already absolute, nothing to do
    spec["path"] = str(abs_path)
    resolved = Path(tempfile.gettempdir()) / f"_resolved_{data_path.name}"
    resolved.write_text(yaml.safe_dump(spec, sort_keys=False))
    return resolved


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the YOLOv8 component detector.")
    p.add_argument("--data", type=Path, default=Path(__file__).with_name("data.yaml"))
    p.add_argument("--model", default="yolov8s.pt",
                   help="Base checkpoint (yolov8n.pt for fast experiments, yolov8s.pt for quality).")
    p.add_argument("--imgsz", type=int, default=768)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="e.g. 0, 0,1, or cpu. None = auto.")
    p.add_argument("--name", default="components", help="Ultralytics run name.")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "yolov8_components.pt",
                   help="Destination for the best weights (used by the inference app).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _ensure_ssl_certs()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. `pip install -r requirements.txt` first.", file=sys.stderr)
        return 1

    data_yaml = _resolve_data_yaml(args.data)
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
    )

    # Resolve the best.pt the run produced and copy it to the deployment path.
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        print(f"Could not find best weights at {best}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out)
    print(f"[yolo] copied {best} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
