"""Train the U-Net wire-trace segmenter.

The model architecture is imported directly from the inference package
(``statics_ocv.segmentation.UNet``) so trained weights always load cleanly. The
checkpoint is saved as ``{"model_kwargs": ..., "state_dict": ...}`` which is the
format ``TraceSegmenter._load_model`` understands; a TorchScript copy is also
exported for the safest deployment path.

Usage (from the repo root)::

    python -m training.unet.train_unet \
        --data datasets/trace_segmentation \
        --epochs 100 --batch 8 --device cuda \
        --out models/digital_unet_trace_segmentation.pt
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Make the repo root importable when run as a script or module.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statics_ocv.segmentation import UNet  # noqa: E402  reuse inference architecture
from training.common.preprocessing import IMAGE_SIZE  # noqa: E402
from training.unet.dataset import TraceSegmentationDataset  # noqa: E402
from training.unet.losses import BCEDiceLoss  # noqa: E402
from training.unet.metrics import SegMetrics, compute_metrics  # noqa: E402

# Keep in sync with statics_ocv.segmentation.UNet defaults.
MODEL_KWARGS = {"in_channels": 3, "out_channels": 1, "features": (32, 64, 128, 256)}


def pick_device() -> str:
    """Prefer CUDA, then Apple MPS (M-series), then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the U-Net wire segmenter.")
    p.add_argument("--data", type=Path, default=REPO_ROOT / "datasets" / "trace_segmentation",
                   help="Dataset root containing images/ and masks/.")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "unet_trace_segmentation.pt",
                   help="Where to write the best state-dict checkpoint.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=pick_device(),
                   help="cuda | mps | cpu (auto-detected).")
    p.add_argument("--pos-weight", type=float, default=None,
                   help="Positive-class weight for BCE; raise if wires are missed.")
    p.add_argument("--no-torchscript", action="store_true",
                   help="Skip exporting a TorchScript copy alongside the checkpoint.")
    p.add_argument("--amp", action="store_true",
                   help="Enable CUDA automatic mixed precision (~half the VRAM, ~same quality). "
                        "No-op on non-CUDA devices; weights stay fp32.")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                   help="AMP compute dtype. bf16 (default; RTX 30/40-series and newer) needs no "
                        "gradient scaler; fp16 enables one automatically.")
    return p


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str) -> SegMetrics:
    model.eval()
    tp = fp = fn = 0.0
    for images, masks in loader:
        logits = model(images.to(device))
        preds = (torch.sigmoid(logits) >= 0.5).float().cpu()
        tp += (preds * masks).sum().item()
        fp += (preds * (1 - masks)).sum().item()
        fn += ((1 - preds) * masks).sum().item()
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return SegMetrics(dice=dice, precision=precision, recall=recall)


def save_checkpoint(model: torch.nn.Module, out_path: Path, device: str, export_ts: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_kwargs": MODEL_KWARGS, "state_dict": model.state_dict()}, out_path)
    print(f"[train] saved checkpoint -> {out_path}")

    if export_ts:
        model.eval()
        example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        with torch.no_grad():
            scripted = torch.jit.trace(model, example)
        ts_path = out_path.with_name(out_path.stem + "_torchscript.pt")
        scripted.save(str(ts_path))
        print(f"[train] saved TorchScript -> {ts_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = args.device
    print(f"[train] device={device} epochs={args.epochs} batch={args.batch} lr={args.lr}")

    train_ds = TraceSegmentationDataset(args.data, "train", size=args.imgsz, augment=True)
    val_ds = TraceSegmentationDataset(args.data, "val", size=args.imgsz, augment=False)
    pin = device == "cuda"  # pin_memory only helps CUDA; warns/no-ops elsewhere
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    model = UNet(**MODEL_KWARGS).to(device)
    criterion = BCEDiceLoss(pos_weight=args.pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision: autocast the forward/loss, and (fp16 only) scale gradients
    # to avoid underflow. bf16 has fp32's exponent range so it needs no scaler.
    use_amp = args.amp and device == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    need_scaler = use_amp and amp_dtype is torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=need_scaler)
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
    if args.amp and not use_amp:
        print(f"[train] --amp ignored: AMP needs CUDA but device={device}; training in fp32.")
    elif use_amp:
        print(f"[train] AMP on: dtype={args.amp_dtype}, grad-scaler={'on' if need_scaler else 'off'}")

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with amp_ctx:
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        scheduler.step()

        metrics = evaluate(model, val_loader, device)
        avg_loss = running / max(1, len(train_loader))
        print(f"[train] epoch {epoch:03d}/{args.epochs}  loss={avg_loss:.4f}  val {metrics}")

        if metrics.dice > best_dice:
            best_dice = metrics.dice
            save_checkpoint(model, args.out, device, export_ts=not args.no_torchscript)
            print(f"[train]   new best dice={best_dice:.4f}")

    print(f"[train] done. best val dice={best_dice:.4f}  weights at {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
