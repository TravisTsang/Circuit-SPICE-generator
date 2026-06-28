# Training

This directory produces the four weight files the inference app expects but does
not ship:

```
models/hand_drawn_unet_trace_segmentation.pt
models/hand_drawn_yolov8_components.pt
models/digital_unet_trace_segmentation.pt
models/digital_yolov8_components.pt
```

There are **two models per domain** (hand-drawn, digital) and **two domains**, so
four trainings total. The U-Net segments wires; YOLOv8 detects components.

## Data strategy: public dataset + synthetic

| Model | Best data source | Why |
|-------|------------------|-----|
| YOLO detector (hand-drawn) | **CGHD** real boxes | 246k real bounding boxes, hand-drawn |
| YOLO detector (digital) | **synthetic** | clean CAD-style renders with free boxes |
| U-Net wires (both) | **synthetic** (primary) | synthetic gives *perfect* wire-only masks |
| U-Net wires (hand-drawn) | + CGHD approx masks (optional) | adds real-domain texture (weak labels) |

CGHD's segmentation maps cover *all* strokes (wires **and** components **and**
text), so they are not directly wire-only labels — `ingest/cghd_wire_masks.py`
approximates wire masks by erasing component boxes, but synthetic data is the
cleaner U-Net target.

## Training on a cloud GPU (recommended)

A laptop without a CUDA/strong GPU can do data prep and debugging, but the real
~4-model training belongs on a free cloud GPU. See [cloud/README.md](cloud/README.md)
and the one-shot [cloud/run_all.sh](cloud/run_all.sh) — clone the repo on Kaggle/
Colab, run one cell, download the four `models/*.pt` files. The steps below are
the same building blocks, broken out for local/iterative use.

## 0. Install

```bash
pip install -r requirements.txt          # core (torch, ultralytics, opencv, …)
pip install -r training/requirements-train.txt   # optional extras
```

The synthetic generator only needs numpy + opencv, so it runs before the heavy
ML stack is installed.

## 1. Generate synthetic data

```bash
python -m training.synthetic.generate --n 800 --domain digital
python -m training.synthetic.generate --n 800 --domain hand_drawn
```

Writes paired images + wire masks + YOLO labels into `datasets/trace_segmentation`
and `datasets/component_detection`. Inspect a few `masks/train/*.png` and overlay
`labels/train/*.txt` to confirm they look right before training.

## 2. (Optional) Ingest CGHD for real hand-drawn detection data

Download CGHD into `datasets/cghd_raw/` (from
<https://github.com/DFKI/cghd>, Zenodo, or Hugging Face `lowercaseonly/cghd`), then:

```bash
python -m training.ingest.cghd_to_yolo --src datasets/cghd_raw     # -> YOLO boxes
python -m training.ingest.cghd_wire_masks --src datasets/cghd_raw  # -> approx wire masks
```

`cghd_to_yolo` prints which CGHD classes it mapped vs skipped; extend `CLASS_MAP`
in that file to capture more. Splits are per-drafter to avoid train/val leakage.
If the stroke maps look inverted, re-run the mask script with `--invert`.

## 3. Train the U-Net wire segmenter

```bash
python -m training.unet.train_unet \
    --data datasets/trace_segmentation \
    --epochs 100 --batch 8 --device cuda \
    --out models/digital_unet_trace_segmentation.pt
```

The model architecture is imported from `statics_ocv.segmentation.UNet`, and the
checkpoint is saved in the `{model_kwargs, state_dict}` format the inference
loader understands (plus a `*_torchscript.pt` copy). Headline metric is **Dice**;
also watch precision (no hallucinated wires) and recall (no broken traces).

## 4. Train the YOLOv8 detector

```bash
python -m training.yolo.train_yolo \
    --data training/yolo/data.yaml \
    --model yolov8s.pt --imgsz 768 --epochs 100 --batch 8 \
    --out models/digital_yolov8_components.pt
```

Use `yolov8n.pt` for fast experiments. Class names in `data.yaml` must stay in
sync with `statics_ocv/config.py` (`class_to_prefix`).

## 5. Wire weights into the app

The backend resolves weights via env vars (see repo README). Point them at the
files you trained, e.g.:

```bash
export DIGITAL_UNET_WEIGHTS=$PWD/models/digital_unet_trace_segmentation.pt
export DIGITAL_YOLO_WEIGHTS=$PWD/models/digital_yolov8_components.pt
export HAND_DRAWN_UNET_WEIGHTS=$PWD/models/hand_drawn_unet_trace_segmentation.pt
export HAND_DRAWN_YOLO_WEIGHTS=$PWD/models/hand_drawn_yolov8_components.pt
```

## 6. Run end-to-end and inspect

```bash
python -m statics_ocv data/input/schematic.png \
    --output data/output/schematic.cir --dump-intermediates --log-level DEBUG
```

Check `*_trace_mask.png`, `*_skeleton.png`, and `*_topology.json`. The first
milestone is **"mask contains only wires, YOLO boxes cover every component"** —
graph recovery and SPICE export are easy to debug only once that holds.

## Class taxonomy

The 11 classes (ids fixed by `data.yaml` / `config.py`):

```
0 resistor   1 capacitor  2 inductor   3 diode
4 voltage_source  5 current_source  6 ground
7 switch  8 bjt  9 mosfet  10 opamp
```

The synthetic generator currently renders classes 0–6 (the others need glyphs in
`training/synthetic/glyphs.py`); CGHD supplies real boxes for 0–4, 6–10.
