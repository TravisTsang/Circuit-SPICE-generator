# Cloud training (free GPU)

Neither target laptop has a training GPU, so train the four models on a free
cloud GPU and copy the weights back. The code is plain PyTorch + Ultralytics and
runs unchanged. [run_all.sh](run_all.sh) does everything: install → generate
synthetic data → (optional CGHD) → train all four models.

The synthetic generator is fast and seeded, so we **regenerate data on the GPU
box** rather than uploading it. Only CGHD (optional, real hand-drawn data) needs
downloading there.

Recommended free option: **Kaggle Notebooks** (30 GPU-hrs/week, P100/T4 — the
most generous free sustained tier). Google Colab also works.

---

## Option A — Kaggle Notebook (recommended)

1. Create a new Notebook, then **Settings → Accelerator → GPU (T4 ×2 or P100)**.
2. In a cell:

   ```python
   !git clone https://github.com/TravisTsang/Circuit-SPICE-generator.git
   %cd Circuit-SPICE-generator
   !bash training/cloud/run_all.sh
   ```

   Tune via env vars, e.g. a quick first pass:

   ```python
   !N_SYNTH=1500 EPOCHS_UNET=50 EPOCHS_YOLO=60 bash training/cloud/run_all.sh
   ```

3. When it finishes, the four `models/*.pt` files are the deliverable. Download
   them (right-click in the file browser) or save them as a Kaggle Dataset
   output.

> Note: if you push your local synthetic data or trained code, remember `.venv/`,
> `runs/`, and generated `datasets/` are gitignored — the cloud box regenerates
> data itself, so a plain `git clone` is all you need.

## Option B — Google Colab

1. **Runtime → Change runtime type → GPU (T4)**.
2. Same three lines as above in a cell. To keep weights after the session,
   mount Drive first and copy `models/*.pt` there:

   ```python
   from google.colab import drive; drive.mount('/content/drive')
   # ... run training ...
   !cp models/*.pt /content/drive/MyDrive/
   ```

## Option C — Rented GPU (vast.ai / RunPod / Lambda, ~$0.30/hr)

SSH in, then:

```bash
git clone https://github.com/TravisTsang/Circuit-SPICE-generator.git
cd Circuit-SPICE-generator
bash training/cloud/run_all.sh
scp models/*.pt you@yourmachine:/path/to/Circuit-SPICE-generator/models/   # or use the provider's file UI
```

---

## Including CGHD (real hand-drawn detection data)

CGHD improves the hand-drawn YOLO detector. On the GPU box, download it into
`datasets/cghd_raw/` first, then set `USE_CGHD=1`:

```python
# Kaggle: add the "lowercaseonly/cghd" dataset to the notebook, then symlink:
!ln -s /kaggle/input/cghd datasets/cghd_raw
!USE_CGHD=1 bash training/cloud/run_all.sh
```

```bash
# Generic: git clone is large (~several GB)
git clone https://github.com/DFKI/cghd datasets/cghd_raw
USE_CGHD=1 bash training/cloud/run_all.sh
```

## Expected time (rough, single T4)

| Model | ~time |
|-------|-------|
| YOLOv8s, 100 epochs, ~2k imgs | ~30–60 min |
| U-Net, 80 epochs, ~2k imgs @768 | ~1–2 hr |

Four models ≈ a few hours. Start with fewer epochs / `yolov8n.pt` for a fast
first end-to-end pass, then scale up.

## After training: use the weights

Put the four files in `models/` on your inference machine and point the backend
at them (see repo README), or run the CLI per domain:

```bash
export SSL_CERT_FILE=$(python -m certifi)   # macOS cert fix for EasyOCR download
python -m statics_ocv data/input/schematic.png \
    --output data/output/schematic.cir \
    --unet-model models/digital_unet_trace_segmentation.pt \
    --yolo-model models/digital_yolov8_components.pt \
    --dump-intermediates
```
