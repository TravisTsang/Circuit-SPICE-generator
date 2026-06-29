#!/usr/bin/env bash
# One-shot HAND-DRAWN training: build data, train the U-Net + YOLO pair, and run
# an end-to-end pipeline check (image -> .cir) to prove everything is wired up.
#
# Produces:
#   models/hand_drawn_unet_trace_segmentation.pt   (+ _torchscript.pt)
#   models/hand_drawn_yolov8_components.pt
#
# Two modes:
#   SMOKE=1  -> fast end-to-end validation (~3-5 min): tiny data, 2 epochs, runs
#              inference and checks a netlist is produced. Use this FIRST.
#   (default)-> the real run: full data + epochs.
#
# CGHD (real hand-drawn boxes) is optional but recommended for the detector.
# Download Zenodo record 17469897 into datasets/cghd_raw/; this auto-detects it.
#
# Usage (from anywhere; the script cd's to the repo root):
#     SMOKE=1 bash training/train_hand_drawn.sh     # validate first
#     bash training/train_hand_drawn.sh             # then the real run
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE="${SMOKE:-0}"
PY="${PY:-python}"
YOLO_BASE="${YOLO_BASE:-yolov8s.pt}"
WORKERS="${WORKERS:-4}"             # set WORKERS=0 if the U-Net DataLoader stalls
IMG="${IMG:-768}"
BATCH_UNET="${BATCH_UNET:-8}"       # 8 fits 8GB VRAM thanks to AMP; drop to 4 if OOM
BATCH_YOLO="${BATCH_YOLO:-8}"       # 8 is safe on 8GB; try 16 on a bigger card

if [ "$SMOKE" = "1" ]; then
  N_SYNTH="${N_SYNTH:-60}"; EPOCHS="${EPOCHS:-2}"
  echo "############################################################"
  echo "#  SMOKE TEST — validates the pipeline runs end to end.    #"
  echo "#  The netlist will NOT be accurate (only 2 epochs); this  #"
  echo "#  just proves data -> train -> weights -> inference works.#"
  echo "############################################################"
else
  N_SYNTH="${N_SYNTH:-1500}"; EPOCHS="${EPOCHS:-100}"
fi

echo "==> Python: $($PY --version)"
$PY -c "import torch; print('==> torch', torch.__version__, '| built for CUDA:', torch.version.cuda)" \
  || { echo "ERROR: PyTorch not importable. Run: $PY -m pip install -r requirements.txt" >&2; exit 1; }

# Hard-require a real CUDA GPU — this targets the RTX 4060, not CPU.
if ! $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: torch.cuda.is_available() is False — you have the CPU-only PyTorch wheel." >&2
  echo "       Install the CUDA build, then retry:" >&2
  echo "         $PY -m pip uninstall -y torch torchvision" >&2
  echo "         $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124" >&2
  exit 1
fi
$PY -c "import torch; print('==> GPU:', torch.cuda.get_device_name(0))"

# TLS fix so Ultralytics can auto-download the yolov8 base weights.
if $PY -c "import certifi" 2>/dev/null; then
  SSL_CERT_FILE="$($PY -m certifi)"; export SSL_CERT_FILE
  export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
fi

echo ""
echo "==> [1/5] Generating $N_SYNTH synthetic hand_drawn images"
$PY -m training.synthetic.generate --n "$N_SYNTH" --domain hand_drawn

echo ""
if [ "$SMOKE" != "1" ] && find datasets/cghd_raw -name '*.xml' -print -quit 2>/dev/null | grep -q .; then
  echo "==> [2/5] CGHD detected — ingesting real hand-drawn boxes + weak wire masks"
  $PY -m training.ingest.cghd_to_yolo --src datasets/cghd_raw
  $PY -m training.ingest.cghd_wire_masks --src datasets/cghd_raw
elif [ "$SMOKE" = "1" ]; then
  echo "==> [2/5] (smoke) skipping CGHD ingest to stay fast"
else
  echo "==> [2/5] No CGHD under datasets/cghd_raw/ — training the detector on synthetic only."
  echo "          For a usable hand-drawn detector, download CGHD (Zenodo 17469897)"
  echo "          into datasets/cghd_raw/ and re-run."
fi

echo ""
echo "==> [3/5] Training U-Net wire segmenter (AMP/bf16, batch $BATCH_UNET, $EPOCHS epochs)"
$PY -m training.unet.train_unet \
    --data datasets/trace_segmentation \
    --epochs "$EPOCHS" --batch "$BATCH_UNET" --imgsz "$IMG" \
    --device cuda --amp --amp-dtype bf16 --workers "$WORKERS" \
    --out models/hand_drawn_unet_trace_segmentation.pt

echo ""
echo "==> [4/5] Training YOLOv8 component detector (batch $BATCH_YOLO, $EPOCHS epochs)"
$PY -m training.yolo.train_yolo \
    --data training/yolo/data.yaml \
    --model "$YOLO_BASE" --imgsz "$IMG" --epochs "$EPOCHS" --batch "$BATCH_YOLO" \
    --device 0 --name hand_drawn_components \
    --out models/hand_drawn_yolov8_components.pt

echo ""
echo "==> [5/5] End-to-end pipeline check (image -> .cir using the new weights)"
SAMPLE="$(ls datasets/component_detection/images/train/synth_hand_drawn_*.png 2>/dev/null | head -n1 || true)"
[ -z "$SAMPLE" ] && SAMPLE="$(find datasets/component_detection/images -name '*.png' -print -quit 2>/dev/null || true)"
CHECK_CIR="data/output/_pipeline_check.cir"
if [ -z "$SAMPLE" ]; then
  echo "    (no sample image found to validate inference — skipping)"
else
  echo "    input: $SAMPLE"
  # First inference downloads the EasyOCR models (~64MB) — needs internet once.
  $PY -m statics_ocv "$SAMPLE" --output "$CHECK_CIR" \
      --unet-model models/hand_drawn_unet_trace_segmentation.pt \
      --yolo-model models/hand_drawn_yolov8_components.pt \
      --dump-intermediates || { echo "PIPELINE CHECK FAILED: inference errored." >&2; exit 1; }
  if [ -s "$CHECK_CIR" ]; then
    echo ""
    echo "    ===================  PIPELINE OK  ==================="
    echo "    Netlist written: $CHECK_CIR"
    echo "    Debug images:    data/output/*_trace_mask.png, *_skeleton.png"
  else
    echo "PIPELINE CHECK FAILED: no netlist was produced." >&2
    exit 1
  fi
fi

echo ""
if [ "$SMOKE" = "1" ]; then
  echo "==> SMOKE TEST PASSED. The pipeline runs end to end."
  echo "    Now do the real run (no SMOKE):   bash training/train_hand_drawn.sh"
else
  echo "==> DONE. Hand-drawn weights:"
  ls -lh models/hand_drawn_*.pt
  echo ""
  echo "Wire them into the backend before serving:"
  echo "  export HAND_DRAWN_UNET_WEIGHTS=\$PWD/models/hand_drawn_unet_trace_segmentation.pt"
  echo "  export HAND_DRAWN_YOLO_WEIGHTS=\$PWD/models/hand_drawn_yolov8_components.pt"
fi
