#!/usr/bin/env bash
# One-shot cloud training: install, build data, train all four models.
#
# Designed for a fresh GPU box (Colab / Kaggle / vast.ai / RunPod / Lambda).
# The synthetic generator is fast and seeded, so we regenerate data ON the GPU
# box instead of uploading it. CGHD is optional (real hand-drawn detection data).
#
# Usage (from the repo root, on the GPU machine):
#     bash training/cloud/run_all.sh
#
# Environment knobs (override inline, e.g. `N_SYNTH=3000 bash ...`):
#     N_SYNTH      images per domain to generate          (default 2000)
#     EPOCHS_UNET  U-Net epochs                            (default 80)
#     EPOCHS_YOLO  YOLO epochs                             (default 100)
#     IMG_UNET     U-Net train resolution                 (default 768)
#     IMG_YOLO     YOLO train resolution                  (default 768)
#     BATCH_UNET   U-Net batch size                        (default 8)
#     BATCH_YOLO   YOLO batch size                         (default 16)
#     USE_CGHD     1 to download+ingest CGHD for hand-drawn (default 0)
#     YOLO_BASE    base detector                           (default yolov8s.pt)
set -euo pipefail

N_SYNTH="${N_SYNTH:-2000}"
EPOCHS_UNET="${EPOCHS_UNET:-80}"
EPOCHS_YOLO="${EPOCHS_YOLO:-100}"
IMG_UNET="${IMG_UNET:-768}"
IMG_YOLO="${IMG_YOLO:-768}"
BATCH_UNET="${BATCH_UNET:-8}"
BATCH_YOLO="${BATCH_YOLO:-16}"
USE_CGHD="${USE_CGHD:-0}"
YOLO_BASE="${YOLO_BASE:-yolov8s.pt}"
PY="${PY:-python}"

echo "==> Python: $($PY --version)"
echo "==> GPU:"; nvidia-smi -L 2>/dev/null || echo "   (no NVIDIA GPU detected — training will be slow)"

echo "==> Installing dependencies"
$PY -m pip install -q -r requirements.txt
# Fix TLS for automatic weight/model downloads on bare boxes.
export SSL_CERT_FILE="$($PY -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

DEVICE="cpu"
if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then DEVICE="cuda"; fi
echo "==> Training device: $DEVICE"

train_domain () {
  local domain="$1"          # digital | hand_drawn
  echo ""
  echo "########################################################"
  echo "# DOMAIN: $domain"
  echo "########################################################"

  echo "==> [$domain] generating $N_SYNTH synthetic images"
  $PY -m training.synthetic.generate --n "$N_SYNTH" --domain "$domain"

  if [ "$domain" = "hand_drawn" ] && [ "$USE_CGHD" = "1" ]; then
    echo "==> [hand_drawn] ingesting CGHD (expects datasets/cghd_raw populated)"
    $PY -m training.ingest.cghd_to_yolo --src datasets/cghd_raw || echo "   (CGHD ingest skipped/failed)"
    $PY -m training.ingest.cghd_wire_masks --src datasets/cghd_raw || echo "   (CGHD masks skipped/failed)"
  fi

  echo "==> [$domain] training U-Net"
  $PY -m training.unet.train_unet \
      --data datasets/trace_segmentation \
      --epochs "$EPOCHS_UNET" --batch "$BATCH_UNET" --imgsz "$IMG_UNET" --device "$DEVICE" \
      --out "models/${domain}_unet_trace_segmentation.pt"

  echo "==> [$domain] training YOLOv8"
  $PY -m training.yolo.train_yolo \
      --data training/yolo/data.yaml \
      --model "$YOLO_BASE" --imgsz "$IMG_YOLO" --epochs "$EPOCHS_YOLO" --batch "$BATCH_YOLO" \
      --device "$DEVICE" --name "${domain}_components" \
      --out "models/${domain}_yolov8_components.pt"
}

# IMPORTANT: clear datasets between domains so a domain trains only on its data.
reset_datasets () {
  find datasets/trace_segmentation datasets/component_detection -type f ! -name '.gitkeep' -delete
}

reset_datasets; train_domain "digital"
reset_datasets; train_domain "hand_drawn"

echo ""
echo "==> DONE. Trained weights:"
ls -lh models/*.pt
echo ""
echo "Download these four files and place them in models/ on your inference machine:"
echo "  models/digital_unet_trace_segmentation.pt"
echo "  models/digital_yolov8_components.pt"
echo "  models/hand_drawn_unet_trace_segmentation.pt"
echo "  models/hand_drawn_yolov8_components.pt"
