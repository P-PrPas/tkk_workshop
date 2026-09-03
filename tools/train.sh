#!/usr/bin/env bash
# เทรนโมเดลดี (yolo11s) บนเครื่องนี้ — V100 GPU
#   bash tools/train.sh              # เทรนเต็ม 80 epoch
#   EPOCHS=10 bash tools/train.sh    # ทดลองสั้น ๆ
#   BATCH=24 bash tools/train.sh     # ถ้า RAM เหลือ (ระวัง cgroup 16GB)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=.venv-train
DATA=datasets/cup_big/dataset.yaml
EPOCHS=${EPOCHS:-80}
# container จำกัด RAM 16GB — batch=32 + workers เยอะ = OOM (ทดสอบแล้ว)
# batch=16 workers=2 รันจบ 1 epoch โดยไม่ล้ม
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-2}
DEVICE=${DEVICE:-0}

[ -x "$VENV/bin/yolo" ] || {
  echo "ยังไม่มี env — สร้างก่อน:"
  echo "  python3 -m venv $VENV"
  echo "  $VENV/bin/pip install -r tools/requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu124"
  exit 1
}
[ -f "$DATA" ] || { echo "ยังไม่มี dataset — รัน: python tools/build_bigdata.py"; exit 1; }

"$VENV/bin/yolo" detect train \
  model=yolo11s.pt data="$DATA" \
  epochs="$EPOCHS" imgsz=640 batch="$BATCH" workers="$WORKERS" device="$DEVICE" \
  patience=15 amp=False seed=0 cache=False project=runs name=cup_big

echo
echo "เสร็จ — วัดผล:  $VENV/bin/python tools/eval.py runs/detect/cup_big/weights/best.pt"
