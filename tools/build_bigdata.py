"""เตรียม dataset โมเดลดี: ดึง Open Images V7 (Coffee cup + Mug) รวมเป็นคลาส cup เดียว

ใช้ครั้งเดียวก่อนเทรน:  python tools/build_bigdata.py
จากนั้นเทรนตาม docs/03-models.md

ต้องมี:  pip install fiftyone
โหลด ~12k ภาพ ใช้เวลาและแบนด์วิดท์พอสมควร เริ่มแต่เนิ่นๆ
"""
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

CLASSES = ["Coffee cup", "Mug"]
EXPORT_DIR = "datasets/cup_big"
OUR_IMAGES = Path("data/images")   # 15 รูปของเราจาก data submodule
OUR_LABELS = Path("data/labels")

ds = foz.load_zoo_dataset(
    "open-images-v7",
    splits=["train", "validation"],
    label_types=["detections"],
    classes=CLASSES,
    max_samples=12000,
)

# รวม Coffee cup + Mug -> cup ก่อน export ไม่งั้นได้ 2 คลาส แล้ว app อ่าน class id ผิด
for sample in ds.iter_samples(progress=True, autosave=True):
    dets = sample.detections
    if dets is None:
        continue
    for d in dets.detections:
        d.label = "cup"

ds.export(
    export_dir=EXPORT_DIR,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    classes=["cup"],
)

# รวมรูป 15 ใบของเราเข้า train เพื่อให้แม่นกับห้องจริง
# label ใน data repo เป็น class 41 (COCO) — ที่นี่ dataset เป็น 1 คลาส remap เป็น 0
train_img_dir = Path(EXPORT_DIR) / "images" / "train"
train_lbl_dir = Path(EXPORT_DIR) / "labels" / "train"
train_img_dir.mkdir(parents=True, exist_ok=True)
train_lbl_dir.mkdir(parents=True, exist_ok=True)
for split in ("train", "val", "test"):
    for img in (OUR_IMAGES / split).glob("*.jpg"):
        shutil.copy(img, train_img_dir / img.name)
        lbl = OUR_LABELS / split / (img.stem + ".txt")
        if lbl.exists():
            rows = [f"0 {' '.join(ln.split()[1:])}"
                    for ln in lbl.read_text().splitlines() if ln.strip()]
            (train_lbl_dir / lbl.name).write_text("\n".join(rows) + "\n")

print("เสร็จ — dataset อยู่ที่", EXPORT_DIR)
print("เทรนต่อ:  yolo detect train model=yolo11s.pt data=%s/dataset.yaml epochs=80 "
      "imgsz=640 batch=32 device=0 patience=15 project=runs seed=0" % EXPORT_DIR)
