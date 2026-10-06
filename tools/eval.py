"""วัดผลโมเดลดี — inference pipeline สำหรับตัดสินว่าโมเดล "ผ่าน" ไหม

    python tools/eval.py runs/detect/train/weights/best.pt

ทำ 3 อย่าง:
  1. mAP บน COCO cup val (datasets/cup_big/images/val)  → เกณฑ์ mAP50 ≥ 0.60 (docs/03)
  2. recall บนเคส "มือบังแก้ว" (datasets/hand_eval ถ้ามี — มาจาก build_openimages_cup.py)
     ยังไม่มีเกณฑ์ผ่าน/ไม่ผ่านตายตัว เพราะยังไม่มี baseline — ดูเลขแล้วเทียบรอบต่อรอบ
  3. ยิงกับรูปในห้องจริง (data/images/{val,test}) → วาดกริดให้ดูด้วยตา
     ⚠️ รูปพวกนี้อยู่ใน train ของโมเดลดีแล้ว — เป็นแค่ sanity check ว่า "ยิงติดไหม"
     ไม่ใช่การวัดผลจริง การวัดผลจริงคือ mAP บน COCO val + ทดสอบกล้องสดในห้อง (docs/03)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BIG_VAL_YAML = Path("datasets/cup_big/dataset.yaml")
HAND_EVAL_YAML = Path("datasets/hand_eval/dataset.yaml")
ROOM = Path("data/images")
ROOM_LBL = Path("data/labels")
CONF = 0.25


def run_val(model, yaml_path, label, pass_threshold=None):
    if not yaml_path.exists():
        print(f"ข้าม {label} — ไม่พบ {yaml_path}")
        return None
    m = model.val(data=str(yaml_path), split="val", conf=0.001, verbose=False)
    print(f"\n=== {label} ===")
    suffix = f"   (เกณฑ์ผ่าน ≥ {pass_threshold:.2f})" if pass_threshold is not None else ""
    print(f"  mAP50     : {m.box.map50:.3f}{suffix}")
    print(f"  mAP50-95  : {m.box.map:.3f}")
    print(f"  precision : {m.box.mp:.3f}   recall: {m.box.mr:.3f}")
    return float(m.box.map50)


def room_test(model):
    imgs = sorted(p for s in ("val", "test") for p in (ROOM / s).glob("IMG_*.jpg"))
    print(f"\n=== sanity check รูปในห้อง ({len(imgs)} ใบ, conf {CONF}) ===")
    tiles, all_hit = [], True
    for p in imgs:
        r = model(str(p), conf=CONF, verbose=False)[0]
        n_pred = len(r.boxes)
        split = "val" if f"{ROOM}/val/" in str(p) else "test"
        n_gt = sum(1 for ln in (ROOM_LBL / split / (p.stem + ".txt")).read_text(encoding="utf-8").splitlines()
                   if ln.strip())
        mark = "ok" if n_pred >= n_gt else "MISS"
        all_hit &= n_pred >= n_gt
        print(f"  {p.name:16} gt={n_gt} pred={n_pred}  [{mark}]  "
              f"conf={[round(float(c), 2) for c in r.boxes.conf]}")
        tiles.append(cv2.resize(r.plot(), (480, 640)))
    if tiles:
        out = Path("datasets/room_eval.jpg")
        cv2.imwrite(str(out), np.hstack(tiles))
        print(f"  กริดผล → {out}")
    return all_hit


def main():
    if len(sys.argv) != 2:
        raise SystemExit("ใช้: python tools/eval.py <path/to/best.pt>")
    weights = sys.argv[1]
    if not Path(weights).exists():
        raise SystemExit(f"ไม่พบไฟล์: {weights}")
    model = YOLO(weights)

    map50 = run_val(model, BIG_VAL_YAML, "COCO cup val", pass_threshold=0.60)
    hand_map50 = run_val(model, HAND_EVAL_YAML, "มือบังแก้ว (hand_eval)")
    all_hit = room_test(model)

    print("\n=== สรุป ===")
    if map50 is not None:
        print(f"  COCO mAP50 {map50:.3f}  {'ผ่าน' if map50 >= 0.60 else 'ไม่ผ่าน (< 0.60)'}")
    if hand_map50 is not None:
        print(f"  มือบัง mAP50 {hand_map50:.3f}  (ยังไม่มีเกณฑ์ตายตัว — เทียบกับรอบเทรนก่อนหน้า)")
    print(f"  รูปในห้อง: {'ยิงติดทุกใบ' if all_hit else 'มีใบที่พลาด — ดู datasets/room_eval.jpg'}")
    print("  เกณฑ์สุดท้าย: ทดสอบกล้องสดในห้องจริง กล่องต้องนิ่ง ไม่กระพริบ (docs/03)")


if __name__ == "__main__":
    main()
