"""เตรียม dataset โมเดลดี: ดึงรูป "cup" จาก COCO 2017 → YOLO format (1 คลาส)

ใช้ครั้งเดียวก่อนเทรน (หรือ `bash tools/train.sh` จะเรียกให้):
    python tools/build_bigdata.py                 # ดึงครบ train ~9.2k / val 390 (~1.5GB)
    python tools/build_bigdata.py --max-train 2000  # จำกัดจำนวน (ทดสอบ/เน็ตช้า)

ผลลัพธ์:
    datasets/cup_big/
      images/{train,val}/   labels/{train,val}/   dataset.yaml

ไม่พึ่ง fiftyone (ที่ต้องมี mongod) — ใช้แค่ urllib + opencv ที่มีอยู่แล้ว
COCO `cup` = แก้วมัค แก้วกาแฟ แก้วกระดาษ แก้วใส ครบตามที่ต้องการ
"""
import argparse
import functools
import json
import shutil
import socket
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

socket.setdefaulttimeout(20)             # กัน urlretrieve ค้างถ้า COCO server ไม่ตอบ
print = functools.partial(print, flush=True)   # progress ออกทันที ไม่ค้างใน buffer

WORK = Path("datasets/_coco_cache")
OUT = Path("datasets/cup_big")
ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{id:012d}.jpg"
CUP_CATEGORY_ID = 47                      # 'cup' ใน COCO instances json
OUR_IMAGES = Path("data/images")         # รูปในห้องจาก data submodule (label class 41)
OUR_LABELS = Path("data/labels")


def valid_jpg(path):
    """รูปครบไหม — JPEG จบด้วย marker FF D9 เสมอ โหลดไม่ครบจะไม่มี (cv2.imread ยังอ่านรูปเพี้ยนได้)"""
    try:
        with open(path, "rb") as f:
            f.seek(-2, 2)
            return f.read() == b"\xff\xd9"
    except OSError:
        return False


def fetch_annotations():
    WORK.mkdir(parents=True, exist_ok=True)
    zip_path = WORK / "ann.zip"
    need = [WORK / "annotations" / f"instances_{s}2017.json" for s in ("train", "val")]
    if all(p.exists() for p in need):
        return need
    if not zip_path.exists():
        print("ดาวน์โหลด COCO annotations (241MB, ครั้งเดียว)...")
        urllib.request.urlretrieve(ANN_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for p in need:
            z.extract(f"annotations/{p.name}", WORK)
    return need


def cup_items(ann_json):
    """image_id -> [(cx,cy,w,h) normalized]  เฉพาะรูปที่มี cup"""
    d = json.load(open(ann_json, encoding="utf-8"))
    meta = {im["id"]: im for im in d["images"]}
    per = {}
    for a in d["annotations"]:
        if a["category_id"] != CUP_CATEGORY_ID or a.get("iscrowd"):
            continue
        m = meta[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        W, H = m["width"], m["height"]
        if bw <= 1 or bh <= 1:
            continue
        per.setdefault(a["image_id"], []).append(
            ((x + bw / 2) / W, (y + bh / 2) / H, bw / W, bh / H))
    return per


def download_split(split, per, out_split, max_n, workers):
    img_dir = OUT / "images" / out_split
    lbl_dir = OUT / "labels" / out_split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(per)[:max_n] if max_n else sorted(per)

    def one(iid):
        dst = img_dir / f"{iid:012d}.jpg"
        lbl = lbl_dir / f"{iid:012d}.txt"
        lbl.write_text("\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                                for cx, cy, w, h in per[iid]) + "\n")
        if dst.exists() and valid_jpg(dst):
            return True
        for _ in range(2):   # โหลดพลาด/ไฟล์ไม่ครบ → ลองใหม่ 1 ครั้ง
            try:
                urllib.request.urlretrieve(IMG_URL.format(split=split, id=iid), dst)
                if valid_jpg(dst):
                    return True
            except Exception:
                pass
        lbl.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
        return False

    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, got in enumerate(ex.map(one, ids), 1):
            ok += got
            if i % 200 == 0:
                print(f"  {out_split}: {i}/{len(ids)}  (สำเร็จ {ok})")
    print(f"{out_split}: {ok}/{len(ids)} รูป")
    return ok


def add_room_images():
    """รูปในห้องจริง (IMG_*) เข้า train — label class 41 → 0
    ข้ามรูป coco_* ใน data/ (มาจาก COCO อยู่แล้ว จะซ้ำกับ val split = leakage)"""
    img_dir = OUT / "images" / "train"
    lbl_dir = OUT / "labels" / "train"
    n = 0
    for split in ("train", "val", "test"):
        for img in (OUR_IMAGES / split).glob("IMG_*.jpg"):
            shutil.copy(img, img_dir / f"room_{img.stem}.jpg")
            src = OUR_LABELS / split / (img.stem + ".txt")
            rows = [f"0 {' '.join(ln.split()[1:])}"
                    for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
            (lbl_dir / f"room_{img.stem}.txt").write_text("\n".join(rows) + "\n")
            n += 1
    print(f"เพิ่มรูปในห้อง {n} ใบเข้า train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=0, help="0 = ครบทุกรูป")
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if not OUR_IMAGES.exists():
        raise SystemExit("ไม่พบ data/images — รัน `git submodule update --init` ก่อน")

    tr_json, val_json = fetch_annotations()
    print("กรอง cup จาก annotations...")
    tr_per = cup_items(tr_json)
    val_per = cup_items(val_json)
    print(f"COCO cup: train {len(tr_per)} รูป, val {len(val_per)} รูป")

    download_split("train2017", tr_per, "train", args.max_train, args.workers)
    download_split("val2017", val_per, "val", args.max_val, args.workers)
    add_room_images()

    # ไม่ใส่ path: -> ultralytics อิงโฟลเดอร์ที่ไฟล์ yaml อยู่เป็น root (path: . จะไปอิง cwd)
    (OUT / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: cup\n", encoding="utf-8")

    n_tr = len(list((OUT / "images" / "train").glob("*.jpg")))
    n_val = len(list((OUT / "images" / "val").glob("*.jpg")))
    print(f"\nเสร็จ — {OUT}/  (train {n_tr} / val {n_val})")
    print("เทรนต่อ:  bash tools/train.sh")
    print("วัดผล:   .venv-train/bin/python tools/eval.py runs/detect/cup_big/weights/best.pt")


if __name__ == "__main__":
    main()
