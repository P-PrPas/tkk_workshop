"""เสริม dataset โมเดลดีด้วย Open Images V7 — เน้น 2 จุดที่ COCO cup ไม่มี:
  1. เคสมือจับ/บังแก้ว (ใช้คลาส Human hand กรองหา แล้วกันส่วนหนึ่งไว้เป็น eval แยก)
  2. hard negative ลด false positive (ขวด/ชาม/แจกัน/คุกเทลที่หน้าตาคล้ายแก้วแต่ไม่ใช่)

รันหลัง build_bigdata.py (เติมเข้า datasets/cup_big/ ที่มีอยู่แล้ว ไม่ทับ):
    python tools/build_openimages_cup.py                  # ดึงครบ (รวม train bbox 2.1GB ครั้งเดียว)
    python tools/build_openimages_cup.py --quick           # ข้าม train บox csv ที่ใหญ่ ใช้แค่ val+test (~100MB)
    python tools/build_openimages_cup.py --max-pos 3000 --max-neg 1000

ผลลัพธ์:
    datasets/cup_big/images|labels/{train,val}/oi_<id>.jpg   ← เติมเข้าชุดเดิม (class 0 เหมือนกัน)
    datasets/hand_eval/images|labels/<id>.jpg                ← กันไว้ต่างหาก ห้ามเอาไปเทรน
                                                                 ใช้วัด recall เคสมือบังที่ COCO ไม่มี

หมายเหตุ: คลาส cup ที่นี่กว้างกว่า COCO เดิม (รวม wine glass ด้วย) เพราะโจทย์คือ
"แก้วอะไรก็ได้" ไม่จำกัดแค่แก้วมัค/กาแฟแบบ workshop — ถ้าไม่ต้องการ ตัด WINE_GLASS ออกจาก POS_MIDS

ไม่ใช้ fiftyone เหมือนเดิม — bbox ใน Open Images เป็นสัดส่วน 0-1 อยู่แล้ว (ไม่ต้องรู้ขนาดรูป)
และรูปดึงตรงจาก S3 bucket เปิดสาธารณะ (ไม่ใช้ OriginalURL ในเมทาดาทาเดิม เพราะลิงก์ flickr ส่วนใหญ่ตายแล้ว)
"""
import argparse
import csv
import functools
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

socket.setdefaulttimeout(20)
print = functools.partial(print, flush=True)

WORK = Path("datasets/_openimages_cache")
OUT = Path("datasets/cup_big")          # เติมเข้าชุดเดิมของ build_bigdata.py
EVAL_OUT = Path("datasets/hand_eval")   # กันแยก ห้ามเทรน

BBOX_CSV_URLS = {
    "train": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
}
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{id}.jpg"
BUCKET = {"train": "train", "validation": "train", "test": "train"}  # ทุก oi split -> train เท่านั้น
# ห้ามเอาเข้า val — val ต้องเป็น COCO val2017 ล้วนๆ เทียบกับรอบก่อนได้ (เคยพลาดเอา test split ไปปนมาแล้ว)

POS_MIDS = {"/m/02jvh9", "/m/02p5f1q", "/m/09tvcd"}   # Mug, Coffee cup, Wine glass -> class 0
HAND_MID = "/m/0k65p"                                  # Human hand -> ใช้กรอง ไม่วาด box
NEG_MIDS = {"/m/04dr76w", "/m/04kkgm", "/m/02s195", "/m/024g6"}  # Bottle, Bowl, Vase, Cocktail


def bbox_to_yolo(xmin, xmax, ymin, ymax):
    """Open Images bbox เป็นสัดส่วน 0-1 อยู่แล้ว (ไม่ต้องหารด้วยขนาดรูปแบบ COCO)"""
    return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin


def _selftest():
    cx, cy, w, h = bbox_to_yolo(0.2, 0.6, 0.1, 0.9)
    assert (round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)) == (0.4, 0.5, 0.4, 0.8)


def valid_jpg(path):
    try:
        with open(path, "rb") as f:
            f.seek(-2, 2)
            return f.read() == b"\xff\xd9"
    except OSError:
        return False


def fetch_csv(oi_split):
    WORK.mkdir(parents=True, exist_ok=True)
    dst = WORK / f"{oi_split}-bbox.csv"
    if not dst.exists():
        print(f"ดาวน์โหลด {oi_split} bbox csv...")
        urllib.request.urlretrieve(BBOX_CSV_URLS[oi_split], dst)
    return dst


def scan(oi_split, pos, hand_ids, neg_ids, id_split):
    """อ่าน bbox csv ของ split เดียว สะสมผลลงใน dict/set ที่ส่งเข้ามา (รวมหลาย split ได้)"""
    with open(fetch_csv(oi_split), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            image_id, _src, label = row[0], row[1], row[2]
            if label != HAND_MID and label not in POS_MIDS and label not in NEG_MIDS:
                continue
            is_group, is_depict = row[10] == "1", row[11] == "1"
            id_split.setdefault(image_id, oi_split)
            if label == HAND_MID:
                hand_ids.add(image_id)
            elif label in POS_MIDS and not is_group and not is_depict:
                xmin, xmax, ymin, ymax = (float(row[i]) for i in (4, 5, 6, 7))
                pos.setdefault(image_id, []).append(bbox_to_yolo(xmin, xmax, ymin, ymax))
            elif label in NEG_MIDS and not is_group and not is_depict:
                neg_ids.add(image_id)


def download_one(oi_split, image_id, img_dir, lbl_dir, rows):
    dst = img_dir / f"oi_{image_id}.jpg"
    lbl = lbl_dir / f"oi_{image_id}.txt"
    body = "\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in rows)
    lbl.write_text(body + "\n" if body else "")
    if dst.exists() and valid_jpg(dst):
        return True
    for _ in range(2):
        try:
            urllib.request.urlretrieve(IMG_URL.format(split=oi_split, id=image_id), dst)
            if valid_jpg(dst):
                return True
        except Exception:
            pass
    lbl.unlink(missing_ok=True)
    dst.unlink(missing_ok=True)
    return False


def download_many(items, img_dir, lbl_dir, workers, label):
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(download_one, split, iid, img_dir, lbl_dir, rows) for split, iid, rows in items]
        for i, fut in enumerate(futs, 1):
            ok += fut.result()
            if i % 200 == 0:
                print(f"  {label}: {i}/{len(futs)} (สำเร็จ {ok})")
    print(f"{label}: {ok}/{len(items)} รูป")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="ข้าม train bbox csv (2.1GB) ใช้แค่ val+test")
    ap.add_argument("--max-pos", type=int, default=0, help="0 = ไม่จำกัด")
    ap.add_argument("--max-neg", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=25, help="จำนวนรูปมือบังที่กันไว้ไม่เทรน")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    _selftest()

    oi_splits = ["validation", "test"] if args.quick else ["train", "validation", "test"]
    pos, hand_ids, neg_ids, id_split = {}, set(), set(), {}
    for oi_split in oi_splits:
        before = len(pos)
        scan(oi_split, pos, hand_ids, neg_ids, id_split)
        print(f"{oi_split}: เจอ cup-like {len(pos) - before} รูปใหม่")

    neg_ids -= pos.keys()   # negative ต้องไม่มี cup-like ติดอยู่ในภาพเดียวกัน
    print(f"รวม: cup-like {len(pos)} รูป (มือบัง {len(pos.keys() & hand_ids)} รูป) / negative {len(neg_ids)} รูป")

    # กันเคสมือบังไว้เป็น eval แยกก่อน ไม่ให้หลุดไปเทรน
    hand_pos = sorted(pos.keys() & hand_ids)
    eval_ids, train_pos_ids = hand_pos[: args.eval_n], set(pos) - set(hand_pos[: args.eval_n])

    eval_items = [(id_split[i], i, pos[i]) for i in eval_ids]
    download_many(eval_items, EVAL_OUT / "images", EVAL_OUT / "labels", args.workers, "hand_eval (กันไว้ไม่เทรน)")
    if eval_items:
        (EVAL_OUT / "dataset.yaml").write_text(
            f"train: images\nval: images\nnames:\n  0: cup\n", encoding="utf-8")

    pos_ids = sorted(train_pos_ids, key=lambda i: i not in hand_ids)[: args.max_pos or None]
    neg_ids = sorted(neg_ids)[: args.max_neg or None]
    for bucket in ("train", "val"):
        items = [(id_split[i], i, pos[i]) for i in pos_ids if BUCKET[id_split[i]] == bucket]
        items += [(id_split[i], i, []) for i in neg_ids if BUCKET[id_split[i]] == bucket]
        if items:
            download_many(items, OUT / "images" / bucket, OUT / "labels" / bucket, args.workers,
                           f"cup_big/{bucket}")

    print(f"\nเสร็จ — เติมเข้า {OUT}/ แล้ว (dataset.yaml เดิมใช้ได้ ไม่ต้องแก้)")
    print(f"eval มือบัง → {EVAL_OUT}/  วัดด้วย: .venv-train/bin/python tools/eval.py <best.pt>")


if __name__ == "__main__":
    main()
