# 02 — ข้อมูล

## Repo
| | URL | บทบาท |
|---|---|---|
| code | `https://github.com/P-PrPas/tkk_workshop.git` | โน้ตบุ๊ก, แอป, สคริปต์, เอกสาร |
| data | `https://github.com/P-PrPas/tkk_workshop-data.git` | รูป + label + วิดีโอสำรอง |

data ผูกเป็น **submodule** ของ code ที่ path `data/` เพื่อความสะดวกตอนพัฒนา
แต่ **โน้ตบุ๊ก clone data repo ตรงๆ ด้วย URL** ไม่พึ่ง `--recursive`

> เหตุผล: ลืม `--recursive` = ได้โฟลเดอร์ว่างเปล่าโดยไม่มี error ให้เห็น
> ซึ่งเป็นอาการล้มที่วินิจฉัยยากที่สุดกลางห้องเรียน submodule มีไว้ให้เครื่องนักพัฒนา
> ไม่ได้มีไว้ให้ผู้เรียน

ตั้งค่าครั้งเดียว:
```bash
git submodule add https://github.com/P-PrPas/tkk_workshop-data.git data
git clone --recursive https://github.com/P-PrPas/tkk_workshop.git   # ฝั่งนักพัฒนา
```

## โครงสร้าง data repo
```
tkk_workshop-data/
├── cup.yaml
├── images/
│   ├── train/   (10 ไฟล์)
│   ├── val/     (2 ไฟล์)
│   └── test/    (3 ไฟล์)
├── labels/
│   ├── train/   (.txt ชื่อตรงกับรูป)
│   ├── val/
│   └── test/
└── sample.mp4   (~20 วินาที ใช้เมื่อกล้องพัง)
```

`cup.yaml`
```yaml
path: .
train: images/train
val: images/val
test: images/test
names:
  0: cup
```

## นิยามคลาส
**`cup` = ภาชนะสำหรับดื่ม** — แก้วใส, แก้วมัค, แก้วกระดาษ, แก้วพลาสติก
**ไม่รวมขวด** (ขวดน้ำ ขวดพลาสติก กระติกน้ำ)

ขวดถูกกันออกโดยตั้งใจ เพื่อให้มีของที่โมเดลจะพลาดไว้โชว์หน้าห้อง

## คู่มือถ่ายรูป (15 ใบ)
ถ่ายด้วยมือถือ **ในห้องและแสงเดียวกับวันงาน** — นี่คือเหตุผลเดียวที่ยอมถ่ายเองแทน
ใช้ dataset สาธารณะ ถ้าถ่ายที่อื่นก็เสียข้อได้เปรียบนี้ไปหมด

| ชุด | จำนวน | เนื้อหา |
|---|---|---|
| train | 10 | แก้ว 2-3 ใบต่างแบบ, มุมต่างกัน, มีทั้งใบเดียวและหลายใบ, มีบางรูปที่มือถือแก้วอยู่ |
| val | 2 | เหมือน train แต่คนละช็อต |
| test | 3 | **ต้องมีอย่างน้อย 1 รูปที่ตั้งใจให้ยาก** — แก้วใสบนพื้นสว่าง, แก้วถูกบังครึ่งใบ, หรือขวดน้ำวางคู่แก้ว |

ข้อกำหนดเทคนิค: JPG, ด้านยาวไม่เกิน 1280px (ไฟล์รวมควร < 5MB เพื่อให้ clone เร็ว)

## Auto-label
`tools/autolabel.py` — ใช้ YOLO ที่ผ่าน COCO มาแล้วยิง label ตั้งต้น แล้วคนไล่แก้

```python
# tools/autolabel.py
from pathlib import Path
from ultralytics import YOLO

COCO_CUP = 41  # class id ของ 'cup' ใน COCO

def autolabel(img_dir: Path, out_dir: Path, conf: float = 0.25) -> None:
    model = YOLO("yolo11x.pt")
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in sorted(img_dir.glob("*.jpg")):
        r = model(img, conf=conf, classes=[COCO_CUP], verbose=False)[0]
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                 for x, y, w, h in r.boxes.xywhn.tolist()]
        (out_dir / f"{img.stem}.txt").write_text("\n".join(lines))
        print(f"{img.name}: {len(lines)} boxes")
```

ขั้นตอน:
1. ใส่รูปลง `images/{train,val,test}/`
2. รัน `python tools/autolabel.py` (ใช้ `yolo11x.pt` ตัวใหญ่สุด — ช้าแต่แม่นกว่า เทรน
   ไม่ได้ใช้มัน ใช้แค่ตอน label)
3. **เปิดดูทุกไฟล์ด้วยตา** ผ่าน `tools/preview_labels.py` แล้วแก้ที่ผิด
4. รูปที่ตั้งใจให้ยาก auto-label มักพลาด — ต้องแก้มือแน่นอน

> ~15 รูปใช้เวลาไล่ตรวจ ~20 นาที เร็วกว่าลากกล่องเองทั้งหมด และให้ผลที่สม่ำเสมอกว่า

## sample.mp4
วิดีโอสำรองสำหรับกรณีกล้องใช้ไม่ได้ ~20 วินาที ต้องมีครบสามฉาก:
มือเปล่าแบ → มือกำ → มือกำแก้ว ถ่ายในห้องเดียวกัน 720p
