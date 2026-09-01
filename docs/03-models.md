# 03 — โมเดล

มีสองโมเดล จงใจให้ต่างกันสุดขั้ว เพราะความต่างคือเนื้อหาของ workshop

| | โมเดลจิ๋ว | โมเดลดี |
|---|---|---|
| ใช้ที่ | โน้ตบุ๊ก (ผู้เรียนเทรนสด) | desktop app |
| สถาปัตยกรรม | yolo11n | yolo11s |
| คลาส | สคีมา COCO 80 คลาส, fine-tune เฉพาะ `cup` (41) | 1 คลาส `cup` |
| ข้อมูล | ~10 รูปในห้อง + ~23 รูปแก้วทั่วไป (COCO) | Open Images V7 (Coffee cup + Mug) |
| epoch | 3 | ~80 |
| เวลาเทรน | ~15 วินาที (CPU) | ~1-2 ชม. (V100) |
| เทรนเมื่อไหร่ | ในคาบ | ล่วงหน้า อยู่ใน GitHub Release |

---

## โมเดลจิ๋ว (ในโน้ตบุ๊ก)

```python
model = YOLO("yolo11n.pt")
model.train(data="data/cup.yaml", epochs=3, imgsz=640, batch=4, seed=0, amp=False, plots=True)
```

- **`yolo11n.pt` ไม่ใช่ `yolo11n.yaml`** — เริ่มจากศูนย์ด้วยข้อมูลเท่านี้จะได้โมเดลที่
  ตรวจไม่เจออะไรเลย ทำให้พาร์ท 3 ทำงานไม่ได้ ทั้ง workshop พังตาม
- **`cup.yaml` ต้องเป็นสคีมา COCO 80 คลาส** — ถ้าลดเหลือ 1 คลาส ultralytics ขึ้น
  `Overriding nc=80 with nc=1` แล้ว **reinit หัว classifier** ทิ้งน้ำหนัก `cup` ของ COCO
  ทดสอบจริงบน 8.3.253: nc=1 ไม่ว่า 3/20/30 epoch, SGD, `nbs=4`, `single_cls` — conf สูงสุด
  ~0.05 ตรวจไม่เจอที่ `conf=0.25` เลยแม้แต่รูป train / ส่วน nc=80 label 41: **3 epoch ~15 วิ
  เจอทุกรูป conf 0.4–0.95**
- label ใน data repo เป็น class **41** / predict ทุกครั้งใส่ `classes=[41]`
- `seed=0` เพื่อให้ผู้เรียนทุกคนได้ผลใกล้เคียงกัน (ยังไม่ deterministic 100% แต่ช่วยได้)
- `batch=4` เพราะ train ยังเล็ก · `amp=False` เพราะบน GPU `amp=True` ทำให้ conf ต่ำผิดปกติ
- ไม่ hardcode `device` — เทรนบน CPU ~15 วิ, ถ้ามี T4 ultralytics ใช้เอง (opt-in ดู 01)
  GPU ช่วยเฉพาะเซลล์กล้อง realtime ไม่ใช่ตอนเทรน

**ข้อความที่ต้องพูดหน้าห้อง:** โมเดลรู้จัก "แก้ว" อยู่แล้วจาก COCO (คลาส 41)
เราเก็บความรู้เดิมไว้แล้วขยับเฉพาะส่วนนั้นด้วยรูปไม่กี่สิบใบ — นั่นคือ transfer learning ของจริง

---

## โมเดลดี (เทรนล่วงหน้าบน V100)

### เตรียมข้อมูล — `tools/build_bigdata.py`
```python
import fiftyone as fo
import fiftyone.zoo as foz

CLASSES = ["Coffee cup", "Mug"]

ds = foz.load_zoo_dataset(
    "open-images-v7",
    splits=["train", "validation"],
    label_types=["detections"],
    classes=CLASSES,
    max_samples=12000,
)
# รวมสองคลาสเป็น cup คลาสเดียว (class 0) — โมเดลดีเป็น 1 คลาส เพราะข้อมูลเยอะพอ
ds.export(
    export_dir="datasets/cup_big",
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="detections",
    classes=["cup"],
)
```
- ต้อง map ทั้ง `Coffee cup` และ `Mug` → `cup` ก่อน export ไม่งั้นได้ 2 คลาส
  แล้ว app จะอ่าน class id ผิด
- โหลด ~12k ภาพ ใช้เวลาและแบนด์วิดท์พอสมควร — เริ่มแต่เนิ่นๆ
- **รวมรูปในห้องของเราเข้าไปใน train ด้วย** เพื่อให้แม่นกับห้องจริง
  (สคริปต์ remap label จาก class 41 → 0 ให้อัตโนมัติ)

### เทรน
```bash
yolo detect train model=yolo11s.pt data=datasets/cup_big/dataset.yaml \
     epochs=80 imgsz=640 batch=32 device=0 patience=15 project=runs seed=0
```
- yolo11s ไม่ใช่ m/l — ต้องรันสดบนแล็ปท็อปวิทยากรที่อาจไม่มี GPU
- `patience=15` ตัดจบเองถ้าไม่ดีขึ้น
- เกณฑ์ผ่าน: mAP50 บน validation ของ Open Images **≥ 0.60** และที่สำคัญกว่านั้นคือ
  ต้อง**ทดสอบด้วยกล้องจริงในห้องจริง** แล้วกล่องนิ่ง ไม่กระพริบ

### ส่งมอบ
```bash
gh release create v1 runs/detect/train/weights/best.pt --title "cup detector v1"
```
แอปโหลดจาก Release URL ไม่เก็บไฟล์ 20MB+ ไว้ใน git

### แผนสำรอง (R5)
ถ้าเทรนไม่ทันหรือผลไม่ผ่านเกณฑ์: ใช้ `yolo11m.pt` COCO ตรงๆ แล้วกรองเฉพาะ
`classes=[41]` — แอปรองรับผ่าน config โดยไม่ต้องแก้โค้ด

---

## Hand pose (ทั้งสองที่ใช้ตัวเดียวกัน)

MediaPipe `HandLandmarker`, float16, โหมด VIDEO

**ทำไมไม่ใช้ YOLO:** Ultralytics ปล่อยเฉพาะ body pose 17 จุด (`yolo11n-pose.pt`)
ส่วนมือมีให้แค่ *dataset* `hand-keypoints.yaml` ไว้ fine-tune เอง ไม่มี checkpoint
สำเร็จรูป และเกร็ดที่ควรรู้: label ของ dataset นั้นถูกสร้างด้วย MediaPipe อยู่แล้ว
การเทรน YOLO บนมันจึงได้เพดานความแม่นเท่า MediaPipe โดยเสียเวลาเทรนเพิ่ม

ตรวจสอบแล้วด้วยการรันจริง: `mediapipe==1.0.1` เป็น wheel `py3-none` ติดตั้งได้
ทั้ง Python 3.12 และ 3.13 จึงไม่ผูกกับเวอร์ชัน Python ของ Colab

model: `hand_landmarker.task` (float16) — mirror ไว้ที่ data repo เผื่อ URL ต้นทางตาย
