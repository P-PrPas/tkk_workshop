# 03 — โมเดล

มีสองโมเดล จงใจให้ต่างกันสุดขั้ว เพราะความต่างคือเนื้อหาของ workshop

| | โมเดลจิ๋ว | โมเดลดี |
|---|---|---|
| ใช้ที่ | โน้ตบุ๊ก (ผู้เรียนเทรนสด) | desktop app |
| สถาปัตยกรรม | yolo11n | yolo11s |
| ข้อมูล | 10 รูป | Open Images V7 (Coffee cup + Mug) |
| epoch | 3 | ~80 |
| เวลาเทรน | ~2 นาที (CPU) | ~1-2 ชม. (V100) |
| เทรนเมื่อไหร่ | ในคาบ | ล่วงหน้า อยู่ใน GitHub Release |

---

## โมเดลจิ๋ว (ในโน้ตบุ๊ก)

```python
model = YOLO("yolo11n.pt")
model.train(data="data/cup.yaml", epochs=3, imgsz=640, batch=4, seed=0, plots=True)
```

- **`yolo11n.pt` ไม่ใช่ `yolo11n.yaml`** — เริ่มจากศูนย์ด้วยข้อมูล 10 รูปจะได้โมเดลที่
  ตรวจไม่เจออะไรเลย ทำให้พาร์ท 3 ทำงานไม่ได้ ทั้ง workshop พังตาม
- `seed=0` เพื่อให้ผู้เรียนทุกคนได้ผลใกล้เคียงกัน (ยังไม่ deterministic 100% แต่ช่วยได้)
- `batch=4` เพราะ train มีแค่ 10 รูป
- ไม่ต้องมี GPU

**ข้อความที่ต้องพูดหน้าห้อง:** ที่มันใช้งานได้คือผลของ COCO ไม่ใช่ผลของรูป 10 ใบ
รูป 10 ใบแค่ปรับจูนให้เข้ากับแก้วในห้องนี้

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
# รวมสองคลาสเป็น cup คลาสเดียว ให้ตรงกับโมเดลจิ๋ว
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
- **รวมรูป 15 ใบของเราเข้าไปใน train ด้วย** เพื่อให้แม่นกับห้องจริง

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
