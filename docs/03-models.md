# 03 — โมเดล

มีสองโมเดล จงใจให้ต่างกันสุดขั้ว เพราะความต่างคือเนื้อหาของ workshop

| | โมเดลจิ๋ว | โมเดลดี |
|---|---|---|
| ใช้ที่ | โน้ตบุ๊ก (ผู้เรียนเทรนสด) | desktop app |
| สถาปัตยกรรม | yolo11n | yolo11s |
| คลาส | สคีมา COCO 80 คลาส, fine-tune เฉพาะ `cup` (41) | 1 คลาส `cup` |
| ข้อมูล | ~10 รูปในห้อง + ~23 รูปแก้วทั่วไป (COCO) | COCO 2017 cup: train 9204 / val 390 |
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
```bash
python tools/build_bigdata.py               # ดึงครบ: train 9204 (9189 COCO + 15 ในห้อง), val 390, ~1.5GB
python tools/build_bigdata.py --max-train 2000   # จำกัดจำนวน ถ้าเน็ตช้า
```
- แหล่งข้อมูล: **COCO 2017 คลาส `cup`** (แก้วมัค/กาแฟ/กระดาษ/ใส ครบ) — ไม่ใช้ Open Images
  แล้ว เพราะ fiftyone ต้องมี mongod ที่ลงยากบางเครื่อง สคริปต์ใหม่ใช้แค่ `urllib` + `opencv`
- ดาวน์โหลด: annotations zip 241MB ครั้งเดียว + รูปจาก `images.cocodataset.org`
  (มี socket timeout กัน server ค้าง · ~15 รูป/วินาที → รูปครบ ~10-15 นาที)
- output: `datasets/cup_big/{images,labels}/{train,val}/` + `dataset.yaml` (1 คลาส `cup` = 0)
- **รวมรูปในห้องของเราเข้า train อัตโนมัติ** (ตั้งชื่อ `room_*`, remap label 41 → 0)
- idempotent: รันซ้ำได้ ข้ามไฟล์ที่โหลดแล้ว

### เทรน
```bash
yolo detect train model=yolo11s.pt data=datasets/cup_big/dataset.yaml \
     epochs=80 imgsz=640 batch=32 device=0 patience=15 amp=False project=runs seed=0
```
- yolo11s ไม่ใช่ m/l — ต้องรันสดบนแล็ปท็อปวิทยากรที่อาจไม่มี GPU
- `patience=15` ตัดจบเองถ้าไม่ดีขึ้น · `amp=False` — บน GPU `amp=True` ทำให้ conf ต่ำผิดปกติ (เจอกับโมเดลจิ๋ว)
- เกณฑ์ผ่าน: `python tools/eval.py runs/detect/train/weights/best.pt` แล้ว
  **mAP50 บน COCO cup val ≥ 0.60** และที่สำคัญกว่า — **ทดสอบด้วยกล้องจริงในห้องจริง**
  กล่องต้องนิ่ง ไม่กระพริบ

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
