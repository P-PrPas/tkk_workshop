# 01 — สเปกโน้ตบุ๊ก (รายเซลล์)

ไฟล์: `notebooks/cv101.ipynb`
กฎ: **markdown เป็นภาษาไทย, โค้ดและชื่อตัวแปรเป็นภาษาอังกฤษ**
กฎ: ผู้เรียนกด Run All ได้ผลลัพธ์ ห้ามมีเซลล์ที่ต้องแก้ค่าก่อนรัน

---

## เซลล์ 0 — ติดตั้ง (code)
```python
!pip install -q ultralytics==8.3.* mediapipe==1.0.1
```
> pin ไว้เพื่อไม่ให้โน้ตบุ๊กเน่าเมื่อ Colab อัปเดต ถ้าจะเปลี่ยนเวอร์ชัน ต้องรันซ้ำทั้งไฟล์ก่อนวันงาน

## เซลล์ 1 — preflight (code)
เช็กแล้ว **พังดังๆ ทันที** ดีกว่าไปพังนาทีที่ 60
```python
import sys, torch, ultralytics, mediapipe, cv2
print("python     :", sys.version.split()[0])
print("torch      :", torch.__version__, "| GPU:", torch.cuda.is_available())
print("ultralytics:", ultralytics.__version__)
print("mediapipe  :", mediapipe.__version__)
assert ultralytics.__version__.startswith("8.3"), "ultralytics เวอร์ชันไม่ตรง"
print("\nพร้อมแล้ว — ไม่ต้องมี GPU ก็รันได้ทั้งโน้ตบุ๊ก")
```
> **ไม่บังคับ GPU** — 10 รูป 3 epoch จบใน ~2 นาทีบน CPU การให้ผู้เรียน 20 คน
> ไปกด Change runtime type พร้อมกันคือการเสียเวลา 5 นาทีโดยไม่ได้อะไร

## เซลล์ 2 — helper กล้อง (code, ยาวที่สุดในโน้ตบุ๊ก ~60 บรรทัด)
หัวใจของทั้งไฟล์ เขียนครั้งเดียวใช้ 3 ที่
```python
def run_webcam(process_frame, seconds=20):
    """เปิดกล้องผ่าน browser ส่งเฟรมมาให้ process_frame(bgr) -> bgr แล้ววาดกลับ"""
```
- ใช้ `google.colab.output.eval_js` + `getUserMedia` ดึงเฟรมเป็น base64
- แปลงเป็น numpy BGR → เรียก `process_frame` → ส่งภาพที่วาดแล้วกลับไปแสดง
- มีปุ่ม **Stop** บนหน้าเว็บ และตัดเองเมื่อครบ `seconds`
- `try/except` รอบ `eval_js`: ถ้าผู้ใช้ปฏิเสธกล้อง ให้ขึ้นข้อความไทยชัดๆ
  ว่าให้ไปใช้เซลล์สำรอง ไม่ใช่ traceback ยาวเหยียด

**เซลล์สำรอง (markdown + code ที่คอมเมนต์ไว้)** ต่อท้ายทันที:
```python
# ถ้ากล้องใช้ไม่ได้ ให้ลบ # ข้างล่างแล้วรันแทน
# run_video("data/sample.mp4", process_frame)
```

## เซลล์ 3 — markdown: พาร์ท 1 Object Detection
อธิบายด้วยภาษาคน: bbox คืออะไร, conf คืออะไร, ทำไมต้องมี label

---

## 1.1 โหลด data

### เซลล์ 4 (code)
```python
!git clone -q https://github.com/P-PrPas/tkk_workshop-data.git data
```
> clone data repo ตรงๆ ไม่พึ่ง `--recursive` (ดู [02-data.md](02-data.md))

### เซลล์ 5 (code)
แสดงกริด 15 รูปพร้อมกล่อง ground-truth ที่วาดจากไฟล์ `.txt` — ให้ผู้เรียนเห็นว่า
"label" ไม่ใช่เวทมนตร์ แต่คือตัวเลข 5 ตัวต่อหนึ่งกล่อง
```python
print("train:", len(train_imgs), "| val:", len(val_imgs), "| test:", len(test_imgs))
# 10 / 2 / 3
```
พร้อมประโยคกำกับ: *"ใช่ครับ สิบรูป จำตัวเลขนี้ไว้"*

---

## 1.2 Training

### เซลล์ 6 (code)
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")          # เริ่มจากน้ำหนักที่ผ่าน COCO มาแล้ว
model.train(data="data/cup.yaml", epochs=3, imgsz=640, batch=4, seed=0, plots=True)
```

### เซลล์ 7 (markdown) — **จุดที่ห้ามข้าม**
> เราไม่ได้เทรนโมเดลนี้ขึ้นมาจากศูนย์ เราเริ่มจาก `yolo11n.pt` ที่เห็นรูปมาแล้วแสนกว่ารูป
> จาก COCO ซึ่งมีคลาส "cup" อยู่แล้ว สิ่งที่รูป 10 ใบของเราทำคือ *ขยับ* โมเดลให้เข้ากับ
> แก้วในห้องนี้เท่านั้น — ถ้าเริ่มจากศูนย์จริงๆ ด้วยข้อมูลเท่านี้ มันจะตรวจไม่เจออะไรเลย

---

## 1.3 Result

### เซลล์ 8 (code) — หลักฐานหลัก
กริดรูป test 3 ใบพร้อมกล่องที่ทำนาย **และต้องมีอย่างน้อย 1 รูปที่โมเดลพลาด**
(ถ้าไม่มีให้เพิ่มรูปแก้วแปลกๆ หรือขวดน้ำเข้าไปใน test)

### เซลล์ 9 (code) — ของแถม
```python
metrics = model.val(split="test")
print("mAP50:", round(metrics.box.map50, 3))
```
markdown กำกับ: *"val มี 2 รูป — ตัวเลขนี้ขยับทีละ 50% ต่อหนึ่งรูป อย่าเอาไปอ้างอิงที่ไหน
ดูรูปข้างบนเถอะ"*

---

## 1.4 Realtime

### เซลล์ 10 (code)
```python
def process_frame(bgr):
    r = model(bgr, conf=0.25, verbose=False)[0]
    return r.plot()

run_webcam(process_frame, seconds=20)
```
markdown: ให้ลองเอาแก้วเข้า-ออกเฟรม, เอียงแก้ว, เอามือบัง, แล้วลองเอา**ขวดน้ำ**
มาให้ดู (มันจะทายว่าเป็นแก้ว หรือไม่เห็นเลย — ทั้งสองอย่างคือบทเรียน)

---

## พาร์ท 2 — Hand Pose

### เซลล์ 11 (markdown)
21 จุดคืออะไร, ทำไม pose ต่างจาก detection

### เซลล์ 12 (code)
```python
!wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
> ต้องทดสอบว่า URL นี้ยังใช้ได้ก่อนวันงาน ถ้าตายให้ mirror ขึ้น data repo

### เซลล์ 13 (code) — กติกา กำ/แบ เขียนเอง
```python
TIPS = [4, 8, 12, 16, 20]
PIPS = [2, 6, 10, 14, 18]

def count_extended(lm):
    """นิ้วเหยียด = ปลายนิ้วอยู่ไกลจากข้อมือกว่าข้อกลาง"""
    w = lm[0]
    d = lambda p: (p.x - w.x) ** 2 + (p.y - w.y) ** 2
    return sum(d(lm[t]) > d(lm[p]) for t, p in zip(TIPS, PIPS))

def hand_state(lm):
    n = count_extended(lm)
    return "FIST" if n <= 1 else "OPEN" if n >= 4 else "UNKNOWN"
```
markdown อธิบายว่าทำไมไม่ใช้ `tip.y < pip.y`: **เพราะพอเอียงมือหรือชี้ลง มันพังทันที**
วัดระยะจากข้อมือแทน ทนการหมุนได้

### เซลล์ 14 (code) — realtime
```python
run_webcam(hand_process_frame, seconds=20)
```
markdown: สังเกตว่าตอนกำๆ แบๆ ค้างระหว่างกลาง ป้ายจะ**กระพริบ** — จำอาการนี้ไว้
เดี๋ยวเราจะกลับมาแก้มันในแอป

---

## พาร์ท 3 — คนถือแก้ว

### เซลล์ 15 (markdown)
"เราไม่ได้เทรนโมเดล 'คนถือแก้ว' — เราเอาผลของสองโมเดลมาต่อกันด้วยกฎ 3 บรรทัด"

### เซลล์ 16 (code)
```python
def boxes_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

def is_holding(hand_bbox, hand_st, cup_boxes):
    return hand_st == "FIST" and any(boxes_overlap(hand_bbox, c) for c in cup_boxes)
```
`hand_bbox` = min/max ของ 21 landmarks

### เซลล์ 17 (code) — realtime รวมร่าง
วาดกล่องแก้ว + โครงมือ + ป้ายใหญ่ `HOLDING` / `NOT HOLDING`

### เซลล์ 18 (markdown) — ส่งไม้ต่อให้แอป
ให้ผู้เรียนสังเกตด้วยตาตัวเองว่ามีปัญหาอะไรบ้าง แล้วค่อยเฉลย:
1. ป้ายกระพริบตลอด (ไม่มีความจำข้ามเฟรม)
2. แก้วหายไปเฟรมเดียวแล้วกลับมา = กลายเป็นแก้วใบใหม่
3. ช้า
4. ถ้าถือแก้วแบบแบมือ (ประคอง) → ตรวจไม่เจอ เพราะกฎเราบังคับว่าต้องกำ
5. ถอดปลั๊กกล้องแล้วทุกอย่างพัง

> "ห้าข้อนี้แหละครับ คือระยะห่างระหว่าง demo กับ product เดี๋ยวผมโชว์ตัวที่แก้ครบแล้ว"

---

## สิ่งที่จงใจ **ไม่** ใส่ในโน้ตบุ๊ก
- exercise ให้ผู้เรียนแก้เอง (ไม่มีเวลาใน 90 นาที)
- คณิตศาสตร์ของ loss / anchor / NMS
- การ export ONNX / TensorRT
- การเทรนโมเดลใหญ่ (อยู่ใน [03-models.md](03-models.md) เป็นสคริปต์แยก)
