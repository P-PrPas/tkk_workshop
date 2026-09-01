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
print("\nพร้อมแล้ว เริ่มได้เลย —",
      "ใช้ GPU (เซลล์กล้องจะลื่น)" if torch.cuda.is_available()
      else "ใช้ CPU (รันได้ครบ; อยากให้เซลล์กล้องลื่นขึ้นสลับเป็น T4)")
```
> **GPU เป็น opt-in ไม่ใช่ข้อบังคับ** — เทรน (10 รูป 3 epoch) จบใน ~15 วิ บน CPU อยู่แล้ว
> โค้ดไม่ hardcode `device` เลย ultralytics ใช้ GPU อัตโนมัติถ้ามี
> เหตุที่ไม่บังคับให้ทุกคนสลับ T4: (1) T4 ฟรีอาจถูกปฏิเสธ/โดน quota กลางคาบ ถ้าโน้ตบุ๊ก
> *ต้อง* มี GPU คนกลุ่มนั้นตกขบวน (2) 20 คนกด Change runtime type พร้อมกัน = งง + เสียเวลา
> ประโยชน์ของ T4 มีแค่ช่วงเซลล์กล้อง (YOLO inference) — hand pose ของ MediaPipe รันบน CPU อยู่ดี
> ใครสลับได้ก็จะเห็น ~15-30fps แทน ~3-8fps · ใครไม่สลับก็รันครบทุกเซลล์เหมือนเดิม

## เซลล์ 2 — helper กล้อง (code, ยาวที่สุดในโน้ตบุ๊ก ~60 บรรทัด)
หัวใจของทั้งไฟล์ เขียนครั้งเดียวใช้ 3 ที่
```python
def run_webcam(process_frame, seconds=20, overlay=True):
    """เปิดกล้องผ่าน browser ส่งเฟรมมาให้ process_frame(bgr) -> bgr"""
```
- ใช้แพตเทิร์นมาตรฐานของ Colab: `createDom()` (สร้าง `<video>` ครั้งเดียว, guard ด้วย
  `if (div !== null)`) + `requestAnimationFrame` loop ที่ resolve promise พร้อมเฟรมล่าสุด
  → `eval_js('stream_frame(label, data)')` คืนเฟรมทีละอันเป็น base64 · `label` = FPS จริง
- **สองโหมด:**
  - `overlay=True` (ทดสอบกล้อง): `<video>` สดเล่นเองที่ ~30fps · `process_frame` diff กับ
    ต้นฉบับ ทำเป็น **PNG โปร่งใส** (เฉพาะกล่อง/ป้าย) วาง `<img>` absolute ทับ → ลื่น
  - `overlay=False` (เซลล์โมเดล 1.4 / 2 / 3): ส่ง**เฟรมที่ประมวลผลแล้วเป็น JPEG ทึบ**
    ทับ `<video>` → ผู้เรียนเห็น FPS จริงของ CPU (กล่องตรงกับเฟรมเป๊ะ ไม่ลอยตามหลัง)
    เพราะ inference ช้า (~3-8fps) การโชว์วิดีโอลื่น 30fps จะทำให้กล่องกับภาพไม่ตรงกัน
- **`.copy()` ก่อนส่งเข้า process_frame สำคัญ** — hand cells วาดทับ array ในที่ ถ้าไม่ copy
  diff (โหมด overlay) จะเป็นศูนย์
- **คลิกที่ภาพ** เพื่อหยุด และตัดเองเมื่อครบ `seconds`
- **ห้ามแยกเป็น `webcamStart()` + `webcamFrame()`** — เฟรมแรกยิงก่อน canvas พร้อม พังเงียบ ๆ
- `try/except` รอบ loop: getUserMedia ถูกปฏิเสธ → propagate มาเป็น exception → พิมพ์
  ข้อความไทย (กด Allow / ย้าย Chrome / รันใหม่) ไม่ใช่ traceback
- มี `run_video(path, process_frame)` ติดมาด้วย — ไม่มีไฟล์วิดีโอใน repo
  ใช้เมื่อกล้องพังจริงๆ โดยวิทยากรอัดคลิปสดแล้วอัปโหลดเข้า Colab

**เซลล์สำรอง (markdown + code ที่คอมเมนต์ไว้)** ต่อท้ายทันที:
```python
# ถ้ากล้องใช้ไม่ได้: อัปโหลดคลิปเข้า Colab แล้วลบ # ข้างล่าง
# run_video("clip.mp4", process_frame)
```

## เซลล์ 2.5 — ทดสอบกล้อง (code)
ต่อจาก markdown "การใช้กล้อง" ทันที — เรียก `run_webcam` ด้วย passthrough สั้น ๆ
```python
def camera_selftest(bgr):
    cv2.putText(bgr, "camera OK", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    return bgr

run_webcam(camera_selftest, seconds=8)
```
เหตุผล: R1 (กล้องถูกบล็อก) เป็นความเสี่ยงอันดับหนึ่ง — ให้ทุกคนกดรันเซลล์นี้ตั้งแต่ต้นคาบ
(ตาม runbook) จะได้เจอปัญหากล้อง/permission ตอนนาทีที่ 5 ไม่ใช่นาทีที่ 40
งบเวลาไม่เพิ่ม เพราะ prompt ขออนุญาตกล้องย้ายมาจากพาร์ท 1.4

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
ตรวจฟอร์แมต label ก่อน แล้วค่อยแสดงผล — label ที่ผิดฟอร์แมตจะเทรนผ่านโดยไม่มี error
แต่ได้โมเดลที่ตรวจไม่เจออะไรเลย ซึ่งวินิจฉัยยากมากกลางห้องเรียน
```python
for txt in Path("data/labels").rglob("*.txt"):
    assert txt.with_suffix(".jpg").name in image_names, f"{txt.name} ไม่มีรูปคู่กัน"
    for line in txt.read_text().split("\n"):
        if not line.strip(): continue
        cls, *box = line.split()
        assert cls == "41", f"{txt.name}: class ต้องเป็น 41 (cup ใน COCO)"
        assert all(0 <= float(v) <= 1 for v in box), f"{txt.name}: พิกัดต้อง normalize"
print("label ผ่านการตรวจทั้งหมด")
```
จากนั้นแสดงกริด 15 รูปพร้อมกล่อง ground-truth ที่วาดจากไฟล์ `.txt` — ให้ผู้เรียนเห็นว่า
"label" ไม่ใช่เวทมนตร์ แต่คือตัวเลข 5 ตัวต่อหนึ่งกล่อง และการเห็นกล่องอยู่ถูกที่
คือการตรวจที่ครอบคลุมกว่า assert ข้างบน
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
model = YOLO("yolo11n.pt")          # โมเดลนี้รู้จัก cup (คลาส 41) อยู่แล้ว
model.train(data="data/cup.yaml", epochs=3, imgsz=640, batch=4, seed=0, plots=True)
```
> `cup.yaml` ใช้สคีมา COCO 80 คลาส (ไม่ใช่ 1 คลาส) — ถ้าลดเหลือ 1 คลาส ultralytics จะ
> reinit หัว classifier ทิ้งน้ำหนัก `cup` ของ COCO แล้ว 3 epoch/10 รูปจะตรวจไม่เจออะไรเลย
> (ทดสอบจริงบน 8.3.253 แล้ว) ดู [03-models.md](03-models.md)

### เซลล์ 7 (markdown) — **จุดที่ห้ามข้าม**
> เราไม่ได้สอนโมเดลให้รู้จัก "แก้ว" ตั้งแต่ต้น — มันรู้จักอยู่แล้วจาก COCO (เป็น 1 ใน 80 อย่าง)
> เราเก็บความรู้เดิมไว้ทั้งหมดแล้วขยับเฉพาะส่วนที่เกี่ยวกับแก้ว ด้วยรูป 10 ใบ
> ถ้าลบความรู้เดิมแล้วเริ่มจากศูนย์ ข้อมูลเท่านี้จะตรวจไม่เจออะไรเลย

---

## 1.3 Result

### เซลล์ 8 (code) — หลักฐานหลัก
กริดรูป test 3 ใบพร้อมกล่องที่ทำนาย เรียก `model(p, conf=0.25, classes=[41])`
(โมเดลจิ๋วเก่งพอบนรูปนิ่ง — บทเรียน "ข้อจำกัด" ย้ายไปอยู่ที่ realtime + พาร์ท 3)

### เซลล์ 9 (code) — ของแถม
```python
metrics = model.val(split="test", classes=[41])
print("mAP50:", round(metrics.box.map50, 3))
```
markdown กำกับ: *"val มี 2 รูป — ตัวเลขนี้ขยับทีละ 50% ต่อหนึ่งรูป อย่าเอาไปอ้างอิงที่ไหน
ดูรูปข้างบนเถอะ"*

---

## 1.4 Realtime

### เซลล์ 10 (code)
```python
def process_frame(bgr):
    r = model(bgr, conf=0.25, classes=[41], verbose=False)[0]
    return r.plot()

run_webcam(process_frame, seconds=20)
```
markdown: ให้ลองเอาแก้วเข้า-ออกเฟรม, เอียงแก้ว, เอามือบัง, แล้วลองเอา**ขวดน้ำ**มาวางข้างๆ
(จะไม่ขึ้นกรอบ เพราะ `classes=[41]` สั่งให้สนใจแค่แก้ว ทั้งที่โมเดลรู้จักขวด — การกำหนดขอบเขต
ให้แคบคือส่วนหนึ่งของการทำระบบให้เชื่อถือได้)

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
เรียก detection ด้วย `model(bgr, conf=0.25, classes=[41])` เหมือนเซลล์ 10

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
