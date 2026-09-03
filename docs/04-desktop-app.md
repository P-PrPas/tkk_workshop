# 04 — Desktop App

## หน้าที่
ทำงานเดียวกับพาร์ท 3 ของโน้ตบุ๊ก แต่ทำให้ **ใช้งานได้จริง** และวิทยากรรันโชว์
หน้าห้องคนเดียว — ไม่ต้องแจกให้ผู้เรียนติดตั้ง

แอปนี้ **ไม่มีโหมดเทียบโมเดล** ใช้โมเดลดีอย่างเดียว ความเปรียบต่างเกิดจากการที่
ผู้ชมเพิ่งเห็นของจริงในโน้ตบุ๊กมาสิบนาทีก่อน

## ไฟล์
```
app/
├── app.py            # ตรรกะทั้งหมด ~270 บรรทัด (Camera / Analyzer / main)
├── config.yaml       # ค่าที่ต้องปรับหน้างาน
├── requirements.txt  # pin เวอร์ชันให้ตรงกับโน้ตบุ๊ก
├── test_app.py       # self-check ตรรกะ (hysteresis / CupMemory / hand_on_cup) — ไม่ต้องมีกล้อง
└── models/best.pt    # โหลดจาก GitHub Release `v1` อัตโนมัติตอนรันครั้งแรก (gitignore ไว้)
tools/
├── diag.py           # เปิดกล้อง พิมพ์สัญญาณทุกเฟรม — ไว้ debug ว่า HOLDING พังตรงไหน
└── optimize.py       # ลอง export ONNX/OpenVINO แล้ววัดว่าเร็วขึ้นบนเครื่องนี้ไหม
```
รัน (บนเครื่องที่มีกล้อง):
```bash
git clone --recursive https://github.com/P-PrPas/tkk_workshop.git
cd tkk_workshop
pip install -r app/requirements.txt
python app/app.py          # ครั้งแรกโหลด best.pt + hand_landmarker.task ให้เอง
```
`hand_landmarker.task` ใช้จาก submodule `data/` ถ้ามี ไม่งั้นโหลดจาก MediaPipe

ไฟล์เดียวโดยตั้งใจ ไม่มี package ไม่มี class hierarchy — วิทยากรต้องเปิดโค้ดโชว์
ได้กลางห้องแล้วผู้เรียนอ่านรู้เรื่องภายในหนึ่งนาที

UI = หน้าต่าง `cv2.imshow` เดียว ไม่ใช้ Qt/Electron เพราะไม่ได้เพิ่มอะไรกับสาร
มีแต่จะเพิ่มเวลาติดตั้งและโอกาสพังหน้างาน

## config.yaml
```yaml
model_path: models/best.pt   # GPU → torch ใช้ CUDA เอง · CPU ล้วน → best.onnx (export ให้อัตโนมัติ)
cup_class: 0                  # 41 ถ้า fallback ไป yolo11m.pt (COCO)
camera_index: 0
mirror: true                 # selfie view
imgsz: 480                   # เล็กลง = เร็วขึ้น (384 เร็วกว่า, 640 แม่นกว่านิด)
conf: 0.25
cup_memory_frames: 15        # แก้วโดนมือบัง → ใช้กล่องเดิมต่ออีกกี่เฟรม
grip_open_max: 3             # นิ้วเหยียด <= ค่านี้ = ยังนับว่ากำ/ประคอง
grip_min_points: 5           # จุดมือ (จาก 21) ต้องอยู่ในกล่องแก้วอย่างน้อยกี่จุด
hold_frames: 3               # เห็นติดกันกี่เฟรมจึงขึ้น HOLDING
release_frames: 6            # หายติดกันกี่เฟรมจึงเลิก (> hold_frames = ไม่กระพริบ)
```
**ทุกตัวเลขที่ต้องจูนหน้างานต้องอยู่ในไฟล์นี้ ห้ามฝังในโค้ด** — แสงในห้องจริงไม่เคย
เหมือนที่ทดสอบ ต้องปรับได้โดยไม่ต้องแก้โค้ด

## สิ่งที่แอปทำแต่โน้ตบุ๊กไม่ทำ (นี่คือเนื้อหาของสาร)

### 1. Threaded inference — จอเดินตาม FPS ที่ detect ได้จริง
`Camera` (เธรดอ่านกล้อง) · `Analyzer` (เธรดหยิบเฟรมล่าสุด รัน YOLO + MediaPipe + กติกา
**วาดผลลงเฟรมเดียวกัน**) · `main` (แค่ `imshow` เฟรมที่ analyzer วาดเสร็จ + รับปุ่ม)
- จอกระตุกเท่า detect FPS แต่**กล่องอยู่บนเฟรมที่โมเดลคิด ไม่ลอยตามหลัง** — ขยับแก้วก็ตามทัน
  (เวอร์ชันก่อนโชว์วิดีโอสด 30fps + กล่องลาก → ดูเหมือน "detect ได้แค่ตอนอยู่นิ่ง")
- `main` เบามาก — inference ไม่บล็อกการรับปุ่ม/การแสดงผล
- ให้เร็วสุด: NVIDIA GPU → ลง torch cu124 (`.pt` บน CUDA ~5ms) · CPU ล้วน → `model_path:
  models/best.onnx` (+ `pip install onnxruntime`) · ลด `imgsz` 480→384 ช่วยทั้งสองทาง

### 2. Tracking ID + ความจำของแก้ว (`CupMemory`)
```python
r = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=..., conf=..., classes=[...])[0]
cups.update([(int(tid), box) for box, tid in ...])   # เก็บกล่องล่าสุดต่อ track ID
cup_boxes = [box for _, box, _coasting in cups.boxes()]
```
- track ID → แต่ละแก้วมีเลขติดตัว "แก้ว #3 ถูกถือมา 4 วินาทีแล้ว"
- **`CupMemory`** — มือกำแก้วบัง object detection จนกล่องแก้ว**หาย** → เก็บกล่องล่าสุดต่ออีก
  `cup_memory_frames` เฟรม (~1 วิที่ 15 FPS) วาดสีจาง + ป้าย `(memory)` → กติกายังมีกล่องให้เช็ก
- **`hand_on_cup`** แทน "กำ + bbox ทับ" — ใช้ได้ทั้งแก้วมีหูและไม่มีหู: มือไม่แบกว้าง
  (`count_extended <= grip_open_max`) **และ** จุด landmark >= `grip_min_points` จุดอยู่ในกล่องแก้ว
  (แก้วไม่มีหูต้องกำตรง ๆ มือไม่เหมือนกำหมัด — เช็กจากจุดที่อยู่บนแก้วแทน)

### 3. State machine + hysteresis
```python
class HoldState:
    """กันป้ายกระพริบ: ขึ้นยาก ลงยากกว่า"""
    def __init__(self, on_n, off_n):
        self.on_n, self.off_n = on_n, off_n
        self.hits = self.misses = 0
        self.holding = False

    def update(self, observed: bool) -> bool:
        if observed:
            self.hits += 1; self.misses = 0
            if self.hits >= self.on_n: self.holding = True
        else:
            self.misses += 1; self.hits = 0
            if self.misses >= self.off_n: self.holding = False
        return self.holding
```
ค่า on/off ไม่เท่ากันโดยตั้งใจ (3 ขึ้น / 6 ลง) — ถ้าเท่ากันจะยังกระพริบตรงขอบ
นี่คือ hysteresis และเป็นคำที่ควรพูดออกไปตรงๆ หน้าห้อง · ยังดูดซับเฟรมที่ pose วืบหายด้วย

### 4. จัดการ error จริง
| เหตุการณ์ | พฤติกรรม |
|---|---|
| เปิดกล้องไม่ได้ตอนเริ่ม | ข้อความไทยบอกว่าต้องทำอะไร แล้วจบโปรแกรม ไม่ใช่ traceback |
| กล้องหลุดกลางทาง | พยายามต่อใหม่ทุก 1 วินาที แสดง "กำลังเชื่อมต่อกล้องใหม่..." บนจอ |
| ไม่เจอมือ / ไม่เจอแก้ว | สถานะปกติ ไม่ใช่ error — วาดเฟรมต่อไปเงียบๆ |
| โหลดโมเดลไม่สำเร็จ | ลองโหลดจาก Release URL เอง; ยังไม่ได้ → บอก path + คำสั่ง `gh release download` + แผนสำรอง yolo11m |

### 5. ปุ่มควบคุม
`q` ออก · `s` บันทึกภาพนิ่ง · `d` สลับโหมด debug (โชว์ ms ของ inference มุมจอ)

## กติกาถือแก้ว
แกนเดียวกับโน้ตบุ๊ก — มือ (ไม่แบกว้าง) อยู่บนแก้ว ห่อด้วย `HoldState`
สิ่งที่เพิ่มเข้ามาไม่ได้ทำให้ *กฎ* ฉลาดขึ้น แต่ทำให้ *สัญญาณเข้ากฎ* ไม่หลุด:
- `CupMemory` — แก้อาการกล่องแก้วหายตอนมือกำบัง
- `hand_on_cup` เช็กจาก "จุดมืออยู่บนแก้ว" แทน "มือกำ" — แก้วไม่มีหูก็จับได้
- `HoldState` ดูดซับเฟรมที่ pose/detection วืบหาย

**ประเด็นที่ต้องเน้นหน้าห้อง: กฎเท่าเดิม — ที่เพิ่มคือ thread + state + memory รอบ ๆ กฎ**

## สิ่งที่จงใจไม่ทำ
- ไม่ทำ installer / py2app / PyInstaller — วิทยากรรัน `python app/app.py` เอง
- ไม่มี GUI framework
- ONNX/OpenVINO เป็น *ทางเลือก* ผ่าน `tools/optimize.py` (บาง CPU เร็วขึ้น บางเครื่องไม่) ไม่ใช่ค่า default
- ไม่รองรับหลายกล้อง
- ไม่บันทึกวิดีโอ/log ลงไฟล์ — `s` เซฟภาพนิ่งพอแล้ว

## เกณฑ์ว่าเสร็จ
- [ ] รันต่อเนื่อง 5 นาทีโดยไม่ crash
- [ ] ขยับแก้ว → กล่องตามทัน (ไม่ค้างที่ตำแหน่งเก่า) · จอเดิน ≥ ~10 fps บนเครื่องที่จะใช้
- [ ] ป้าย HOLDING ไม่กระพริบเมื่อถือแก้วนิ่ง
- [ ] ถอดปลั๊กกล้องแล้วเสียบกลับ → กลับมาทำงานเองโดยไม่ต้องรีสตาร์ท
- [ ] แก้วสองใบพร้อมกัน แต่ละใบมี ID ของตัวเองที่ไม่สลับกัน
