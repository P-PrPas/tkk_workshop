"""คนถือแก้ว — เวอร์ชัน "ใช้งานได้จริง" ที่วิทยากรรันโชว์หน้าห้อง

กฎยังเหมือนโน้ตบุ๊ก: มือ (ไม่แบกว้าง) อยู่บนแก้ว → กำลังถือ
ที่เพิ่มคือวิศวกรรมรอบ ๆ กฎ:
  1. threaded capture — อ่านกล้องอีกเธรด เก็บเฟรมล่าสุด ไม่มีดีเลย์สะสม
  2. threaded inference — YOLO + MediaPipe อยู่อีกเธรด วาดผลลงเฟรมที่มันวิเคราะห์
     main แค่แสดง → จอเดินเท่า detect FPS · กล่องอยู่บนเฟรมที่ถูกต้องเสมอ ไม่ลอยตามหลัง
  3. tracking ID + CupMemory — แก้วที่โดนมือกำบังจนตรวจไม่เจอ ยังจำกล่องไว้ต่อ
  4. HoldState hysteresis — ป้ายไม่กระพริบ (ขึ้นยาก ลงยากกว่า)
  5. จัดการ error จริง — กล้องหลุดต่อใหม่, โมเดล/กล้องหาย ขึ้นข้อความไทย

หน้าต่างเดียว (`cv2`) — ลากขอบปรับขนาดได้ · กด F เต็มจอ · HUD วาดลงบนเฟรม
รัน:  python app/app.py
ค่าที่ต้องจูนหน้างานอยู่ใน app/config.yaml ทั้งหมด — ห้ามแก้ไฟล์นี้หน้างาน
"""
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

HERE = Path(__file__).parent
HAND_TASK = HERE / "hand_landmarker.task"
HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/1/hand_landmarker.task")
HAND_TASK_MIRROR = HERE.parent / "data" / "hand_landmarker.task"
RELEASE = "https://github.com/P-PrPas/tkk_workshop/releases/download/v1"   # best.pt · best.onnx (imgsz 480)

TIPS = [4, 8, 12, 16, 20]     # ปลายนิ้วทั้งห้า
PIPS = [2, 6, 10, 14, 18]     # ข้อกลางของแต่ละนิ้ว
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ─────────────────────────── กติกาถือแก้ว ───────────────────────────
def count_extended(lm):
    """นับนิ้วที่เหยียด: ปลายนิ้วอยู่ไกลจากข้อมือกว่าข้อกลาง (ทนการหมุนมือ)"""
    w = lm[0]
    d = lambda p: (p.x - w.x) ** 2 + (p.y - w.y) ** 2
    return sum(d(lm[t]) > d(lm[p]) for t, p in zip(TIPS, PIPS))


def hand_state(lm):
    """กำ / แบ / กลาง — เหมือนพาร์ท 2 ของโน้ตบุ๊ก (โชว์ให้ดู ไม่ได้ใช้ตัดสิน HOLDING)"""
    n = count_extended(lm)
    return "FIST" if n <= 1 else "OPEN" if n >= 4 else "MID"


def hand_on_cup(lm, w, h, cup_boxes, min_pts, max_ratio=3.0, margin=0.0):
    """มือ "จับ" แก้วไหม — ไม่ดูว่ากำหรือแบ (แก้วไม่มีหูต้องจับตรง ๆ มือดูเหมือนแบ)
    ดูจาก: มือกับแก้วขนาดใกล้เคียงกัน (ไม่ใช่มือชี้จากไกล) และจุด landmark >= min_pts
    จุดตกอยู่ในกล่องแก้ว (ขยายขอบ margin เท่าตัวแก้ว — แก้วมีหูจับที่หู มือจะอยู่ *ข้าง* กล่อง)"""
    hx = [p.x * w for p in lm]
    hy = [p.y * h for p in lm]
    ha = (max(hx) - min(hx)) * (max(hy) - min(hy))
    for x1, y1, x2, y2 in cup_boxes:
        cw, ch = x2 - x1, y2 - y1
        ca = cw * ch
        if ca <= 0 or not (1 / max_ratio <= ha / ca <= max_ratio):
            continue                        # มือใหญ่/เล็กกว่าแก้วมาก = คนละระยะ ไม่ได้จับ
        mx, my = margin * cw, margin * ch
        if sum(x1 - mx <= x <= x2 + mx and y1 - my <= y <= y2 + my
               for x, y in zip(hx, hy)) >= min_pts:
            return True
    return False


class HoldState:
    """กันป้ายกระพริบ: เห็นติดกัน on_n เฟรมจึงขึ้น HOLDING,
    หายติดกัน off_n เฟรมจึงเลิก — on_n < off_n โดยตั้งใจ (hysteresis)"""

    def __init__(self, on_n, off_n):
        self.on_n, self.off_n = on_n, off_n
        self.hits = self.misses = 0
        self.holding = False
        self.since = 0.0                     # เวลาที่เริ่ม HOLDING — ไว้โชว์ "held for 4.2s"

    def update(self, observed: bool) -> bool:
        if observed:
            self.hits += 1
            self.misses = 0
            if self.hits >= self.on_n and not self.holding:
                self.holding = True
                self.since = time.time()
        else:
            self.misses += 1
            self.hits = 0
            if self.misses >= self.off_n:
                self.holding = False
        return self.holding


class CupMemory:
    """จำกล่องแก้วรายตัว (ตาม track ID) — มือกำบังจนตรวจไม่เจอชั่วคราว
    ก็ยังถือว่าแก้วอยู่ที่เดิมอีก `keep` เฟรม แล้วค่อยลืม"""

    def __init__(self, keep):
        self.keep = keep
        self.tracks = {}   # tid -> [box, misses]

    def update(self, detections):        # detections: list[(tid, box)]
        alive = set()
        for tid, box in detections:
            self.tracks[tid] = [box, 0]
            alive.add(tid)
        for tid, t in list(self.tracks.items()):
            if tid not in alive:
                t[1] += 1
                if t[1] > self.keep:
                    del self.tracks[tid]

    def boxes(self):
        """(tid, box, coasting) — coasting=True คือกล่องจากความจำ ไม่ใช่ detection สด"""
        return [(tid, box, miss > 0) for tid, (box, miss) in self.tracks.items()]


# ─────────── threaded capture — เก็บแค่เฟรมล่าสุด กันดีเลย์สะสม ───────────
class Camera:
    """อ่านกล้องในเธรดแยกแบบไม่หยุด เก็บเฉพาะเฟรมล่าสุด (mirror ให้ด้วยถ้า mirror=True)
    กล้องหลุด → ต่อใหม่ทุก 1 วินาที (`read()[0]` เป็น False ระหว่างนั้น)"""

    def __init__(self, index, mirror=True, width=None, height=None):
        self.index, self.mirror = index, mirror
        self.width, self.height = width, height
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self._stop = False
        self.cap = self._open()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.index)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _loop(self):
        while not self._stop:
            ret, f = (self.cap.read() if self.cap.isOpened() else (False, None))
            if not ret:
                with self.lock:
                    self.ok = False
                self.cap.release()
                time.sleep(1.0)
                self.cap = self._open()
                continue
            if self.mirror:
                f = cv2.flip(f, 1)
            with self.lock:
                self.frame, self.ok = f, True

    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self._stop = True
        self.cap.release()


# ─────────── threaded inference — YOLO + MediaPipe + กติกา ───────────
class Analyzer:
    """เธรดวิเคราะห์ — หยิบเฟรมล่าสุด รันโมเดล **วาดผลลงเฟรมเดียวกัน** เก็บไว้ให้ main แสดง
    จอจึงเดินเท่า FPS ที่ detect ได้จริง (กระตุกกว่า แต่กล่องอยู่บนเฟรมที่มันคิด ไม่ลอยตามหลัง)"""

    def __init__(self, cam, model, hands, cfg):
        self.cam, self.model, self.hands, self.cfg = cam, model, hands, cfg
        self.device = pick_device(cfg)          # auto: cuda → mps → cpu (onnx บังคับ cpu)
        self.device_label = self.device.upper() + (" · onnx" if str(cfg["model_path"]).endswith(".onnx") else "")
        self.hold = HoldState(cfg["hold_frames"], cfg["release_frames"])
        self.cups = CupMemory(cfg.get("cup_memory_frames", 15))
        self.lock = threading.Lock()
        self.frame = None            # เฟรมที่วาดผลแล้ว พร้อมแสดง
        self.fps = 0.0
        self.debug = False
        self.toast = None            # (ข้อความ, เวลาหมดอายุ) — ป็อปอัปสั้น ๆ เวลากดปุ่ม
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        c = self.cfg
        prev = time.time()
        while not self._stop:
            try:
                ok, frame = self.cam.read()
                if not ok or frame is None:
                    with self.lock:
                        self.frame = splash("reconnecting camera")
                    time.sleep(0.1)
                    continue
                h, w = frame.shape[:2]

                r = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                     imgsz=c.get("imgsz", 480), conf=c["conf"],
                                     device=self.device, classes=[c["cup_class"]],
                                     verbose=False)[0]
                dets = []
                if r.boxes is not None and r.boxes.id is not None:
                    for box, tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist()):
                        dets.append((int(tid), box))
                self.cups.update(dets)
                id_boxes = [(tid, box) for tid, box, _ in self.cups.boxes()]
                cup_boxes = [box for _, box in id_boxes]

                res = self.hands.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    int(time.monotonic() * 1000))
                mgn = c.get("grip_box_margin", 0.35)
                hands_out, observed, held_id = [], False, None
                for lm in (res.hand_landmarks or []):
                    on_cup = hand_on_cup(lm, w, h, cup_boxes,
                                         c.get("grip_min_points", 10),
                                         c.get("grip_max_size_ratio", 4.0), mgn)
                    observed = observed or on_cup
                    pts_in = max((sum(x1 - mgn * (x2 - x1) <= p.x * w <= x2 + mgn * (x2 - x1)
                                      and y1 - mgn * (y2 - y1) <= p.y * h <= y2 + mgn * (y2 - y1)
                                      for p in lm)
                                  for x1, y1, x2, y2 in cup_boxes), default=0)
                    if on_cup:               # แก้วใบที่มือทับจุดมากสุด = ใบที่กำลังถือ
                        held_id = max(((sum(x1 <= p.x * w <= x2 and y1 <= p.y * h <= y2
                                            for p in lm), tid)
                                       for tid, (x1, y1, x2, y2) in id_boxes),
                                      default=(0, None))[1]
                    hands_out.append(([(int(p.x * w), int(p.y * h)) for p in lm],
                                      on_cup, hand_state(lm), pts_in))
                holding = self.hold.update(observed)

                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 / max(now - prev, 1e-3)
                prev = now
                held_s = now - self.hold.since if holding else 0.0
                toast = self.toast[0] if self.toast and now < self.toast[1] else None
                view = draw(frame, self.cups.boxes(), hands_out, holding, held_s, held_id,
                            self.fps, self.device_label, self.debug, toast)
                with self.lock:
                    self.frame = view
            except Exception as e:
                if self._stop:
                    break               # กำลังปิดโปรแกรม — เงียบไว้
                print("analyzer:", e)
                time.sleep(0.2)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self._stop = True
        self._thread.join(timeout=3)
        try:
            self.hands.close()          # กัน mediapipe บ่นตอนปิดโปรแกรม
        except Exception:
            pass


# ─────────────────────────── setup ───────────────────────────
def load_config():
    # encoding ระบุชัด — Windows default เป็น cp1252 อ่านคอมเมนต์ไทยใน yaml ไม่ได้
    with open(HERE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fetch(name):
    """โหลดไฟล์โมเดลจาก GitHub Release มาไว้ที่ app/models/ ถ้ายังไม่มี"""
    dst = HERE / "models" / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"โหลด {name} จาก GitHub Release ครั้งแรก...")
        try:
            urllib.request.urlretrieve(f"{RELEASE}/{name}", dst)
        except urllib.error.URLError:
            pass
    return dst if dst.exists() else None


def pick_device(cfg):
    """เลือก device เอง: config ระบุมา (cuda/mps/cpu) ใช้ตามนั้น · เว้นว่าง/auto → ไล่หาที่เร็วสุด
    onnx บังคับ cpu เสมอ (onnxruntime CUDA EP พังง่ายบนบางเครื่อง)"""
    d = str(cfg.get("device") or "auto").lower()
    if d not in ("auto", "none", ""):
        return d
    if str(cfg["model_path"]).endswith(".onnx"):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():        # Apple Silicon
        return "mps"
    return "cpu"


def load_model(cfg):
    """default = best.pt (GPU ใช้ CUDA เอง) · ตั้ง model_path เป็น .onnx สำหรับ CPU (ต้องมี onnxruntime)
    ไฟล์มาจาก Release ถ้าโหลดไม่ได้ก็ export .onnx จาก best.pt ให้เอง"""
    p = Path(cfg["model_path"])
    if not p.is_absolute():
        p = HERE / p

    if not p.exists() and p.name in ("best.pt", "best.onnx"):
        _fetch(p.name)
    if not p.exists() and p.name == "best.onnx":          # Release โหลดไม่ได้ → export เอง
        pt = _fetch("best.pt")
        if pt:
            print(f"export best.pt -> onnx (imgsz {cfg.get('imgsz', 480)})...")
            out = YOLO(str(pt)).export(format="onnx", imgsz=cfg.get("imgsz", 480),
                                       dynamic=False, verbose=False)
            Path(out).replace(p)

    if not p.exists():
        raise SystemExit(
            f"\nหาไฟล์โมเดลไม่เจอ: {p}\n"
            "โหลดเอง:  gh release download v1 -R P-PrPas/tkk_workshop -p best.pt -D app/models\n"
            "หรือใช้แผนสำรอง: model_path: yolo11m.pt  +  cup_class: 41  ใน config.yaml\n"
        )
    dev = pick_device(cfg)
    auto = str(cfg.get("device") or "auto").lower() in ("auto", "none", "")
    if dev == "cuda":
        print("YOLO device: CUDA", torch.cuda.get_device_name(0))
    elif dev == "mps":
        print("YOLO device: MPS (Apple GPU)")
    elif dev == "cpu" and auto and p.suffix == ".pt":
        print("──────────────────────────────────────────────────────────────")
        print("  YOLO auto → CPU (~5 FPS) เพราะไม่เจอ GPU:")
        print("  · NVIDIA (Win/Linux): pip install --force-reinstall torch torchvision \\")
        print("        --index-url https://download.pytorch.org/whl/cu124   (torch ตอนนี้เป็นตัว +cpu?)")
        print("  · Apple Silicon: pip install torch torchvision  (PyPI มี MPS อยู่แล้ว)")
        print("  · CPU ล้วน / mac Intel: model_path: models/best.onnx  +  pip install onnxruntime")
        print("──────────────────────────────────────────────────────────────")
    else:
        print("YOLO device:", dev.upper(), "(onnx)" if p.suffix == ".onnx" else "")
    return YOLO(str(p))


def load_hand_landmarker():
    if not HAND_TASK.exists():
        if HAND_TASK_MIRROR.exists():
            HAND_TASK.write_bytes(HAND_TASK_MIRROR.read_bytes())
        else:
            print("ดาวน์โหลด hand_landmarker.task ครั้งแรก...")
            try:
                urllib.request.urlretrieve(HAND_TASK_URL, HAND_TASK)
            except urllib.error.URLError as e:
                raise SystemExit(
                    f"\nโหลด hand_landmarker.task ไม่ได้ ({e})\n"
                    f"ดาวน์โหลดเองจาก {HAND_TASK_URL}\nแล้ววางไว้ที่ {HAND_TASK}\n"
                )
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_TASK)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


# ─────────────────────────── หน้าตา (ธีม + ตัวช่วยวาด) ───────────────────────────
# ทุกข้อความบนจอเป็นภาษาอังกฤษ — DejaVu ไม่มี glyph ไทย (ไทยไปที่ terminal)
INK   = (22, 24, 28)      # พื้นแผงโปร่งแสง  (BGR)
FG    = (245, 247, 249)   # อักษรหลัก
MUTED = (150, 156, 165)   # อักษรรอง
OK    = (105, 205, 100)   # HOLDING / มือจับแก้ว  (เขียว)
WARN  = (70, 180, 255)    # แก้วจากความจำ  (เหลืองอำพัน)
BAD   = (78, 82, 235)     # NOT HOLDING  (แดง)
CUP   = (225, 195, 95)    # กล่องแก้วสด  (ฟ้า)
STATE_COLOR = {"FIST": OK, "OPEN": (205, 205, 210), "MID": WARN}   # สีโครงมือตามท่า (พาร์ท 2)

try:                        # DejaVu มากับ matplotlib (dep ของ ultralytics อยู่แล้ว)
    import matplotlib
    _FDIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    _REG, _BOLD = _FDIR / "DejaVuSans.ttf", _FDIR / "DejaVuSans-Bold.ttf"
except Exception:
    _REG = _BOLD = None
_FONTS = {}


def font(px, bold=False):
    px = max(9, int(px))
    hit = _FONTS.get((px, bold))
    if hit is None:
        for cand in ((_BOLD if bold else _REG),
                     ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
            try:
                hit = ImageFont.truetype(str(cand), px)
                break
            except (OSError, TypeError):
                hit = None
        hit = hit or ImageFont.load_default(px)
        _FONTS[(px, bold)] = hit
    return hit


def _round_rect(img, x1, y1, x2, y2, r, color):
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def panel(frame, box, alpha=0.5, r=14, color=INK):
    """สี่เหลี่ยมมุมมนโปร่งแสง — พื้นหลังของแผง HUD ทุกอัน"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    ov = roi.copy()
    _round_rect(ov, 0, 0, x2 - x1 - 1, y2 - y1 - 1, r, color)
    cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, roi)


def scrim(frame, strength=0.55):
    """ไล่เฉดมืดขอบบน-ล่าง — อ่านตัวหนังสือออกไม่ว่าพื้นหลังจะสว่างแค่ไหน"""
    h = frame.shape[0]
    b = max(1, int(h * 0.26))
    ramp = np.linspace(strength, 0.0, b, dtype=np.float32)[:, None, None]
    frame[:b] = (frame[:b] * (1 - ramp)).astype(np.uint8)
    frame[h - b:] = (frame[h - b:] * (1 - ramp[::-1])).astype(np.uint8)


def _put(items, x, y, s, ft, col, anchor="la"):
    items.append((int(x), int(y), s, ft, (col[2], col[1], col[0]), anchor))


def _flush(frame, items):
    """เขียนข้อความทั้งหมดทีเดียวด้วย PIL (คมกว่า Hershey มาก) แล้วคืนเป็น BGR"""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for x, y, s, ft, col, anchor in items:
        d.text((x, y), s, font=ft, fill=col, anchor=anchor)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _chip(frame, T, x, y, s, col, u):
    ft = font(15 * u, True)
    pad = max(4, int(7 * u))
    asc = ft.getmetrics()[0]
    panel(frame, (x, y, x + ft.getlength(s) + pad * 2, y + asc + pad * 2), 0.62, pad)
    _put(T, x + pad, y + pad, s, ft, col)


def draw(frame, cups, hands, holding, held_s, held_id, fps, device, debug, toast):
    """วาด HUD ทั้งหมดลงบนเฟรม แล้วคืนเฟรมที่วาดข้อความเสร็จ · u = สเกลตามความสูงจอ"""
    h, w = frame.shape[:2]
    u = h / 720.0
    px = lambda v: max(1, int(v * u))
    scrim(frame)
    T = []

    # ── กล่องแก้ว: มุมเหลี่ยม (look แบบ detection) + ป้ายชื่อ ──
    # กล่อง coasting (จาก CupMemory) ไม่วาด — กติกายังใช้เช็กอยู่เบื้องหลัง แค่ไม่โชว์บนจอ
    cups = [c for c in cups if not c[2]]
    for tid, box, _coasting in cups:
        x1, y1, x2, y2 = map(int, box)
        active = holding and tid == held_id
        col = OK if active else CUP
        L, t = px(26), px(2)
        for cx, sx in ((x1, 1), (x2, -1)):
            for cy, sy in ((y1, 1), (y2, -1)):
                cv2.line(frame, (cx, cy), (cx + sx * L, cy), col, t, cv2.LINE_AA)
                cv2.line(frame, (cx, cy), (cx, cy + sy * L), col, t, cv2.LINE_AA)
        _chip(frame, T, x1, y1 - px(32), f"cup #{tid}" + ("  held" if active else ""), col, u)

    # ── โครงมือ: สีตามท่า (พาร์ท 2) · เขียวเมื่อจับแก้ว ──
    for pts, on_cup, state, pts_in in hands:
        col = OK if on_cup else STATE_COLOR[state]
        for a, b in HAND_EDGES:
            cv2.line(frame, pts[a], pts[b], col, px(2), cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (x, y), px(3), col, -1, cv2.LINE_AA)
        msg = state + ("  on cup" if on_cup else "") + (f"   pts:{pts_in}" if debug else "")
        _put(T, pts[0][0], pts[0][1] + px(18), msg, font(15 * u, True), col, "lm")

    # ── แผงสถานะ (บนซ้าย) ──
    m = px(24)
    big, small = font(38 * u, True), font(15 * u)
    word = "HOLDING" if holding else "NOT HOLDING"
    col = OK if holding else BAD
    sub = f"held for {held_s:0.1f}s" if holding else "waiting for a hand on a cup"
    dot, pad = px(9), px(18)
    inner = dot * 2 + px(12) + max(big.getlength(word), small.getlength(sub))
    panel(frame, (m, m, m + inner + pad * 2, m + px(76)), 0.5, px(16))
    dcx, dcy = m + pad + dot, m + px(28)
    cv2.circle(frame, (dcx, dcy), dot, col, -1, cv2.LINE_AA)
    if holding:
        cv2.circle(frame, (dcx, dcy), dot + px(5), col, px(1), cv2.LINE_AA)
    _put(T, dcx + dot + px(12), dcy, word, big, col, "lm")
    _put(T, m + pad, m + px(50), sub, small, MUTED)

    # ── แผงสถิติ (บนขวา) — FPS นี้คือ detect FPS จริง ไม่ใช่ FPS วิดีโอ ──
    rx = w - m
    _put(T, rx, m + px(4), f"{fps:0.1f}", font(30 * u, True), FG, "ra")
    _put(T, rx, m + px(30), "DETECT FPS", font(12 * u, True), MUTED, "ra")
    _put(T, rx, m + px(52), device, font(13 * u, True), MUTED, "ra")
    _put(T, rx, m + px(70),
         f"{len(hands)} hand{'s' * (len(hands) != 1)}  ·  {len(cups)} cup{'s' * (len(cups) != 1)}",
         font(13 * u), MUTED, "ra")

    # ── แถบปุ่ม (ล่างกลาง) ──
    keys = [("Q", "quit"), ("S", "shot"), ("D", "debug"), ("F", "fullscreen")]
    kf, lf = font(15 * u, True), font(14 * u)
    sz = px(28)
    seg = [sz + px(7) + lf.getlength(lab) + px(18) for _, lab in keys]
    x, y = (w - (sum(seg) - px(18))) / 2, h - m - sz
    for (k, lab), segw in zip(keys, seg):
        panel(frame, (x, y, x + sz, y + sz), 0.55, px(7), (58, 62, 70))
        _put(T, x + sz / 2, y + sz / 2, k, kf, FG, "mm")
        _put(T, x + sz + px(7), y + sz / 2, lab, lf, MUTED, "lm")
        x += segw

    # ── toast (ลอยเหนือแถบปุ่ม ตอนกดปุ่ม) ──
    if toast:
        tf = font(15 * u, True)
        cx, ty, tw = w / 2, h - m - sz - px(46), tf.getlength(toast)
        panel(frame, (cx - tw / 2 - px(16), ty, cx + tw / 2 + px(16), ty + px(32)), 0.62, px(15))
        _put(T, cx, ty + px(16), toast, tf, FG, "mm")

    return _flush(frame, T)


def splash(msg, width=1280):
    """จอรอ (เปิดกล้อง / ต่อกล้องใหม่ / โหลดโมเดล) — สปินเนอร์ + ชื่อแอป"""
    w = int(width)
    h = w * 9 // 16
    img = np.full((h, w, 3), 13, np.uint8)
    cx, cy, r = w // 2, int(h * 0.45), max(16, h // 20)
    a = time.time() * 90
    cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 360, (36, 39, 45), max(2, h // 300), cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r, r), 0, a % 360, a % 360 + 90, OK, max(2, h // 240), cv2.LINE_AA)
    T = []
    _put(T, cx, cy + r + h // 12, "cup-holding detector", font(h / 26, True), FG, "ma")
    _put(T, cx, cy + r + h // 12 + h // 18, msg, font(h / 40), MUTED, "ma")
    return _flush(img, T)


# ─────────────────────────── main ───────────────────────────
WINDOW = "cup-holding detector"


def main():
    cfg = load_config()
    model = load_model(cfg)
    hands = load_hand_landmarker()
    cam = Camera(cfg["camera_index"], mirror=cfg.get("mirror", True),
                 width=cfg.get("camera_width"), height=cfg.get("camera_height"))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    ww = int(cfg.get("window_width", 1280))
    cv2.resizeWindow(WINDOW, ww, ww * 9 // 16)

    t0 = time.time()
    while not cam.read()[0]:
        cv2.imshow(WINDOW, splash("opening camera", ww))
        if cv2.waitKey(80) & 0xFF in (ord("q"), 27):
            cam.release(); cv2.destroyAllWindows(); return
        if time.time() - t0 > 10:
            cam.release()
            cv2.destroyAllWindows()
            raise SystemExit(
                f"\nเปิดกล้องไม่ได้ (camera_index = {cfg['camera_index']})\n"
                "เช็กว่ากล้องเสียบอยู่ ไม่มีโปรแกรมอื่นแย่งใช้ แล้วลองเปลี่ยน "
                "camera_index ใน config.yaml เป็น 1 หรือ 2\n"
            )

    analyzer = Analyzer(cam, model, hands, cfg)
    full = False

    # main แค่แสดงเฟรมที่ analyzer วาดผลเสร็จแล้ว + รับปุ่ม (จอเดินเท่า detect FPS)
    while True:
        frame = analyzer.latest()
        cv2.imshow(WINDOW, frame if frame is not None else splash("starting model", ww))
        k = cv2.waitKey(15) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("f"):
            full = not full
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        elif k == ord("s") and frame is not None:
            fn = f"shot_{int(time.time())}.png"
            cv2.imwrite(fn, frame)
            print("เซฟภาพ:", fn)
            analyzer.toast = (f"saved  {fn}", time.time() + 2.0)
        elif k == ord("d"):
            analyzer.debug = not analyzer.debug
            analyzer.toast = (f"debug {'on' if analyzer.debug else 'off'}", time.time() + 1.5)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break                       # กดปิดหน้าต่าง

    analyzer.release()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
