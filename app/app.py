"""เช็กลิสต์ผู้ตรวจ — เวอร์ชัน "ใช้งานได้จริง" ที่วิทยากรรันโชว์หน้าห้อง

โจทย์: operator หยิบชิ้นงานไหนออกมาตรวจแล้วบ้าง
กฎยังเหมือนโน้ตบุ๊ก: มือ (ไม่แบกว้าง) อยู่บนแก้ว → กำลังถือ
ที่เพิ่มคือวิศวกรรมรอบ ๆ กฎ:
  1. threaded capture — อ่านกล้องอีกเธรด เก็บเฟรมล่าสุด ไม่มีดีเลย์สะสม
  2. threaded inference — YOLO + MediaPipe อยู่อีกเธรด วาดผลลงเฟรมที่มันวิเคราะห์
     GUI แค่แสดง → ภาพเดินเท่า detect FPS · กล่องอยู่บนเฟรมที่ถูกต้องเสมอ ไม่ลอยตามหลัง
  3. tracking ID + CupMemory — แก้วที่โดนมือกำบังจนตรวจไม่เจอ ยังจำกล่องไว้ต่อ
  4. HoldState hysteresis — ป้ายไม่กระพริบ (ขึ้นยาก ลงยากกว่า)
  5. Inspection — เช็กลิสต์ว่าชิ้นไหนถูกหยิบไปตรวจแล้ว + เส้นทางที่มันเคลื่อนที่มา
     ต้องถูกจับติดกันหลายเฟรมจึงติ๊ก มือเฉียดผ่านไม่นับ · กด R เริ่มรอบใหม่
  6. จัดการ error จริง — กล้องหลุดต่อใหม่, โมเดล/กล้องหาย ขึ้นข้อความไทย

GUI = Tkinter (มากับ Python ไม่ต้องลงเพิ่ม): ภาพซ้าย · เช็กลิสต์+ปุ่มขวา · แถบสถานะล่าง
รัน:  python app/app.py
ค่าที่ต้องจูนหน้างานอยู่ใน app/config.yaml ทั้งหมด — ห้ามแก้ไฟล์นี้หน้างาน
"""
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageTk
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

try:
    import tkinter as tk
except ImportError:                 # ลินุกซ์บางดิสโทรแยก tk ออกจาก python
    raise SystemExit("\nไม่มี tkinter — ติดตั้งก่อน:  sudo apt install python3-tk\n")

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


class Inspection:
    """เช็กลิสต์รอบตรวจ — ชิ้นไหนถูกหยิบออกมาตรวจแล้ว + เส้นทางที่มันเคลื่อนที่มา

    ต้องถูกจับสะสมครบ `need` เฟรมจึงติ๊กถูก (มือเฉียดผ่านไม่นับ) หลุดไปเฟรมเดียว
    ถอยแค่หนึ่ง ไม่ล้างทิ้ง — ตัวนับแบบรั่ว ทนเฟรมที่ pose วืบหายเหมือน HoldState
    ติ๊กแล้วติ๊กเลย วางคืนก็ยังตรวจแล้ว จนกว่าจะกด reset เริ่มรอบใหม่"""

    def __init__(self, need, trail_len, forget_seconds):
        self.need, self.trail_len, self.forget = need, trail_len, forget_seconds
        self.reset()

    def reset(self):
        self.trails = {}      # tid -> deque จุดกึ่งกลางกล่อง
        self.hits = {}        # tid -> เฟรมสะสมที่มือจับอยู่
        self.picked = {}      # tid -> เวลาที่นับว่าตรวจแล้ว
        self.seen = {}        # tid -> เวลาที่เห็นล่าสุด
        self.started = time.time()

    def update(self, id_boxes, held_ids):
        now = time.time()
        for tid, (x1, y1, x2, y2) in id_boxes:
            self.seen[tid] = now
            self.trails.setdefault(tid, deque(maxlen=self.trail_len)).append(
                (int((x1 + x2) / 2), int((y1 + y2) / 2)))
        for tid in set(self.hits) | set(held_ids):
            self.hits[tid] = max(0, min(self.need, self.hits.get(tid, 0)
                                        + (1 if tid in held_ids else -1)))
            if self.hits[tid] >= self.need:
                self.picked.setdefault(tid, now)
        # ลืม id ที่หายไปนานและไม่เคยถูกหยิบ — tracker แจก id ใหม่เรื่อย ๆ เช็กลิสต์จะรก
        for tid, last in list(self.seen.items()):
            if now - last > self.forget and tid not in self.picked:
                del self.seen[tid]
                self.trails.pop(tid, None)
                self.hits.pop(tid, None)

    def rows(self):
        """[(tid, ตรวจแล้ว?)] เรียงตามเลข — ให้ GUI เอาไปวาดเช็กลิสต์"""
        return sorted((tid, tid in self.picked) for tid in set(self.seen) | set(self.picked))


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
        self.insp = Inspection(cfg.get("pick_frames", 8), cfg.get("trail_length", 60),
                               cfg.get("forget_seconds", 4.0))
        self.lock = threading.Lock()
        self.frame = None            # เฟรมที่วาดผลแล้ว พร้อมแสดง
        self.seq = 0                 # เลขเฟรม — GUI ใช้เช็กว่ามีของใหม่ค่อยแปลงภาพ
        self.status = {"holding": False, "held_s": 0.0, "fps": 0.0, "hands": 0,
                       "rows": [], "device": self.device_label}
        self.debug = False
        self._reset = False          # ตั้งจาก GUI — เคลียร์ในเธรดนี้ ไม่ไปยุ่งกับ tracker ข้ามเธรด
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def reset(self):
        """เริ่มรอบตรวจใหม่ — ขอไว้ก่อน เธรดวิเคราะห์จะทำให้ตอนต้นเฟรมถัดไป"""
        self._reset = True

    def _do_reset(self):
        self.insp.reset()
        self.cups = CupMemory(self.cfg.get("cup_memory_frames", 15))
        self.hold = HoldState(self.cfg["hold_frames"], self.cfg["release_frames"])
        try:                        # ให้ ByteTrack เริ่มนับ id ใหม่จาก 1 ด้วย
            for t in self.model.predictor.trackers:
                t.reset()
        except Exception:
            pass                    # ยังไม่เคย track สักเฟรม / ultralytics เปลี่ยน API — ไม่ใช่เรื่องคอขาดบาดตาย

    def _loop(self):
        c = self.cfg
        prev = time.time()
        while not self._stop:
            try:
                if self._reset:
                    self._reset = False
                    self._do_reset()
                ok, frame = self.cam.read()
                if not ok or frame is None:
                    with self.lock:
                        self.frame, self.seq = splash("reconnecting camera"), self.seq + 1
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
                hands_out, held_ids = [], set()
                for lm in (res.hand_landmarks or []):
                    on_cup = hand_on_cup(lm, w, h, cup_boxes,
                                         c.get("grip_min_points", 10),
                                         c.get("grip_max_size_ratio", 4.0), mgn)
                    pts_in = max((sum(x1 - mgn * (x2 - x1) <= p.x * w <= x2 + mgn * (x2 - x1)
                                      and y1 - mgn * (y2 - y1) <= p.y * h <= y2 + mgn * (y2 - y1)
                                      for p in lm)
                                  for x1, y1, x2, y2 in cup_boxes), default=0)
                    if on_cup:               # แก้วใบที่มือทับจุดมากสุด = ใบที่มือนี้ถืออยู่
                        best = max(((sum(x1 <= p.x * w <= x2 and y1 <= p.y * h <= y2
                                         for p in lm), tid)
                                    for tid, (x1, y1, x2, y2) in id_boxes), default=(0, None))
                        if best[1] is not None:
                            held_ids.add(best[1])
                    hands_out.append(([(int(p.x * w), int(p.y * h)) for p in lm],
                                      on_cup, hand_state(lm), pts_in))
                holding = self.hold.update(bool(held_ids))
                self.insp.update(id_boxes, held_ids)

                now = time.time()
                fps = 0.9 * self.status["fps"] + 0.1 / max(now - prev, 1e-3)
                prev = now
                view = draw(frame, self.cups.boxes(), hands_out, held_ids,
                            self.insp, self.debug)
                with self.lock:
                    self.frame, self.seq = view, self.seq + 1
                    self.status = {
                        "holding": holding,
                        "held_s": now - self.hold.since if holding else 0.0,
                        "fps": fps, "hands": len(hands_out),
                        "rows": self.insp.rows(), "device": self.device_label,
                        "round_s": now - self.insp.started,
                    }
            except Exception as e:
                if self._stop:
                    break               # กำลังปิดโปรแกรม — เงียบไว้
                print("analyzer:", e)
                time.sleep(0.2)

    def latest(self):
        """(เลขเฟรม, เฟรม, สถานะ) — GUI เทียบเลขเฟรมก่อน ไม่ต้องแปลงภาพเดิมซ้ำ"""
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return self.seq, frame, dict(self.status)

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
OK    = (105, 205, 100)   # มือจับแก้ว  (เขียว)
WARN  = (70, 180, 255)    # เหลืองอำพัน
DONE  = (105, 205, 100)   # แก้วที่ตรวจแล้ว  (เขียว)
TODO  = (60, 190, 245)    # แก้วที่ยังไม่ตรวจ  (เหลือง)
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


def draw(frame, cups, hands, held_ids, insp, debug):
    """วาดลงบนภาพเฉพาะสิ่งที่ต้องอยู่ *ตรงตำแหน่ง* — เส้นทาง กล่องแก้ว โครงมือ
    สถานะ/FPS/เช็กลิสต์เป็นวิดเจ็ตของ GUI ไม่ต้องเขียนทับภาพ (และพิมพ์ไทยได้)
    u = สเกลตามความสูงเฟรม ทำให้เส้น/ตัวหนังสือหนาเท่ากันทุกความละเอียด"""
    u = frame.shape[0] / 720.0
    px = lambda v: max(1, int(v * u))
    T = []

    # ── แก้ว: เส้นทางที่เคลื่อนมา + กรอบมุมเหลี่ยม + ป้าย id (ติ๊กถูกถ้าตรวจแล้ว) ──
    # กล่อง coasting (จาก CupMemory) ไม่วาด — กติกายังใช้เช็กอยู่เบื้องหลัง แค่ไม่โชว์บนจอ
    for tid, box, coasting in cups:
        if coasting:
            continue
        done = tid in insp.picked
        col = DONE if done else TODO
        trail = insp.trails.get(tid)
        if trail and len(trail) > 1:
            cv2.polylines(frame, [np.array(trail, np.int32)], False, col, px(2), cv2.LINE_AA)
        x1, y1, x2, y2 = map(int, box)
        L, t = px(26), px(4 if tid in held_ids else 2)
        for cx, sx in ((x1, 1), (x2, -1)):
            for cy, sy in ((y1, 1), (y2, -1)):
                cv2.line(frame, (cx, cy), (cx + sx * L, cy), col, t, cv2.LINE_AA)
                cv2.line(frame, (cx, cy), (cx, cy + sy * L), col, t, cv2.LINE_AA)
        _chip(frame, T, x1, y1 - px(32), f"#{tid}" + ("  ✓" if done else ""), col, u)

    # ── โครงมือ: สีตามท่า (พาร์ท 2) · เขียวเมื่อจับแก้ว ──
    for pts, on_cup, state, pts_in in hands:
        col = OK if on_cup else STATE_COLOR[state]
        for a, b in HAND_EDGES:
            cv2.line(frame, pts[a], pts[b], col, px(2), cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (x, y), px(3), col, -1, cv2.LINE_AA)
        msg = state + ("  on cup" if on_cup else "") + (f"   pts:{pts_in}" if debug else "")
        _put(T, pts[0][0], pts[0][1] + px(18), msg, font(15 * u, True), col, "lm")

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


# ─────────────────────────── GUI (tkinter) ───────────────────────────
# Segoe UI / Helvetica มี glyph ไทย — ดังนั้นข้อความบน GUI เป็นไทยได้ (ต่างจากบนเฟรม)
UI = {"win32": "Segoe UI", "darwin": "Helvetica"}.get(sys.platform, "Noto Sans Thai")
BG, CARD, LINE = "#0f1216", "#171b22", "#252b35"
TXT, DIM = "#eef1f5", "#98a2b0"
G, Y, R = "#69cd64", "#f5be3c", "#eb524e"


def _button(parent, text, cmd, accent=False):
    return tk.Button(parent, text=text, command=cmd, cursor="hand2",
                     bg=G if accent else CARD, fg="#0f1216" if accent else TXT,
                     activebackground=G if accent else LINE,
                     activeforeground="#0f1216" if accent else TXT,
                     font=(UI, 11, "bold" if accent else "normal"),
                     relief="flat", bd=0, padx=10, pady=9, highlightthickness=0)


class App:
    """หน้าต่างเดียว: ภาพซ้าย · เช็กลิสต์ขวา · แถบสถานะล่าง
    GUI ไม่คิดอะไรเอง แค่หยิบเฟรม+สถานะล่าสุดจาก Analyzer มาแสดงทุก 20ms"""

    def __init__(self, root, analyzer, cfg):
        self.root, self.an, self.cfg = root, analyzer, cfg
        self.seq, self.photo, self.shown_rows, self.full = -1, None, None, False
        self.msg = ("", 0.0)              # ข้อความชั่วคราวบนแถบสถานะ
        self.t0 = time.time()

        vw = int(cfg.get("window_width", 1280))
        root.title("เช็กลิสต์ผู้ตรวจ — cup inspection")
        root.configure(bg=BG)
        root.geometry(f"{vw + 330}x{vw * 9 // 16 + 60}")
        root.minsize(900, 520)
        root.protocol("WM_DELETE_WINDOW", self.quit)

        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True)
        self.video = tk.Label(body, bg="#000000", bd=0)
        self.video.pack(side="left", fill="both", expand=True)
        self._sidebar(body)
        self._statusbar(root)

        for key, fn in (("q", self.quit), ("<Escape>", self.quit), ("r", self.reset),
                        ("s", self.shot), ("d", self.toggle_debug), ("f", self.fullscreen)):
            root.bind(key if key.startswith("<") else f"<KeyPress-{key}>", lambda e, f=fn: f())
        self.tick()

    # ── โครงหน้าตา ──
    def _sidebar(self, parent):
        side = tk.Frame(parent, bg=BG, width=310)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="รายการตรวจ", bg=BG, fg=TXT, font=(UI, 15, "bold")).pack(
            anchor="w", padx=18, pady=(16, 0))
        tk.Label(side, text="ชิ้นงานไหนถูกหยิบออกมาตรวจแล้ว", bg=BG, fg=DIM,
                 font=(UI, 9)).pack(anchor="w", padx=18)

        card = tk.Frame(side, bg=CARD)
        card.pack(fill="x", padx=14, pady=12)
        self.count = tk.Label(card, text="0 / 0", bg=CARD, fg=TXT, font=(UI, 26, "bold"))
        self.count.pack(anchor="w", padx=14, pady=(12, 0))
        tk.Label(card, text="ตรวจแล้ว", bg=CARD, fg=DIM, font=(UI, 9)).pack(anchor="w", padx=14)
        self.bar = tk.Canvas(card, height=6, bg=LINE, highlightthickness=0)
        self.bar.pack(fill="x", padx=14, pady=(10, 14))

        self.list = tk.Frame(side, bg=BG)     # ponytail: ไม่มี scrollbar — เกิน ~10 ชิ้นค่อยใส่
        self.list.pack(fill="both", expand=True, padx=14)

        keys = tk.Frame(side, bg=BG)
        keys.pack(fill="x", padx=14, pady=(0, 12))
        _button(keys, "เริ่มรอบใหม่   R", self.reset, accent=True).pack(fill="x", pady=(0, 6))
        row = tk.Frame(keys, bg=BG)
        row.pack(fill="x")
        for text, fn in (("บันทึกภาพ  S", self.shot), ("เต็มจอ  F", self.fullscreen),
                         ("ออก  Q", self.quit)):
            _button(row, text, fn).pack(side="left", expand=True, fill="x", padx=2)

    def _statusbar(self, parent):
        bar = tk.Frame(parent, bg=CARD, height=36)
        bar.pack(fill="x", side="bottom")
        self.dot = tk.Label(bar, text="●", bg=CARD, fg=DIM, font=(UI, 12))
        self.dot.pack(side="left", padx=(14, 6), pady=7)
        self.state = tk.Label(bar, text="กำลังเริ่ม...", bg=CARD, fg=TXT, font=(UI, 11, "bold"))
        self.state.pack(side="left")
        self.stats = tk.Label(bar, text="", bg=CARD, fg=DIM, font=(UI, 10))
        self.stats.pack(side="right", padx=14)

    # ── ปุ่ม ──
    def reset(self):
        self.an.reset()
        self.flash("เริ่มรอบตรวจใหม่แล้ว")

    def shot(self):
        _, frame, _ = self.an.latest()
        if frame is not None:
            fn = f"shot_{int(time.time())}.png"
            cv2.imwrite(fn, frame)
            print("เซฟภาพ:", fn)
            self.flash(f"บันทึก {fn}")

    def toggle_debug(self):
        self.an.debug = not self.an.debug
        self.flash(f"debug {'เปิด' if self.an.debug else 'ปิด'}")

    def fullscreen(self):
        self.full = not self.full
        self.root.attributes("-fullscreen", self.full)

    def quit(self):
        self.root.destroy()

    def flash(self, text, seconds=2.0):
        self.msg = (text, time.time() + seconds)

    # ── วนแสดงผล ──
    def tick(self):
        seq, frame, st = self.an.latest()
        if frame is None:
            frame = splash("starting model", int(self.cfg.get("window_width", 1280)))
            seq = -2
        if seq != self.seq:               # เฟรมเดิมไม่ต้องแปลงซ้ำ (แปลงภาพแพงกว่าที่คิด)
            self.seq = seq
            self._show(frame)
        self._sync(st)
        self.root.after(20, self.tick)

    def _show(self, frame):
        lw, lh = self.video.winfo_width(), self.video.winfo_height()
        if lw < 20 or lh < 20:            # ยังไม่ได้ layout รอบแรก
            return
        h, w = frame.shape[:2]
        s = min(lw / w, lh / h)
        small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=self.photo)

    def _sync(self, st):
        rows = st.get("rows", [])
        done = sum(1 for _, ok in rows if ok)
        self.count.configure(text=f"{done} / {len(rows)}")
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, self.bar.winfo_width() * done / max(len(rows), 1), 6,
                                  fill=G, width=0)
        if rows != self.shown_rows:
            self.shown_rows = rows
            self._rebuild(rows)

        text, expire = self.msg
        if text and time.time() < expire:
            self.dot.configure(fg=DIM)
            self.state.configure(text=text)
        elif st["holding"]:
            self.dot.configure(fg=G)
            self.state.configure(text=f"กำลังหยิบตรวจ · {st['held_s']:0.1f} วิ")
        else:
            self.dot.configure(fg=Y if rows else DIM)
            self.state.configure(text="รอมือมาหยิบชิ้นงาน")
        self.stats.configure(
            text=f"{st['fps']:0.1f} FPS · {st['device']} · {st['hands']} มือ · "
                 f"รอบนี้ {st.get('round_s', 0) / 60:0.0f} นาที")

    def _rebuild(self, rows):
        for w in self.list.winfo_children():
            w.destroy()
        if not rows:
            tk.Label(self.list, text="ยังไม่เห็นชิ้นงานในเฟรม", bg=BG, fg=DIM,
                     font=(UI, 10)).pack(anchor="w", pady=8)
            return
        for tid, ok in rows:
            row = tk.Frame(self.list, bg=CARD)
            row.pack(fill="x", pady=3)
            tk.Label(row, text="✓" if ok else "○", bg=CARD, fg=G if ok else Y,
                     font=(UI, 13, "bold"), width=2).pack(side="left", padx=(10, 2), pady=8)
            tk.Label(row, text=f"ชิ้น #{tid}", bg=CARD, fg=TXT,
                     font=(UI, 12, "bold")).pack(side="left")
            tk.Label(row, text="ตรวจแล้ว" if ok else "ยังไม่ตรวจ", bg=CARD, fg=G if ok else DIM,
                     font=(UI, 10)).pack(side="right", padx=12)


# ─────────────────────────── main ───────────────────────────
def main():
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        v = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"เตือน: Python {v} ยังไม่ทดสอบ — สแต็กนี้ใช้ 3.12/3.13 (mediapipe/torch อาจพัง)")
        print(f"       venv ใหม่:  py -3.12 -m venv .venv  &&  .venv\\Scripts\\activate\n")
    cfg = load_config()
    model = load_model(cfg)
    hands = load_hand_landmarker()
    cam = Camera(cfg["camera_index"], mirror=cfg.get("mirror", True),
                 width=cfg.get("camera_width"), height=cfg.get("camera_height"))

    print("กำลังเปิดกล้อง...")
    t0 = time.time()
    while not cam.read()[0]:
        time.sleep(0.05)
        if time.time() - t0 > 10:
            cam.release()
            raise SystemExit(
                f"\nเปิดกล้องไม่ได้ (camera_index = {cfg['camera_index']})\n"
                "เช็กว่ากล้องเสียบอยู่ ไม่มีโปรแกรมอื่นแย่งใช้ แล้วลองเปลี่ยน "
                "camera_index ใน config.yaml เป็น 1 หรือ 2\n"
            )

    analyzer = Analyzer(cam, model, hands, cfg)
    root = tk.Tk()
    App(root, analyzer, cfg)
    root.mainloop()

    analyzer.release()
    cam.release()


if __name__ == "__main__":
    main()
