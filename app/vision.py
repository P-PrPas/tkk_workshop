"""แกนการมองเห็น — กล้อง โมเดล กติกา และเช็กลิสต์

สามไฟล์ สามหน้าที่:  vision.py = คิด · overlay.py = วาดลงเฟรม · app.py = แสดงผล
ไฟล์นี้ไม่รู้จัก GUI เลยสักบรรทัด

  1. threaded capture — อ่านกล้องอีกเธรด เก็บเฟรมล่าสุด ไม่มีดีเลย์สะสม
  2. threaded inference — YOLO + MediaPipe อยู่อีกเธรด วาดผลลงเฟรมที่มันวิเคราะห์
     GUI แค่แสดง → ภาพเดินเท่า detect FPS · กล่องอยู่บนเฟรมที่ถูกต้องเสมอ ไม่ลอยตามหลัง
  3. tracking ID + CupMemory — แก้วที่โดนมือกำบังจนตรวจไม่เจอ ยังจำกล่องไว้ต่อ
  4. HoldState hysteresis — ป้ายไม่กระพริบ (ขึ้นยาก ลงยากกว่า)
  5. Inspection — เช็กลิสต์ว่าชิ้นไหนถูกหยิบไปตรวจแล้ว + เส้นทางที่มันเคลื่อนที่มา
  6. จัดการ error จริง — กล้องหลุดต่อใหม่, โมเดล/กล้องหาย ขึ้นข้อความไทย
"""
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

# macOS: ปิด GPU ของ MediaPipe ก่อน import — hand_landmarker.task มีโหนด palm detector ที่
# ไป init Metal (DrishtiMetalHelper) แล้ว abort ทั้งโปรเซส ถ้า Metal service ไม่พร้อม
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import mediapipe as mp
import torch
import yaml
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

from overlay import draw, splash

HERE = Path(__file__).parent
HAND_TASK = HERE / "hand_landmarker.task"
HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/1/hand_landmarker.task")
HAND_TASK_MIRROR = HERE.parent / "data" / "hand_landmarker.task"
RELEASE = "https://github.com/P-PrPas/tkk_workshop/releases/download/v1"   # best.pt · best.onnx (imgsz 480)

TIPS = [4, 8, 12, 16, 20]     # ปลายนิ้วทั้งห้า
PIPS = [2, 6, 10, 14, 18]     # ข้อกลางของแต่ละนิ้ว


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
        """[(tid, ตรวจแล้ว?, ความคืบหน้า 0..1, ตรวจไปกี่วินาทีแล้ว)] เรียงตามเลข

        `ความคืบหน้า` คือตัวนับแบบรั่วที่กำลังไต่ขึ้น — GUI เอาไปวาดเป็นแถบที่ค่อย ๆ
        เต็มตอนมือจับอยู่ ทำให้ผู้ชมเห็นว่าเครื่อง "กำลังมั่นใจขึ้น" ไม่ใช่ติ๊กมาเฉย ๆ"""
        now = time.time()
        out = []
        for tid in set(self.seen) | set(self.picked):
            done = tid in self.picked
            out.append((tid, done,
                        1.0 if done else min(1.0, self.hits.get(tid, 0) / max(self.need, 1)),
                        int(now - self.picked[tid]) if done else 0))
        return sorted(out)


# ─────────── threaded capture — เก็บแค่เฟรมล่าสุด กันดีเลย์สะสม ───────────
class Camera:
    """อ่านกล้องในเธรดแยกแบบไม่หยุด เก็บเฉพาะเฟรมล่าสุด (mirror ให้ด้วยถ้า mirror=True)
    กล้องหลุด → ต่อใหม่ทุก 1 วินาที (`read()[0]` เป็น False ระหว่างนั้น)"""

    def __init__(self, index, mirror=True, width=None, height=None, available=None):
        self.index, self.mirror = index, mirror
        self.width, self.height = width, height
        self.available = available or [index]   # index ที่เปิดได้ตอนสำรวจ — ให้ปุ่มเลือกกล้องใน GUI
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self._stop = False
        self._switch = False
        self.cap = self._open()
        threading.Thread(target=self._loop, daemon=True).start()

    def switch(self, index):
        """สลับกล้องระหว่างรัน — ยกธง เธรด _loop เปิดตัวใหม่ให้ตอนวนรอบถัดไป (ไม่แตะ cap ข้ามเธรด)"""
        with self.lock:
            self.index, self._switch, self.ok = index, True, False

    def _open(self):
        cap = cv2.VideoCapture(self.index)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _loop(self):
        while not self._stop:
            if self._switch:
                self._switch = False
                self.cap.release()
                self.cap = self._open()
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
    """เธรดวิเคราะห์ — หยิบเฟรมล่าสุด รันโมเดล **วาดผลลงเฟรมเดียวกัน** เก็บไว้ให้ GUI แสดง
    จอจึงเดินเท่า FPS ที่ detect ได้จริง (กระตุกกว่า แต่กล่องอยู่บนเฟรมที่มันคิด ไม่ลอยตามหลัง)"""

    def __init__(self, cam, model, hands, cfg):
        self.cam, self.model, self.hands, self.cfg = cam, model, hands, cfg
        self.device = pick_device(cfg)          # auto: cuda → mps → cpu (onnx บังคับ cpu)
        self.device_label = self.device.upper() + (" · ONNX" if str(cfg["model_path"]).endswith(".onnx") else "")
        self.hold = HoldState(cfg["hold_frames"], cfg["release_frames"])
        self.cups = CupMemory(cfg.get("cup_memory_frames", 15))
        self.insp = Inspection(cfg.get("pick_frames", 8), cfg.get("trail_length", 60),
                               cfg.get("forget_seconds", 4.0))
        self.lock = threading.Lock()
        self.frame = None            # เฟรมที่วาดผลแล้ว พร้อมแสดง
        self.seq = 0                 # เลขเฟรม — GUI ใช้เช็กว่ามีของใหม่ค่อยแปลงภาพ
        self.status = {"holding": False, "held_s": 0.0, "fps": 0.0, "hands": 0,
                       "rows": [], "device": self.device_label, "camera": True}
        self.events = deque(maxlen=40)   # (เวลา, ข้อความ) — ป้อนแถบ ticker ล่างจอ
        self.debug = False
        self._picked_seen = set()    # ไว้ยิง event ตอนมีชิ้นใหม่ถูกติ๊ก
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
        self._picked_seen.clear()
        self.events.clear()
        try:                        # ให้ ByteTrack เริ่มนับ id ใหม่จาก 1 ด้วย
            for t in self.model.predictor.trackers:
                t.reset()
        except Exception:
            pass                    # ยังไม่เคย track สักเฟรม / ultralytics เปลี่ยน API — ไม่ใช่เรื่องคอขาดบาดตาย

    def log(self, text):
        self.events.append((time.strftime("%H:%M:%S"), text))

    def _loop(self):
        c = self.cfg
        prev = time.time()
        while not self._stop:
            try:
                if self._reset:
                    self._reset = False
                    self._do_reset()
                    self.log("รอบตรวจใหม่")
                ok, frame = self.cam.read()
                if not ok or frame is None:
                    with self.lock:
                        self.frame, self.seq = splash("reconnecting camera"), self.seq + 1
                        self.status = dict(self.status, camera=False)
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
                for tid in sorted(set(self.insp.picked) - self._picked_seen):
                    self._picked_seen.add(tid)
                    self.log(f"ตรวจแล้ว · ชิ้น #{tid}")

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
                        "fps": fps, "hands": len(hands_out), "camera": True,
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
def list_cameras(probe=3):
    """ลองเปิด index 0..probe-1 คืนเฉพาะตัวที่อ่านเฟรมได้ — ช้า ~1 วิ/ตัวบน Windows ทำครั้งเดียวตอนเปิดแอป"""
    found = []
    for i in range(max(1, int(probe))):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            found.append(i)
    return found or [0]


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
        # delegate=CPU บังคับไว้ — บน macOS ตัว .task (float16) จะไปเรียก GPU/Metal delegate เอง
        # แล้ว abort ทันที: "Check failed: service_ Service is unavailable" ที่ DrishtiMetalHelper
        # โมเดลมือจิ๋วมาก รันบน CPU (XNNPACK) ก็เร็วพอ ไม่ต้องใช้ GPU
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_TASK),
                                           delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def start(cfg):
    """โหลดโมเดล เปิดกล้อง แล้วคืน (analyzer, camera) ที่พร้อมทำงาน
    เปิดกล้องไม่ได้ใน 10 วินาที = ออกพร้อมข้อความไทยที่บอกว่าต้องไปแก้อะไร ไม่ใช่ traceback"""
    if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
        v = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"เตือน: Python {v} ยังไม่ทดสอบ — สแต็กนี้ใช้ 3.12/3.13 (mediapipe/torch อาจพัง)")
        print(f"       venv ใหม่:  py -3.12 -m venv .venv  &&  .venv\\Scripts\\activate\n")
    model = load_model(cfg)
    hands = load_hand_landmarker()
    cams = list_cameras(cfg.get("camera_probe", 3))          # ทำรายการก่อนเปิดตัวจริง (สองตัวพร้อมกันไม่ได้)
    idx = cfg["camera_index"] if cfg["camera_index"] in cams else cams[0]
    print("กล้องที่เปิดได้:", cams, "→ ใช้", idx)
    cam = Camera(idx, mirror=cfg.get("mirror", True), available=cams,
                 width=cfg.get("camera_width"), height=cfg.get("camera_height"))

    print("กำลังเปิดกล้อง...")
    t0 = time.time()
    while not cam.read()[0]:
        time.sleep(0.05)
        if time.time() - t0 > 10:
            cam.release()
            raise SystemExit(
                f"\nเปิดกล้องไม่ได้ (camera_index = {idx})\n"
                "เช็กว่ากล้องเสียบอยู่ ไม่มีโปรแกรมอื่นแย่งใช้ แล้วลองเปลี่ยน "
                "camera_index ใน config.yaml เป็น 1 หรือ 2\n"
            )
    return Analyzer(cam, model, hands, cfg), cam
