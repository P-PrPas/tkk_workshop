"""คนถือแก้ว — เวอร์ชัน "ใช้งานได้จริง" ที่วิทยากรรันโชว์หน้าห้อง

กฎยังเหมือนโน้ตบุ๊ก: มือ (ไม่แบกว้าง) อยู่บนแก้ว → กำลังถือ
ที่เพิ่มคือวิศวกรรมรอบ ๆ กฎ:
  1. threaded capture — อ่านกล้องอีกเธรด เก็บเฟรมล่าสุด ไม่มีดีเลย์สะสม
  2. threaded inference — YOLO + MediaPipe อยู่อีกเธรด วาดผลลงเฟรมที่มันวิเคราะห์
     main แค่แสดง → จอเดินเท่า detect FPS · กล่องอยู่บนเฟรมที่ถูกต้องเสมอ ไม่ลอยตามหลัง
  3. tracking ID + CupMemory — แก้วที่โดนมือกำบังจนตรวจไม่เจอ ยังจำกล่องไว้ต่อ
  4. HoldState hysteresis — ป้ายไม่กระพริบ (ขึ้นยาก ลงยากกว่า)
  5. จัดการ error จริง — กล้องหลุดต่อใหม่, โมเดล/กล้องหาย ขึ้นข้อความไทย

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

# ponytail: OpenCV วาดฟอนต์ไทยไม่ได้ ข้อความบนจอเลยเป็นอังกฤษ ส่วนไทยไปที่ terminal


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

    def update(self, observed: bool) -> bool:
        if observed:
            self.hits += 1
            self.misses = 0
            if self.hits >= self.on_n:
                self.holding = True
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

    def __init__(self, index, mirror=True):
        self.index, self.mirror = index, mirror
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self._stop = False
        self.cap = cv2.VideoCapture(self.index)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            ret, f = (self.cap.read() if self.cap.isOpened() else (False, None))
            if not ret:
                with self.lock:
                    self.ok = False
                self.cap.release()
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.index)
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
        # onnx → บังคับ CPU (onnxruntime CUDA EP พังง่ายบนบางเครื่อง) · pt → ให้ torch เลือกเอง
        self.device = "cpu" if str(cfg["model_path"]).endswith(".onnx") else None
        self.hold = HoldState(cfg["hold_frames"], cfg["release_frames"])
        self.cups = CupMemory(cfg.get("cup_memory_frames", 15))
        self.lock = threading.Lock()
        self.frame = None            # เฟรมที่วาดผลแล้ว พร้อมแสดง
        self.fps = 0.0
        self.debug = False
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
                        self.frame = draw_waiting("reconnecting camera...")
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
                cup_boxes = [box for _, box, _ in self.cups.boxes()]

                res = self.hands.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    int(time.monotonic() * 1000))
                mgn = c.get("grip_box_margin", 0.35)
                hands_out, observed = [], False
                for lm in (res.hand_landmarks or []):
                    on_cup = hand_on_cup(lm, w, h, cup_boxes,
                                         c.get("grip_min_points", 10),
                                         c.get("grip_max_size_ratio", 4.0), mgn)
                    observed = observed or on_cup
                    pts_in = max((sum(x1 - mgn * (x2 - x1) <= p.x * w <= x2 + mgn * (x2 - x1)
                                      and y1 - mgn * (y2 - y1) <= p.y * h <= y2 + mgn * (y2 - y1)
                                      for p in lm)
                                  for x1, y1, x2, y2 in cup_boxes), default=0)
                    hands_out.append(([(int(p.x * w), int(p.y * h)) for p in lm],
                                      on_cup, hand_state(lm), pts_in))
                holding = self.hold.update(observed)

                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 / max(now - prev, 1e-3)
                prev = now
                draw(frame, self.cups.boxes(), hands_out, holding, self.fps, self.debug)
                with self.lock:
                    self.frame = frame
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
    if p.suffix == ".pt" and not torch.cuda.is_available():
        print("──────────────────────────────────────────────────────────────")
        print("  YOLO จะรันบน CPU → ~5 FPS. ถ้าเครื่องนี้มี NVIDIA GPU ให้ลง torch CUDA:")
        print("    pip install --force-reinstall torch torchvision \\")
        print("        --index-url https://download.pytorch.org/whl/cu124")
        print("──────────────────────────────────────────────────────────────")
    else:
        print("YOLO device:", "CUDA " + torch.cuda.get_device_name(0)
              if torch.cuda.is_available() else "CPU (onnx)")
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


# ─────────────────────────── วาด ───────────────────────────
STATE_COLOR = {"FIST": (0, 220, 0), "OPEN": (200, 200, 200), "MID": (0, 210, 255)}


def draw(frame, cups, hands, holding, fps, debug):
    h = frame.shape[0]
    for tid, box, coasting in cups:
        x1, y1, x2, y2 = map(int, box)
        col = (140, 140, 90) if coasting else (255, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"cup #{tid}" + (" (memory)" if coasting else ""),
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for pts, on_cup, state, pts_in in hands:
        col = STATE_COLOR[state]                 # สีโครงมือ = ท่ามือ (พาร์ท 2)
        for a, b in HAND_EDGES:
            cv2.line(frame, pts[a], pts[b], col, 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, col, -1)
        tag = state + (" -> holding cup" if on_cup else "")
        if debug:
            tag += f"  (pts:{pts_in})"
        cv2.putText(frame, tag, (pts[0][0] - 10, pts[0][1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    cv2.putText(frame, "HOLDING" if holding else "NOT HOLDING", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0) if holding else (0, 0, 255), 4)
    cv2.putText(frame, f"{fps:4.1f} FPS", (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_waiting(msg):
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(img, msg, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


# ─────────────────────────── main ───────────────────────────
WINDOW = "cup-holding   [q] quit  [s] save shot  [d] debug"


def main():
    cfg = load_config()
    model = load_model(cfg)
    hands = load_hand_landmarker()
    cam = Camera(cfg["camera_index"], mirror=cfg.get("mirror", True))

    t0 = time.time()
    while not cam.read()[0]:
        cv2.imshow(WINDOW, draw_waiting("opening camera..."))
        cv2.waitKey(100)
        if time.time() - t0 > 10:
            cam.release()
            cv2.destroyAllWindows()
            raise SystemExit(
                f"\nเปิดกล้องไม่ได้ (camera_index = {cfg['camera_index']})\n"
                "เช็กว่ากล้องเสียบอยู่ ไม่มีโปรแกรมอื่นแย่งใช้ แล้วลองเปลี่ยน "
                "camera_index ใน config.yaml เป็น 1 หรือ 2\n"
            )

    analyzer = Analyzer(cam, model, hands, cfg)

    # main แค่แสดงเฟรมที่ analyzer วาดผลเสร็จแล้ว + รับปุ่ม (จอเดินเท่า detect FPS)
    while True:
        frame = analyzer.latest()
        cv2.imshow(WINDOW, frame if frame is not None else draw_waiting("starting model..."))
        k = cv2.waitKey(15) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s") and frame is not None:
            fn = f"shot_{int(time.time())}.png"
            cv2.imwrite(fn, frame)
            print("เซฟภาพ:", fn)
        elif k == ord("d"):
            analyzer.debug = not analyzer.debug

    analyzer.release()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
