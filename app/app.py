"""คนถือแก้ว — เวอร์ชัน "ใช้งานได้จริง" ที่วิทยากรรันโชว์หน้าห้อง

กฎถือแก้วเหมือนโน้ตบุ๊กเป๊ะ: กรอบมือซ้อนกรอบแก้ว **และ** มือกำ
ที่เพิ่มเข้ามาคือวิศวกรรมรอบ ๆ กฎ:
  1. tracking ID ต่อเนื่อง (bytetrack) — แก้วแต่ละใบมีเลขติดตัว
  2. state machine + hysteresis — ป้ายไม่กระพริบ (ขึ้นยาก ลงยากกว่า)
  3. threaded capture — อ่านกล้องอีกเธรด เก็บเฟรมล่าสุด ไม่มีดีเลย์สะสม
  4. จัดการ error จริง — กล้องหลุดแล้วต่อใหม่, โมเดลหาย/กล้องเปิดไม่ได้ ขึ้นข้อความไทย

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
import yaml
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

HERE = Path(__file__).parent
HAND_TASK = HERE / "hand_landmarker.task"
HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/1/hand_landmarker.task")
HAND_TASK_MIRROR = HERE.parent / "data" / "hand_landmarker.task"   # data submodule
MODEL_RELEASE_URL = "https://github.com/P-PrPas/tkk_workshop/releases/download/v1/best.pt"

TIPS = [4, 8, 12, 16, 20]     # ปลายนิ้วทั้งห้า
PIPS = [2, 6, 10, 14, 18]     # ข้อกลางของแต่ละนิ้ว
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# ponytail: OpenCV วาดฟอนต์ไทยไม่ได้ ข้อความบนจอเลยเป็นอังกฤษ ส่วนไทยไปที่ terminal


# ─────────────────────── กติกา (ยกมาจากโน้ตบุ๊กตรง ๆ) ───────────────────────
def count_extended(lm):
    """นับนิ้วที่เหยียด: ปลายนิ้วอยู่ไกลจากข้อมือกว่าข้อกลาง (ทนการหมุนมือ)"""
    w = lm[0]
    d = lambda p: (p.x - w.x) ** 2 + (p.y - w.y) ** 2
    return sum(d(lm[t]) > d(lm[p]) for t, p in zip(TIPS, PIPS))


def hand_bbox(lm, w, h):
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    return [min(xs), min(ys), max(xs), max(ys)]


def boxes_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


# ─────────────────────── state machine + hysteresis ───────────────────────
class HoldState:
    """กันป้ายกระพริบ: ต้องเห็นติดกัน on_n เฟรมจึงขึ้น HOLDING,
    ต้องหายติดกัน off_n เฟรมจึงเลิก — on_n < off_n โดยตั้งใจ"""

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
    """จำกล่องแก้วรายตัว (ตาม track ID) — ตอนมือกำบังจนตรวจไม่เจอชั่วคราว
    ก็ยังถือว่าแก้วอยู่ที่เดิมอีก `keep` เฟรม แล้วค่อยลืม
    นี่คือ "เก็บเป็น state" ที่แก้อาการแก้วหายตอนโดนบัง"""

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
    """อ่านกล้องในเธรดแยกแบบไม่หยุด เก็บเฉพาะเฟรมล่าสุด
    กล้องหลุด → พยายามต่อใหม่ทุก 1 วินาที (`ok` เป็น False ระหว่างนั้น)"""

    def __init__(self, index):
        self.index = index
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self._stop = False
        self.cap = cv2.VideoCapture(self.index)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            ret, f = (False, None)
            if self.cap.isOpened():
                ret, f = self.cap.read()
            if not ret:
                with self.lock:
                    self.ok = False
                self.cap.release()
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.index)
                continue
            with self.lock:
                self.frame, self.ok = f, True

    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self._stop = True
        self.cap.release()


# ─────────────────────────── setup ───────────────────────────
def load_config():
    # encoding ระบุชัด — Windows default เป็น cp1252 อ่านคอมเมนต์ไทยใน yaml ไม่ได้
    with open(HERE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(path):
    p = Path(path)
    if not p.is_absolute():
        p = HERE / p
    if not p.exists() and p.name == "best.pt":
        p.parent.mkdir(parents=True, exist_ok=True)
        print("โหลด best.pt จาก GitHub Release ครั้งแรก...")
        try:
            urllib.request.urlretrieve(MODEL_RELEASE_URL, p)
        except urllib.error.URLError:
            pass
    if not p.exists():
        raise SystemExit(
            f"\nหาไฟล์โมเดลไม่เจอ: {p}\n"
            "โหลดเอง:  gh release download v1 -R P-PrPas/tkk_workshop -p best.pt -D app/models\n"
            "หรือใช้แผนสำรอง: model_path: yolo11m.pt  +  cup_class: 41  ใน config.yaml\n"
        )
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
def draw_hand(img, lm, w, h, color):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_EDGES:
        cv2.line(img, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(img, (x, y), 3, color, -1)


def draw_hud(img, holding, fps, ms):
    h = img.shape[0]
    text = "HOLDING" if holding else "NOT HOLDING"
    color = (0, 200, 0) if holding else (0, 0, 255)
    cv2.putText(img, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
    cv2.putText(img, f"{fps:4.1f} FPS   {ms:5.1f} ms/frame", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_waiting(msg):
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(img, msg, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


# ─────────────────────────── main ───────────────────────────
WINDOW = "cup-holding   [q] quit  [s] save shot  [d] debug"


def main():
    cfg = load_config()
    model = load_model(cfg["model_path"])
    hands = load_hand_landmarker()
    cam = Camera(cfg["camera_index"])

    # รอกล้องพร้อมครั้งแรก — ถ้าเกิน 10 วิยังไม่มา แปลว่าเปิดไม่ได้จริง
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

    hold = HoldState(cfg["hold_frames"], cfg["release_frames"])
    cups = CupMemory(cfg.get("cup_memory_frames", 15))
    debug = False
    prev = time.time()
    fps = 0.0
    frame_i = 0

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            cv2.imshow(WINDOW, draw_waiting("reconnecting camera..."))
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        if cfg.get("mirror", True):          # selfie view — ขยับขวาไปขวา
            frame = cv2.flip(frame, 1)

        t = time.time()
        h, w = frame.shape[:2]
        frame_i += 1

        # แก้ว: track ให้แต่ละใบมี ID ติดตัว → CupMemory จำกล่องต่อแม้ detection หายตอนมือบัง
        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        conf=cfg["conf"], classes=[cfg["cup_class"]], verbose=False)[0]
        detections = []
        if r.boxes is not None and r.boxes.id is not None:
            for box, tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist()):
                detections.append((int(tid), box))
        cups.update(detections)

        cup_boxes = []
        for tid, box, coasting in cups.boxes():
            cup_boxes.append(box)
            x1, y1, x2, y2 = map(int, box)
            col = (140, 140, 90) if coasting else (255, 180, 0)   # จาง = จากความจำ
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            tag = f"cup #{tid}" + (" (memory)" if coasting else "")
            cv2.putText(frame, tag, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        # มือ: MediaPipe โหมด VIDEO ต้องการ timestamp ที่เพิ่มขึ้นเรื่อย ๆ
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                     int(time.monotonic() * 1000))

        observed = False
        for lm in (res.hand_landmarks or []):
            n_ext = count_extended(lm)
            gripping = n_ext <= cfg["fist_threshold"]   # กำแก้วไม่ใช่กำแน่น เกณฑ์เลยหลวม
            hb = hand_bbox(lm, w, h)
            if gripping and any(boxes_overlap(hb, c) for c in cup_boxes):
                observed = True
            draw_hand(frame, lm, w, h, (0, 255, 0) if gripping else (200, 200, 200))
            if debug:
                cv2.putText(frame, f"ext:{n_ext}", (int(hb[0]), int(hb[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        holding = hold.update(observed)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-3)
        prev = now
        draw_hud(frame, holding, fps, (now - t) * 1000)

        cv2.imshow(WINDOW, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            fn = f"shot_{int(time.time())}.png"
            cv2.imwrite(fn, frame)
            print("เซฟภาพ:", fn)
        elif k == ord("d"):
            debug = not debug

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
