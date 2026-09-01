"""คนถือแก้ว — เวอร์ชัน "ใช้งานได้จริง" ที่วิทยากรรันโชว์หน้าห้อง

กฎถือแก้วเหมือนโน้ตบุ๊กเป๊ะ: hand bbox ซ้อน cup bbox และมือกำ
ที่ต่างคือวิศวกรรมรอบๆ — tracking ID, state machine + hysteresis,
threaded capture, และการจัดการ error จริง

รัน:  python app/app.py
ค่าที่ต้องจูนหน้างานอยู่ใน app/config.yaml ทั้งหมด ห้ามแก้ไฟล์นี้หน้างาน
"""
import threading
import time
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

TIPS = [4, 8, 12, 16, 20]
PIPS = [2, 6, 10, 14, 18]
# ปลายนิ้ว-โคนนิ้ว 21 จุดของ MediaPipe ไว้วาดโครงมือ
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


# ─────────────────────────── กติกา (เหมือนโน้ตบุ๊ก) ───────────────────────────
def count_extended(lm):
    """นิ้วเหยียด = ปลายนิ้วอยู่ไกลจากข้อมือกว่าข้อกลาง (ทนการหมุนมือ)"""
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
    """กันป้ายกระพริบ: ขึ้นยาก ลงยากกว่า (on != off โดยตั้งใจ = hysteresis)"""

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


# ─────────────────────── threaded capture (เก็บเฟรมล่าสุดเท่านั้น) ───────────────────────
class Camera:
    """เธรดอ่านกล้องไม่หยุด ทิ้งเฟรมเก่า กันดีเลย์สะสม + ต่อกล้องใหม่เองเมื่อหลุด"""

    def __init__(self, index):
        self.index = index
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stop = False
        self.cap = cv2.VideoCapture(self.index)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop:
            if not self.cap.isOpened():
                self.ok = False
                time.sleep(1.0)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.index)
                continue
            ret, f = self.cap.read()
            if not ret:
                self.ok = False
                self.cap.release()
                time.sleep(1.0)
                self.cap = cv2.VideoCapture(self.index)
                continue
            with self.lock:
                self.frame = f
                self.ok = True

    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self.stop = True
        self.cap.release()


# ─────────────────────────── setup ───────────────────────────
def load_config():
    with open(HERE / "config.yaml") as f:
        return yaml.safe_load(f)


def load_model(path):
    p = Path(path)
    if not p.is_absolute():
        p = HERE / p
    if not p.exists():
        raise SystemExit(
            f"หาโมเดลไม่เจอที่: {p}\n"
            "ดาวน์โหลด best.pt จาก GitHub Release แล้ววางไว้ที่ path นั้น:\n"
            "  gh release download v1 -R P-PrPas/tkk_workshop -p best.pt\n"
            "หรือแก้ model_path ใน config.yaml (fallback: yolo11m.pt + cup_class: 41)"
        )
    return YOLO(str(p))


def load_hand_landmarker():
    if not HAND_TASK.exists():
        print("ดาวน์โหลด hand_landmarker.task ครั้งเดียว...")
        urllib.request.urlretrieve(HAND_TASK_URL, HAND_TASK)
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


def draw_hud(img, holding, fps, ms, debug, n_ext=None):
    h, w = img.shape[:2]
    label = "HOLDING" if holding else "NOT HOLDING"
    color = (0, 200, 0) if holding else (0, 0, 255)
    cv2.putText(img, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)
    cv2.putText(img, f"{fps:4.1f} FPS  {ms:5.1f} ms", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if debug and n_ext is not None:
        cv2.putText(img, f"extended fingers: {n_ext}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


# ─────────────────────────── main ───────────────────────────
def main():
    cfg = load_config()
    model = load_model(cfg["model_path"])
    hands = load_hand_landmarker()
    cam = Camera(cfg["camera_index"])

    # รอกล้องพร้อมครั้งแรก
    t0 = time.time()
    while not cam.read()[0]:
        if time.time() - t0 > 10:
            cam.release()
            raise SystemExit(
                "เปิดกล้องไม่ได้ ตรวจว่ากล้องเสียบอยู่และไม่มีโปรแกรมอื่นใช้อยู่ "
                f"แล้วลองแก้ camera_index ใน config.yaml (ตอนนี้ = {cfg['camera_index']})"
            )
        time.sleep(0.2)

    hold = HoldState(cfg["hold_frames"], cfg["release_frames"])
    debug = False
    win = "cup-holding (q ออก | s เซฟภาพ | d debug)"
    prev = time.time()
    fps = 0.0
    frame_i = 0

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "กำลังเชื่อมต่อกล้องใหม่...", (60, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow(win, blank)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        t = time.time()
        h, w = frame.shape[:2]
        frame_i += 1

        # แก้ว: track ให้แต่ละใบมี ID ติดตัว
        r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                        conf=cfg["conf"], classes=[cfg["cup_class"]], verbose=False)[0]
        cup_boxes = []
        if r.boxes is not None and r.boxes.id is not None:
            for box, tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist()):
                cup_boxes.append(box)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
                cv2.putText(frame, f"cup #{int(tid)}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        # มือ
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = hands.detect_for_video(mp_img, int(frame_i * 1000 / 30))

        observed = False
        n_ext_dbg = None
        for lm in (res.hand_landmarks or []):
            n_ext = count_extended(lm)
            n_ext_dbg = n_ext
            is_fist = n_ext <= cfg["fist_threshold"]
            hb = hand_bbox(lm, w, h)
            touching = any(boxes_overlap(hb, c) for c in cup_boxes)
            if is_fist and touching:
                observed = True
            draw_hand(frame, lm, w, h, (0, 255, 0) if is_fist else (200, 200, 200))

        holding = hold.update(observed)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-3))
        prev = now
        draw_hud(frame, holding, fps, (now - t) * 1000, debug, n_ext_dbg)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            fn = f"shot_{int(time.time())}.png"
            cv2.imwrite(fn, frame)
            print("เซฟ", fn)
        elif k == ord("d"):
            debug = not debug

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
