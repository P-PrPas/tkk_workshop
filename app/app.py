"""คนถือแก้ว — เวอร์ชัน "ใช้งานได้จริง" ที่วิทยากรรันโชว์หน้าห้อง

กฎยังเหมือนโน้ตบุ๊ก: มือ (ไม่แบกว้าง) อยู่บนแก้ว → กำลังถือ
ที่เพิ่มคือวิศวกรรมรอบ ๆ กฎ:
  1. threaded capture — อ่านกล้องอีกเธรด เก็บเฟรมล่าสุด ไม่มีดีเลย์สะสม
  2. threaded inference — YOLO + MediaPipe อยู่อีกเธรด · จอวาดผลล่าสุดทับเฟรมสด
     → ภาพลื่น ~เท่า FPS กล้อง แม้ inference จะช้ากว่า
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
from collections import namedtuple
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
HAND_TASK_MIRROR = HERE.parent / "data" / "hand_landmarker.task"
MODEL_RELEASE_URL = "https://github.com/P-PrPas/tkk_workshop/releases/download/v1/best.pt"

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


def hand_on_cup(lm, w, h, cup_boxes, open_max, min_pts):
    """มืออยู่บนแก้วไหม — ใช้ได้ทั้งแก้วมีหูและไม่มีหู:
    (1) มือไม่แบกว้าง (นิ้วเหยียด <= open_max)  (2) จุดมืออย่างน้อย min_pts จุดอยู่ในกล่องแก้ว"""
    if count_extended(lm) > open_max:
        return False
    for x1, y1, x2, y2 in cup_boxes:
        if sum(x1 <= p.x * w <= x2 and y1 <= p.y * h <= y2 for p in lm) >= min_pts:
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
Result = namedtuple("Result", "cups hands holding infer_ms")
# cups  : list[(tid, (x1,y1,x2,y2), coasting)]
# hands : list[(pts, on_cup)]   pts = list[(x,y)] พิกัดจริง
Result.EMPTY = Result([], [], False, 0.0)


class Analyzer:
    """เธรดวิเคราะห์ — หยิบเฟรมล่าสุดจากกล้อง รันโมเดล เก็บผลไว้ให้จอไปวาด
    ไม่รอจอ ไม่บล็อกจอ → จอลื่นเท่า FPS กล้อง"""

    def __init__(self, cam, model, hands, cfg):
        self.cam, self.model, self.hands, self.cfg = cam, model, hands, cfg
        self.hold = HoldState(cfg["hold_frames"], cfg["release_frames"])
        self.cups = CupMemory(cfg.get("cup_memory_frames", 15))
        self.lock = threading.Lock()
        self.result = Result.EMPTY
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        c = self.cfg
        while not self._stop:
            ok, frame = self.cam.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            t = time.time()
            h, w = frame.shape[:2]

            r = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 imgsz=c.get("imgsz", 480), conf=c["conf"],
                                 classes=[c["cup_class"]], verbose=False)[0]
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
            hands_out, observed = [], False
            for lm in (res.hand_landmarks or []):
                on_cup = hand_on_cup(lm, w, h, cup_boxes,
                                     c.get("grip_open_max", 3), c.get("grip_min_points", 5))
                observed = observed or on_cup
                hands_out.append(([(int(p.x * w), int(p.y * h)) for p in lm], on_cup))

            holding = self.hold.update(observed)
            with self.lock:
                self.result = Result(self.cups.boxes(), hands_out, holding,
                                     (time.time() - t) * 1000)

    def latest(self):
        with self.lock:
            return self.result

    def release(self):
        self._stop = True
        self._thread.join(timeout=2)
        self.hands.close()          # กัน mediapipe บ่นตอนปิดโปรแกรม


# ─────────────────────────── setup ───────────────────────────
def load_config():
    # encoding ระบุชัด — Windows default เป็น cp1252 อ่านคอมเมนต์ไทยใน yaml ไม่ได้
    with open(HERE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fetch_best_pt():
    """คืน path ของ best.pt — โหลดจาก Release ถ้ายังไม่มี"""
    pt = (HERE / "models" / "best.pt")
    if not pt.exists():
        pt.parent.mkdir(parents=True, exist_ok=True)
        print("โหลด best.pt จาก GitHub Release ครั้งแรก...")
        try:
            urllib.request.urlretrieve(MODEL_RELEASE_URL, pt)
        except urllib.error.URLError:
            pass
    return pt if pt.exists() else None


def load_model(cfg):
    """model_path ชี้ไป .onnx → export จาก best.pt ให้อัตโนมัติ (imgsz ตรงกับ config)
    onnxruntime เร็วกว่า pytorch บน CPU · ถ้ามี GPU ให้ `pip install onnxruntime-gpu`"""
    p = Path(cfg["model_path"])
    if not p.is_absolute():
        p = HERE / p

    if p.suffix == ".onnx" and not p.exists():
        pt = _fetch_best_pt()
        if pt:
            print(f"export {pt.name} -> onnx (imgsz {cfg.get('imgsz', 480)}) ครั้งแรก...")
            out = YOLO(str(pt)).export(format="onnx", imgsz=cfg.get("imgsz", 480),
                                       dynamic=False, verbose=False)
            Path(out).replace(p)
    elif p.name == "best.pt" and not p.exists():
        _fetch_best_pt()

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
def draw(frame, result, disp_fps, debug):
    h = frame.shape[0]
    for tid, box, coasting in result.cups:
        x1, y1, x2, y2 = map(int, box)
        col = (140, 140, 90) if coasting else (255, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, f"cup #{tid}" + (" (memory)" if coasting else ""),
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for pts, on_cup in result.hands:
        col = (0, 255, 0) if on_cup else (200, 200, 200)
        for a, b in HAND_EDGES:
            cv2.line(frame, pts[a], pts[b], col, 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 3, col, -1)

    text = "HOLDING" if result.holding else "NOT HOLDING"
    cv2.putText(frame, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 200, 0) if result.holding else (0, 0, 255), 4)
    hud = f"{disp_fps:4.1f} FPS"
    if debug:
        hud += f"   infer {result.infer_ms:5.0f} ms"
    cv2.putText(frame, hud, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


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
    debug = False
    prev = time.time()
    disp_fps = 0.0

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            cv2.imshow(WINDOW, draw_waiting("reconnecting camera..."))
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        now = time.time()
        disp_fps = 0.9 * disp_fps + 0.1 / max(now - prev, 1e-3)
        prev = now

        draw(frame, analyzer.latest(), disp_fps, debug)
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

    analyzer.release()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
