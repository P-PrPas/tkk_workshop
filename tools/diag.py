"""diag — เปิดกล้อง แล้วพิมพ์ทุกอย่างที่ app ใช้ตัดสิน HOLDING ทีละเฟรม
ไว้หาว่าปัญหาอยู่ที่ (ก) ตรวจแก้วไม่เจอ (ข) มือไม่เป็น FIST (ค) กล่องไม่ทับกัน

    python tools/diag.py                    # ใช้ best.pt, conf 0.10
    python tools/diag.py yolo11m.pt 0.15    # ลองโมเดล/conf อื่น

กด q ออก
"""
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from app import count_extended, hand_bbox, boxes_overlap, HAND_TASK, HAND_TASK_MIRROR  # noqa

WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "app/models/best.pt"
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10

model = YOLO(WEIGHTS)
CUP_CLS = [k for k, v in model.names.items() if v == "cup"] or [0]
print(f"model: {WEIGHTS}  names={model.names}  -> cup class {CUP_CLS}  conf {CONF}")

if not HAND_TASK.exists() and HAND_TASK_MIRROR.exists():
    HAND_TASK.write_bytes(HAND_TASK_MIRROR.read_bytes())
hands = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(HAND_TASK)),
    running_mode=mp_vision.RunningMode.VIDEO, num_hands=2))

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        print("อ่านกล้องไม่ได้"); break
    frame = cv2.flip(frame, 1)              # mirror ให้ตรงกับ app
    h, w = frame.shape[:2]

    r = model(frame, conf=CONF, classes=CUP_CLS, verbose=False)[0]
    cups = r.boxes.xyxy.tolist() if r.boxes is not None else []
    confs = [round(float(c), 2) for c in r.boxes.conf] if r.boxes is not None else []

    res = hands.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        int(time.monotonic() * 1000))
    hand_info, holding = [], False
    for lm in (res.hand_landmarks or []):
        n = count_extended(lm)
        st = "FIST" if n <= 1 else "OPEN" if n >= 4 else "MID"
        hb = hand_bbox(lm, w, h)
        touch = any(boxes_overlap(hb, c) for c in cups)
        if st == "FIST" and touch:
            holding = True
        hand_info.append(f"{st}(ext={n},touch={touch})")
        cv2.rectangle(frame, (int(hb[0]), int(hb[1])), (int(hb[2]), int(hb[3])), (0, 255, 0), 2)

    for (x1, y1, x2, y2), c in zip(cups, confs):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), 2)
        cv2.putText(frame, f"cup {c}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

    print(f"cups={confs or '-':<28} hands={hand_info or '-'}  HOLDING={holding}")
    cv2.putText(frame, "HOLDING" if holding else "no", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 0) if holding else (0, 0, 255), 3)
    cv2.imshow("diag (q ออก)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
