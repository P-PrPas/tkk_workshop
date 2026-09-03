"""ลองแปลง best.pt เป็นฟอร์แมตอื่น แล้ววัดว่าฟอร์แมตไหนเร็วสุด "บนเครื่องนี้"
(OpenVINO/ONNX เร็วขึ้นบน CPU Intel รุ่นใหม่ แต่บางเครื่องช้าลง — ต้องวัดจริง)

    python tools/optimize.py                        # best.pt, imgsz 480
    python tools/optimize.py app/models/best.pt 384 # imgsz อื่น

ผลบอกว่าให้ตั้ง config.yaml -> model_path เป็นอะไร
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "app/models/best.pt")
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 480
if not SRC.exists():
    raise SystemExit(f"ไม่พบ {SRC} — รัน app/app.py ครั้งแรกให้มันโหลด best.pt ก่อน")

img = np.random.randint(0, 255, (480, 640, 3), np.uint8)


def bench(path, tag):
    try:
        m = YOLO(str(path))
        for _ in range(3):
            m.predict(img, imgsz=IMGSZ, conf=0.25, verbose=False)
        t = time.time()
        for _ in range(25):
            m.predict(img, imgsz=IMGSZ, conf=0.25, verbose=False)
        ms = (time.time() - t) / 25 * 1000
        print(f"  {tag:10} {ms:6.1f} ms/frame  (~{1000/ms:.0f} fps)")
        return ms, path
    except Exception as e:
        print(f"  {tag:10} ใช้ไม่ได้: {e}")
        return 1e9, None


print(f"benchmark @ imgsz {IMGSZ} (เครื่องนี้)")
runs = [bench(SRC, "pytorch")]
for fmt in ("onnx", "openvino"):
    try:
        out = YOLO(str(SRC)).export(format=fmt, imgsz=IMGSZ, verbose=False)
        runs.append(bench(out, fmt))
    except Exception as e:
        print(f"  {fmt:10} export ไม่ได้: {e}")

best_ms, best_path = min(runs, key=lambda x: x[0])
print(f"\nเร็วสุด: {best_path}")
if best_path and Path(best_path) != SRC:
    print(f"ตั้ง config.yaml ->  model_path: models/{Path(best_path).name}"
          f"   (คัดลอกโฟลเดอร์/ไฟล์ไป app/models/ ก่อน)")
else:
    print("pytorch เร็วสุดอยู่แล้ว — ไม่ต้องแปลง ลองลด imgsz เป็น 384 แทน")
