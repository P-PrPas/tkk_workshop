"""เรนเดอร์หน้าจอแอปเป็น PNG โดยไม่ต้องมีกล้อง — เครื่องพัฒนาไม่มีกล้อง แต่ต้องตรวจงานดีไซน์ได้

ป้อน Analyzer ปลอมที่คืนเฟรมสังเคราะห์ + เช็กลิสต์ตัวอย่างให้ `app.Station`
ใช้เวลาแก้ layout/สี: แก้แล้วรันอันนี้ ดูรูป ไม่ต้องรอวันที่มีกล้อง

    python tools/ui_preview.py                    # ฉากปกติ -> docs/app-ui.png
    python tools/ui_preview.py empty out.png      # สถานะอื่น: empty · done · many · lost
"""
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import overlay                                                   # noqa: E402
from PySide6.QtWidgets import QApplication                       # noqa: E402

import app as ui                                                 # noqa: E402

CUPS = [(1, (170, 250, 330, 470)), (2, (430, 300, 570, 500)),
        (3, (690, 240, 830, 450)), (4, (940, 290, 1080, 490))]
PICKED, HELD = {1, 2}, {3}


class FakeInspection:
    """หน้าตาเหมือน vision.Inspection เท่าที่ overlay.draw() กับ Rack ใช้ — ไม่ต้องพึ่ง torch"""

    def __init__(self):
        self.picked = {tid: time.time() - 30 for tid in PICKED}
        self.trails = {tid: deque([(int((b[0] + b[2]) / 2) + int(60 * np.sin(i / 7)),
                                    int((b[1] + b[3]) / 2) - i * 2) for i in range(40)])
                       for tid, b in CUPS}

    def rows(self):
        # (id, ตรวจแล้ว?, ความคืบหน้า, ตรวจไปกี่วิ) — ใบที่ 3 กำลังไต่ ยังไม่ถึงเกณฑ์
        return [(tid, tid in PICKED, 1.0 if tid in PICKED else (0.62 if tid in HELD else 0.0),
                 30 if tid in PICKED else 0) for tid, _ in CUPS]


def fake_frame():
    """ฉากจำลอง: พื้นโต๊ะไล่เฉด + แก้วสี่ใบ + มือหนึ่งข้างจับใบที่ 3 แล้ววาดด้วย overlay.draw ตัวจริง"""
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = np.linspace(46, 20, h, dtype=np.uint8)[:, None, None]
    for _, (x1, y1, x2, y2) in CUPS:                     # แก้วหลอก ๆ ให้ภาพไม่ว่างเปล่า
        cv2.rectangle(frame, (x1 + 14, y1 + 20), (x2 - 14, y2), (86, 92, 100), -1)
        cv2.ellipse(frame, ((x1 + x2) // 2, y1 + 20), ((x2 - x1) // 2 - 14, 14),
                    0, 0, 360, (120, 128, 138), -1, cv2.LINE_AA)

    insp = FakeInspection()
    cx, cy = 760, 350                                    # มือหลอกรอบแก้วใบที่ 3
    pts = [(cx, cy + 120)] + [(int(cx + 44 * np.cos(a)), int(cy + 52 * np.sin(a) + 40))
                              for a in np.linspace(-2.6, 0.6, 20)]
    hands = [(pts, True, "MID", 14)]
    cups = [(tid, box, False) for tid, box in CUPS]
    return overlay.draw(frame, cups, hands, HELD, insp, False), insp


# สถานะที่ต้องดูตอนแก้ดีไซน์ — ว่างเปล่า / ครบทุกชิ้น / ของเยอะเกินรายการ / กล้องหลุด
STATES = {
    "live":  lambda r: (r, True),
    "empty": lambda r: ([], True),
    "done":  lambda r: ([(t, True, 1.0, 30 + t) for t, _, _, _ in r], True),
    "many":  lambda r: ([(t, t % 3 == 0, 1.0 if t % 3 == 0 else t / 20, 12) for t in range(1, 12)], True),
    "lost":  lambda r: (r, False),
}


class FakeAnalyzer:
    def __init__(self, frame, insp, state):
        self.frame, self.insp, self.debug = frame, insp, False
        self.rows, self.camera = STATES[state](insp.rows())
        self.events = deque([(time.strftime("%H:%M:%S"), f"ตรวจแล้ว · ชิ้น #{t}")
                             for t, ok, *_ in self.rows if ok][:3])

    def latest(self):
        return 1, self.frame.copy(), {
            "holding": self.camera, "held_s": 3.4, "fps": 28.6, "hands": 1,
            "camera": self.camera, "device": "CUDA", "rows": self.rows, "round_s": 252,
        }

    def reset(self):
        pass


def main():
    state = sys.argv[1] if len(sys.argv) > 1 else "live"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "app-ui.png"
    frame, insp = fake_frame()
    if state == "lost":
        frame = overlay.splash("reconnecting camera")
    qt = QApplication(sys.argv[:1])
    win = ui.Station(FakeAnalyzer(frame, insp, state), {"window_width": 1280})
    win.resize(1640, 812)
    win.show()
    qt.processEvents()
    win.tick()
    qt.processEvents()
    win.grab().save(str(out))
    print("wrote", out)


if __name__ == "__main__":
    main()
