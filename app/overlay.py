"""วาดผลลงบนเฟรม — สี ฟอนต์ กรอบแก้ว โครงมือ เส้นทาง และจอรอ

แยกออกมาจาก `vision.py` เพราะไฟล์นี้ **ไม่พึ่ง torch / ultralytics / mediapipe เลย**
เปิดแอปดูหน้าตาบนเครื่องที่ไม่มีสแต็ก CV ก็ยังได้ (tools/ui_preview.py ใช้ทางนี้)

เส้นแบ่ง: อะไรที่ต้องอยู่ *ตรงตำแหน่งบนภาพ* วาดที่นี่ · อะไรที่เป็นข้อความ/รายการ/ปุ่ม
เป็นวิดเจ็ต Qt ใน app.py (และพิมพ์ไทยได้ ต่างจากบนเฟรมที่ DejaVu ไม่มี glyph ไทย)
"""
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ทุกข้อความบนจอเป็นภาษาอังกฤษ — DejaVu ไม่มี glyph ไทย (ไทยไปที่วิดเจ็ต Qt)
# สีชุดเดียวกับ UI ใน app.py — ภาพกับแผงข้างต้องดูเป็นเครื่องเดียวกัน (BGR)
INK   = (27, 20, 16)      # พื้นแผงโปร่งแสง        #10141B
FG    = (243, 236, 231)   # อักษรหลัก              #E7ECF3
MUTED = (150, 132, 122)   # อักษรรอง               #7A8496
HALO  = (10, 12, 16)      # ขอบดำใต้ทุกเส้น — ทำให้ bbox/โครงมือตัดกับพื้นสว่างของโต๊ะ

# --- สองสีสัญญาณของ "การตรวจ" อยู่บน bbox ของแก้วเท่านั้น (ล็อก — ตรงกับ app.py) ---
OK    = (140, 214, 61)    # ตรวจแล้ว               #3DD68C   = app.SIGNAL
WARN  = (75, 179, 242)    # ยังไม่ตรวจ             #F2B34B   = app.AMBER
DONE, TODO = OK, WARN

# --- โครงมือเป็นคนละแกนสี (น้ำเงิน = เย็น) จะได้ไม่ปนกับ bbox · สว่างขึ้นเมื่อมือกำเข้า ---
HAND_OPEN = (192, 168, 146)   # แบ    #92A8C0  น้ำเงินหม่น เกือบเทา
HAND_MID  = (232, 165, 107)   # กลาง  #6BA5E8
HAND_FIST = (246, 130, 59)    # กำ    #3B82F6  น้ำเงินสด
STATE_COLOR = {"FIST": HAND_FIST, "OPEN": HAND_OPEN, "MID": HAND_MID}   # สีโครงมือตามท่า (พาร์ท 2)

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

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


def _line(frame, p1, p2, col, w, halo):
    """เส้นสี + ขอบดำใต้เส้น — เส้น bbox / โครงมือ อ่านออกทั้งบนพื้นมืดและพื้นสว่าง"""
    cv2.line(frame, p1, p2, HALO, w + halo, cv2.LINE_AA)
    cv2.line(frame, p1, p2, col, w, cv2.LINE_AA)


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
            path = [np.array(trail, np.int32)]
            cv2.polylines(frame, path, False, HALO, px(5), cv2.LINE_AA)
            cv2.polylines(frame, path, False, col, px(2), cv2.LINE_AA)
        x1, y1, x2, y2 = map(int, box)
        held = tid in held_ids
        t, halo = px(6 if held else 4), px(3)
        L = max(px(14), min(px(36), (x2 - x1) // 3, (y2 - y1) // 3))   # วงเล็บมุมสั้นลงตามกล่องเล็ก ไม่ให้ชนกลาง
        cv2.rectangle(frame, (x1, y1), (x2, y2), HALO, px(3), cv2.LINE_AA)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, px(1), cv2.LINE_AA)   # เส้นเต็มบาง ๆ ให้เห็นทั้งกล่อง
        for cx, sx in ((x1, 1), (x2, -1)):
            for cy, sy in ((y1, 1), (y2, -1)):                              # วงเล็บมุมหนา — ตัวเน้น
                _line(frame, (cx, cy), (cx + sx * L, cy), col, t, halo)
                _line(frame, (cx, cy), (cx, cy + sy * L), col, t, halo)
        _chip(frame, T, x1, y1 - px(32), f"#{tid}" + ("  ✓" if done else ""), col, u)

    # ── โครงมือ: สีตามท่า (พาร์ท 2) · เขียวเมื่อจับแก้ว ──
    for pts, on_cup, state, pts_in in hands:
        col = STATE_COLOR[state]                    # สีตามท่า (กำ/แบ/กลาง) — ไม่เปลี่ยนเป็นสี bbox ตอนจับแก้ว
        w = px(4 if on_cup else 3)                  # จับแก้วอยู่ = เส้นหนาขึ้น (นอกจากป้าย on cup)
        for a, b in HAND_EDGES:
            _line(frame, pts[a], pts[b], col, w, px(2))
        for x, y in pts:
            cv2.circle(frame, (x, y), px(3) + px(2), HALO, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), px(3), col, -1, cv2.LINE_AA)
        msg = state + ("  on cup" if on_cup else "") + (f"   pts:{pts_in}" if debug else "")
        _put(T, pts[0][0], pts[0][1] + px(18), msg, font(15 * u, True), col, "lm")

    return _flush(frame, T)


def splash(msg, width=1280):
    """จอรอ (เปิดกล้อง / ต่อกล้องใหม่ / โหลดโมเดล) — สปินเนอร์ + ชื่อสถานี"""
    w = int(width)
    h = w * 9 // 16
    img = np.full((h, w, 3), 11, np.uint8)
    cx, cy, r = w // 2, int(h * 0.45), max(16, h // 20)
    a = time.time() * 90
    cv2.ellipse(img, (cx, cy), (r, r), 0, 0, 360, (40, 34, 30), max(2, h // 300), cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r, r), 0, a % 360, a % 360 + 90, OK, max(2, h // 240), cv2.LINE_AA)
    T = []
    _put(T, cx, cy + r + h // 12, "INSPECTION STATION", font(h / 30, True), FG, "ma")
    _put(T, cx, cy + r + h // 12 + h // 18, msg, font(h / 42), MUTED, "ma")
    return _flush(img, T)
