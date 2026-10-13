"""CUPTICK — หน้าจอสถานีตรวจชิ้นงาน (Qt / PySide6)

ไฟล์นี้ *แสดงผลอย่างเดียว* ทุก 20ms หยิบ (เลขเฟรม, เฟรม, สถานะ) จาก `vision.Analyzer`
มาวาด — ไม่มีการตัดสินใจเรื่อง CV อยู่ในนี้เลย ตรรกะทั้งหมดอยู่ที่ `app/vision.py`

ภาษาการออกแบบ: **แพลตฟอร์มเครื่องมือ CV ระดับงานจริง** (อ้างอิงระบบดีไซน์ของ Intel Geti)
  · พื้นเทากลางไร้โทนสี แผงแบนมุม 4px เส้นขอบ 1px — ไม่มีเงา ไม่มีลวดลายตกแต่ง
  · น้ำเงิน (Intel blue) = ส่วนควบคุมเท่านั้น: ปุ่มหลัก · แท็บที่เลือก · วงโฟกัส
  · ตัวเลขทุกตัวเป็นฟอนต์ mono ความกว้างเท่ากัน — อ่านข้ามห้องแล้วไม่เต้น
  · สีที่เป็นสัญญาณคือเขียว (ตรวจแล้ว) กับอำพัน (ยังไม่ตรวจ) ที่เหลือเป็นเทา
  · ฟอนต์ Inter + Noto Sans Thai + JetBrains Mono แพ็กมากับแอป (app/fonts) — หน้าตาเหมือนกันทุกเครื่อง

รัน:  python app/app.py        ค่าที่ต้องจูนหน้างานอยู่ใน app/config.yaml ทั้งหมด
"""
import functools
import math
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import cv2

from zone import polygons_overlap
from overlay import FONT_DIR          # vision (torch/ultralytics) นำเข้าตอน main() เท่านั้น —
                                    # เปิดดูหน้าตา/รัน tools/ui_preview.py บนเครื่องที่ไม่มีสแต็ก CV ได้

try:
    from PySide6.QtCore import QObject, Qt, QPoint, QRectF, QTimer, QVariantAnimation, QPointF, Signal
    from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetrics, QImage, QPainter,
                               QPalette, QPen, QPixmap, QPolygonF)
    from PySide6.QtWidgets import (QAbstractButton, QApplication, QDialog, QDoubleSpinBox, QLabel, QHBoxLayout,
                                   QMenu, QMessageBox, QProgressBar, QScrollArea, QVBoxLayout, QWidget)
except ImportError:
    raise SystemExit("\nไม่มี PySide6 — ติดตั้งก่อน:  pip install -r app/requirements.txt\n")


# ─────────────────────────── ธีม ───────────────────────────
# ห้องเดโมปิดไฟ ฉายโปรเจกเตอร์ — พื้นยิ่งเข้ม ตัวอักษรยิ่งสว่าง อ่านจากท้ายห้องได้
# สองสีสัญญาณ (SIGNAL/AMBER) ล็อกไว้ ต้องตรงกับ overlay.py (ที่นั่นเป็น BGR) — แก้ต้องแก้คู่กัน
GROUND = "#0B0B0C"   # พื้นหน้าต่าง
PANEL  = "#1A1A1C"   # หัวแถบ / แผงข้าง / แถบสถานะ
CARD   = "#252528"   # แถวเช็กลิสต์ ช่องกรอก (สว่างกว่าแผงให้แถวลอยออกมา)
HOVER  = "#303034"   # ปุ่ม/แท็บตอนเมาส์อยู่
LINE   = "#3A3A40"   # เส้นแบ่ง 1px / ขอบ
TXT    = "#F3F3F5"   # อักษรหลัก
HEAD   = "#D8D8DC"   # ป้ายหัวข้อของแต่ละแผง
DIM    = "#ABABB3"   # อักษรรอง
FAINT  = "#8D8D96"   # ป้ายกำกับจาง (timestamp, hint) — ผ่าน 4.5:1 บนพื้น PANEL
ACCENT = "#0068B5"   # ส่วนควบคุม: ปุ่มหลัก (อักษรขาวบนสีนี้ 5.8:1)
ACCENT_HI = "#00C7FD"   # ส่วนควบคุม: วงโฟกัส · ขีดใต้แท็บ · ข้อความแจ้งผลการกด
SIGNAL = "#3DD68C"   # ตรวจแล้ว / กำลังหยิบ            (ล็อก — ตรงกับ overlay.OK)
AMBER  = "#F2B34B"   # ยังไม่ตรวจ / กล้องหลุด          (ล็อก — ตรงกับ overlay.WARN)
VIOLET = "#A78BFA"   # โหมด zone: อยู่ที่ A / polygon A        (ล็อก — ตรงกับ overlay.READY)
BLACK  = "#000000"   # พื้นหลังกรอบภาพ
R = 4                # มุมมนของทุกกรอบ — Geti ใช้มุมเล็ก ไม่ใช่ปุ่มแคปซูล

# โหมด zone: สีและชื่อสถานะของแก้ว (เทา → ม่วง → อำพัน → เขียว) — ตรงกับ overlay.ZONE_COLOR
ZONE_HEX = {"idle": "#7A8496", "at_a": VIOLET, "transit": AMBER, "done": SIGNAL}
ZONE_NAME = {"idle": "IDLE", "at_a": "AT A", "transit": "IN TRANSIT", "done": "DONE"}

# Inter ไม่มี glyph ไทย — ไล่ family ให้ Qt เลือกรายตัวอักษร (ไทยตกไปที่ Noto Sans Thai)
UI_FAMILIES = ["Inter", "Noto Sans Thai", "Segoe UI", "Helvetica Neue", "Tahoma", "sans-serif"]
MONO_FAMILIES = ["JetBrains Mono", "Cascadia Mono", "SF Mono", "Consolas",
                 "DejaVu Sans Mono", "monospace"]


@functools.cache
def load_fonts():
    """ลงฟอนต์ที่แพ็กมากับแอปให้ Qt — เรียกครั้งเดียว (cache) ตอนสร้างฟอนต์ตัวแรก"""
    for f in sorted(FONT_DIR.glob("*.ttf")):
        if QFontDatabase.addApplicationFont(str(f)) < 0:
            print("โหลดฟอนต์ไม่ได้:", f)          # ไฟล์เสีย → Qt เงียบแล้วตกไปฟอนต์เครื่อง หน้าตาเพี้ยนโดยไม่มีใครรู้


def face(size, weight=QFont.Weight.Normal, mono=False):
    """ฟอนต์หนึ่งตัว — mono สำหรับ *ตัวเลขทุกตัว*"""
    load_fonts()
    f = QFont()
    f.setFamilies(MONO_FAMILIES if mono else UI_FAMILIES)
    f.setPixelSize(size)
    f.setWeight(weight)
    return f


def pen(color, width=1.0):
    return QPen(QColor(color), width)


def text(p, x, y, s, f, color, align=Qt.AlignmentFlag.AlignLeft, w=0):
    """เขียนข้อความโดยให้ y เป็นขอบบนของบรรทัด (คิด layout ง่ายกว่าอ้าง baseline)"""
    p.setFont(f)
    p.setPen(pen(color))
    box = QRectF(x, y, w or 2000, QFontMetrics(f).height() + 2)
    p.drawText(box, int(align | Qt.AlignmentFlag.AlignTop), s)


def ground(w, color):
    """ระบายพื้นหลังวิดเจ็ตด้วย palette — ไม่ใช้ stylesheet เพราะ QSS ไหลลงลูกทุกตัว"""
    w.setAutoFillBackground(True)
    pal = w.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(color))
    w.setPalette(pal)


def section(p, x, y, s):
    """ป้ายหัวข้อแผง — ตัวหนาธรรมดา ไม่มีเครื่องประดับ"""
    text(p, x, y, s, face(13, QFont.Weight.DemiBold), HEAD)


def cup_mark(p, cx, cy, s, color):
    """โลโก้แก้วมัคมีไอน้ำ กว้างประมาณ 0.8*s สูงประมาณ 0.95*s — รูปทรงเดียวกับไอคอนตัวติดตั้ง (installer/make_assets.py: cup)"""
    c = QColor(color)
    bw = 0.52 * s
    x0, y0 = cx - 0.405 * s, cy - 0.10 * s
    x1 = x0 + bw
    stroke = max(1.5, 0.085 * s)
    p.setBrush(Qt.BrushStyle.NoBrush)                   # หู: วงแหวน (ครึ่งซ้ายถูกตัวแก้วทับ)
    p.setPen(QPen(c, stroke))
    hr = 0.19 * s - stroke / 2
    p.drawEllipse(QPointF(x1 + 0.10 * s, y0 + 0.25 * s), hr, hr)
    p.setPen(Qt.PenStyle.NoPen)                         # ตัวแก้ว: ก้นโค้ง ปากตรง + ขอบปาก
    p.setBrush(c)
    p.drawRoundedRect(QRectF(x0, y0, bw, bw), 0.15 * s, 0.15 * s)
    p.drawRect(QRectF(x0, y0, bw, bw / 2))
    p.drawRoundedRect(QRectF(x0 - 0.035 * s, y0 - 0.05 * s, bw + 0.07 * s, 0.085 * s), 0.035 * s, 0.035 * s)
    steam = QPen(c, max(1.2, 0.055 * s))                # ไอน้ำ: เส้นคดสองเส้น
    steam.setCapStyle(Qt.PenCapStyle.RoundCap)
    steam.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(steam)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for frac, phase in ((0.28, 0.0), (0.68, math.pi)):
        px = x0 + bw * frac
        p.drawPolyline(QPolygonF([QPointF(px + 0.035 * s * math.sin(t * 2 * math.pi + phase), y0 - 0.11 * s - t * 0.27 * s)
                                  for t in (i / 24 for i in range(25))]))


def spinner(p, cx, cy, r):
    """สปินเนอร์วาดด้วย QPainter — มุมคิดจากเวลาจริง ไม่นับเฟรม: ถ้าเธรดหลักสะดุดมันกระโดด ไม่ช้าลง"""
    box = QRectF(cx - r, cy - r, 2 * r, 2 * r)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(pen(LINE, 3))
    p.drawEllipse(box)
    arc = pen(ACCENT_HI, 3)
    arc.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(arc)
    p.drawArc(box, int(-(time.time() * 330 % 360) * 16), 100 * 16)


# ─────────────────────────── ปุ่ม ───────────────────────────
class Button(QAbstractButton):
    """ปุ่มวาดเอง — มุมเล็ก antialias + ไล่สีตอน hover ใน 160ms (QSS ของ Qt ทำ transition ไม่ได้)
    เป็น QAbstractButton จึงได้ focus ring, กด Space/Enter, และ signal clicked มาฟรี"""

    def __init__(self, label, hint="", primary=False):
        super().__init__()
        self.label, self.hint, self.primary = label, hint, primary
        self._glow = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(40 if primary else 36)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.anim = QVariantAnimation(self)
        self.anim.valueChanged.connect(self._set_glow)

    def _set_glow(self, v):
        self._glow = v
        self.update()

    def set_label(self, s):
        self.label = s
        self.update()

    def _to(self, target):
        """วิ่งจากค่าปัจจุบันเสมอ — เข้า-ออกเร็ว ๆ จะได้ไม่กระตุกกลับไปเริ่มใหม่"""
        self.anim.stop()
        self.anim.setStartValue(self._glow)
        self.anim.setEndValue(target)
        self.anim.setDuration(max(1, int(160 * abs(target - self._glow))))
        self.anim.start()

    def enterEvent(self, e):
        self._to(1.0)

    def leaveEvent(self, e):
        self._to(0.0)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        g = self._glow * (0.55 if self.isDown() else 1.0)
        if self.primary:                         # ปุ่มหลักของหน้า: น้ำเงินทึบ
            p.setBrush(QColor(ACCENT).lighter(int(100 + 22 * g)))
            p.setPen(Qt.PenStyle.NoPen)
            ink, sub = QColor("#FFFFFF"), QColor(255, 255, 255, 170)
        else:                                    # ปุ่มรอง: โปร่ง มีขอบ — hover แล้วพื้นสว่างขึ้น
            p.setBrush(QColor(HOVER) if g > 0.5 else QColor(CARD) if g > 0 else Qt.BrushStyle.NoBrush)
            p.setPen(pen(QColor(LINE).lighter(int(100 + 40 * g))))
            ink, sub = QColor(TXT), QColor(FAINT)
        p.drawRoundedRect(r, R, R)
        if self.hasFocus():                      # a11y — เห็นชัดว่าโฟกัสอยู่ปุ่มไหน
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(pen(ACCENT_HI, 2))
            p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), R + 1, R + 1)

        f = face(13, QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(pen(ink))
        pad = 14 if self.primary else 11
        room = 0 if self.primary or self.width() > 200 else 16   # ปุ่มแคบ: กันป้ายชนคีย์ลัด
        p.drawText(r.adjusted(pad, 0, -pad - room, 0),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                   if self.primary else int(Qt.AlignmentFlag.AlignCenter), self.label)
        if self.hint:                            # คีย์ลัดกำกับทุกปุ่ม
            p.setFont(face(11, QFont.Weight.DemiBold, mono=True))
            p.setPen(pen(sub))
            p.drawText(r.adjusted(0, 0, -pad, 0),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), self.hint)


# ─────────────────────────── กรอบภาพ ───────────────────────────
class Viewport(QWidget):
    """ภาพจากกล้อง วางกลางพื้นดำ + วงเล็บมุมแบบขอบเครื่องวัด + ชิปบอกค่าเครื่อง
    เฟรมถูกย่อด้วย cv2 ครั้งเดียวตอนได้ของใหม่ (ถูกกว่าให้ Qt ย่อทุกครั้งที่ repaint)"""

    def __init__(self):
        super().__init__()
        self.pix = None
        self.raw = None                 # เฟรมล่าสุด (ยังไม่ย่อ) — ไว้ให้ปุ่ม S เซฟ
        self.st = {}
        self.beat = 0.0
        self.busy = ""                  # ข้อความจอรอ ("กำลังเชื่อมต่อกล้อง…") — ไม่ว่าง = วาดสปินเนอร์แทนภาพ
        self.box = None                 # สี่เหลี่ยมของภาพบนวิดเจ็ต — ไว้แปลงตำแหน่งเมาส์ ↔ สัดส่วนเฟรม
        self.zones = {"A": None, "B": None}   # polygon สัดส่วน 0-1 (Station เป็นเจ้าของ ที่นี่แค่วาด)
        self.show_zones = False         # True เฉพาะโหมด zone — กลับโหมดตรวจแล้วกรอบต้องหายจากจอ (ค่ากรอบยังจำไว้ใช้ต่อ)
        self.stage = None               # "A"/"B" = กำลังวาดอันนั้น · None = ไม่ได้วาด
        self.pts = []                   # จุดของ polygon ที่กำลังวาด
        self.hover = None
        self.on_point = self.on_close = self.on_undo = lambda *a: None   # Station ผูกให้
        self.setMouseTracking(True)
        self.setMinimumSize(480, 270)

    def set_frame(self, frame):
        self.busy = ""
        self.raw = frame
        h, w = frame.shape[:2]
        s = min(self.width() / w, self.height() / h)
        small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA if s < 0.7 else cv2.INTER_LINEAR)   # ย่อเล็กน้อย: LINEAR เร็วกว่า ~4 เท่า ภาพไม่ต่าง
        img = QImage(small.data, small.shape[1], small.shape[0], small.strides[0],
                     QImage.Format.Format_BGR888)
        self.pix = QPixmap.fromImage(img)            # fromImage คัดลอกพิกเซลให้แล้ว บัฟเฟอร์ numpy ทิ้งได้
        self.update()

    def wait(self, msg):
        """ไม่มีภาพให้โชว์ — ทิ้งภาพเก่า (กล้องใหม่ต้องไม่เห็นภาพค้างของกล้องก่อนหน้า) แล้วโชว์ข้อความรอ"""
        self.busy, self.pix, self.raw, self.box = msg, None, None, None
        self.update()

    def pulse(self):
        """เฟรมเดิม: วาดซ้ำเฉพาะมุมซ้ายบนที่จุด LIVE เต้นอยู่ ไม่ต้องวาดทั้งภาพ 50 ครั้ง/วินาที"""
        if self.box is None:
            self.update()
        else:
            self.update(int(self.box.left()), int(self.box.top()), 170, 60)

    def resizeEvent(self, e):
        if self.raw is not None:
            self.set_frame(self.raw)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(BLACK))
        if self.busy:
            spinner(p, self.width() / 2, self.height() / 2 - 14, 22)
            text(p, 0, self.height() / 2 + 28, self.busy, face(14, QFont.Weight.DemiBold), DIM,
                 Qt.AlignmentFlag.AlignCenter, self.width())
            return
        if self.pix is None:
            return
        x = (self.width() - self.pix.width()) // 2
        y = (self.height() - self.pix.height()) // 2
        p.drawPixmap(x, y, self.pix)
        box = QRectF(x, y, self.pix.width(), self.pix.height())
        self.box = box

        # วงเล็บมุมสี่มุม — ภาษาเดียวกับกรอบแก้วในเฟรม ทำให้ chrome กับภาพเป็นเครื่องเดียวกัน
        p.setPen(pen(QColor(255, 255, 255, 46), 1.0))
        L = 22
        for cx, sx in ((box.left() + 6, 1), (box.right() - 6, -1)):
            for cy, sy in ((box.top() + 6, 1), (box.bottom() - 6, -1)):
                p.drawLine(QPointF(cx, cy), QPointF(cx + sx * L, cy))
                p.drawLine(QPointF(cx, cy), QPointF(cx, cy + sy * L))

        live = self.st.get("camera", True)
        self._marker(p, box.left() + 22, box.top() + 34, live)
        n = self.st.get("hands", 0)      # กล้องหลุด = ตัวเลขค้างอยู่ อย่าโชว์ให้เข้าใจผิดว่ายังเดินอยู่
        chips = [f"{self.st.get('fps', 0):.1f} FPS" if live else "— FPS",
                 self.st.get("device", "—"),
                 f"{n} HAND" + ("S" if n != 1 else "") if live else "— HANDS"]
        cx = box.left() + 20
        for c in chips:
            cx += self._chip(p, cx, box.bottom() - 42, c) + 8
        self._paint_zones(p, box)

    # ── โหมด zone: วาด polygon A/B + รับเมาส์ ──
    def _norm(self, pos):
        if self.box is None or not self.box.contains(pos):
            return None
        return ((pos.x() - self.box.left()) / self.box.width(),
                (pos.y() - self.box.top()) / self.box.height())

    def mousePressEvent(self, e):
        if not self.stage:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            pt = self._norm(e.position())
            if pt:
                self.on_point(pt)
        elif e.button() == Qt.MouseButton.RightButton:
            self.on_undo()

    def mouseDoubleClickEvent(self, e):
        if self.stage and e.button() == Qt.MouseButton.LeftButton:
            self.on_close()

    def mouseMoveEvent(self, e):
        self.hover = self._norm(e.position()) if self.stage else None

    def leaveEvent(self, e):
        self.hover = None

    def _to_widget(self, box, pt):
        return QPointF(box.left() + pt[0] * box.width(), box.top() + pt[1] * box.height())

    def _paint_zones(self, p, box):
        if not self.show_zones:
            return
        for name, col in (("A", VIOLET), ("B", SIGNAL)):
            poly = self.zones.get(name)
            if not poly:
                continue
            pts = [self._to_widget(box, q) for q in poly]
            fill = QColor(col)
            fill.setAlpha(52)
            p.setPen(pen(col, 2.5))
            p.setBrush(fill)
            p.drawPolygon(QPolygonF(pts))
            # ป้ายอยู่มุมซ้ายบนของกรอบ — กลางกรอบคือที่ที่แก้ววางอยู่ ป้ายตรงนั้นจะบังตัวแก้ว
            chip = QRectF(min(q.x() for q in pts), min(q.y() for q in pts) - 34, 30, 30)
            if chip.top() < box.top() + 2:                       # กรอบชิดขอบบนของภาพ → ย้ายเข้ามาอยู่ในมุมกรอบ
                chip.moveTo(chip.left() + 4, chip.top() + 38)
            chip.moveLeft(max(box.left() + 2, min(chip.left(), box.right() - 32)))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(6, 9, 14, 215))
            p.drawRoundedRect(chip, R, R)
            text(p, chip.left(), chip.top() + 6, name, face(17, QFont.Weight.Bold, mono=True), col,
                 Qt.AlignmentFlag.AlignCenter, 30)
        if not self.stage:
            return
        col = VIOLET if self.stage == "A" else SIGNAL
        pts = [self._to_widget(box, q) for q in self.pts]
        if self.hover and pts:
            pts.append(self._to_widget(box, self.hover))       # เส้นยางไปหาเมาส์
        p.setPen(pen(col, 2.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if len(pts) > 1:
            p.drawPolyline(QPolygonF(pts))
        p.setBrush(QColor(col))
        for q in pts[:len(self.pts)]:
            p.drawEllipse(q, 5, 5)
        msg = (f"วาดกรอบ {self.stage}  ·  คลิกซ้าย = เพิ่มจุด  ·  ดับเบิลคลิก/Enter = ปิดกรอบ  ·  "
               "คลิกขวา/Backspace = ย้อน  ·  Esc = ล้าง")
        f = face(13, QFont.Weight.DemiBold)
        w = QFontMetrics(f).horizontalAdvance(msg) + 28
        x = box.left() + (box.width() - w) / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(6, 9, 14, 225))
        p.drawRoundedRect(QRectF(x, box.top() + 14, w, 34), R, R)
        p.setPen(pen(col, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(x + 0.5, box.top() + 14.5, w - 1, 33), R, R)
        text(p, x, box.top() + 22, msg, f, TXT, Qt.AlignmentFlag.AlignCenter, w)

    def _marker(self, p, x, y, live):
        """จุดสถานะกล้อง — เต้นเบา ๆ ตอนภาพสด ค้างสีอำพันตอนกล้องหลุด"""
        col = QColor(SIGNAL if live else AMBER)
        pulse = 0.55 + 0.45 * abs((self.beat % 2.0) - 1.0) if live else 1.0
        halo = QColor(col)
        halo.setAlphaF(0.22 * pulse)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawEllipse(QPointF(x + 4, y + 6), 8, 8)
        p.setBrush(col)
        p.drawEllipse(QPointF(x + 4, y + 6), 3.5, 3.5)
        text(p, x + 18, y - 2, "Live" if live else "Reconnecting",
             face(13, QFont.Weight.DemiBold), col.name())

    def _chip(self, p, x, y, s):
        f = face(12, QFont.Weight.DemiBold, mono=True)
        w = QFontMetrics(f).horizontalAdvance(s) + 22
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(6, 9, 14, 205))
        p.drawRoundedRect(QRectF(x, y, w, 28), R, R)
        p.setPen(pen(QColor(255, 255, 255, 40)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(x + 0.5, y + 0.5, w - 1, 27), R, R)
        p.setFont(f)
        p.setPen(pen(TXT))
        p.drawText(QRectF(x, y, w, 28), int(Qt.AlignmentFlag.AlignCenter), s)
        return w


# ─────────────────────────── เช็กลิสต์ ───────────────────────────
ROW_H = 46       # ponytail: ไม่มี scrollbar — เกินที่ว่างสรุปเป็น "+ อีก N ชิ้น" บรรทัดเดียว
LOG_ROWS = 5     # บันทึกเหตุการณ์ท้ายแผง


class Rack(QWidget):
    """แผงเช็กลิสต์ทั้งแผง วาดใน paintEvent เดียว — คุมเส้น 1px ได้ทุกเส้น
    และไม่ต้องสร้าง/ทำลายวิดเจ็ตทุกครั้งที่รายการเปลี่ยน

    หน่วยนับคือ "ช่อง" ไม่ใช่ "แถบเปอร์เซ็นต์": มิเตอร์ด้านบนมีช่องเท่าจำนวนชิ้นพอดี
    เห็นทั้งจำนวนและความคืบหน้าในภาพเดียว · แถวที่ยังไม่ตรวจมีเส้นอำพันไต่ที่ขอบล่าง
    คือตัวนับแบบรั่วของ Inspection กำลังสะสม — ผู้ชมเห็นเครื่อง "กำลังมั่นใจขึ้น"
    ท้ายแผงเป็นบันทึกเหตุการณ์ของรอบนี้ — ตอบว่า "เมื่อกี้เกิดอะไรขึ้น" ได้โดยไม่ต้องจ้องจอ"""

    def __init__(self):
        super().__init__()
        self.rows, self.log = [], []
        self.zone = False        # True = แถวเป็น (tid, สถานะ) ของโหมด zone แทน (tid, ตรวจแล้ว?, ความคืบหน้า, วินาที)
        self.setMinimumWidth(340)

    def set_rows(self, rows, events, zone=False):
        log = list(events)[-LOG_ROWS:][::-1]
        if (rows, log, zone) != (self.rows, self.log, self.zone):
            self.rows, self.log, self.zone = rows, log, zone
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)   # ไม่ระบายพื้น — ปล่อยให้ขีดสเกลของ Rail ทะลุขึ้นมา
        W = self.width()
        pad = 22
        inner = W - pad * 2
        zone = self.zone
        counted = [r for r in self.rows if r[1] != "idle"] if zone else self.rows   # zone: IDLE ไม่นับ
        done = sum(1 for r in counted if (r[1] == "done" if zone else r[1]))

        section(p, pad, 26, "รอบ Zone A → B" if zone else "รอบตรวจปัจจุบัน")

        # ── ตัวเลขพระเอก: อ่านจากท้ายห้องได้ ──
        big = face(54, QFont.Weight.Bold, mono=True)
        p.setFont(big)
        p.setPen(pen(SIGNAL if counted and done == len(counted) else TXT))
        p.drawText(QRectF(pad, 46, inner, 64), int(Qt.AlignmentFlag.AlignLeft), f"{done:02d}")
        wd = QFontMetrics(big).horizontalAdvance(f"{done:02d}")
        p.setFont(face(30, QFont.Weight.Normal, mono=True))
        p.setPen(pen(QColor(255, 255, 255, 110)))
        p.drawText(QRectF(pad + wd + 10, 68, inner, 46), int(Qt.AlignmentFlag.AlignLeft),
                   f"/ {len(counted):02d}")
        text(p, pad, 114, "แก้วที่ถึง B แล้ว" if zone else "ชิ้นงานที่ตรวจแล้ว", face(13), DIM)

        # ── มิเตอร์แบบช่อง: หนึ่งช่องต่อหนึ่งชิ้น ──
        y = 142
        n = len(counted)
        if n:
            gap, seg = 4, (inner - 4 * (n - 1)) / n
            for i, row in enumerate(counted):
                x = pad + i * (seg + gap)
                ok, prog = (row[1] == "done", 0) if zone else (row[1], row[2])
                p.setPen(Qt.PenStyle.NoPen)
                if zone:                                   # ช่องระบายสีตามสถานะ
                    p.setBrush(QColor(ZONE_HEX[row[1]]))
                else:
                    p.setBrush(QColor(SIGNAL) if ok else QColor(255, 255, 255, 36))
                p.drawRoundedRect(QRectF(x, y, seg, 6), 2, 2)
                if not ok and prog > 0:                    # ช่องกำลังไต่
                    p.setBrush(QColor(AMBER))
                    p.drawRoundedRect(QRectF(x, y, max(3.0, seg * prog), 6), 2, 2)
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 28))
            p.drawRoundedRect(QRectF(pad, y, inner, 6), 2, 2)

        p.setPen(pen(LINE))
        p.drawLine(pad, y + 30, W - pad, y + 30)
        section(p, pad, y + 44, "รายการชิ้นงาน")

        # ── แถวชิ้นงาน — จำนวนแถวที่โชว์คิดจากที่ว่างจริง ไม่ใช่ค่าคงที่ ──
        # (จอเล็ก/จอโปรเจกเตอร์สูงไม่เท่ากัน ตัวเลขตายตัวจะไปทับบันทึกเหตุการณ์)
        # รายการได้ที่ก่อน บันทึกเหตุการณ์หดลงเหลืออย่างน้อย 2 บรรทัดเมื่อชิ้นงานเยอะ
        top = y + 68
        want = len(self.rows) * (ROW_H + 6)
        log_n = max(2, min(LOG_ROWS, int((self.height() - top - want - 44) // 22)))
        compact = self.height() < top + ROW_H + 6 + 26 + 14 + 34 + 44
        base = self.height() - log_n * 22 - 34           # ขอบบนของบันทึกเหตุการณ์
        if compact:
            base = self.height()  # จอเตี้ย: ให้ที่รายการก่อน เหตุการณ์ล่าสุดยังอยู่ใน footer
        room = base - 14 - top
        cap = max(0, int(room // (ROW_H + 6)))
        if not self.rows:
            text(p, pad, top + 10, "ยังไม่เห็นแก้วในเฟรม" if zone else "ยังไม่เห็นชิ้นงานในเฟรม",
                 face(12), FAINT)
            text(p, pad, top + 32,
                 "วางแก้วในกรอบ A แล้วยกไปวางที่ B" if zone else "วางแก้วให้กล้องเห็น แล้วหยิบขึ้นมาตรวจ",
                 face(11), FAINT)
        else:
            shown = self.rows[:cap]
            if len(shown) < len(self.rows) and cap * (ROW_H + 6) + 20 > room:
                shown = shown[:-1]               # ยอมทิ้งอีกแถวเพื่อให้ "+ อีก N ชิ้น" มีที่ยืน
            for i, row in enumerate(shown):
                if zone:
                    self._zone_row(p, pad, top + i * (ROW_H + 6), inner, *row)
                else:
                    self._row(p, pad, top + i * (ROW_H + 6), inner, *row)
            if len(shown) < len(self.rows):
                text(p, pad, top + len(shown) * (ROW_H + 6) + 6,
                     f"+ อีก {len(self.rows) - len(shown)} ชิ้น", face(11), FAINT)

        # ── บันทึกเหตุการณ์ — ยึดขอบล่างของแผง ใหม่สุดอยู่บน ──
        if compact:
            return
        p.setPen(pen(LINE))
        p.drawLine(pad, base - 14, W - pad, base - 14)
        section(p, pad, base, "บันทึกเหตุการณ์")
        for i in range(log_n):
            ly = base + 22 + i * 22
            if i < len(self.log):
                when, what = self.log[i]
                text(p, pad, ly, when, face(11, mono=True), FAINT)
                text(p, pad + 70, ly - 1, what, face(11), DIM if i else TXT)
            else:                       # บรรทัดว่างของสมุดบันทึก — ที่ว่างตรงนี้ตั้งใจเว้น ไม่ใช่ layout พัง
                p.setPen(pen(QColor(255, 255, 255, 12)))
                p.drawLine(pad, ly + 9, W - pad, ly + 9)

    def _card(self, p, x, y, w):
        p.setPen(pen(LINE))
        p.setBrush(QColor(CARD))
        p.drawRoundedRect(QRectF(x + 0.5, y + 0.5, w - 1, ROW_H - 1), R, R)

    def _tag(self, p, x, y, s, col, f):
        """ป้ายสถานะ: พื้นโปร่งสีเดียวกับตัวอักษร — ภาษาเดียวกับป้ายบนกรอบแก้ว"""
        w = QFontMetrics(f).horizontalAdvance(s) + 20
        fill = QColor(col)
        fill.setAlpha(36)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(QRectF(x, y + 11, w, ROW_H - 22), R, R)
        text(p, x, y + 14, s, f, col.name(), Qt.AlignmentFlag.AlignCenter, w)

    def _zone_row(self, p, x, y, w, tid, state):
        col = QColor(ZONE_HEX[state])
        self._card(p, x, y, w)
        text(p, x + 14, y + 12, f"#{tid:02d}", face(18, QFont.Weight.Bold, mono=True), TXT)
        self._tag(p, x + 70, y, ZONE_NAME[state], col, face(12, QFont.Weight.DemiBold, mono=True))

    def _row(self, p, x, y, w, tid, ok, prog, held_s):
        col = QColor(SIGNAL if ok else AMBER)
        self._card(p, x, y, w)
        text(p, x + 14, y + 12, f"#{tid:02d}", face(18, QFont.Weight.Bold, mono=True), TXT)
        self._tag(p, x + 70, y, "ตรวจแล้ว" if ok else "ยังไม่ตรวจ", col, face(12, QFont.Weight.DemiBold))
        if ok:
            text(p, x, y + 17, f"{held_s:5.0f}s", face(11, mono=True), FAINT,
                 Qt.AlignmentFlag.AlignRight, w - 14)
        elif prog > 0:                              # ตัวนับแบบรั่วกำลังสะสม
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(AMBER))
            p.drawRoundedRect(QRectF(x + 1, y + ROW_H - 3, (w - 2) * prog, 2), 1, 1)


# ─────────────────────────── หน้าต่างหลัก ───────────────────────────
class Station(QWidget):
    """หน้าต่างเดียว: หัวแถบ · ภาพ+แผงข้าง · แถบสถานะ
    ไม่คิดอะไรเอง — ทุก 20ms ดึงของล่าสุดจาก Analyzer มาวาด"""

    def __init__(self, analyzer, cfg):
        super().__init__()
        self.an, self.cfg = analyzer, cfg
        self.seq, self.full = -1, False
        self.msg = ("", 0.0)
        self.setWindowTitle("Cuptick — เช็กลิสต์ผู้ตรวจ")
        vw = int(cfg.get("window_width", 1280))
        self.resize(vw + 360, vw * 9 // 16 + 92)
        self.setMinimumSize(980, 620)
        ground(self, GROUND)

        self.view = Viewport()
        self.rack = Rack()
        self.head = Header()
        self.head.on_mode = self.set_mode
        self.foot = Footer()

        self.new_round = Button("เริ่มรอบตรวจใหม่", "R", primary=True)
        self.new_round.clicked.connect(self.reset)
        self.cam_btn = Button("เลือกกล้อง", "C")
        self.cam_btn.clicked.connect(self.pick_camera)
        self.mode = "inspect"                       # inspect = เช็กลิสต์ตรวจแก้ว · zone = ย้ายแก้ว A → B
        self.redraw_btn = Button("วาดใหม่", "X")
        self.redraw_btn.clicked.connect(self.start_drawing)
        self.redraw_btn.hide()
        self.view.on_point, self.view.on_close, self.view.on_undo = (
            self.add_point, self.close_polygon, self.undo_point)
        keys = QVBoxLayout()
        keys.setContentsMargins(22, 0, 22, 20)
        keys.setSpacing(8)
        # ค่า threshold ทั้งหมดอยู่ในหน้าต่างลอยแยก (ปุ่ม "ตั้งค่า" / คีย์ T) — แผงข้างจะได้ไม่รก
        self.settings = QDialog(self)
        self.settings.setWindowTitle("ตั้งค่า · threshold")
        self.settings.setWindowFlag(Qt.WindowType.Tool)      # ลอยเหนือหน้าต่างหลัก ไม่ขึ้นแถบงาน · ไม่บล็อกหน้าหลัก
        self.settings.setFixedWidth(380)
        ground(self.settings, PANEL)
        form = QVBoxLayout(self.settings)
        form.setContentsMargins(22, 18, 22, 18)
        form.setSpacing(10)
        hint = QLabel("มีผลทันทีระหว่างรัน · ไม่บันทึกลง config.yaml")
        hint.setFont(face(11))
        hint.setStyleSheet(f"color: {FAINT}")
        form.addWidget(hint)
        conf_row = QHBoxLayout()
        conf_label = QLabel("เกณฑ์ conf · cup")
        conf_label.setFont(face(12, QFont.Weight.DemiBold))
        conf_label.setStyleSheet(f"color: {HEAD}")
        self.conf_input = QDoubleSpinBox()
        self.conf_input.setRange(0.01, 1.00)
        self.conf_input.setDecimals(2)
        self.conf_input.setSingleStep(0.05)
        self.conf_input.setValue(float(cfg.get("conf", 0.25)))
        self.conf_input.setKeyboardTracking(False)
        self.conf_input.setFixedSize(112, 36)
        self.conf_input.setFont(face(14, mono=True))
        self.conf_input.setAccessibleName("เกณฑ์ confidence ของ class cup")
        self.conf_input.setToolTip("เกณฑ์เริ่ม ID แก้วใหม่ · ID เดิมใช้คะแนนต่ำช่วยติดตามได้\nลดค่าเพื่อตรวจพบง่ายขึ้น · เพิ่มค่าเพื่อลดการตรวจผิด\nมีผลระหว่างรัน ไม่บันทึกลง config.yaml")
        self.conf_input.setStyleSheet(
            f"QDoubleSpinBox {{background:{CARD}; color:{TXT}; border:1px solid {LINE};"
            f"border-radius:{R}px; padding:4px 8px;}}"
            f"QDoubleSpinBox:focus {{border:1px solid {ACCENT_HI};}}"
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {width:24px;}")
        self.conf_input.valueChanged.connect(self.set_cup_conf)
        conf_label.setBuddy(self.conf_input)
        conf_row.addWidget(conf_label, 1)
        conf_row.addWidget(self.conf_input)
        form.addLayout(conf_row)
        duration_row = QHBoxLayout()
        duration_label = QLabel("เวลาถือก่อนตรวจแล้ว")
        duration_label.setFont(face(12, QFont.Weight.DemiBold))
        duration_label.setStyleSheet(f"color: {HEAD}")
        self.duration_input = QDoubleSpinBox()
        self.duration_input.setRange(0.1, 30.0)
        self.duration_input.setDecimals(1)
        self.duration_input.setSingleStep(0.1)
        self.duration_input.setSuffix(" s")
        self.duration_input.setValue(float(cfg.get("pick_seconds", 1.0)))
        self.duration_input.setKeyboardTracking(False)
        self.duration_input.setFixedSize(112, 36)
        self.duration_input.setFont(face(14, mono=True))
        self.duration_input.setStyleSheet(self.conf_input.styleSheet())
        self.duration_input.setAccessibleName("เวลาถือชิ้นงานก่อนนับว่าตรวจแล้ว หน่วยวินาที")
        self.duration_input.setToolTip("เวลาถือสะสมก่อนนับว่าตรวจแล้ว (วินาที)\nเปลี่ยนค่าแล้วเริ่มสะสมใหม่ ชิ้นที่ตรวจแล้วคงเดิม\nใช้ระหว่างรัน ไม่บันทึกลง config.yaml")
        self.duration_input.valueChanged.connect(self.set_pick_seconds)
        duration_label.setBuddy(self.duration_input)
        duration_row.addWidget(duration_label, 1)
        duration_row.addWidget(self.duration_input)
        form.addLayout(duration_row)
        self.hand_conf_inputs = {}
        self.hand_conf_timer = QTimer(self)
        self.hand_conf_timer.setSingleShot(True)
        self.hand_conf_timer.setInterval(350)
        self.hand_conf_timer.timeout.connect(self.set_hand_confidences)
        for key, label, default in (
            ("min_hand_detection_confidence", "มือ · Detection", 0.3),
            ("min_hand_presence_confidence", "มือ · Presence", 0.5),
            ("min_tracking_confidence", "มือ · Tracking", 0.5),
        ):
            row = QHBoxLayout()
            title = QLabel(label)
            title.setFont(face(12, QFont.Weight.DemiBold))
            title.setStyleSheet(f"color: {HEAD}")
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(float(cfg.get(key, default)))
            spin.setKeyboardTracking(False)
            spin.setFixedSize(112, 30)
            spin.setFont(face(14, mono=True))
            spin.setStyleSheet(self.conf_input.styleSheet())
            spin.setAccessibleName(key)
            spin.setToolTip(f"MediaPipe · {key}\nปรับระหว่างรัน ไม่บันทึกลง config.yaml")
            spin.valueChanged.connect(lambda _: self.hand_conf_timer.start())
            title.setBuddy(spin)
            row.addWidget(title, 1)
            row.addWidget(spin)
            form.addLayout(row)
            self.hand_conf_inputs[key] = spin
        keys.addWidget(self.new_round)
        self.settings_btn = Button("ตั้งค่า", "T")
        self.settings_btn.clicked.connect(self.toggle_settings)
        keys.addWidget(self.settings_btn)
        cam_row = QHBoxLayout()                     # กล้อง (· วาดใหม่ เฉพาะโหมด zone) อยู่แถวเดียว —
        cam_row.setSpacing(8)                       # แถวใหม่จะกิน 44px จากรายการชิ้นงาน (Rack) จนแถวหาย
        cam_row.addWidget(self.cam_btn, 1)
        cam_row.addWidget(self.redraw_btn, 1)
        keys.addLayout(cam_row)
        strip = QHBoxLayout()
        strip.setSpacing(8)
        for label, hint, fn in (("บันทึกภาพ", "S", self.shot), ("เต็มจอ", "F", self.fullscreen),
                                ("ออก", "Q", self.close)):
            b = Button(label, hint)
            b.clicked.connect(fn)
            strip.addWidget(b)
        keys.addLayout(strip)

        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(0)
        side.addWidget(self.rack, 1)
        side.addLayout(keys)

        wrap = Rail()
        wrap.setLayout(side)
        self.rack.setMinimumHeight(300)
        rail_scroll = QScrollArea()
        rail_scroll.setWidgetResizable(True)
        rail_scroll.setWidget(wrap)
        rail_scroll.setFixedWidth(366)
        rail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rail_scroll.setStyleSheet(f"QScrollArea {{border:0; background:{PANEL};}}")

        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(0)
        mid.addWidget(self.view, 1)
        mid.addWidget(rail_scroll)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.head)
        root.addLayout(mid, 1)
        root.addWidget(self.foot)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)     # ค่าปริยายปัดเป็น ~15ms บน Windows → 20ms กลายเป็น 31ms
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)
        self._sync_cam_btn()

    # ── ปุ่ม ──
    def set_cup_conf(self, value):
        self.an.set_cup_conf(value)
        self.flash(f"เกณฑ์ conf ของ cup = {value:.2f}")

    def set_hand_confidences(self):
        self.an.set_hand_confidences({k: spin.value() for k, spin in self.hand_conf_inputs.items()})
        self.flash("กำลังปรับค่า MediaPipe")

    def set_pick_seconds(self, value):
        self.an.set_pick_seconds(value)
        self.flash(f"เวลาถือก่อนตรวจแล้ว = {value:.1f} วินาที · เริ่มสะสมใหม่")

    def reset(self):
        self.an.reset()
        self.flash("เริ่มรอบ Zone ใหม่แล้ว" if self.mode == "zone" else "เริ่มรอบตรวจใหม่แล้ว")

    def toggle_settings(self):
        if self.settings.isVisible():
            self.settings.hide()
            return
        self.settings.adjustSize()
        # วางที่มุมขวาบนของพื้นที่ภาพ ติดแผงข้าง — ไม่บังเช็กลิสต์ และใกล้ปุ่มที่เพิ่งกด
        corner = self.view.mapToGlobal(QPoint(self.view.width() - self.settings.width() - 12, 12))
        self.settings.move(corner)
        self.settings.show()

    # ── โหมด zone: สลับโหมด + วาดกรอบ A/B (จำเฉพาะ session นี้ ไม่บันทึกลงไฟล์) ──
    def set_mode(self, mode):
        if mode != self.mode:
            self.toggle_mode()

    def toggle_mode(self):
        if self.mode == "inspect":
            self.mode = "zone"
            self.view.show_zones = True
            self.new_round.set_label("เริ่มรอบ Zone ใหม่")
            self.redraw_btn.show()
            self.an.set_mode("zone")
            if not all(self.view.zones.values()):
                self.start_drawing()
        else:
            self.mode = "inspect"
            self.view.stage, self.view.pts, self.view.show_zones = None, [], False
            self.new_round.set_label("เริ่มรอบตรวจใหม่")
            self.redraw_btn.hide()
            self.an.set_mode("inspect")
        self.head.mode = self.mode
        self.head.update()

    def start_drawing(self):
        self.view.zones = {"A": None, "B": None}
        self.view.stage, self.view.pts = "A", []
        self.an.set_zones(None, None)               # วาดไม่ครบ = แก้วทุกใบเป็น IDLE

    def add_point(self, pt):
        self.view.pts.append(pt)

    def undo_point(self):
        if self.view.pts:
            self.view.pts.pop()

    def close_polygon(self):
        v = self.view
        if not v.stage:
            return
        if len(v.pts) < 3:
            self.flash("กรอบต้องมีอย่างน้อย 3 จุด")
            return
        poly = list(v.pts)
        if v.stage == "B" and polygons_overlap(v.zones["A"], poly):
            v.pts = []
            self.flash("กรอบ B ซ้อนกับ A — วาด B ใหม่")
            return
        v.zones[v.stage], v.pts = poly, []
        if v.stage == "A":
            v.stage = "B"
        else:
            v.stage = None
            self.an.set_zones(v.zones["A"], v.zones["B"])
            self.flash("พร้อม — ย้ายแก้วจาก A ไป B")

    # ── เลือกกล้อง — vision.start() สำรวจ index ที่เปิดได้ไว้ใน cam.available ตอนเปิดแอป ──
    def _cam(self):
        return getattr(self.an, "cam", None)          # ui_preview ไม่มี cam จริง

    def _sync_cam_btn(self):
        cam = self._cam()
        self.cam_btn.set_label(f"กล้อง {cam.index}" if cam else "กล้อง —")

    def pick_camera(self):
        cam = self._cam()
        if not cam:
            return
        m = QMenu(self)                               # popup แยกชั้น — QSS ไม่ไหลลงหน้าต่างหลัก
        m.setStyleSheet(
            f"QMenu{{background:{PANEL};color:{TXT};border:1px solid {LINE};padding:4px}}"
            f"QMenu::item{{padding:8px 26px 8px 16px;border-radius:{R}px}}"
            f"QMenu::item:selected{{background:{HOVER}}}")
        for i in cam.available:
            act = m.addAction(f"กล้อง {i}" + ("   ●" if i == cam.index else ""))
            act.triggered.connect(lambda _=False, n=i: self.set_camera(n))
        m.exec(self.cam_btn.mapToGlobal(QPoint(0, 0)))   # Qt เลื่อนขึ้นเองถ้าชนขอบล่างจอ

    def set_camera(self, i):
        cam = self._cam()
        if cam and i != cam.index:
            cam.switch(i)
            self.flash(f"สลับไปกล้อง {i}")
        self._sync_cam_btn()

    def cycle_camera(self):
        cam = self._cam()
        if cam and len(cam.available) > 1:
            j = cam.available.index(cam.index) if cam.index in cam.available else -1
            self.set_camera(cam.available[(j + 1) % len(cam.available)])

    def shot(self):
        if self.view.raw is not None:
            fn = str(shots_dir() / f"shot_{int(time.time())}.png")
            cv2.imwrite(fn, self.view.raw)
            print("เซฟภาพ:", fn)
            self.flash(f"บันทึกภาพ {fn}")

    def fullscreen(self):
        self.full = not self.full
        self.showFullScreen() if self.full else self.showNormal()

    def flash(self, s, seconds=2.5):
        self.msg = (s, time.time() + seconds)

    def keyPressEvent(self, e):
        k = e.key()
        # กดค้างทำให้ Qt ส่ง keyPress ซ้ำ ๆ — ปุ่มสลับ (C กล้อง · Z โหมด · T ตั้งค่า · F เต็มจอ · D debug) จะสลับไปมาจนไม่รู้ว่าจบที่ไหน
        # (ยกเว้น Backspace: กดค้างเพื่อย้อนจุดตอนวาดกรอบเป็นการใช้งานปกติ)
        if e.isAutoRepeat() and k != Qt.Key.Key_Backspace:
            return
        if self.view.stage and k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.close_polygon()
        elif self.view.stage and k == Qt.Key.Key_Backspace:
            self.undo_point()
        elif self.view.stage and k == Qt.Key.Key_Escape:     # กำลังวาด: ล้างจุด · ไม่มีจุดแล้ว = ออกจากโหมด zone
            if self.view.pts:
                self.view.pts = []
            else:
                self.toggle_mode()
        elif k == Qt.Key.Key_T:
            self.toggle_settings()
        elif k == Qt.Key.Key_Z:
            self.toggle_mode()
        elif k == Qt.Key.Key_X and self.mode == "zone":
            self.start_drawing()
        elif k == Qt.Key.Key_Escape and self.mode == "zone":   # วาดเสร็จแล้ว: Esc แค่ออกจากโหมด ไม่ปิดแอปกลางเดโม (Q ยังปิด)
            self.toggle_mode()
        elif k in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        elif k == Qt.Key.Key_R:
            self.reset()
        elif k == Qt.Key.Key_S:
            self.shot()
        elif k == Qt.Key.Key_F:
            self.fullscreen()
        elif k == Qt.Key.Key_C:
            self.cycle_camera()
        elif k == Qt.Key.Key_D:
            self.an.debug = not self.an.debug
            self.flash(f"debug {'เปิด' if self.an.debug else 'ปิด'}")

    # ── วนแสดงผล ──
    def tick(self):
        seq, frame, st = self.an.latest()
        if not self.hand_conf_timer.isActive():
            for key, value in st.get("hand_conf", {}).items():
                spin = self.hand_conf_inputs[key]
                if not spin.hasFocus():
                    spin.blockSignals(True)
                    spin.setValue(value)
                    spin.blockSignals(False)
        self.view.st = st
        self.view.beat = time.time()
        if frame is None:                # ยังไม่มีภาพ: กล้องหลุด/กำลังสลับ หรือรอเฟรมแรก — วาดจอรอ ไม่ใช่ภาพ
            self.seq = -1
            self.view.wait("กำลังเชื่อมต่อกล้อง…" if not st.get("camera", True) else "กำลังเริ่มวิเคราะห์ภาพ…")
        elif seq != self.seq:            # เฟรมเดิมไม่ต้องแปลงซ้ำ (แปลงภาพแพงกว่าที่คิด)
            self.seq = seq
            self.view.set_frame(frame)
        else:
            self.view.pulse()            # จุดสถานะยังต้องเต้นแม้เฟรมไม่มา
        self.rack.set_rows(st.get("rows", []), self.an.events, st.get("mode") == "zone")
        self.head.set(st)
        self.foot.set(st, self.msg, self.an.events)


class Rail(QWidget):
    """กรอบของแผงข้าง — พื้นแผง + เส้นแบ่ง 1px ทางซ้าย"""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(348)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(pen(LINE))
        p.drawLine(0, 0, 0, self.height())


TABS = (("inspect", "เช็กลิสต์ตรวจ"), ("zone", "ย้ายแก้ว A → B"))


class Header(QWidget):
    """หัวแถบ: โลโก้ · ชื่อ · แท็บโหมด (ขีดใต้แท็บที่เลือก) · เวลาปัจจุบัน + ความยาวรอบตรวจ"""

    def __init__(self):
        super().__init__()
        self.st, self.clock = {}, ""
        self.mode, self.hover = "inspect", None
        self.on_mode = lambda m: None            # Station ผูกให้
        self.setFixedHeight(52)
        self.setMouseTracking(True)

    def set(self, st):
        now = time.strftime("%H:%M:%S")
        if now != self.clock:            # หัวแถบเปลี่ยนแค่วินาทีละครั้ง ไม่ต้องวาด 50 ครั้ง/วิ
            self.st, self.clock = st, now
            self.update()

    def _tabs(self):
        """[(โหมด, ป้าย, สี่เหลี่ยม)] — วาดกับรับเมาส์ใช้ชุดเดียวกัน"""
        f, x, out = face(13, QFont.Weight.DemiBold), 166, []
        for mode, label in TABS:
            w = QFontMetrics(f).horizontalAdvance(label) + 36
            out.append((mode, label, QRectF(x, 0, w, self.height())))
            x += w
        return out

    def _tab_at(self, pos):
        return next((m for m, _, r in self._tabs() if r.contains(pos)), None)

    def mouseMoveEvent(self, e):
        hit = self._tab_at(e.position())
        self.setCursor(Qt.CursorShape.PointingHandCursor if hit else Qt.CursorShape.ArrowCursor)
        if hit != self.hover:
            self.hover = hit
            self.update()

    def leaveEvent(self, e):
        self.hover = None
        self.update()

    def mousePressEvent(self, e):
        hit = self._tab_at(e.position())
        if hit and e.button() == Qt.MouseButton.LeftButton:
            self.on_mode(hit)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(pen(LINE))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

        cup_mark(p, 32, 28, 28, ACCENT_HI)             # โลโก้: แก้วมัค
        text(p, 52, 15, "Cuptick", face(16, QFont.Weight.Bold), TXT)
        p.setPen(pen(LINE))
        p.drawLine(148, 14, 148, 38)

        for mode, label, r in self._tabs():
            on = mode == self.mode
            if on:
                p.fillRect(QRectF(r.left(), r.bottom() - 2, r.width(), 2), QColor(ACCENT_HI))
            elif mode == self.hover:
                p.fillRect(QRectF(r.left(), 8, r.width(), r.height() - 17), QColor(HOVER))
            text(p, r.left(), 17, label, face(13, QFont.Weight.DemiBold), TXT if on else DIM,
                 Qt.AlignmentFlag.AlignCenter, r.width())

        r = int(self.st.get("round_s", 0))
        text(p, 0, 9, self.clock, face(16, QFont.Weight.DemiBold, mono=True),
             TXT, Qt.AlignmentFlag.AlignRight, self.width() - 22)
        text(p, 0, 29, f"รอบนี้ {r // 60:02d}:{r % 60:02d}", face(12, mono=True), DIM,
             Qt.AlignmentFlag.AlignRight, self.width() - 22)


class Footer(QWidget):
    """แถบสถานะ: ไฟสถานะ + ประโยคบอกว่ากำลังเกิดอะไร · ขวาสุดคือ ticker เหตุการณ์ล่าสุด"""

    def __init__(self):
        super().__init__()
        self.line = ("", DIM, "")
        self.setFixedHeight(40)

    def set(self, st, msg, events):
        note, expire = msg
        if note and time.time() < expire:
            state, col = note, ACCENT_HI
        elif not st.get("camera", True):
            state, col = "กล้องหลุด — กำลังเชื่อมต่อใหม่", AMBER
        elif st.get("mode") == "zone":
            done, total = st.get("zone_counts", (0, 0))
            if not st.get("zone_ready"):
                state, col = "วาดกรอบ A และ B บนภาพให้ครบก่อน", AMBER
            elif total and done == total:
                state, col = f"ถึง B ครบทุกใบแล้ว · {done}/{total}", SIGNAL
            elif total:
                state, col = f"กำลังติดตาม · ถึง B แล้ว {done}/{total}", AMBER
            else:
                state, col = "วางแก้วในกรอบ A เพื่อเริ่ม", DIM
        elif st.get("holding"):
            state, col = f"กำลังหยิบตรวจ · {st['held_s']:0.1f} วินาที", SIGNAL
        elif st.get("rows"):
            state, col = "รอมือมาหยิบชิ้นงาน", AMBER
        else:
            state, col = "กำลังมองหาชิ้นงาน", DIM
        last = f"{events[-1][0]}   {events[-1][1]}" if events else ""
        if (state, col, last) != self.line:
            self.line = (state, col, last)
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(pen(LINE))
        p.drawLine(0, 0, self.width(), 0)
        state, col, last = self.line
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(col))
        p.drawEllipse(QPointF(24, 20), 4, 4)
        text(p, 38, 11, state, face(13, QFont.Weight.DemiBold), col)
        if last:
            text(p, 0, 12, last, face(12, mono=True), FAINT,
                 Qt.AlignmentFlag.AlignRight, self.width() - 22)


# ─────────────────────────── main ───────────────────────────
FROZEN = getattr(sys, "frozen", False)      # รันจากตัวติดตั้ง ไม่ใช่ python app/app.py


def shots_dir():
    """รูปที่กด S บันทึก — ตัวติดตั้งไม่มีโฟลเดอร์ทำงานที่เขียนได้ ลงโฟลเดอร์รูปภาพของผู้ใช้"""
    pics = Path.home() / "Pictures"
    return pics if FROZEN and pics.is_dir() else Path.cwd()


def to_log_file():
    """แอปที่ติดตั้งไม่มีหน้าต่าง console (และ Windows ไม่มี stdout เลย) — ส่งข้อความทั้งหมดลงไฟล์
    เก็บแค่ครั้งล่าสุด (app.log) ไว้ให้ส่งมาดูเวลามีปัญหา · คืน path ของไฟล์"""
    from userdir import migrate_legacy_user_dir, user_dir          # ไม่ import vision ที่นี่ — ต้องตั้งไฟล์ log ก่อนที่ torch/ultralytics จะพิมพ์อะไร
    migrate_legacy_user_dir()
    d = user_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "app.log"
    sys.stdout = sys.stderr = open(path, "w", encoding="utf-8", buffering=1)
    return path


def fatal(msg):
    """ข้อผิดพลาดที่ทำให้เปิดแอปไม่ได้ — แอปที่ติดตั้งไม่มี console ต้องขึ้นหน้าต่างให้ผู้ใช้เห็น"""
    print(msg)
    box = QMessageBox(QMessageBox.Icon.Critical, "Cuptick", "เปิดแอปไม่สำเร็จ")
    box.setInformativeText(str(msg).strip())
    box.exec()


class Boot(QWidget):
    """หน้าต่างรอตอนเปิดแอป (โหลดไลบรารี โมเดล กล้อง ใช้เวลาหลายวินาที) — ตัวโหลดทำงานในเธรดอื่น
    เธรดหลักจึงว่างพอวาดข้อความ + เวลาที่ผ่านไป · แถบวิ่งเป็น QProgressBar แบบไม่รู้ความคืบหน้า: Qt แอนิเมตมันใน C++
    ไม่ต้องรอ Python — ตอน import torch ถือ GIL ค้างเป็นช่วง ๆ แถบก็ยังวิ่งต่อ ไม่ดูเหมือนแอปค้าง"""

    def __init__(self):
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.msg, self.t0 = "กำลังเริ่มต้น", time.time()
        self.setFixedSize(460, 280)
        self.bar = QProgressBar(self)
        self.bar.setRange(0, 0)
        self.bar.setTextVisible(False)
        self.bar.setGeometry(120, 178, 220, 4)
        self.bar.setStyleSheet(f"QProgressBar {{background:{LINE}; border:0; border-radius:2px;}}"
                               f"QProgressBar::chunk {{background:{ACCENT_HI}; border-radius:2px;}}")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(250)                  # แค่ให้ตัวเลขวินาทีเดิน

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(pen(LINE))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))
        w = self.width()
        cup_mark(p, w / 2, 62, 44, ACCENT_HI)
        text(p, 0, 112, "Cuptick", face(22, QFont.Weight.Bold), TXT, Qt.AlignmentFlag.AlignCenter, w)
        text(p, 0, 198, self.msg, face(13, QFont.Weight.DemiBold), DIM, Qt.AlignmentFlag.AlignCenter, w)
        s = int(time.time() - self.t0)
        text(p, 0, 224, f"{s // 60:02d}:{s % 60:02d}", face(12, mono=True), FAINT, Qt.AlignmentFlag.AlignCenter, w)


class Loader(QObject):
    """โหลดของหนักในเธรดแยก แล้วส่งผลกลับเธรดหลักผ่าน signal — สร้างวิดเจ็ตต้องทำที่เธรดหลักเท่านั้น"""
    step = Signal(str)
    ready = Signal(object, object, object)       # cfg, analyzer, cam
    failed = Signal(object)                      # ข้อความที่จะโชว์ผู้ใช้

    def run(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self.step.emit("กำลังโหลดไลบรารี AI")
            import vision                          # torch/ultralytics/mediapipe — เป็นส่วนที่ช้าที่สุดของการเปิดแอป

            cfg = vision.load_config()
            analyzer, cam = vision.start(cfg, self.step.emit)
            self.ready.emit(cfg, analyzer, cam)
        except SystemExit as e:                    # vision.start() ออกด้วยข้อความไทยที่บอกว่าต้องแก้อะไร
            self.failed.emit(e.code if isinstance(e.code, str) else "ออกจากโปรแกรม")
        except Exception:
            import traceback
            self.failed.emit(traceback.format_exc())


def selftest_hands(out):
    """โหมดลูกของ --selftest: โหลดตัวตรวจมือ MediaPipe แล้วตรวจเฟรมว่างหนึ่งใบ — แยกโปรเซสเพราะ MediaPipe บน macOS
    สั่ง abort ทั้งโปรเซสเมื่อสร้างตัวตรวจมือไม่ได้ (เคยเกิดบน macOS) ซึ่ง try/except จับไม่ได้"""
    import vision
    import mediapipe as mp
    import numpy as np
    hands = vision.load_hand_landmarker(vision.load_config())
    hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8)), 1)
    Path(out).write_text("HANDS OK\n", encoding="utf-8")
    return 0


def selftest_hands_in_child(out):
    import subprocess
    marker = Path(str(out) + ".hands")
    marker.unlink(missing_ok=True)
    cmd = [sys.executable] + ([] if FROZEN else [str(Path(__file__).resolve())]) + ["--selftest-hands", str(marker)]
    r = subprocess.run(cmd, timeout=300)
    if r.returncode == 0 and marker.exists():
        return "HANDS OK"
    raise RuntimeError(f"hand tracker self-test failed (exit {r.returncode})")


def selftest(out):
    """`Cuptick --selftest <ไฟล์>` — โหลดโมเดล + MediaPipe + วาดเฟรมหนึ่งใบ + สร้างวิดเจ็ต Qt โดยไม่ต้องมีกล้อง/จอ
    ตัวสร้างติดตั้ง (CI) ใช้เช็กว่าที่แพ็กมาครบจริง · เขียนผลลงไฟล์เพราะแอปแบบไม่มี console พิมพ์ออกหน้าจอไม่ได้"""
    import traceback
    lines = []
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import vision                      # ต้องมาก่อน mediapipe: vision ตั้ง MEDIAPIPE_DISABLE_GPU ก่อน import (เหมือนตอนรันจริง)
        import mediapipe as mp
        import numpy as np

        app = QApplication(sys.argv)
        cfg = vision.load_config()
        cfg["model_path"] = vision.resolve_model_path(cfg)
        model = vision.load_model(cfg)
        frame = np.zeros((480, 640, 3), np.uint8)
        dev = vision.pick_device(cfg)
        model.predict(frame, imgsz=cfg.get("imgsz", 480), device=dev, verbose=False, **vision.PREDICT_ARGS)
        insp = vision.Inspection(8, 5, 4.0)
        shown = vision.draw(frame.copy(), [(1, [100, 100, 300, 400], False)], [], set(), insp, False,
                            confidences={1: 0.9})
        view = Viewport()
        view.resize(640, 360)
        view.set_frame(shown)
        lines += [f"SELFTEST OK model={cfg['model_path']} device={dev}",
                  f"python={sys.version.split()[0]} frozen={FROZEN} resources={vision.HERE}",
                  selftest_hands_in_child(out)]
        code = 0
    except BaseException:
        lines += ["SELFTEST FAILED", traceback.format_exc()]
        code = 1
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return code


def main():
    multiprocessing.freeze_support()           # PyInstaller + torch/ultralytics อาจเปิดโปรเซสลูก
    if "--selftest-hands" in sys.argv:
        i = sys.argv.index("--selftest-hands")
        raise SystemExit(selftest_hands(sys.argv[i + 1]))
    if "--selftest" in sys.argv:
        i = sys.argv.index("--selftest")
        raise SystemExit(selftest(sys.argv[i + 1] if len(sys.argv) > i + 1 else "selftest.txt"))
    if FROZEN:
        to_log_file()

    app = QApplication(sys.argv)
    app.setApplicationName("Cuptick")
    boot = Boot()
    boot.show()
    loaded = {}                                # เก็บไว้ปิดตอนออก — analyzer/cam สร้างในเธรดตัวโหลด

    def on_ready(cfg, analyzer, cam):
        loaded.update(analyzer=analyzer, cam=cam)
        try:
            loaded["win"] = win = Station(analyzer, cfg)
            win.show()
        except Exception:                      # ไม่ดักไว้ = หน้าต่างรอค้างอยู่ตลอดไป ไม่มีใครรู้ว่าเกิดอะไร
            import traceback
            on_failed(traceback.format_exc())
            return
        boot.close()

    def on_failed(msg):
        boot.hide()                            # hide ไม่ใช่ close: close = หน้าต่างสุดท้ายปิด → Qt ออกจากแอปก่อนกล่องข้อความขึ้น
        fatal(msg)
        app.exit(1)

    loader = Loader()
    loader.step.connect(lambda s: setattr(boot, "msg", s))
    loader.ready.connect(on_ready)
    loader.failed.connect(on_failed)
    QTimer.singleShot(0, loader.run)           # เริ่มหลังวนอีเวนต์รอบแรก — หน้าต่างรอขึ้นจอก่อนเสมอ
    code = app.exec()

    if loaded:
        loaded["analyzer"].release()
        loaded["cam"].release()
    sys.exit(code)


if __name__ == "__main__":
    main()
