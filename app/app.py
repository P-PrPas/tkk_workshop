"""INSPECTION STATION — หน้าจอสถานีตรวจชิ้นงาน (Qt / PySide6)

ไฟล์นี้ *แสดงผลอย่างเดียว* ทุก 20ms หยิบ (เลขเฟรม, เฟรม, สถานะ) จาก `vision.Analyzer`
มาวาด — ไม่มีการตัดสินใจเรื่อง CV อยู่ในนี้เลย ตรรกะทั้งหมดอยู่ที่ `app/vision.py`

ภาษาการออกแบบ: **แผงเครื่องมือ ไม่ใช่หน้าเว็บ**
  · ตัวเลขทุกตัวเป็นฟอนต์ monospace ความกว้างเท่ากัน — อ่านข้ามห้องแล้วไม่เต้น
  · เส้นทุกเส้นหนา 1px และมีขีดสเกลเหมือนขอบเครื่องวัด
  · มุมกรอบภาพเป็นวงเล็บมุม — ภาษาเดียวกับกรอบแก้วที่ vision.draw() วาด จอเดียวกันจริง ๆ
  · สีเดียวที่เป็นสัญญาณคือเขียว (ตรวจแล้ว) กับอำพัน (ยังไม่ตรวจ) ที่เหลือเป็นเทา

รัน:  python app/app.py        ค่าที่ต้องจูนหน้างานอยู่ใน app/config.yaml ทั้งหมด
"""
import sys
import time

import cv2

from zone import polygons_overlap
from overlay import splash          # vision (torch/ultralytics) นำเข้าตอน main() เท่านั้น —
                                    # เปิดดูหน้าตา/รัน tools/ui_preview.py บนเครื่องที่ไม่มีสแต็ก CV ได้

try:
    from PySide6.QtCore import Qt, QPoint, QRectF, QTimer, QVariantAnimation, QPointF
    from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QPainter,
                               QPalette, QPen, QPixmap, QPolygonF)
    from PySide6.QtWidgets import (QAbstractButton, QApplication, QDoubleSpinBox, QLabel, QHBoxLayout,
                                   QMenu, QScrollArea, QVBoxLayout, QWidget)
except ImportError:
    raise SystemExit("\nไม่มี PySide6 — ติดตั้งก่อน:  pip install -r app/requirements.txt\n")


# ─────────────────────────── ธีม ───────────────────────────
# ห้องเดโมปิดไฟ ฉายโปรเจกเตอร์ — พื้นยิ่งเข้ม ตัวอักษรยิ่งสว่าง อ่านจากท้ายห้องได้
# สองสีสัญญาณ (SIGNAL/AMBER) ล็อกไว้ ต้องตรงกับ vision.py (ที่นั่นเป็น BGR) — แก้ต้องแก้คู่กัน
GROUND = "#080B10"   # พื้นหน้าต่าง
PANEL  = "#0E121A"   # แผงข้าง / แถบสถานะ
CARD   = "#1B2330"   # แถวเช็กลิสต์ (สว่างกว่าแผงชัด ๆ ให้แถวลอยออกมา)
LINE   = "#303B4C"   # เส้นแบ่ง 1px / ขอบ
TXT    = "#F4F7FC"   # อักษรหลัก
HEAD   = "#C7D0DE"   # ป้ายหัวข้อของแต่ละแผง — สว่างพออ่านข้ามห้อง
DIM    = "#A7B3C6"   # อักษรรอง
FAINT  = "#8C99AE"   # ป้ายกำกับจาง (timestamp, hint) — ยังผ่าน 4.5:1 บนพื้น PANEL
SIGNAL = "#3DD68C"   # ตรวจแล้ว / กำลังหยิบ            (ล็อก — ตรงกับ overlay.OK)
AMBER  = "#F2B34B"   # ยังไม่ตรวจ / กล้องหลุด          (ล็อก — ตรงกับ overlay.WARN)
VIOLET = "#A78BFA"   # โหมด zone: อยู่ที่ A / polygon A        (ล็อก — ตรงกับ overlay.READY)
BLACK  = "#04060A"   # พื้นหลังกรอบภาพ

# โหมด zone: สีและชื่อสถานะของแก้ว (เทา → ม่วง → อำพัน → เขียว) — ตรงกับ overlay.ZONE_COLOR
ZONE_HEX = {"idle": "#7A8496", "at_a": VIOLET, "transit": AMBER, "done": SIGNAL}
ZONE_NAME = {"idle": "IDLE", "at_a": "AT A", "transit": "IN TRANSIT", "done": "DONE"}

# Inter ไม่มี glyph ไทย — ไล่ family ให้ Qt เลือกรายตัวอักษร (ไทยตกไปที่ Segoe UI / Noto)
UI_FAMILIES = ["Inter", "Segoe UI", "Helvetica Neue", "Noto Sans Thai", "Tahoma", "sans-serif"]
MONO_FAMILIES = ["JetBrains Mono", "Cascadia Mono", "SF Mono", "Consolas",
                 "DejaVu Sans Mono", "monospace"]


def face(size, weight=QFont.Weight.Normal, mono=False, track=0.0, caps=False):
    """ฟอนต์หนึ่งตัว — mono สำหรับ *ตัวเลขทุกตัว* · track = ระยะห่างตัวอักษร (ใช้กับป้ายตัวพิมพ์ใหญ่)"""
    f = QFont()
    f.setFamilies(MONO_FAMILIES if mono else UI_FAMILIES)
    f.setPixelSize(size)
    f.setWeight(weight)
    if track:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, track)
    if caps:
        f.setCapitalization(QFont.Capitalization.AllUppercase)
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
    """ป้ายหัวข้อแผง — ขีดตั้งสั้นนำหน้า (ภาษาเดียวกับขีดสถานะในแถว) + ตัวอักษรสว่าง ใหญ่กว่าเดิม"""
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(HEAD))
    p.drawRoundedRect(QRectF(x, y + 3, 3, 12), 1.5, 1.5)
    text(p, x + 12, y, s, face(12, QFont.Weight.Bold, track=1.0), HEAD)


# ─────────────────────────── ปุ่ม ───────────────────────────
class Button(QAbstractButton):
    """ปุ่มวาดเอง — มุมมน antialias + ไล่สีตอน hover ใน 160ms (QSS ของ Qt ทำ transition ไม่ได้)
    เป็น QAbstractButton จึงได้ focus ring, กด Space/Enter, และ signal clicked มาฟรี"""

    def __init__(self, label, hint="", primary=False):
        super().__init__()
        self.label, self.hint, self.primary = label, hint, primary
        self._glow = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(44 if primary else 36)
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
        if self.primary:
            base = QColor(SIGNAL).darker(int(100 + 14 * (1 - g)))
            p.setBrush(base)
            p.setPen(Qt.PenStyle.NoPen)
            ink, sub = QColor("#06110B"), QColor(0, 0, 0, 150)
        else:
            p.setBrush(QColor(CARD).lighter(int(100 + 16 * g)))
            p.setPen(pen(QColor(LINE).lighter(int(100 + 40 * g))))
            ink, sub = QColor(TXT), QColor(DIM)
        p.drawRoundedRect(r, 9, 9)
        if self.hasFocus():                      # a11y — เห็นชัดว่าโฟกัสอยู่ปุ่มไหน
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(pen(SIGNAL, 1.6))
            p.drawRoundedRect(r.adjusted(-1.5, -1.5, 1.5, 1.5), 10.5, 10.5)

        wide = self.primary or self.width() > 240   # ปุ่มเต็มแถว (primary + "เลือกกล้อง") ใหญ่กว่าปุ่มในแถบสามช่อง
        f = face(14 if wide else 12, QFont.Weight.Bold if self.primary else QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(pen(ink))
        pad = 14 if self.primary else 11
        p.drawText(r.adjusted(pad, 0, -pad, 0),
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
        self.box = None                 # สี่เหลี่ยมของภาพบนวิดเจ็ต — ไว้แปลงตำแหน่งเมาส์ ↔ สัดส่วนเฟรม
        self.zones = {"A": None, "B": None}   # polygon สัดส่วน 0-1 (Station เป็นเจ้าของ ที่นี่แค่วาด)
        self.stage = None               # "A"/"B" = กำลังวาดอันนั้น · None = ไม่ได้วาด
        self.pts = []                   # จุดของ polygon ที่กำลังวาด
        self.hover = None
        self.on_point = self.on_close = self.on_undo = lambda *a: None   # Station ผูกให้
        self.setMouseTracking(True)
        self.setMinimumSize(480, 270)

    def set_frame(self, frame):
        self.raw = frame
        h, w = frame.shape[:2]
        s = min(self.width() / w, self.height() / h)
        small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                           interpolation=cv2.INTER_AREA)
        img = QImage(small.data, small.shape[1], small.shape[0], small.strides[0],
                     QImage.Format.Format_BGR888)
        self.pix = QPixmap.fromImage(img)            # fromImage คัดลอกพิกเซลให้แล้ว บัฟเฟอร์ numpy ทิ้งได้
        self.update()

    def resizeEvent(self, e):
        if self.raw is not None:
            self.set_frame(self.raw)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(BLACK))
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
            cx = sum(q.x() for q in pts) / len(pts)
            cy = sum(q.y() for q in pts) / len(pts)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(6, 9, 14, 215))
            p.drawRoundedRect(QRectF(cx - 15, cy - 15, 30, 30), 7, 7)
            text(p, cx - 15, cy - 9, name, face(17, QFont.Weight.Bold, mono=True), col,
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
        p.drawRoundedRect(QRectF(x, box.top() + 14, w, 34), 8, 8)
        p.setPen(pen(col, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(x + 0.5, box.top() + 14.5, w - 1, 33), 8, 8)
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
        text(p, x + 18, y - 2, "LIVE" if live else "RECONNECTING",
             face(12, QFont.Weight.Bold, track=1.6, caps=True), col.name())

    def _chip(self, p, x, y, s):
        f = face(12, QFont.Weight.DemiBold, mono=True)
        w = QFontMetrics(f).horizontalAdvance(s) + 22
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(6, 9, 14, 205))
        p.drawRoundedRect(QRectF(x, y, w, 28), 6, 6)
        p.setPen(pen(QColor(255, 255, 255, 40)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(x + 0.5, y + 0.5, w - 1, 27), 6, 6)
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
                p.drawRoundedRect(QRectF(x, y, seg, 7), 2, 2)
                if not ok and prog > 0:                    # ช่องกำลังไต่
                    p.setBrush(QColor(AMBER))
                    p.drawRoundedRect(QRectF(x, y, max(3.0, seg * prog), 7), 2, 2)
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 28))
            p.drawRoundedRect(QRectF(pad, y, inner, 7), 2, 2)

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
                 face(11), QColor(FAINT).darker(125).name())
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
                text(p, pad, ly, when, face(11, mono=True), QColor(FAINT).darker(130).name())
                text(p, pad + 70, ly - 1, what, face(11), DIM if i else TXT)
            else:                       # บรรทัดว่างของสมุดบันทึก — ที่ว่างตรงนี้ตั้งใจเว้น ไม่ใช่ layout พัง
                p.setPen(pen(QColor(255, 255, 255, 12)))
                p.drawLine(pad, ly + 9, W - pad, ly + 9)

    def _zone_row(self, p, x, y, w, tid, state):
        col = QColor(ZONE_HEX[state])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(CARD))
        p.drawRoundedRect(QRectF(x, y, w, ROW_H), 8, 8)
        p.setBrush(col)                             # ขีดสถานะซ้าย — สีเดียวกับกรอบแก้วบนภาพ
        p.drawRoundedRect(QRectF(x + 5, y + 11, 3, ROW_H - 22), 1.5, 1.5)
        text(p, x + 18, y + 12, f"#{tid:02d}", face(18, QFont.Weight.Bold, mono=True),
             col.name() if state != "idle" else TXT)
        text(p, x + 76, y + 15, ZONE_NAME[state], face(13, QFont.Weight.DemiBold, mono=True), col.name())

    def _row(self, p, x, y, w, tid, ok, prog, held_s):
        col = QColor(SIGNAL if ok else AMBER)
        r = QRectF(x, y, w, ROW_H)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(CARD))
        p.drawRoundedRect(r, 8, 8)

        p.setBrush(col)                             # ขีดสถานะซ้าย — เขียว=ตรวจแล้ว อำพัน=ยังไม่ตรวจ
        p.drawRoundedRect(QRectF(x + 5, y + 11, 3, ROW_H - 22), 1.5, 1.5)

        text(p, x + 18, y + 12, f"#{tid:02d}", face(18, QFont.Weight.Bold, mono=True),
             col.name() if ok else TXT)
        text(p, x + 76, y + 15, "ตรวจแล้ว" if ok else "ยังไม่ตรวจ",
             face(13, QFont.Weight.DemiBold), col.name())
        if ok:
            text(p, x, y + 17, f"{held_s:5.0f}s", face(11, mono=True), FAINT,
                 Qt.AlignmentFlag.AlignRight, w - 14)
        elif prog > 0:                              # ตัวนับแบบรั่วกำลังสะสม
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(AMBER))
            p.drawRoundedRect(QRectF(x + 3, y + ROW_H - 3, (w - 6) * prog, 2), 1, 1)


# ─────────────────────────── หน้าต่างหลัก ───────────────────────────
class Station(QWidget):
    """หน้าต่างเดียว: หัวแถบ · ภาพ+แผงข้าง · แถบสถานะ
    ไม่คิดอะไรเอง — ทุก 20ms ดึงของล่าสุดจาก Analyzer มาวาด"""

    def __init__(self, analyzer, cfg):
        super().__init__()
        self.an, self.cfg = analyzer, cfg
        self.seq, self.full = -1, False
        self.msg = ("", 0.0)
        self.setWindowTitle("Inspection Station — เช็กลิสต์ผู้ตรวจ")
        vw = int(cfg.get("window_width", 1280))
        self.resize(vw + 360, vw * 9 // 16 + 92)
        self.setMinimumSize(980, 620)
        ground(self, GROUND)

        self.view = Viewport()
        self.rack = Rack()
        self.head = Header()
        self.foot = Footer()

        new_round = Button("เริ่มรอบตรวจใหม่", "R", primary=True)
        new_round.clicked.connect(self.reset)
        self.cam_btn = Button("เลือกกล้อง", "C")
        self.cam_btn.clicked.connect(self.pick_camera)
        self.mode = "inspect"                       # inspect = เช็กลิสต์ตรวจแก้ว · zone = ย้ายแก้ว A → B
        self.mode_btn = Button("โหมด: ตรวจ", "Z")
        self.mode_btn.clicked.connect(self.toggle_mode)
        self.redraw_btn = Button("วาดใหม่", "X")
        self.redraw_btn.clicked.connect(self.start_drawing)
        self.redraw_btn.hide()
        self.view.on_point, self.view.on_close, self.view.on_undo = (
            self.add_point, self.close_polygon, self.undo_point)
        keys = QVBoxLayout()
        keys.setContentsMargins(22, 0, 22, 20)
        keys.setSpacing(8)
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
        self.conf_input.setMinimumSize(100, 36)
        self.conf_input.setFont(face(14, mono=True))
        self.conf_input.setAccessibleName("เกณฑ์ confidence ของ class cup")
        self.conf_input.setToolTip("เกณฑ์เริ่ม ID แก้วใหม่ · ID เดิมใช้คะแนนต่ำช่วยติดตามได้\nลดค่าเพื่อตรวจพบง่ายขึ้น · เพิ่มค่าเพื่อลดการตรวจผิด\nมีผลระหว่างรัน ไม่บันทึกลง config.yaml")
        self.conf_input.setStyleSheet(
            f"QDoubleSpinBox {{background:{CARD}; color:{TXT}; border:1px solid {LINE};"
            "border-radius:6px; padding:4px 8px;}"
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {width:24px;}")
        self.conf_input.valueChanged.connect(self.set_cup_conf)
        conf_label.setBuddy(self.conf_input)
        conf_row.addWidget(conf_label, 1)
        conf_row.addWidget(self.conf_input)
        keys.addLayout(conf_row)
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
        self.duration_input.setMinimumSize(100, 36)
        self.duration_input.setFont(face(14, mono=True))
        self.duration_input.setStyleSheet(self.conf_input.styleSheet())
        self.duration_input.setAccessibleName("เวลาถือชิ้นงานก่อนนับว่าตรวจแล้ว หน่วยวินาที")
        self.duration_input.setToolTip("เวลาถือสะสมก่อนนับว่าตรวจแล้ว (วินาที)\nเปลี่ยนค่าแล้วเริ่มสะสมใหม่ ชิ้นที่ตรวจแล้วคงเดิม\nใช้ระหว่างรัน ไม่บันทึกลง config.yaml")
        self.duration_input.valueChanged.connect(self.set_pick_seconds)
        duration_label.setBuddy(self.duration_input)
        duration_row.addWidget(duration_label, 1)
        duration_row.addWidget(self.duration_input)
        keys.addLayout(duration_row)
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
            spin.setMinimumSize(100, 30)
            spin.setFont(face(14, mono=True))
            spin.setStyleSheet(self.conf_input.styleSheet())
            spin.setAccessibleName(key)
            spin.setToolTip(f"MediaPipe · {key}\nปรับระหว่างรัน ไม่บันทึกลง config.yaml")
            spin.valueChanged.connect(lambda _: self.hand_conf_timer.start())
            title.setBuddy(spin)
            row.addWidget(title, 1)
            row.addWidget(spin)
            keys.addLayout(row)
            self.hand_conf_inputs[key] = spin
        keys.addWidget(new_round)
        cam_row = QHBoxLayout()                     # กล้อง · โหมด (· วาดใหม่ เฉพาะโหมด zone) อยู่แถวเดียว —
        cam_row.setSpacing(8)                       # แถวใหม่จะกิน 44px จากรายการชิ้นงาน (Rack) จนแถวหาย
        cam_row.addWidget(self.cam_btn, 5)
        cam_row.addWidget(self.mode_btn, 7)
        cam_row.addWidget(self.redraw_btn, 6)
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
        self.timer.timeout.connect(self.tick)
        self.timer.start(20)
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
        self.flash("เริ่มรอบตรวจใหม่แล้ว")

    # ── โหมด zone: สลับโหมด + วาดกรอบ A/B (จำเฉพาะ session นี้ ไม่บันทึกลงไฟล์) ──
    def toggle_mode(self):
        if self.mode == "inspect":
            self.mode = "zone"
            self.mode_btn.set_label("โหมด: Zone")
            self.redraw_btn.show()
            self.an.set_mode("zone")
            if not all(self.view.zones.values()):
                self.start_drawing()
        else:
            self.mode = "inspect"
            self.view.stage, self.view.pts = None, []
            self.mode_btn.set_label("โหมด: ตรวจ")
            self.redraw_btn.hide()
            self.an.set_mode("inspect")

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
            f"QMenu{{background:{PANEL};color:{TXT};border:1px solid {LINE};padding:6px}}"
            f"QMenu::item{{padding:8px 26px 8px 16px;border-radius:6px}}"
            f"QMenu::item:selected{{background:{CARD}}}")
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
            fn = f"shot_{int(time.time())}.png"
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
        if self.view.stage and k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.close_polygon()
        elif self.view.stage and k == Qt.Key.Key_Backspace:
            self.undo_point()
        elif self.view.stage and k == Qt.Key.Key_Escape:     # กำลังวาด: ล้างจุด · ไม่มีจุดแล้ว = ออกจากโหมด zone
            if self.view.pts:
                self.view.pts = []
            else:
                self.toggle_mode()
        elif k == Qt.Key.Key_Z:
            self.toggle_mode()
        elif k == Qt.Key.Key_X and self.mode == "zone":
            self.start_drawing()
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
        if frame is None:
            frame, seq = splash("starting model", int(self.cfg.get("window_width", 1280))), -2
        self.view.st = st
        self.view.beat = time.time()
        if seq != self.seq:              # เฟรมเดิมไม่ต้องแปลงซ้ำ (แปลงภาพแพงกว่าที่คิด)
            self.seq = seq
            self.view.set_frame(frame)
        else:
            self.view.update()           # จุดสถานะยังต้องเต้นแม้เฟรมไม่มา
        self.rack.set_rows(st.get("rows", []), self.an.events, st.get("mode") == "zone")
        self.head.set(st)
        self.foot.set(st, self.msg, self.an.events)


class Rail(QWidget):
    """กรอบของแผงข้าง — เส้นแบ่ง 1px กับขีดสเกลที่ขอบ ทำให้ดูเป็นขอบเครื่องวัด ไม่ใช่ขอบการ์ดเว็บ"""

    def __init__(self):
        super().__init__()
        self.setFixedWidth(348)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(pen(LINE))
        p.drawLine(0, 0, 0, self.height())
        p.setPen(pen(QColor(255, 255, 255, 30)))
        for y in range(14, self.height() - 6, 14):
            p.drawLine(1, y, 5 if y % 70 == 14 else 3, y)     # ขีดยาวทุกห้าขีด เหมือนไม้บรรทัด


class Header(QWidget):
    """หัวแถบ: เครื่องหมายสถานี · ชื่อ · เวลาปัจจุบัน + ความยาวรอบตรวจ"""

    def __init__(self):
        super().__init__()
        self.st, self.clock = {}, ""
        self.setFixedHeight(58)

    def set(self, st):
        now = time.strftime("%H:%M:%S")
        if now != self.clock:            # หัวแถบเปลี่ยนแค่วินาทีละครั้ง ไม่ต้องวาด 50 ครั้ง/วิ
            self.st, self.clock = st, now
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(PANEL))
        p.setPen(pen(LINE))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

        p.setPen(Qt.PenStyle.NoPen)                    # เครื่องหมาย: สี่เหลี่ยมตะแคง
        p.setBrush(QColor(SIGNAL))
        p.translate(32, 29)
        p.rotate(45)
        p.drawRoundedRect(QRectF(-7, -7, 14, 14), 2, 2)
        p.resetTransform()

        text(p, 52, 11, "INSPECTION STATION",
             face(15, QFont.Weight.Bold, track=2.6, caps=True), TXT)
        zone = self.st.get("mode") == "zone"
        text(p, 52, 33, "โหมด Zone · ย้ายแก้วจากกรอบ A ไปกรอบ B" if zone else
             "เช็กลิสต์ผู้ตรวจ · operator หยิบชิ้นงานไหนออกมาตรวจแล้วบ้าง", face(12), DIM)

        r = int(self.st.get("round_s", 0))
        text(p, 0, 10, self.clock, face(17, QFont.Weight.DemiBold, mono=True),
             TXT, Qt.AlignmentFlag.AlignRight, self.width() - 28)
        text(p, 0, 34, f"รอบนี้ {r // 60:02d}:{r % 60:02d}", face(12, mono=True), DIM,
             Qt.AlignmentFlag.AlignRight, self.width() - 28)


class Footer(QWidget):
    """แถบสถานะ: ไฟสถานะ + ประโยคบอกว่ากำลังเกิดอะไร · ขวาสุดคือ ticker เหตุการณ์ล่าสุด"""

    def __init__(self):
        super().__init__()
        self.line = ("", DIM, "")
        self.setFixedHeight(44)

    def set(self, st, msg, events):
        note, expire = msg
        if note and time.time() < expire:
            state, col = note, SIGNAL
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
        p.drawEllipse(QPointF(28, 22), 5, 5)
        text(p, 44, 13, state, face(13, QFont.Weight.DemiBold), col)
        if last:
            text(p, 0, 14, last, face(12, mono=True), FAINT,
                 Qt.AlignmentFlag.AlignRight, self.width() - 28)


# ─────────────────────────── main ───────────────────────────
def main():
    import vision

    cfg = vision.load_config()
    analyzer, cam = vision.start(cfg)

    app = QApplication(sys.argv)
    app.setApplicationName("Inspection Station")
    win = Station(analyzer, cfg)
    win.show()
    app.exec()

    analyzer.release()
    cam.release()


if __name__ == "__main__":
    main()
