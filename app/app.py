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

from overlay import splash          # vision (torch/ultralytics) นำเข้าตอน main() เท่านั้น —
                                    # เปิดดูหน้าตา/รัน tools/ui_preview.py บนเครื่องที่ไม่มีสแต็ก CV ได้

try:
    from PySide6.QtCore import Qt, QPoint, QRectF, QTimer, QVariantAnimation, QPointF
    from PySide6.QtGui import (QColor, QFont, QFontMetrics, QImage, QPainter,
                               QPalette, QPen, QPixmap)
    from PySide6.QtWidgets import (QAbstractButton, QApplication, QHBoxLayout,
                                   QMenu, QVBoxLayout, QWidget)
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
BLACK  = "#04060A"   # พื้นหลังกรอบภาพ

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
        self.setMinimumWidth(340)

    def set_rows(self, rows, events):
        log = list(events)[-LOG_ROWS:][::-1]
        if (rows, log) != (self.rows, self.log):
            self.rows, self.log = rows, log
            self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)   # ไม่ระบายพื้น — ปล่อยให้ขีดสเกลของ Rail ทะลุขึ้นมา
        W = self.width()
        pad = 22
        inner = W - pad * 2
        done = sum(1 for r in self.rows if r[1])

        section(p, pad, 26, "รอบตรวจปัจจุบัน")

        # ── ตัวเลขพระเอก: อ่านจากท้ายห้องได้ ──
        big = face(54, QFont.Weight.Bold, mono=True)
        p.setFont(big)
        p.setPen(pen(SIGNAL if self.rows and done == len(self.rows) else TXT))
        p.drawText(QRectF(pad, 46, inner, 64), int(Qt.AlignmentFlag.AlignLeft), f"{done:02d}")
        wd = QFontMetrics(big).horizontalAdvance(f"{done:02d}")
        p.setFont(face(30, QFont.Weight.Normal, mono=True))
        p.setPen(pen(QColor(255, 255, 255, 110)))
        p.drawText(QRectF(pad + wd + 10, 68, inner, 46), int(Qt.AlignmentFlag.AlignLeft),
                   f"/ {len(self.rows):02d}")
        text(p, pad, 114, "ชิ้นงานที่ตรวจแล้ว", face(13), DIM)

        # ── มิเตอร์แบบช่อง: หนึ่งช่องต่อหนึ่งชิ้น ──
        y = 142
        n = len(self.rows)
        if n:
            gap, seg = 4, (inner - 4 * (n - 1)) / n
            for i, (_, ok, prog, _) in enumerate(self.rows):
                x = pad + i * (seg + gap)
                p.setPen(Qt.PenStyle.NoPen)
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
        base = self.height() - log_n * 22 - 34           # ขอบบนของบันทึกเหตุการณ์
        room = base - 14 - top
        cap = max(1, int(room // (ROW_H + 6)))
        if not self.rows:
            text(p, pad, top + 10, "ยังไม่เห็นชิ้นงานในเฟรม", face(12), FAINT)
            text(p, pad, top + 32, "วางแก้วให้กล้องเห็น แล้วหยิบขึ้นมาตรวจ",
                 face(11), QColor(FAINT).darker(125).name())
        else:
            shown = self.rows[:cap]
            if len(shown) < len(self.rows) and cap * (ROW_H + 6) + 20 > room:
                shown = shown[:-1] or shown       # ยอมทิ้งอีกแถวเพื่อให้ "+ อีก N ชิ้น" มีที่ยืน
            for i, (tid, ok, prog, held_s) in enumerate(shown):
                self._row(p, pad, top + i * (ROW_H + 6), inner, tid, ok, prog, held_s)
            if len(shown) < len(self.rows):
                text(p, pad, top + len(shown) * (ROW_H + 6) + 6,
                     f"+ อีก {len(self.rows) - len(shown)} ชิ้น", face(11), FAINT)

        # ── บันทึกเหตุการณ์ — ยึดขอบล่างของแผง ใหม่สุดอยู่บน ──
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
        keys = QVBoxLayout()
        keys.setContentsMargins(22, 0, 22, 20)
        keys.setSpacing(8)
        keys.addWidget(new_round)
        keys.addWidget(self.cam_btn)
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

        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(0)
        mid.addWidget(self.view, 1)
        mid.addWidget(wrap)

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
    def reset(self):
        self.an.reset()
        self.flash("เริ่มรอบตรวจใหม่แล้ว")

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
        if k in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
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
        if frame is None:
            frame, seq = splash("starting model", int(self.cfg.get("window_width", 1280))), -2
        self.view.st = st
        self.view.beat = time.time()
        if seq != self.seq:              # เฟรมเดิมไม่ต้องแปลงซ้ำ (แปลงภาพแพงกว่าที่คิด)
            self.seq = seq
            self.view.set_frame(frame)
        else:
            self.view.update()           # จุดสถานะยังต้องเต้นแม้เฟรมไม่มา
        self.rack.set_rows(st.get("rows", []), self.an.events)
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
        text(p, 52, 33, "เช็กลิสต์ผู้ตรวจ · operator หยิบชิ้นงานไหนออกมาตรวจแล้วบ้าง",
             face(12), DIM)

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
