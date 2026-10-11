"""โหมด zone — ย้ายแก้วจากจุด A ไปจุด B แล้วบอกว่าแก้วแต่ละใบอยู่สถานะไหน

ไฟล์นี้ **ไม่พึ่ง torch / ultralytics / mediapipe** (เหมือน overlay.py) — เทสต์ได้โดยไม่ต้องมีกล้อง

สถานะของแก้วแต่ละใบ (ผูกกับ track id):
    IDLE     ยังไม่เคยอยู่ใน A                         เทา
    AT_A     จุดกึ่งกลางกล่องอยู่ใน A                    ม่วง
    TRANSIT  ออกจาก A แล้ว ยังไม่ถึง B                  อำพัน   (คงอยู่แม้มือปล่อย/วางทิ้งกลางทาง)
    DONE     อยู่ใน B ต่อเนื่อง settle_seconds ขณะ TRANSIT   เขียว   (คงอยู่จนกด reset)

เข้า TRANSIT ได้ 2 ทาง: มือจับแก้วใน A ต่อเนื่อง hold_seconds วินาที · หรือจุดกลางอยู่นอก A ต่อเนื่อง
settle_seconds วินาที (ตาข่ายนิรภัยตอน MediaPipe ตรวจมือพลาด) กลับเข้า A โดยไม่มีมือจับต่อเนื่อง
settle_seconds วินาที = กลับเป็น AT_A · ทุกตัวนับเป็นวินาที ไม่ใช่เฟรม — CPU 3 FPS กับ GPU 30 FPS ต้องได้ผลเท่ากัน
DONE ต้องอยู่ใน B ต่อเนื่องด้วย: id ผี (ghost) ที่โผล่ใน B เฟรมเดียวหลังรับสถานะต่อ จะไม่ทำให้แก้ว "ถึง B" แบบถาวร

track id หลุดระหว่างทาง (มือบัง) แล้วได้เลขใหม่ → id ใหม่รับสถานะ AT_A/TRANSIT ของแก้วที่หายไป
(ไม่เกิน handoff_seconds) โดยเลือกใบที่ใกล้สุด — ไม่ใช้ระยะเป็นเกณฑ์ปัด เพราะคนยกแก้วข้ามโต๊ะ 3 วินาที
เคลื่อนที่ไกลเกินรัศมีใดๆ ที่ตั้งไว้ได้ · ข้อยกเว้น: id ที่โผล่ *ใน A* รับ TRANSIT ไม่ได้ (แก้วใบใหม่ที่เพิ่งวาง
ลงใน A ไม่ใช่แก้วที่กำลังเดินทาง) · DONE นับไปแล้วและคงอยู่เอง จึงรับต่อเฉพาะ id ที่โผล่ใกล้มาก

zone เป็นพิกัดสัดส่วน 0-1 ของเฟรม (ไม่ผูกกับความละเอียดกล้อง) — แปลงเป็นพิกเซลตอน update
"""
import cv2
import numpy as np

IDLE, AT_A, TRANSIT, DONE = "idle", "at_a", "transit", "done"


def point_in_polygon(x, y, poly):
    inside = False
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def polygons_overlap(a, b, n=200):
    """A กับ B ซ้อนกันไหม — ระบายทั้งคู่ลงตาราง n×n แล้วดูว่ามีช่องที่ทับกัน"""
    masks = []
    for poly in (a, b):
        m = np.zeros((n, n), np.uint8)
        cv2.fillPoly(m, [np.round(np.array(poly) * (n - 1)).astype(np.int32)], 1)
        masks.append(m)
    return bool((masks[0] & masks[1]).any())


class ZoneTracker:
    def __init__(self, hold_seconds=0.2, settle_seconds=0.5, handoff_seconds=3.0, handoff_dist=1.5):
        self.hold_seconds, self.settle_seconds = hold_seconds, settle_seconds
        self.handoff_seconds, self.handoff_dist = handoff_seconds, handoff_dist
        self.a = self.b = None
        self.reset()

    @property
    def ready(self):
        return self.a is not None and self.b is not None

    def set_zones(self, a, b):
        """a, b = polygon (list of (x, y) สัดส่วน 0-1) หรือ None · เปลี่ยน zone = เริ่มรอบใหม่"""
        self.a, self.b = a, b
        self.reset()

    def reset(self):
        # tid -> {state, center, size, last_live, และเวลาที่เริ่ม "ต่อเนื่อง" ของแต่ละเงื่อนไข (None = ไม่ได้เป็นอยู่):
        #         held_since มือจับใน A · out_since อยู่นอก A · back_since วางคืนใน A · in_b_since อยู่ใน B}
        self.cups = {}

    def _set(self, tid, cup, state, events, text):
        cup.update(state=state, held_since=None, out_since=None, back_since=None, in_b_since=None)
        events.append(f"#{tid} {text}")

    def _inherit(self, tid, center, in_a, live, now):
        """ID ใหม่ → รับสถานะต่อจากแก้ว (ที่ไม่ใช่ IDLE) ที่เพิ่งหายไปและอยู่ใกล้ที่สุด แล้วคืน (id เดิม, record)
        AT_A/TRANSIT ไม่จำกัดระยะ (ดูหัวไฟล์) · TRANSIT ไม่รับต่อให้ id ที่โผล่ใน A · DONE ต้องอยู่ใน handoff_dist × ด้านยาวกล่อง"""
        best = None
        for old, c in self.cups.items():
            if old in live or c["state"] == IDLE or now - c["last_live"] > self.handoff_seconds:
                continue
            if c["state"] == TRANSIT and in_a:
                continue
            d = np.hypot(center[0] - c["center"][0], center[1] - c["center"][1])
            if c["state"] == DONE and d > self.handoff_dist * c["size"]:
                continue
            if best is None or d < best[0]:
                best = (d, old)
        return (best[1], self.cups.pop(best[1])) if best else (None, None)

    def update(self, dets, held_ids, size, now):
        """dets = [(tid, box)] เฉพาะที่ตรวจเจอสดในเฟรมนี้ (ไม่รวมกล่องจากความจำ)
        คืน list ข้อความเหตุการณ์ (ไว้ลงบันทึก)"""
        events = []
        w, h = size
        pa = [(x * w, y * h) for x, y in self.a] if self.ready else None
        pb = [(x * w, y * h) for x, y in self.b] if self.ready else None
        live = {tid for tid, _ in dets}

        for tid, (x1, y1, x2, y2) in sorted(dets):
            center, sz = ((x1 + x2) / 2, (y1 + y2) / 2), max(x2 - x1, y2 - y1)
            in_a = in_b = False
            if self.ready:
                in_a, in_b = point_in_polygon(*center, pa), point_in_polygon(*center, pb)
            cup = self.cups.get(tid)
            if cup is None:
                old, cup = self._inherit(tid, center, in_a, live, now) if self.ready else (None, None)
                if cup is None:
                    cup = {"state": IDLE, "held_since": None, "out_since": None, "back_since": None, "in_b_since": None}
                else:                    # ตัวนับเดิมไว้ครบ — แก้วออกจาก A มา 2 วินาทีแล้วไม่ต้องรอใหม่เพราะ id เปลี่ยน
                    cup["in_b_since"] = None   # ยกเว้น "อยู่ใน B": id ใหม่ต้องพิสูจน์เองว่าอยู่ใน B ต่อเนื่อง (กัน ghost)
                    events.append(f"#{tid} รับสถานะต่อจาก #{old}")
                self.cups[tid] = cup
            cup.update(center=center, size=sz, last_live=now)
            if not self.ready:
                continue

            if cup["state"] == IDLE and in_a:
                self._set(tid, cup, AT_A, events, "อยู่ที่ A")
            if cup["state"] == AT_A:
                if in_a:
                    cup["out_since"] = None
                    if tid not in held_ids:
                        cup["held_since"] = None
                    else:
                        if cup["held_since"] is None:
                            cup["held_since"] = now
                        if now - cup["held_since"] >= self.hold_seconds:
                            self._set(tid, cup, TRANSIT, events, "ออกเดินทาง (มือจับ)")
                else:
                    cup["held_since"] = None
                    if cup["out_since"] is None:
                        cup["out_since"] = now
                    if now - cup["out_since"] >= self.settle_seconds or in_b:
                        self._set(tid, cup, TRANSIT, events, "ออกเดินทาง (ออกจาก A)")
            if cup["state"] == TRANSIT:
                if in_b:                                  # ต้องอยู่ใน B ต่อเนื่อง ไม่ใช่แค่แตะขอบ/เฟรมเดียว
                    cup["back_since"] = None
                    if cup["in_b_since"] is None:
                        cup["in_b_since"] = now
                    if now - cup["in_b_since"] >= self.settle_seconds:
                        self._set(tid, cup, DONE, events, "ถึง B แล้ว")
                else:
                    cup["in_b_since"] = None
                    if in_a and tid not in held_ids:     # วางคืนที่ A (มือที่กำลังยกอยู่ไม่นับ)
                        if cup["back_since"] is None:
                            cup["back_since"] = now
                        if now - cup["back_since"] >= self.settle_seconds:
                            self._set(tid, cup, AT_A, events, "กลับมาที่ A")
                    else:
                        cup["back_since"] = None

        for tid, cup in list(self.cups.items()):         # แก้วที่หายไป: เก็บไว้รอรับ id ใหม่ แล้วค่อยลืม
            if tid in live or cup["state"] == DONE:
                continue
            if now - cup["last_live"] > self.settle_seconds:   # หายไปนาน = ไม่ต่อเนื่องใน B (กัน tracker คืน id เดิมของผี)
                cup["in_b_since"] = None
            if cup["state"] == IDLE or now - cup["last_live"] > self.handoff_seconds:
                del self.cups[tid]
        return events

    def rows(self):
        """[(tid, state)] เรียงตามเลข — แก้วที่เห็นอยู่ + แก้วที่ไม่ใช่ IDLE ที่ยังไม่ถูกลืม"""
        return sorted((tid, c["state"]) for tid, c in self.cups.items())

    def counts(self):
        """(สำเร็จ, ทั้งหมดที่เข้ากติกา) — IDLE ไม่นับ"""
        states = [c["state"] for c in self.cups.values()]
        return states.count(DONE), sum(s != IDLE for s in states)
