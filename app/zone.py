"""โหมด zone — ย้ายแก้วจากจุด A ไปจุด B แล้วบอกว่าแก้วแต่ละใบอยู่สถานะไหน

ไฟล์นี้ **ไม่พึ่ง torch / ultralytics / mediapipe** (เหมือน overlay.py) — เทสต์ได้โดยไม่ต้องมีกล้อง

สถานะของแก้วแต่ละใบ (ผูกกับ track id):
    IDLE     ยังไม่เคยอยู่ใน A                         เทา
    AT_A     จุดกึ่งกลางกล่องอยู่ใน A                    ม่วง
    TRANSIT  ออกจาก A แล้ว ยังไม่ถึง B                  อำพัน   (คงอยู่แม้มือปล่อย/วางทิ้งกลางทาง)
    DONE     เข้า B ขณะ TRANSIT                        เขียว   (คงอยู่จนกด reset)

เข้า TRANSIT ได้ 2 ทาง: มือจับแก้วใน A ติดกัน hold_frames เฟรม · หรือจุดกลางออกนอก A ติดกัน
settle_frames เฟรม (ตาข่ายนิรภัยตอน MediaPipe ตรวจมือพลาด) กลับเข้า A โดยไม่มีมือจับ
settle_frames เฟรม = กลับเป็น AT_A

track id หลุดระหว่างทาง (มือบัง) แล้วได้เลขใหม่ → แก้วใหม่ที่โผล่ใกล้ตำแหน่งสุดท้ายภายใน
handoff_seconds รับสถานะต่อ

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
    def __init__(self, hold_frames=3, settle_frames=6, handoff_seconds=3.0, handoff_dist=1.5):
        self.hold_frames, self.settle_frames = hold_frames, settle_frames
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
        self.cups = {}      # tid -> {state, held, out, back, center, size, last_live}

    def _set(self, tid, cup, state, events, text):
        cup["state"], cup["held"], cup["out"], cup["back"] = state, 0, 0, 0
        events.append(f"#{tid} {text}")

    def _inherit(self, tid, center, live, now):
        """ID ใหม่ที่โผล่ใกล้แก้ว (ที่ไม่ใช่ IDLE) ซึ่งเพิ่งหายไป → รับสถานะต่อ แล้วคืน (id เดิม, record)"""
        best = None
        for old, c in self.cups.items():
            if old in live or c["state"] == IDLE or now - c["last_live"] > self.handoff_seconds:
                continue
            d = np.hypot(center[0] - c["center"][0], center[1] - c["center"][1])
            if d <= self.handoff_dist * c["size"] and (best is None or d < best[0]):
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
            cup = self.cups.get(tid)
            if cup is None:
                old, cup = self._inherit(tid, center, live, now) if self.ready else (None, None)
                if cup is None:
                    cup = {"state": IDLE, "held": 0, "out": 0, "back": 0}
                else:
                    cup["held"] = cup["out"] = cup["back"] = 0
                    events.append(f"#{tid} รับสถานะต่อจาก #{old}")
                self.cups[tid] = cup
            cup.update(center=center, size=sz, last_live=now)
            if not self.ready:
                continue

            in_a, in_b = point_in_polygon(*center, pa), point_in_polygon(*center, pb)
            if cup["state"] == IDLE and in_a:
                self._set(tid, cup, AT_A, events, "อยู่ที่ A")
            if cup["state"] == AT_A:
                if in_a:
                    cup["out"] = 0
                    cup["held"] = cup["held"] + 1 if tid in held_ids else 0
                    if cup["held"] >= self.hold_frames:
                        self._set(tid, cup, TRANSIT, events, "ออกเดินทาง (มือจับ)")
                else:
                    cup["out"] += 1
                    if cup["out"] >= self.settle_frames or in_b:
                        self._set(tid, cup, TRANSIT, events, "ออกเดินทาง (ออกจาก A)")
            if cup["state"] == TRANSIT:
                if in_b:
                    self._set(tid, cup, DONE, events, "ถึง B แล้ว")
                elif in_a and tid not in held_ids:       # วางคืนที่ A (มือที่กำลังยกอยู่ไม่นับ)
                    cup["back"] += 1
                    if cup["back"] >= self.settle_frames:
                        self._set(tid, cup, AT_A, events, "กลับมาที่ A")
                else:
                    cup["back"] = 0

        for tid, cup in list(self.cups.items()):         # แก้วที่หายไป: เก็บไว้รอรับ id ใหม่ แล้วค่อยลืม
            if tid in live or cup["state"] == DONE:
                continue
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
