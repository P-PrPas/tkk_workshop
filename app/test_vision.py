"""self-check ตรรกะล้วน ๆ ของ vision.py — ไม่ต้องมีกล้อง/โมเดล

รัน:  python app/test_vision.py
"""
import time
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from ultralytics.engine.results import Boxes
from zone import AT_A, DONE, IDLE, TRANSIT, ZoneTracker, point_in_polygon, polygons_overlap
from vision import CupMemory, CupTracker, HoldState, Inspection, hand_on_cup, hand_state, match_hand_to_cup


def ticks(ins):
    """เช็กลิสต์ย่อเหลือ (id, ตรวจแล้ว?) — rows() คืนความคืบหน้า/เวลามาด้วย ไม่ใช้ในเทสต์นี้"""
    return [(tid, ok) for tid, ok, *_ in ins.rows()]


def _hand(pts):
    """สร้าง landmark 21 จุดจาก list ของ (x, y) normalized"""
    return [SimpleNamespace(x=x, y=y) for x, y in pts]


def test_hysteresis():
    """ขึ้นต้อง 3 เฟรมติด ลงต้อง 5 เฟรมติด — สั่น 1 เฟรมไม่ทำให้เปลี่ยนสถานะ"""
    s = HoldState(on_n=3, off_n=5)
    assert [s.update(True) for _ in range(3)] == [False, False, True]
    for _ in range(4):
        assert s.update(False) is True     # หายเฟรมเดียวแล้วกลับ → ยังค้าง
        assert s.update(True) is True
    assert [s.update(False) for _ in range(5)] == [True, True, True, True, False]


def test_cup_memory():
    """แก้วหายตอนมือบัง → กล่องยังอยู่ (coasting) อีก keep เฟรม แล้วค่อยลืม"""
    cm = CupMemory(keep=3)
    cm.update([(1, [10, 10, 50, 50]), (2, [100, 100, 140, 140])])
    assert {t: c for t, _, c in cm.boxes()} == {1: False, 2: False}
    for i in range(1, 4):
        cm.update([(1, [10, 10, 50, 50])])
        assert (2, True) in [(t, c) for t, _, c in cm.boxes()], f"เฟรม {i}: cup2 ควร coasting"
    cm.update([(1, [10, 10, 50, 50])])
    assert [t for t, _, _ in cm.boxes()] == [1]        # cup2 ถูกลืม
    cm.update([(1, [10, 10, 50, 50]), (2, [99, 99, 139, 139])])
    assert {t: c for t, _, c in cm.boxes()} == {1: False, 2: False}   # กลับมา = สด


def test_hand_state():
    """โชว์ท่ามือ (พาร์ท 2) — ปลายนิ้วห่างข้อมือ = เหยียด"""
    wrist = (0.5, 0.9)
    curled = [wrist] + [(0.5, 0.85)] * 20                       # ทุกจุดชิดข้อมือ
    assert hand_state(_hand(curled)) == "FIST"
    splayed = [wrist] + [(0.5, 0.85)] * 20
    for t in (4, 8, 12, 16, 20):
        splayed[t] = (0.5, 0.1)                                 # ปลายนิ้วไกลออกไป
    assert hand_state(_hand(splayed)) == "OPEN"


def test_hand_on_cup():
    """แก้วไม่มีหู: มือ *ห่อ* แก้ว (มือดูเหมือนแบ) จุดส่วนใหญ่ทับกล่อง → ต้องนับว่าถือ
    มือชี้จากไกล (ใหญ่กว่าแก้วมาก) หรืออยู่ไม่ตรงแก้ว → ไม่นับ"""
    W = H = 200
    cup = [(80, 60, 140, 160)]                          # กล่องแก้ว ~60x100
    # มือห่อแก้ว: จุด landmark กระจายในกล่องแก้ว (มือขนาดพอ ๆ กับแก้ว)
    grip = _hand([(0.4 + 0.15 * (i % 3) / 2, 0.35 + 0.55 * (i // 3) / 6) for i in range(21)])
    assert hand_on_cup(grip, W, H, cup, min_pts=10, max_ratio=3.0)
    # มือชี้จากไกล = มือเต็มเฟรม (ใหญ่กว่าแก้วมาก) แม้จุดจะทับกล่อง
    big = _hand([(0.05 + 0.9 * (i % 5) / 4, 0.05 + 0.9 * (i // 5) / 4) for i in range(21)])
    assert not hand_on_cup(big, W, H, cup, min_pts=10, max_ratio=3.0)
    # มือขนาดพอดีแต่ไม่ทับแก้ว
    away = _hand([(0.05 + 0.1 * (i % 3) / 2, 0.8 + 0.15 * (i // 3) / 6) for i in range(21)])
    assert not hand_on_cup(away, W, H, cup, min_pts=10, max_ratio=3.0)


def test_hand_on_cup_eared():
    """แก้วมีหู: จับที่หู มือเยื้องไป *ข้าง* กล่องแก้ว — ทับกล่องดิบ ๆ ไม่พอ
    แต่อยู่ในระยะ margin → ต้องนับว่าถือ"""
    W = H = 200
    cup = [(100, 50, 150, 160)]                                     # 50x110
    beside = _hand([(0.35 + 0.20 * (i % 3) / 2, 0.28 + 0.60 * (i // 3) / 6) for i in range(21)])
    assert not hand_on_cup(beside, W, H, cup, min_pts=12, max_ratio=4.0, margin=0.0)
    assert hand_on_cup(beside, W, H, cup, min_pts=12, max_ratio=4.0, margin=0.5)


def test_inspection():
    """เช็กลิสต์: เฉียดผ่านไม่ติ๊ก · ถือครบ need เฟรมถึงติ๊ก · ติ๊กแล้วปล่อยก็ยังติ๊ก · reset ล้างหมด"""
    ins = Inspection(need=4, trail_len=5, forget_seconds=999)
    cup1, cup2 = (0, [0, 0, 10, 10]), (2, [50, 50, 60, 60])

    for _ in range(3):                                  # มือเฉียด cup 0 อยู่ 3 เฟรม (ไม่ถึง 4)
        ins.update([cup1, cup2], {0})
    assert ticks(ins) == [(0, False), (2, False)], "เฉียดผ่านไม่ควรติ๊ก"

    for _ in range(4):
        ins.update([cup1, cup2], {2})                   # จับ cup 2 ครบ 4 เฟรม
    assert ticks(ins) == [(0, False), (2, True)]
    for _ in range(10):
        ins.update([cup1, cup2], set())                 # ปล่อยแล้ว ยังต้องติ๊กค้าง
    assert ticks(ins) == [(0, False), (2, True)]

    assert len(ins.trails[2]) == 5, "เส้นทางต้องเก็บแค่ trail_len จุดล่าสุด"
    ins.reset()
    assert ticks(ins) == []


def test_inspection_survives_flicker():
    """หลุดสลับเฟรมเว้นเฟรม ตัวนับถอยแค่ทีละหนึ่ง — สะสมจนติ๊กได้ ไม่ล้างทิ้งทุกครั้งที่วืบ"""
    ins = Inspection(need=3, trail_len=5, forget_seconds=999)
    for hit in (True, False, True, False, True, True, True):
        ins.update([(7, [0, 0, 10, 10])], {7} if hit else set())
    assert ticks(ins) == [(7, True)]


def test_inspection_forgets_untouched():
    """แก้วที่ไม่เคยถูกตรวจและหายไปนาน = หลุดจากเช็กลิสต์ · ที่ตรวจแล้วอยู่ยาว"""
    ins = Inspection(need=1, trail_len=5, forget_seconds=0.05)
    ins.update([(1, [0, 0, 10, 10]), (2, [9, 9, 20, 20])], {2})
    time.sleep(0.06)
    ins.update([], set())
    assert ticks(ins) == [(2, True)]


def test_inspection_seconds():
    cup = [(1, [0, 0, 10, 10])]
    for fps in (5, 30):
        ins = Inspection(8, 5, 999, seconds=1.0)
        for i in range(fps + 1):
            with patch("vision.time.monotonic", return_value=i / fps):
                ins.update(cup, {1})
            assert (1 in ins.picked) == (i == fps)
        ins.set_duration(2.0)
        assert 1 in ins.picked  # Changing duration keeps completed items.
        ins.reset()
        assert not ins.picked

    ins = Inspection(8, 5, 999, seconds=1.0, grace_seconds=0)
    for tick, held in ((0, {1}), (0.5, {1}), (0.75, set())):
        with patch("vision.time.monotonic", return_value=tick):
            ins.update(cup, held)
    assert ins.rows()[0][2] == 0.25
    ins.pause()
    with patch("vision.time.monotonic", return_value=100):
        ins.update(cup, {1})
    assert ins.rows()[0][2] == 0.25  # Disconnection adds no time.
    ins.set_duration(0.5)
    assert ins.rows()[0][2] == 0


def test_hand_matching_uses_valid_candidates():
    hand = _hand([(0.35 + 0.20 * (i % 3) / 2, 0.28 + 0.60 * (i // 3) / 6)
                  for i in range(21)])
    good = (1, [100, 50, 150, 160])
    # Large distractor contains every point but fails the hand/cup size ratio.
    bad = (99, [0, 0, 200, 200])
    for candidates in ([good, bad], [bad, good]):
        tid, count = match_hand_to_cup(hand, 200, 200, candidates, 12, 4.0, 0.5)
        assert tid == 1 and count >= 12
    assert match_hand_to_cup(hand, 200, 200, [bad], 12, 4.0, 0.5)[0] is None
    assert match_hand_to_cup(hand, 200, 200, [good, (2, good[1])], 12, 4.0, 0.5)[0] is None


def test_inspection_grace():
    ins = Inspection(8, 5, 999, seconds=1.0, grace_seconds=0.4)
    cups = [(1, [0, 0, 10, 10]), (2, [50, 50, 60, 60])]

    def step(tick, held):
        with patch("vision.time.monotonic", return_value=tick):
            ins.update(cups, held)

    step(0, {1})
    step(0.5, {1})
    step(0.6, set())
    step(0.8, set())
    assert ins.hits[1] == 0.5 and not ins.picked
    step(0.85, {1})
    assert ins.hits[1] == 0.5  # Reappearance does not count the invisible interval.
    step(1.0, {1})
    assert abs(ins.hits[1] - 0.65) < 1e-9
    step(1.2, {2})
    assert ins.hits[2] == 0 and ins.hits[1] == 0.65
    step(1.5, {2})
    assert abs(ins.hits[1] - 0.55) < 1e-9  # Only decay after the grace deadline.
    assert abs(ins.hits[2] - 0.3) < 1e-9
    step(2.2, set())
    assert not ins.picked
    ins.set_duration(2.0)
    assert not ins.hits and not ins._last_held
    ins.reset()
    assert not ins._held_before


def test_tracker_low_confidence_and_live_threshold():
    tracker = CupTracker({"conf": 0.5, "track_low_conf": 0.1, "track_buffer": 30})

    def boxes(*rows):
        return Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6), (400, 400))

    high = [10, 10, 60, 100, 0.9, 0]
    weak = [12, 10, 62, 100, 0.2, 0]
    other = [200, 10, 250, 100, 0.2, 0]
    tracks = tracker.update(boxes(high, other), 0.5)
    assert len(tracks) == 1
    original_id = tracks[0][0]
    for _ in range(3):
        tracks = tracker.update(boxes(weak, other), 0.5)
        assert len(tracks) == 1 and tracks[0][0] == original_id
        assert abs(tracks[0][2] - 0.2) < 1e-6
    assert tracker.update(boxes(), 0.5) == []
    tracks = tracker.update(boxes(high), 0.5)
    assert tracks[0][0] == original_id  # Brief total occlusion retains identity.
    medium = [200, 10, 250, 100, 0.6, 0]
    for _ in range(2):
        tracks = tracker.update(boxes(high, medium), 0.8)
        assert len(tracks) == 1  # Raising UI conf affects new tracks immediately.
    for _ in range(2):
        tracks = tracker.update(boxes(high, medium), 0.5)
    assert len(tracks) == 2
    assert original_id in {tid for tid, _, _ in tracks}
    assert tracker.detection_conf(0.05) == 0.05
    tracker.reset()
    tracks = tracker.update(boxes(high), 0.5)
    assert len(tracks) == 1 and tracks[0][0] == 1
    low_tracker = CupTracker({"conf": 0.01, "track_low_conf": 0.01})
    faint = [10, 10, 60, 100, 0.02, 0]
    first = low_tracker.update(boxes(faint), 0.01)[0][0]
    for _ in range(3):
        tracks = low_tracker.update(boxes(faint), 0.01)
        assert len(tracks) == 1 and tracks[0][0] == first



# ─────────── โหมด zone ───────────
ZA = [(0.0, 0.0), (0.3, 0.0), (0.3, 1.0), (0.0, 1.0)]       # ซ้าย
ZB = [(0.7, 0.0), (1.0, 0.0), (1.0, 1.0), (0.7, 1.0)]       # ขวา
SIZE = (100, 100)


def zbox(cx, cy=50, s=10):
    return [cx - s, cy - s, cx + s, cy + s]


def zstep(z, t, cups, held=()):
    """cups = {tid: cx} · คืนสถานะของแก้วทุกใบเป็น dict"""
    ev = z.update([(tid, zbox(cx)) for tid, cx in cups.items()], set(held), SIZE, t)
    return ev, dict(z.rows())


def test_zone_geometry():
    assert point_in_polygon(10, 50, [(x * 100, y * 100) for x, y in ZA])
    assert not point_in_polygon(50, 50, [(x * 100, y * 100) for x, y in ZA])
    assert not polygons_overlap(ZA, ZB)
    assert polygons_overlap(ZA, [(0.2, 0.2), (0.6, 0.2), (0.6, 0.8), (0.2, 0.8)])


def test_zone_idle_until_in_a_and_not_ready():
    z = ZoneTracker()
    assert zstep(z, 0, {1: 10})[1] == {1: IDLE}               # ยังไม่ได้วาด zone
    z.set_zones(ZA, ZB)
    assert zstep(z, 1, {1: 50})[1] == {1: IDLE}               # อยู่กลางโต๊ะ ไม่อยู่ zone ไหน
    assert zstep(z, 2, {1: 85})[1] == {1: IDLE}               # เริ่มที่ B ไม่นับ
    assert zstep(z, 3, {1: 50, 2: 10})[1] == {1: IDLE, 2: AT_A}


def test_zone_full_path_with_hand():
    z = ZoneTracker(hold_frames=3)
    z.set_zones(ZA, ZB)
    zstep(z, 0, {1: 10})
    for t in (1, 2):                                          # จับยังไม่ครบ 3 เฟรม
        assert zstep(z, t, {1: 10}, held={1})[1] == {1: AT_A}
    assert zstep(z, 3, {1: 10}, held={1})[1] == {1: TRANSIT}
    for t, x in enumerate((30, 50, 60), 4):                   # คงสีข้ามพื้นที่นอก zone
        assert zstep(z, t, {1: x}, held={1})[1] == {1: TRANSIT}
    ev, st = zstep(z, 8, {1: 85})
    assert st == {1: DONE} and ev == ["#1 ถึง B แล้ว"]
    assert zstep(z, 9, {1: 50})[1] == {1: DONE}               # สำเร็จแล้วคงอยู่
    assert z.counts() == (1, 1)
    z.reset()
    assert z.rows() == [] and z.ready                          # reset ล้างสถานะ คง zone


def test_zone_brush_past_does_not_trigger():
    z = ZoneTracker(hold_frames=3)
    z.set_zones(ZA, ZB)
    zstep(z, 0, {1: 10})
    for t, held in enumerate(({1}, {1}, set(), {1}, {1}), 1):  # จับไม่ติดกัน 3 เฟรม
        st = zstep(z, t, {1: 10}, held=held)[1]
    assert st == {1: AT_A}


def test_zone_leave_without_hand_is_safety_net():
    z = ZoneTracker(settle_frames=4)
    z.set_zones(ZA, ZB)
    zstep(z, 0, {1: 10})
    for t in range(1, 4):                                     # ออกจาก A ไม่ถึง 4 เฟรมติดกัน
        assert zstep(z, t, {1: 50})[1] == {1: AT_A}
    assert zstep(z, 4, {1: 50})[1] == {1: TRANSIT}            # ไม่มีมือเลยก็ถือว่าถูกย้าย
    assert zstep(z, 5, {1: 85})[1] == {1: DONE}


def test_zone_transit_persists_when_released_and_returns_to_a():
    z = ZoneTracker(hold_frames=1, settle_frames=3)
    z.set_zones(ZA, ZB)
    zstep(z, 0, {1: 10})
    zstep(z, 1, {1: 10}, held={1})                            # → TRANSIT (ยังไม่ทันออกจาก A)
    assert zstep(z, 2, {1: 10}, held={1})[1] == {1: TRANSIT}  # มือยังถืออยู่ใน A ไม่ตกกลับ
    assert zstep(z, 3, {1: 50})[1] == {1: TRANSIT}            # วางทิ้งกลางทาง คงสี
    assert zstep(z, 4, {1: 50})[1] == {1: TRANSIT}
    for t in (5, 6):                                          # วางคืน A แต่ยังไม่นิ่งพอ
        assert zstep(z, t, {1: 10})[1] == {1: TRANSIT}
    assert zstep(z, 7, {1: 10})[1] == {1: AT_A}


def test_zone_id_handoff():
    z = ZoneTracker(hold_frames=1, handoff_seconds=3.0)
    z.set_zones(ZA, ZB)
    zstep(z, 0, {1: 10})
    zstep(z, 1, {1: 10}, held={1})
    zstep(z, 2, {1: 45}, held={1})                            # TRANSIT, ถูกมือบังจน id 1 หาย
    ev, st = zstep(z, 3, {5: 48})                              # id 5 โผล่ใกล้ตำแหน่งสุดท้าย
    assert st == {5: TRANSIT} and ev == ["#5 รับสถานะต่อจาก #1"]
    assert zstep(z, 4, {5: 85})[1] == {5: DONE}


def test_zone_handoff_limits():
    z = ZoneTracker(hold_frames=1, handoff_seconds=3.0, handoff_dist=1.5)
    z.set_zones(ZA, ZB)
    zstep(z, 0, {1: 10})
    zstep(z, 1, {1: 10}, held={1})
    assert zstep(z, 2, {9: 90})[1][9] == IDLE                  # ไกลเกิน ไม่รับต่อ (id 1 ยังรอ id ใหม่อยู่)
    z2 = ZoneTracker(hold_frames=1, handoff_seconds=3.0)
    z2.set_zones(ZA, ZB)
    zstep(z2, 0, {1: 10})
    zstep(z2, 1, {1: 10}, held={1})
    assert zstep(z2, 6, {5: 12})[1] == {5: AT_A}              # เกินเวลา 3 วิ ไม่รับต่อ (และ id 1 ถูกลืม)
    z3 = ZoneTracker(hold_frames=1)
    z3.set_zones(ZA, ZB)
    zstep(z3, 0, {1: 10, 2: 25})
    zstep(z3, 1, {1: 10, 2: 25}, held={1, 2})
    st = zstep(z3, 2, {7: 24})[1]                              # สองใบหาย → ใบใหม่รับต่อจากใบที่ใกล้กว่า
    assert st[7] == TRANSIT and 2 not in st and 1 in st


def test_zone_multiple_cups_independent_and_counts():
    z = ZoneTracker(hold_frames=1)
    z.set_zones(ZA, ZB)
    zstep(z, 0, {1: 10, 2: 20, 3: 50})
    zstep(z, 1, {1: 10, 2: 20, 3: 50}, held={1})
    zstep(z, 2, {1: 85, 2: 20, 3: 50})
    assert dict(z.rows()) == {1: DONE, 2: AT_A, 3: IDLE}
    assert z.counts() == (1, 2)                                # IDLE ไม่นับในตัวหาร


def test_zone_redraw_resets_states():
    z = ZoneTracker(hold_frames=1)
    z.set_zones(ZA, ZB)
    zstep(z, 0, {1: 10})
    z.set_zones(ZA, None)                                      # วาดใหม่ระหว่างทาง
    assert not z.ready and z.rows() == []
    assert zstep(z, 1, {1: 10})[1] == {1: IDLE}


if __name__ == "__main__":
    test_zone_geometry()
    test_zone_idle_until_in_a_and_not_ready()
    test_zone_full_path_with_hand()
    test_zone_brush_past_does_not_trigger()
    test_zone_leave_without_hand_is_safety_net()
    test_zone_transit_persists_when_released_and_returns_to_a()
    test_zone_id_handoff()
    test_zone_handoff_limits()
    test_zone_multiple_cups_independent_and_counts()
    test_zone_redraw_resets_states()
    test_hand_matching_uses_valid_candidates()
    test_inspection_grace()
    test_tracker_low_confidence_and_live_threshold()
    test_inspection_seconds()
    test_hysteresis()
    test_cup_memory()
    test_inspection()
    test_inspection_survives_flicker()
    test_inspection_forgets_untouched()
    test_hand_state()
    test_hand_on_cup()
    test_hand_on_cup_eared()
    print("ok")
