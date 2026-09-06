"""self-check ตรรกะล้วน ๆ ของ vision.py — ไม่ต้องมีกล้อง/โมเดล

รัน:  python app/test_vision.py
"""
import time
from types import SimpleNamespace

from vision import CupMemory, HoldState, Inspection, hand_on_cup, hand_state


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


if __name__ == "__main__":
    test_hysteresis()
    test_cup_memory()
    test_inspection()
    test_inspection_survives_flicker()
    test_inspection_forgets_untouched()
    test_hand_state()
    test_hand_on_cup()
    test_hand_on_cup_eared()
    print("ok")
