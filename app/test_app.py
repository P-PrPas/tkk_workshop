"""self-check ตรรกะล้วน ๆ ของ app.py — ไม่ต้องมีกล้อง/โมเดล

รัน:  python app/test_app.py
"""
from types import SimpleNamespace

from app import CupMemory, HoldState, hand_on_cup


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


def test_hand_on_cup():
    """แก้วไม่มีหู: มือประคอง (นิ้วไม่กำแน่น) แต่จุดมืออยู่บนแก้ว → ต้องนับว่าถือ
    ส่วนมือแบกว้างเอื้อม → ไม่นับ"""
    W = H = 100
    cup = [(40, 40, 70, 70)]                            # กล่องแก้ว
    # มือประคอง: ทุกจุดกองอยู่กลางกล่องแก้ว, นิ้วไม่เหยียด (tip อยู่ใกล้ wrist)
    grip = _hand([(0.55, 0.55)] * 21)
    assert hand_on_cup(grip, W, H, cup, open_max=3, min_pts=5)
    # มือแบกว้าง: ปลายนิ้ว (index 4,8,12,16,20) เหยียดไกลจากข้อมือ (index 0)
    wide_pts = [(0.55, 0.55)] * 21
    for t in (4, 8, 12, 16, 20):
        wide_pts[t] = (0.95, 0.95)                      # ปลายนิ้วไกลออกไป
    wide = _hand(wide_pts)
    assert not hand_on_cup(wide, W, H, cup, open_max=3, min_pts=5)
    # มือกำแต่ไม่อยู่บนแก้ว → ไม่นับ
    away = _hand([(0.1, 0.1)] * 21)
    assert not hand_on_cup(away, W, H, cup, open_max=3, min_pts=5)


if __name__ == "__main__":
    test_hysteresis()
    test_cup_memory()
    test_hand_on_cup()
    print("ok")
