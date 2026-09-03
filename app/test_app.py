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


if __name__ == "__main__":
    test_hysteresis()
    test_cup_memory()
    test_hand_on_cup()
    print("ok")
