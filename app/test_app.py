"""self-check ตรรกะล้วน ๆ ของ app.py — ไม่ต้องมีกล้อง/โมเดล

รัน:  python app/test_app.py
"""
from app import CupMemory, HoldState, boxes_overlap


def test_boxes_overlap():
    assert boxes_overlap([0, 0, 10, 10], [5, 5, 15, 15])
    assert not boxes_overlap([0, 0, 10, 10], [20, 20, 30, 30])
    assert not boxes_overlap([0, 0, 10, 10], [10, 0, 20, 10])   # แตะขอบ = ไม่ซ้อน


def test_hysteresis():
    """ขึ้นต้อง 3 เฟรมติด ลงต้อง 5 เฟรมติด — สั่น 1 เฟรมไม่ทำให้เปลี่ยนสถานะ"""
    s = HoldState(on_n=3, off_n=5)
    assert [s.update(True) for _ in range(3)] == [False, False, True]
    # เห็นบ้างไม่เห็นบ้าง: ยังไม่ถึง off_n ติดกัน → ค้าง HOLDING
    for _ in range(4):
        assert s.update(False) is True
        assert s.update(True) is True
    # หายจริง 5 เฟรมติด → เลิก
    assert [s.update(False) for _ in range(5)] == [True, True, True, True, False]


def test_cup_memory():
    """แก้วหายตอนมือบัง → กล่องยังอยู่ (coasting) อีก keep เฟรม แล้วค่อยลืม"""
    cm = CupMemory(keep=3)
    cm.update([(1, [10, 10, 50, 50]), (2, [100, 100, 140, 140])])
    assert {t: c for t, _, c in cm.boxes()} == {1: False, 2: False}   # สด ทั้งคู่

    for i in range(1, 4):                       # cup 2 หายไป 3 เฟรม (ยังไม่เกิน keep)
        cm.update([(1, [10, 10, 50, 50])])
        assert (2, True) in [(t, c) for t, _, c in cm.boxes()], f"เฟรม {i}: cup2 ควร coasting"

    cm.update([(1, [10, 10, 50, 50])])          # เฟรมที่ 4 > keep → ลืม cup 2
    ids = [t for t, _, _ in cm.boxes()]
    assert ids == [1]

    cm.update([(1, [10, 10, 50, 50]), (2, [99, 99, 139, 139])])   # cup 2 กลับมา = สด
    assert {t: c for t, _, c in cm.boxes()} == {1: False, 2: False}


if __name__ == "__main__":
    test_boxes_overlap()
    test_hysteresis()
    test_cup_memory()
    print("ok")
