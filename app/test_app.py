"""self-check ตรรกะล้วน ๆ ของ app.py — ไม่ต้องมีกล้อง/โมเดล

รัน:  python app/test_app.py
"""
from app import HoldState, boxes_overlap


def test_boxes_overlap():
    assert boxes_overlap([0, 0, 10, 10], [5, 5, 15, 15])
    assert not boxes_overlap([0, 0, 10, 10], [20, 20, 30, 30])
    assert not boxes_overlap([0, 0, 10, 10], [10, 0, 20, 10])   # แตะขอบ = ไม่ซ้อน


def test_hysteresis():
    """ขึ้นต้อง 3 เฟรมติด ลงต้อง 5 เฟรมติด — สั่น 1 เฟรมไม่ทำให้เปลี่ยนสถานะ"""
    s = HoldState(on_n=3, off_n=5)
    assert [s.update(True) for _ in range(3)] == [False, False, True]
    # เห็นบ้างไม่เห็นบ้าง: ยังไม่ถึง off_n ติดกัน → ค้าง HOLDING
    for miss_then_hit in range(4):
        assert s.update(False) is True
        assert s.update(True) is True
    # หายจริง 5 เฟรมติด → เลิก
    assert [s.update(False) for _ in range(5)] == [True, True, True, True, False]


if __name__ == "__main__":
    test_boxes_overlap()
    test_hysteresis()
    print("ok")
