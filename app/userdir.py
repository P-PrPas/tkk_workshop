"""โฟลเดอร์ของผู้ใช้ (config ที่แก้เอง · โมเดลที่วางเอง · app.log) — แยกจาก vision.py เพราะ app.py ต้องตั้งไฟล์ log
ให้เสร็จก่อน import torch/ultralytics (แอปที่ติดตั้งบน Windows ไม่มี stdout เลย) · ไม่พึ่งอะไรนอกจาก stdlib"""
import os
import sys
from pathlib import Path

APP_DIR_NAME = "InspectionStation"


def user_dir():
    """ตัวแอปที่ติดตั้งแล้วแก้ไม่ได้ (macOS .app เซ็นไว้ / Windows Program Files) ของที่ผู้ใช้ต้องแก้จึงอยู่ที่นี่"""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_DIR_NAME
