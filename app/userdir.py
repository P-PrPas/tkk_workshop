"""โฟลเดอร์ของผู้ใช้ (config ที่แก้เอง · โมเดลที่วางเอง · app.log) — แยกจาก vision.py เพราะ app.py ต้องตั้งไฟล์ log
ให้เสร็จก่อน import torch/ultralytics (แอปที่ติดตั้งบน Windows ไม่มี stdout เลย) · ไม่พึ่งอะไรนอกจาก stdlib"""
import os
import shutil
import sys
from pathlib import Path

APP_DIR_NAME = "Cuptick"
LEGACY_DIR_NAMES = ("TKKWorkshop", "InspectionStation")   # ชื่อเดิมของแอป (TKK Workshop · ก่อนหน้านั้น Inspection Station) — ใหม่สุดก่อน


def user_dir():
    """ตัวแอปที่ติดตั้งแล้วแก้ไม่ได้ (macOS .app เซ็นไว้ / Windows Program Files) ของที่ผู้ใช้ต้องแก้จึงอยู่ที่นี่"""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / APP_DIR_NAME


def migrate_legacy_user_dir():
    """รุ่นที่ยังใช้ชื่อเดิมเก็บ config / โมเดลที่ผู้ใช้วางเองไว้ในโฟลเดอร์ชื่อเดิม — คัดลอกมาให้ครั้งเดียวตอนเปิดแอป
    (เฉพาะเมื่อโฟลเดอร์ใหม่ยังไม่มี · เอาโฟลเดอร์เดิมที่ใหม่สุดอันเดียว · ไม่แตะของเดิม · ไม่เอา log ไป)"""
    new = user_dir()
    old = next((d for d in (new.with_name(n) for n in LEGACY_DIR_NAMES) if d.is_dir()), None)
    if not new.exists() and old:
        shutil.copytree(old, new, ignore=shutil.ignore_patterns("app.log", "runs"))
