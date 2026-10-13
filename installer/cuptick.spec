# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: แพ็กแอป (python + torch + ultralytics + mediapipe + Qt + โมเดล) เป็นโฟลเดอร์/แอปที่รันได้เอง

    pyinstaller installer/cuptick.spec --noconfirm       # หรือ python installer/build.py (ทำครบทั้งชุด)

ผลลัพธ์: dist/Cuptick/ (Windows/Linux) · dist/Cuptick.app (macOS)
ตัวติดตั้ง (Inno Setup / .dmg) ห่อผลลัพธ์นี้อีกชั้น — ดู installer/build.py
"""
import os
import platform
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
APP = ROOT / "app"
ASSETS = Path(SPECPATH) / "assets"
VERSION = (ROOT / "VERSION").read_text().strip()
# macOS: ตั้ง MACOS_CODESIGN_IDENTITY ("Developer ID Application: ...") = เซ็นจริง + hardened runtime · ไม่ตั้ง = เซ็นแบบ ad-hoc
SIGN_ID = os.environ.get("MACOS_CODESIGN_IDENTITY") or None
ENTITLEMENTS = str(ROOT / "installer" / "entitlements.plist") if SIGN_ID else None

# ── ไฟล์ที่ต้องมากับแอป (อ่านอย่างเดียว ที่ _internal/ หรือ Contents/Resources/) ──
datas = [
    (str(APP / "config.yaml"), "."),
    (str(ROOT / "data" / "hand_landmarker.task"), "."),     # โมเดลมือของ MediaPipe — ไม่ต้องดาวน์โหลดตอนเปิด
    (str(APP / "fonts"), "fonts"),                          # Inter / Noto Sans Thai / JetBrains Mono — หน้าตาเหมือนกันทุกเครื่อง
]
models = sorted(p for p in (APP / "models").glob("cup_v*") if p.suffix in (".pt", ".onnx"))
if not any(p.name == "cup_v2.pt" for p in models):
    raise SystemExit("ไม่พบ app/models/cup_v2.pt — วางโมเดลก่อนแพ็ก (ดู docs/05-installer.md)")
datas += [(str(p), "models") for p in models]

binaries, hiddenimports = [], []
for pkg in ("ultralytics", "mediapipe", "torchvision"):   # ไฟล์ yaml/โมเดลเสริม + torchvision ต้องมี .so ของ ops (nms) ไม่งั้น ultralytics import ไม่ได้
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(APP / "app.py")],
    pathex=[str(APP)],                                     # vision / overlay / zone อยู่ข้าง app.py
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["vision", "overlay", "zone", "userdir", "lap"],
    excludes=["tkinter", "pytest", "IPython", "notebook", "sphinx",
              "polars", "_polars_runtime_32"],   # ultralytics เรียกแบบ lazy เฉพาะตอนเทรน/benchmark — ไม่ใช้ตอนตรวจ (ลดไป ~180 MB)
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Cuptick",
    console=False,                                         # ไม่มีหน้าต่างดำ — ข้อความไปที่ app.log (ดู app.py: to_log_file)
    icon=str(ASSETS / ("icon.ico" if sys.platform == "win32" else "icon.icns")),
    codesign_identity=SIGN_ID,
    entitlements_file=ENTITLEMENTS,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Cuptick")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Cuptick.app",
        icon=str(ASSETS / "icon.icns"),
        bundle_identifier="dev.tkk.cuptick",
        version=VERSION,
        codesign_identity=SIGN_ID,
        entitlements_file=ENTITLEMENTS,
        info_plist={
            "CFBundleDisplayName": "Cuptick",
            "CFBundleShortVersionString": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "13.0" if platform.machine() == "x86_64" else "12.0",   # ต้องตรงกับ MAC_MIN_OS ใน build.py
            # ไม่มีข้อความนี้ macOS จะปิดแอปทันทีที่เปิดกล้อง และไม่ขึ้นหน้าต่างขอสิทธิ์
            "NSCameraUsageDescription": "Cuptick ใช้กล้องเพื่อตรวจจับแก้วและมือแบบเรียลไทม์",
        },
    )
