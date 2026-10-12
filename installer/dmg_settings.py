"""dmgbuild: หน้าต่าง .dmg (ลากแอปไปวางที่ Applications) — python installer/build.py เรียกให้

    dmgbuild -s installer/dmg_settings.py -D app="dist/Inspection Station.app" "Inspection Station" out.dmg

พื้นหลังคือ installer/assets/dmg_background.png (660x400 pt ที่ 2x) — ไอคอนสองตัววางที่ x=180 กับ x=480 ตรงปลายลูกศรในภาพ
ตัวติดตั้งยังไม่ได้เซ็นรับรอง: วิธีเปิดครั้งแรกเขียนไว้ในพื้นหลังแล้ว
"""
import os

application = defines.get("app", "dist/Inspection Station.app")      # noqa: F821 (dmgbuild ใส่ defines ให้)
appname = os.path.basename(application)

format = "UDZO"
filesystem = "HFS+"
files = [application]
symlinks = {"Applications": "/Applications"}
icon = "installer/assets/icon.icns"
background = "installer/assets/dmg_background.png"
window_rect = ((200, 120), (660, 400))
default_view = "icon-view"
show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False
icon_size = 128
text_size = 13
icon_locations = {appname: (180, 190), "Applications": (480, 190)}
