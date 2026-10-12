"""สร้างตัวติดตั้งของระบบที่รันอยู่ — Windows → .exe (Inno Setup) · macOS → .dmg (dmgbuild) · Linux → แค่แพ็ก+ทดสอบ

    pip install -r app/requirements.txt -r installer/requirements-build.txt
    python installer/build.py                  # แพ็ก → ทดสอบตัวที่แพ็ก → ห่อเป็นตัวติดตั้ง ที่ dist/
    python installer/build.py --no-package     # หยุดหลังแพ็ก+ทดสอบ (Linux หรือเครื่องที่ไม่มี Inno Setup)

ก่อนรัน: วางโมเดลไว้ที่ app/models/ (cup_v2.pt เป็นอย่างน้อย · เครื่อง Windows ใส่ .onnx ด้วย) และ git submodule update --init
ผลลัพธ์: dist/InspectionStation-<VERSION>-win-x64-setup.exe · dist/InspectionStation-<VERSION>-macos-arm64.dmg
"""
import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = (ROOT / "VERSION").read_text().strip()


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, **kw)


def built_exe(dist):
    if sys.platform == "darwin":
        return dist / "Inspection Station.app" / "Contents" / "MacOS" / "InspectionStation"
    return dist / "InspectionStation" / ("InspectionStation.exe" if sys.platform == "win32" else "InspectionStation")


def selftest(exe, work):
    """รันตัวที่แพ็กแล้วในโหมด --selftest: โหลดโมเดล+MediaPipe+วาดเฟรม+สร้างวิดเจ็ต — จับของที่แพ็กไม่ครบก่อนส่งให้ใครติดตั้ง"""
    out = work / "selftest.txt"
    out.unlink(missing_ok=True)
    r = subprocess.run([str(exe), "--selftest", str(out)], timeout=900)
    text = out.read_text(encoding="utf-8") if out.exists() else f"(ไม่มีผลลัพธ์ — โปรเซสออกด้วยรหัส {r.returncode})"
    print(text)
    if r.returncode != 0 or "SELFTEST OK" not in text:
        raise SystemExit("selftest ของตัวที่แพ็กไม่ผ่าน — ห้ามส่งตัวติดตั้งนี้ให้ใคร")


def package_windows(dist):
    iscc = shutil.which("iscc") or next((str(p) for p in (
        Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"), Path("C:/Program Files/Inno Setup 6/ISCC.exe")) if p.exists()), None)
    if not iscc:
        raise SystemExit("ไม่พบ Inno Setup (iscc) — ติดตั้ง: choco install innosetup  หรือ https://jrsoftware.org/isinfo.php")
    run([iscc, f"/DAppVersion={VERSION}", "installer/windows.iss"])
    return dist / f"InspectionStation-{VERSION}-win-x64-setup.exe"


def package_macos(dist):
    arch = platform.machine().replace("x86_64", "x64")
    out = dist / f"InspectionStation-{VERSION}-macos-{arch}.dmg"
    out.unlink(missing_ok=True)
    run([sys.executable, "-m", "dmgbuild", "-s", "installer/dmg_settings.py",
         "-D", f"app={dist / 'Inspection Station.app'}", "Inspection Station", out])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-package", action="store_true", help="ข้ามขั้นห่อเป็นตัวติดตั้ง")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--dist", default=str(ROOT / "dist"))
    ap.add_argument("--work", default=str(ROOT / "build"))
    args = ap.parse_args()
    dist, work = Path(args.dist), Path(args.work)

    if not (ROOT / "data" / "hand_landmarker.task").exists():
        raise SystemExit("ไม่พบ data/hand_landmarker.task — รัน: git submodule update --init")
    shutil.rmtree(dist / "InspectionStation", ignore_errors=True)
    shutil.rmtree(dist / "Inspection Station.app", ignore_errors=True)
    run([sys.executable, "-m", "PyInstaller", "installer/inspection_station.spec", "--noconfirm",
         "--distpath", dist, "--workpath", work])

    if not args.skip_selftest:
        selftest(built_exe(dist), work)
    if args.no_package:
        print(f"\nแพ็กเสร็จ (ยังไม่ห่อเป็นตัวติดตั้ง): {dist}")
        return
    if sys.platform == "win32":
        result = package_windows(dist)
    elif sys.platform == "darwin":
        result = package_macos(dist)
    else:
        print(f"\nLinux: ไม่มีตัวติดตั้ง — แพ็กเสร็จที่ {dist}/InspectionStation")
        return
    print(f"\nเสร็จ: {result}  ({result.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
