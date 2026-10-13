"""สร้างตัวติดตั้งของระบบที่รันอยู่ — Windows → .exe (Inno Setup) · macOS → .pkg (pkgbuild + productbuild) · Linux → แค่แพ็ก+ทดสอบ

    pip install -r app/requirements.txt -r installer/requirements-build.txt
    python installer/build.py                  # แพ็ก → ทดสอบตัวที่แพ็ก → ห่อเป็นตัวติดตั้ง ที่ dist/
    python installer/build.py --no-package     # หยุดหลังแพ็ก+ทดสอบ (Linux หรือเครื่องที่ไม่มี Inno Setup)

ก่อนรัน: วางโมเดลไว้ที่ app/models/ (cup_v2.pt เป็นอย่างน้อย · เครื่อง Windows ใส่ .onnx ด้วย) และ git submodule update --init
ผลลัพธ์: dist/Cuptick-<VERSION>-win-x64-setup.exe · dist/Cuptick-<VERSION>-macos-arm64.pkg (Apple Silicon) · -macos-x64.pkg (Mac Intel)
แต่ละเครื่องสร้างได้เฉพาะชิปของตัวเอง — Mac Intel ต้องสร้างบนเครื่อง Intel (CI ใช้ macos-15-intel)
macOS เซ็น + notarize ให้เมื่อมีตัวแปรสภาพแวดล้อมครบ (ดู docs/05-installer.md): MACOS_CODESIGN_IDENTITY, MACOS_INSTALLER_IDENTITY,
APPLE_API_KEY_PATH, APPLE_API_KEY_ID, APPLE_API_ISSUER — ถ้าไม่มี จะได้ตัวติดตั้งที่ยังไม่เซ็น
"""
import argparse
import os
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
        return dist / "Cuptick.app" / "Contents" / "MacOS" / "Cuptick"
    return dist / "Cuptick" / ("Cuptick.exe" if sys.platform == "win32" else "Cuptick")


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
    return dist / f"Cuptick-{VERSION}-win-x64-setup.exe"


def notarize(path):
    """ส่งให้ Apple ตรวจ (notarization) แล้วแนบผลกลับเข้าไฟล์ — ต้องเซ็นด้วย Developer ID ก่อน และตั้งค่า API key ครบ"""
    key, key_id, issuer = (os.environ.get(k) for k in ("APPLE_API_KEY_PATH", "APPLE_API_KEY_ID", "APPLE_API_ISSUER"))
    if not (key and key_id and issuer):
        print("ไม่ได้ notarize (ไม่มี APPLE_API_*) — ผู้ใช้จะเจอหน้าต่างเตือนของ macOS ครั้งแรก")
        return
    run(["xcrun", "notarytool", "submit", path, "--key", key, "--key-id", key_id, "--issuer", issuer, "--wait"])
    run(["xcrun", "stapler", "staple", path])


# ชิป → ข้อความในหน้าต้อนรับ · macOS ขั้นต่ำ (13: wheel ของ PySide6 / onnxruntime / onnx ที่ลงในตัวแอปติดแท็ก macosx_13_0)
MAC_CHIP = {"arm64": {"en": "an Apple silicon chip (M1 or later)", "th": "ชิป Apple silicon (M1 ขึ้นไป)"},
            "x86_64": {"en": "an Intel processor", "th": "ซีพียู Intel"}}
MAC_MIN_OS = {"arm64": "12.0", "x86_64": "13.0"}


def package_macos(dist, work):
    """ตัวติดตั้ง macOS แบบมาตรฐาน (.pkg): ต้อนรับ → ปลายทาง → ติดตั้ง → สรุป · สคริปต์หลังติดตั้งล้างธง quarantine ของแอป"""
    chip = platform.machine()                      # arm64 | x86_64
    out = dist / f"Cuptick-{VERSION}-macos-{chip.replace('x86_64', 'x64')}.pkg"
    out.unlink(missing_ok=True)
    stage = work / "pkg"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "root").mkdir(parents=True)
    run(["ditto", dist / "Cuptick.app", stage / "root" / "Cuptick.app"])   # ditto: คงลายเซ็นและ symlink
    plist = stage / "component.plist"
    run(["pkgbuild", "--analyze", "--root", stage / "root", plist])
    run(["plutil", "-replace", "0.BundleIsRelocatable", "-bool", "NO", plist])                  # ติดตั้งที่ /Applications เสมอ
    run(["pkgbuild", "--root", stage / "root", "--component-plist", plist, "--identifier", "dev.tkk.cuptick",
         "--version", VERSION, "--install-location", "/Applications", "--scripts", "installer/macos/scripts", stage / "app.pkg"])
    min_os = MAC_MIN_OS[chip]
    shutil.copytree(ROOT / "installer" / "macos" / "resources", stage / "resources")
    for lang in ("en", "th"):                      # หน้าต้อนรับบอกชิป/macOS ขั้นต่ำของตัวติดตั้งตัวนี้
        page = stage / "resources" / f"{lang}.lproj" / "welcome.html"
        page.write_text(page.read_text(encoding="utf-8").replace("@CHIP@", MAC_CHIP[chip][lang]).replace("@MINOS@", min_os.split(".")[0]),
                        encoding="utf-8")
    shutil.copy(ROOT / "installer" / "assets" / "pkg_logo.png", stage / "resources" / "logo.png")
    (stage / "distribution.xml").write_text(
        (ROOT / "installer" / "macos" / "distribution.xml").read_text(encoding="utf-8")
        .replace("@VERSION@", VERSION).replace("@ARCH@", chip).replace("@MINOS@", min_os), encoding="utf-8")
    cmd = ["productbuild", "--distribution", stage / "distribution.xml", "--resources", stage / "resources", "--package-path", stage]
    if os.environ.get("MACOS_INSTALLER_IDENTITY"):
        cmd += ["--sign", os.environ["MACOS_INSTALLER_IDENTITY"]]
    run(cmd + [out])
    notarize(out)
    return out


def main():
    for stream in (sys.stdout, sys.stderr):        # คอนโซล Windows เป็น cp1252 — ข้อความไทยทำให้สคริปต์ล้มตอนท้าย
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-package", action="store_true", help="ข้ามขั้นห่อเป็นตัวติดตั้ง")
    ap.add_argument("--skip-selftest", action="store_true")
    ap.add_argument("--dist", default=str(ROOT / "dist"))
    ap.add_argument("--work", default=str(ROOT / "build"))
    args = ap.parse_args()
    dist, work = Path(args.dist), Path(args.work)

    if not (ROOT / "data" / "hand_landmarker.task").exists():
        raise SystemExit("ไม่พบ data/hand_landmarker.task — รัน: git submodule update --init")
    shutil.rmtree(dist / "Cuptick", ignore_errors=True)
    shutil.rmtree(dist / "Cuptick.app", ignore_errors=True)
    run([sys.executable, "-m", "PyInstaller", "installer/cuptick.spec", "--noconfirm",
         "--distpath", dist, "--workpath", work])

    if not args.skip_selftest:
        selftest(built_exe(dist), work)
    if args.no_package:
        print(f"\nแพ็กเสร็จ (ยังไม่ห่อเป็นตัวติดตั้ง): {dist}")
        return
    if sys.platform == "win32":
        result = package_windows(dist)
    elif sys.platform == "darwin":
        result = package_macos(dist, work)
    else:
        print(f"\nLinux: ไม่มีตัวติดตั้ง — แพ็กเสร็จที่ {dist}/Cuptick")
        return
    print(f"\nเสร็จ: {result}  ({result.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
