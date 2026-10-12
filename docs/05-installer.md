# 05 — ตัวติดตั้ง (Windows / macOS)

ห่อแอปทั้งตัว (Python + torch + ultralytics + MediaPipe + Qt + โมเดล) ให้ทีม sales ติดตั้งแล้วใช้ได้เลย
ไม่ต้องลง Python ไม่ต้องใช้อินเทอร์เน็ต

| | Windows | macOS |
|---|---|---|
| ไฟล์ | `InspectionStation-<รุ่น>-win-x64-setup.exe` | `InspectionStation-<รุ่น>-macos-arm64.pkg` |
| รูปแบบ | วิซาร์ดมาตรฐาน (Inno Setup) | วิซาร์ดมาตรฐานของ macOS (`.pkg`) |
| ภาษา | **เลือกได้** อังกฤษ / ไทย ที่หน้าแรก | ตามภาษาของระบบ (อังกฤษ / ไทย) — macOS ไม่มีตัวเลือกภาษาในตัวติดตั้ง |
| ชิป | x64 | **Apple Silicon (M1–M4) เท่านั้น** (ไม่มี Mac Intel) |
| ตรวจแก้ว | ONNX บน CPU (torch ในตัวติดตั้งเป็นรุ่น CPU) | torch + GPU ของชิป (MPS) |
| เซ็นรับรอง | ยังไม่เซ็น → SmartScreen เตือน | เซ็น + notarize ได้เมื่อมี Apple Developer ID (ดูด้านล่าง) · ยังไม่มี → macOS เตือน |

เครื่องพัฒนาเป็น Linux สร้างตัวติดตั้ง Windows/macOS เองไม่ได้ จึงสร้างบน **GitHub Actions**
(`.github/workflows/build-installers.yml`) ที่ใช้เครื่อง Windows/macOS จริง

## ติดตั้ง (สำหรับผู้ใช้)

**Windows** — ดับเบิลคลิก `.exe` → เลือกภาษา (English / ไทย) → Next → Install → Finish
- ถ้าขึ้นหน้าต่างสีน้ำเงิน "Windows protected your PC" (SmartScreen เพราะยังไม่ได้เซ็นรับรอง): กด **More info** → **Run anyway**
- ติดตั้งแบบรายผู้ใช้ ไม่ต้องใช้สิทธิ์ admin ลงที่ `%LOCALAPPDATA%\Programs\Inspection Station` · ติดตั้งรุ่นใหม่ทับรุ่นเดิมได้เลย · ถอนการติดตั้งที่ Settings → Apps
- ครั้งแรกที่เปิดแอป กด Allow เมื่อ Windows ถามสิทธิ์กล้อง

**macOS** — ดับเบิลคลิก `.pkg` → Continue → Install (ใส่รหัสผ่านเครื่อง) → Close แล้วเปิดจาก Launchpad
- แอปที่ติดตั้งผ่านตัวติดตั้งนี้ไม่ถูกบล็อกอีก (สคริปต์หลังติดตั้งล้างธง quarantine ให้) แต่ **ตัวไฟล์ `.pkg` เองที่โหลดจากเว็บ** ยังเจอหน้าต่างเตือนของ macOS ครั้งเดียว จนกว่าจะเซ็น (ดูหัวข้อถัดไป)
- กด **อนุญาต** เมื่อ macOS ถามสิทธิ์กล้อง (ถ้าพลาด: ตั้งค่าระบบ → ความเป็นส่วนตัวและความปลอดภัย → กล้อง)

### macOS บอกว่า "Apple could not verify … is free of malware" มีแต่ปุ่ม Done / Move to Trash
เป็นหน้าต่างมาตรฐานของ macOS สำหรับไฟล์ที่ยังไม่ผ่านการรับรอง (notarization) ของ Apple **ไม่ใช่ไวรัส** วิธีผ่าน เลือกอย่างใดอย่างหนึ่ง:
1. กด Done แล้วไป **ตั้งค่าระบบ → ความเป็นส่วนตัวและความปลอดภัย** เลื่อนลงล่าง กด **Open Anyway** ตรงรายการที่ขึ้นชื่อไฟล์ แล้วใส่รหัสผ่าน
2. หรือใช้ Terminal (ไม่ผ่านด่านนี้เลย): `sudo installer -pkg ~/Downloads/InspectionStation-1.0.0-macos-arm64.pkg -target /`
3. หรือล้างธงก่อนเปิด: `xattr -d com.apple.quarantine ~/Downloads/InspectionStation-1.0.0-macos-arm64.pkg`

**ที่ทำให้หายขาดมีทางเดียว:** เซ็นรับรองและ notarize ด้วยบัญชี Apple Developer Program (ราว $99/ปี) ตัวสร้างรองรับแล้ว ดูหัวข้อ "เซ็นรับรองและ notarize" ด้านล่าง

## ไฟล์ของผู้ใช้ (ตัวแอปที่ติดตั้งแล้วแก้ไม่ได้)

| | Windows | macOS |
|---|---|---|
| โฟลเดอร์ | `%APPDATA%\InspectionStation` | `~/Library/Application Support/InspectionStation` |
| `app.log` | บันทึกการทำงานครั้งล่าสุด — **ส่งไฟล์นี้มาเวลาเปิดแอปไม่ขึ้น** | เหมือนกัน |
| `config.yaml` | ใส่เฉพาะค่าที่อยากเปลี่ยน เช่น `model_name: cup_v1` ทับค่าที่มากับแอป | เหมือนกัน |
| `models/` | วางโมเดลสำรองเอง แอปมองที่นี่ก่อนของที่มากับแอป | เหมือนกัน |

รูปที่กด **S** บันทึกไปที่โฟลเดอร์ "รูปภาพ" ของผู้ใช้ ค่าที่ปรับในหน้าต่างตั้งค่า (T) ไม่ถูกบันทึก (เหมือนเดิม)

มากับตัวติดตั้ง: `cup_v2` (ค่าเริ่มต้น) + `cup_v1` (สำรอง — ตั้ง `model_name: cup_v1` ใน `config.yaml` ของผู้ใช้) · `hand_landmarker.task`

## สร้างตัวติดตั้ง (ผู้พัฒนา)

### ครั้งแรก: แนบโมเดลเป็น Release (ทำครั้งเดียว — สร้างไว้แล้วเมื่อ 2026-10-12)
ตัวสร้างบน GitHub ไม่มีไฟล์โมเดล (ไม่อยู่ใน git) จึงดึงจาก Release `models-v2` (v1 ดึงจาก Release `v1` ที่มีอยู่แล้ว):
```bash
gh release create models-v2 app/models/cup_v2.pt app/models/cup_v2.onnx \
  --title "cup_v2 weights (for the installer build)" --notes "Weights bundled into the installers. Not an app release."
```
⚠️ repo เป็น public — **ทุกคนดาวน์โหลดโมเดลได้** คำสั่งนี้เผยแพร่ไฟล์ จึงรันเมื่อพร้อมเท่านั้น

### สร้างบน GitHub Actions
- **ลองสร้าง:** Actions → *Build installers* → *Run workflow* → ดาวน์โหลดไฟล์จาก **Artifacts** ของ run นั้น (สั่งมือได้เมื่อไฟล์ workflow อยู่บน `main` แล้ว) · PR ที่แก้ไฟล์ใน `installer/` จะสร้างให้เองด้วย
- **ปล่อยจริง:** `git tag app-v1.0.0 && git push origin app-v1.0.0` → ตัวติดตั้งทั้งสองแนบกับ Release `app-v1.0.0` (เปลี่ยนเลขรุ่นที่ไฟล์ `VERSION` ก่อน)

ในตัวสร้างทำตามนี้ — ถ้าขั้นไหนพังจะไม่ได้ไฟล์:
`pip install` → ดึงโมเดล → `pyinstaller` → **`--selftest` กับตัวที่แพ็กแล้ว** (โหลดโมเดล + MediaPipe + วาดเฟรม + สร้างวิดเจ็ต Qt) → Inno Setup (Windows) / pkgbuild + productbuild (macOS) → ติดตั้งจริงแล้วรันซ้ำ

### สร้างเองบนเครื่อง Windows หรือ Mac
```bash
git submodule update --init
# วางโมเดลที่ app/models/ (cup_v2.pt อย่างน้อย · Windows ใส่ .onnx ด้วย · cup_v1.* ถ้าอยากแพ็กตัวสำรอง)
pip install -r app/requirements.txt -r installer/requirements-build.txt
python installer/build.py             # ผลลัพธ์อยู่ที่ dist/
```
Windows ต้องมี [Inno Setup 6](https://jrsoftware.org/isinfo.php) (`choco install innosetup`) · `--no-package` = แพ็กและทดสอบแต่ยังไม่ห่อเป็นตัวติดตั้ง

### ไฟล์ในโฟลเดอร์ `installer/`
| ไฟล์ | หน้าที่ |
|---|---|
| `build.py` | ตัวควบคุม: แพ็ก → selftest → ห่อเป็นตัวติดตั้ง (+ เซ็น/notarize บน macOS ถ้าตั้งค่าไว้) |
| `inspection_station.spec` | PyInstaller: อะไรเข้าตัวแอป · `.app` + `NSCameraUsageDescription` บน macOS |
| `windows.iss` | ตัวติดตั้ง Windows (Inno Setup) — ภาษาอังกฤษ + ไทย (`compiler:Languages\Thai.isl` มากับ Inno Setup 6.5+) |
| `macos/` | ตัวติดตั้ง macOS: `distribution.xml`, หน้าต้อนรับ/สรุป `resources/{en,th}.lproj/*.html`, `scripts/postinstall` |
| `entitlements.plist` | สิทธิ์ของ hardened runtime (ใช้เมื่อเซ็นด้วย Developer ID) |
| `make_assets.py` → `assets/` | สร้างไอคอน/ภาพประกอบทั้งหมดจากโค้ด (ผลลัพธ์ commit ไว้แล้ว) |
| `requirements-build.txt` | PyInstaller, onnx + onnxruntime (Windows) |

### เซ็นรับรองและ notarize (macOS) — ให้หน้าต่างเตือนหายไป
1. สมัคร [Apple Developer Program](https://developer.apple.com/programs/) (ราว $99/ปี)
2. ที่ developer.apple.com → Certificates สร้าง **Developer ID Application** และ **Developer ID Installer** ส่งออกเป็น `.p12` (ใส่รหัสผ่านเดียวกัน)
3. ที่ App Store Connect → Users and Access → Integrations สร้าง **API key** (ดาวน์โหลด `.p8` ได้ครั้งเดียว จด Key ID กับ Issuer ID)
4. ใส่ GitHub Secrets (Settings → Secrets and variables → Actions):

| Secret | ค่า |
|---|---|
| `APPLE_CERT_APP_P12_BASE64` | `base64 -i app.p12` ของใบรับรอง Developer ID Application |
| `APPLE_CERT_INSTALLER_P12_BASE64` | เช่นเดียวกัน ของ Developer ID Installer |
| `APPLE_CERT_PASSWORD` | รหัสผ่านของไฟล์ `.p12` |
| `APPLE_API_KEY_P8` | เนื้อหาไฟล์ `.p8` ทั้งไฟล์ |
| `APPLE_API_KEY_ID`, `APPLE_API_ISSUER` | จากข้อ 3 |

พอมี `APPLE_CERT_APP_P12_BASE64` workflow จะนำเข้าใบรับรอง เซ็นแอปด้วย hardened runtime เซ็น `.pkg` ส่งให้ Apple ตรวจ (`notarytool`) และแนบผลกลับ (`stapler`) เอง — ส่วนนี้ **ยังไม่เคยรันจริง** (ยังไม่มีบัญชี) ครั้งแรกอาจต้องแก้เล็กน้อย
Windows: ถ้าต้องการให้ SmartScreen หายต้องซื้อใบรับรอง code signing (หรือ Azure Trusted Signing) แล้วเพิ่ม `SignTool` ใน `windows.iss`

## เรื่องที่ควรรู้ / ยังไม่ได้ทำ

- **Mac Intel ไม่รองรับ:** torch รุ่นสุดท้ายที่มี wheel สำหรับ Mac Intel คือ 2.2 ต้องแยกการสร้างและทดสอบต่างหาก
- **Windows ไม่ใช้ GPU NVIDIA:** torch ในตัวติดตั้งเป็นรุ่น CPU (รุ่น CUDA ทำให้ตัวติดตั้งใหญ่หลาย GB) ตรวจแก้วด้วย ONNX บน CPU แทน
- **ไม่มีการอัปเดตอัตโนมัติ:** รุ่นใหม่ = ติดตั้งทับตัวเดิม (Windows: `AppId` เดิม จึงอัปเกรดทับได้ · macOS: ลากทับ)
- **ตัวสร้างทดสอบ "ติดตั้งจริงแล้วรัน" ให้เอง** ทุกครั้ง: Windows ติดตั้งแบบเงียบเป็นภาษาไทย → รัน `--selftest` ของแอปที่ติดตั้ง → ถอนการติดตั้ง · macOS ติดตั้ง `.pkg` ด้วย `installer` → `codesign --verify` → ตรวจว่าไม่มีธง quarantine → รัน `--selftest` ของแอปที่ติดตั้ง
- **MediaPipe บน macOS:** รุ่น 1.0.1 abort ทั้งโปรเซสตอนสร้างตัวตรวจมือ (เจอบน Mac จริง macOS 26.2 และบน CI) จึงใช้ 0.10.21 เฉพาะ macOS ดู `app/requirements.txt` — CI ตรวจตัวตรวจมือในตัวติดตั้งทุกครั้งและล้มถ้าไม่ผ่าน
- **เครื่องเสมือน macOS ของ GitHub ไม่มี GPU:** selftest จึงใช้ CPU ส่วน MPS บน Mac จริงยังไม่ได้ตรวจ
- **ยังไม่มีใครตรวจ:** หน้าตาตัวติดตั้งจริงทั้งสองระบบ, หน้าต่างเตือนของ macOS/SmartScreen, หน้าต่างขอสิทธิ์กล้อง, ความเร็ว ONNX บน CPU ของเครื่อง Windows จริง
- **ก่อนแจกทีม sales ต้องลองบน Mac จริงหนึ่งเครื่อง:** ติดตั้งจาก `.pkg` → เปิดแอป → เห็นภาพกล้อง → มือถูกตรวจ (กรอบโครงมือขึ้น) → เปิด/ปิดโหมด zone ได้ ถ้าไม่ขึ้นหรือแอปดับ ส่ง `app.log` (ดูตารางไฟล์ของผู้ใช้) และรายงานการแครชใน Console.app → Crash Reports มาให้ดู
