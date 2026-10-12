# 05 — ตัวติดตั้ง (Windows / macOS)

ห่อแอปทั้งตัว (Python + torch + ultralytics + MediaPipe + Qt + โมเดล) ให้ทีม sales ติดตั้งแล้วใช้ได้เลย
ไม่ต้องลง Python ไม่ต้องใช้อินเทอร์เน็ต

| | Windows | macOS |
|---|---|---|
| ไฟล์ | `InspectionStation-<รุ่น>-win-x64-setup.exe` | `InspectionStation-<รุ่น>-macos-arm64.dmg` |
| เครื่องมือ | PyInstaller → Inno Setup | PyInstaller → dmgbuild |
| ชิป | x64 | **Apple Silicon (M1–M4) เท่านั้น** (ไม่มีรุ่น Intel) |
| ตรวจแก้ว | ONNX บน CPU (torch ในตัวติดตั้งเป็นรุ่น CPU) | torch + GPU ของชิป (MPS) |
| เซ็นรับรอง | ยังไม่เซ็น → SmartScreen เตือน | ยังไม่เซ็น → Gatekeeper บล็อกจนกด Open Anyway |

เครื่องพัฒนาเป็น Linux สร้างตัวติดตั้ง Windows/macOS เองไม่ได้ จึงสร้างบน **GitHub Actions**
(`.github/workflows/build-installers.yml`) ที่ใช้เครื่อง Windows/macOS จริง

## ติดตั้ง (สำหรับผู้ใช้)

**Windows** — ดับเบิลคลิก `.exe`
1. ถ้าขึ้นหน้าต่างสีน้ำเงิน "Windows protected your PC": กด **More info** → **Run anyway** (ตัวติดตั้งยังไม่ได้เซ็นรับรอง)
2. กด Next ตามหน้าจอ — ติดตั้งแบบรายผู้ใช้ ไม่ต้องมีสิทธิ์ admin ลงที่ `%LOCALAPPDATA%\Programs\Inspection Station`
3. เปิดจากไอคอนบนเดสก์ท็อป ครั้งแรกกด Allow เมื่อ Windows ถามสิทธิ์กล้อง

**macOS** — เปิด `.dmg` แล้วลากไอคอนไปวางที่ **Applications**
1. เปิดแอปจาก Launchpad — ถ้า macOS บอกว่าเปิดไม่ได้: **ตั้งค่าระบบ → ความเป็นส่วนตัวและความปลอดภัย** เลื่อนลงล่าง กด **Open Anyway** แล้วเปิดอีกครั้ง
2. หรือใช้ Terminal: `xattr -dr com.apple.quarantine "/Applications/Inspection Station.app"`
3. กด **อนุญาต** เมื่อ macOS ถามสิทธิ์กล้อง (ถ้าพลาด: ตั้งค่าระบบ → ความเป็นส่วนตัวและความปลอดภัย → กล้อง)

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

### ครั้งแรก: แนบโมเดลเป็น Release (ทำครั้งเดียว)
ตัวสร้างบน GitHub ไม่มีไฟล์โมเดล (ไม่อยู่ใน git) จึงดึงจาก Release `models-v2` (v1 ดึงจาก Release `v1` ที่มีอยู่แล้ว):
```bash
gh release create models-v2 app/models/cup_v2.pt app/models/cup_v2.onnx \
  --title "cup_v2 weights (for the installer build)" --notes "Weights bundled into the installers. Not an app release."
```
⚠️ repo เป็น public — **ทุกคนดาวน์โหลดโมเดลได้** คำสั่งนี้เผยแพร่ไฟล์ จึงรันเมื่อพร้อมเท่านั้น

### สร้างบน GitHub Actions
- **ลองสร้าง:** Actions → *Build installers* → *Run workflow* → ดาวน์โหลดไฟล์จาก **Artifacts** ของ run นั้น
- **ปล่อยจริง:** `git tag app-v1.0.0 && git push origin app-v1.0.0` → ตัวติดตั้งทั้งสองแนบกับ Release `app-v1.0.0` (เปลี่ยนเลขรุ่นที่ไฟล์ `VERSION` ก่อน)

ในตัวสร้างทำตามนี้ — ถ้าขั้นไหนพังจะไม่ได้ไฟล์:
`pip install` → ดึงโมเดล → `pyinstaller` → **`--selftest` กับตัวที่แพ็กแล้ว** (โหลดโมเดล + MediaPipe + วาดเฟรม + สร้างวิดเจ็ต Qt) → Inno Setup / dmgbuild

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
| `build.py` | ตัวควบคุม: แพ็ก → selftest → ห่อเป็นตัวติดตั้ง |
| `inspection_station.spec` | PyInstaller: อะไรเข้าตัวแอป · `.app` + `NSCameraUsageDescription` บน macOS |
| `windows.iss`, `windows-notes.txt` | หน้าตาและข้อความตัวติดตั้ง Windows (UTF-8 มี BOM — ห้ามบันทึกซ้ำเป็นไม่มี BOM) |
| `dmg_settings.py` | หน้าต่าง `.dmg` |
| `make_assets.py` → `assets/` | สร้างไอคอน/ภาพประกอบทั้งหมดจากโค้ด (ผลลัพธ์ commit ไว้แล้ว) |
| `requirements-build.txt` | PyInstaller, dmgbuild, onnx + onnxruntime (Windows) |

## เรื่องที่ควรรู้ / ยังไม่ได้ทำ

- **ยังไม่เซ็นรับรอง:** ถ้ามีใบรับรอง Windows หรือ Apple Developer ID ให้เพิ่มขั้นเซ็นใน `windows.iss` (`SignTool`) และใน `inspection_station.spec` (`codesign_identity`, `entitlements_file`) แล้ว notarize `.dmg` — เมื่อเซ็นแล้วคำเตือนข้างต้นหายไป
- **Mac Intel ไม่รองรับ:** torch รุ่นสุดท้ายที่มี wheel สำหรับ Mac Intel คือ 2.2 ต้องแยกการสร้างและทดสอบต่างหาก
- **Windows ไม่ใช้ GPU NVIDIA:** torch ในตัวติดตั้งเป็นรุ่น CPU (รุ่น CUDA ทำให้ตัวติดตั้งใหญ่หลาย GB) ตรวจแก้วด้วย ONNX บน CPU แทน
- **ไม่มีการอัปเดตอัตโนมัติ:** รุ่นใหม่ = ติดตั้งทับตัวเดิม (Windows: `AppId` เดิม จึงอัปเกรดทับได้ · macOS: ลากทับ)
- **สถานะการทดสอบ:** แพ็กและ selftest ทดสอบบน Linux เป็นตัวแทนเท่านั้น ตัวสร้างบน Windows/macOS (Inno Setup, dmgbuild, หน้าตาตัวติดตั้ง, การขอสิทธิ์กล้อง) ต้องรันบน GitHub Actions และเครื่องจริงถึงจะรู้ผล
