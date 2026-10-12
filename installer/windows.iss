; Inno Setup 6 — ห่อ dist\InspectionStation (ผลลัพธ์ PyInstaller) เป็นตัวติดตั้งไฟล์เดียว
;   iscc /DAppVersion=1.0.0 installer\windows.iss     (หรือ python installer/build.py)
; ติดตั้งแบบรายผู้ใช้ (ไม่ต้องมีสิทธิ์ admin) ลง %LOCALAPPDATA%\Programs\Inspection Station
; ไฟล์นี้ต้องเป็น UTF-8 มี BOM — ไม่งั้นข้อความไทยในหน้าตัวติดตั้งจะเพี้ยน

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "Inspection Station"
#define AppExe "InspectionStation.exe"

[Setup]
AppId={{7D3B2C1A-5E4F-4A8B-9C6D-1F2E3A4B5C6D}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Connected Tech
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=InspectionStation-{#AppVersion}-win-x64-setup
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
WizardImageFile=assets\wizard_large.bmp,assets\wizard_large_2x.bmp
WizardSmallImageFile=assets\wizard_small.bmp,assets\wizard_small_2x.bmp
InfoBeforeFile=windows-notes.txt
ShowLanguageDialog=no
Compression=lzma2/max
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
WelcomeLabel1=ยินดีต้อนรับสู่ [name]
WelcomeLabel2=ตัวติดตั้งนี้จะติดตั้ง [name/ver] ลงในเครื่องของคุณ รวมทุกอย่างที่ต้องใช้แล้ว (โมเดลตรวจจับแก้วและมือ) ไม่ต้องติดตั้งอะไรเพิ่มและไม่ต้องใช้อินเทอร์เน็ต%n%nThis installs [name/ver] with everything it needs, including the detection models. No extra software and no internet connection required.%n%nแนะนำให้ปิดโปรแกรมอื่นที่ใช้กล้องก่อนเริ่ม
FinishedHeadingLabel=ติดตั้ง [name] เรียบร้อยแล้ว
FinishedLabel=เปิดแอปได้จากไอคอนบนเดสก์ท็อปหรือ Start menu%n%nครั้งแรกที่เปิด Windows อาจถามสิทธิ์ใช้กล้อง — กด Allow

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\InspectionStation\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
