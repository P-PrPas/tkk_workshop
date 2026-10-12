; Inno Setup 6.5+ - wraps dist\TKKWorkshop (the PyInstaller output) into one installer.
;   iscc /DAppVersion=1.0.0 installer\windows.iss      (or: python installer/build.py)
; A standard wizard: language choice (English / Thai) -> welcome -> folder -> shortcuts -> install -> finish.
; Per-user install (no administrator rights), upgrades in place (same AppId), closes a running copy first.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "TKK Workshop"
#define AppExe "TKKWorkshop.exe"
#define AppPublisher "Connected Tech"

[Setup]
AppId={{6E335DF4-E3A6-4923-8A83-FC1BF51C33C4}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoProductName={#AppName}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
OutputDir=..\dist
OutputBaseFilename=TKKWorkshop-{#AppVersion}-win-x64-setup
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
WizardImageFile=assets\wizard_large.bmp,assets\wizard_large_2x.bmp
WizardSmallImageFile=assets\wizard_small.bmp,assets\wizard_small_2x.bmp
ShowLanguageDialog=yes
UsePreviousLanguage=no
Compression=lzma2/max
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "thai"; MessagesFile: "compiler:Languages\Thai.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\TKKWorkshop\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
