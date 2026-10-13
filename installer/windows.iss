; Inno Setup 6.5+ - wraps dist\Cuptick (the PyInstaller output) into one installer.
;   iscc /DAppVersion=1.0.0 installer\windows.iss      (or: python installer/build.py)
; A standard wizard: language choice (English / Thai) -> welcome -> folder -> shortcuts -> install -> finish.
; Per-user install (no administrator rights). Running it again reinstalls cleanly: a running copy is closed, the old
; program files are removed first (nothing stale is left behind), and the installs under the former names
; 'TKK Workshop' and 'Inspection Station' (different AppIds) are uninstalled silently.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#define AppName "Cuptick"
#define AppExe "Cuptick.exe"
#define AppPublisher "Connected Tech"

[Setup]
AppId={{E67DD168-0483-44BD-BBD5-C53B5D28B157}
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
OutputBaseFilename=Cuptick-{#AppVersion}-win-x64-setup
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

[InstallDelete]
; Reinstall / upgrade over an existing copy: remove the old program files first so files that no longer exist stay gone.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\Cuptick\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
const
  // AppIds of the former product names (registry key = AppId + '_is1'): 'TKK Workshop' and before it 'Inspection Station'
  UninstallKeyPrefix = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\';
  LegacyTkkWorkshop = '{6E335DF4-E3A6-4923-8A83-FC1BF51C33C4}_is1';
  LegacyInspectionStation = '{7D3B2C1A-5E4F-4A8B-9C6D-1F2E3A4B5C6D}_is1';

function LegacyUninstaller(Id: String): String;
begin
  if not RegQueryStringValue(HKCU, UninstallKeyPrefix + Id, 'UninstallString', Result) then
    if not RegQueryStringValue(HKLM, UninstallKeyPrefix + Id, 'UninstallString', Result) then
      Result := '';
end;

// a running old copy would lock its files
procedure RemoveLegacy(Id, ExeName: String);
var
  Uninstaller: String;
  Code: Integer;
begin
  Uninstaller := LegacyUninstaller(Id);
  if Uninstaller <> '' then
  begin
    Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /IM ' + ExeName, '', SW_HIDE, ewWaitUntilTerminated, Code);
    Exec(RemoveQuotes(Uninstaller), '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART', '', SW_HIDE, ewWaitUntilTerminated, Code);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
  begin
    RemoveLegacy(LegacyTkkWorkshop, 'TKKWorkshop.exe');
    RemoveLegacy(LegacyInspectionStation, 'InspectionStation.exe');
  end;
end;
