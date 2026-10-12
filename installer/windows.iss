; Inno Setup 6.5+ - wraps dist\TKKWorkshop (the PyInstaller output) into one installer.
;   iscc /DAppVersion=1.0.0 installer\windows.iss      (or: python installer/build.py)
; A standard wizard: language choice (English / Thai) -> welcome -> folder -> shortcuts -> install -> finish.
; Per-user install (no administrator rights). Running it again reinstalls cleanly: a running copy is closed, the old
; program files are removed first (nothing stale is left behind), and the earlier 'Inspection Station' install
; (the product's former name, different AppId) is uninstalled silently.

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

[InstallDelete]
; Reinstall / upgrade over an existing copy: remove the old program files first so files that no longer exist stay gone.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\TKKWorkshop\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
const
  // AppId of the former product name 'Inspection Station' (registry key = AppId + '_is1')
  LegacyUninstallKey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{7D3B2C1A-5E4F-4A8B-9C6D-1F2E3A4B5C6D}_is1';

function LegacyUninstaller(): String;
begin
  if not RegQueryStringValue(HKCU, LegacyUninstallKey, 'UninstallString', Result) then
    if not RegQueryStringValue(HKLM, LegacyUninstallKey, 'UninstallString', Result) then
      Result := '';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Uninstaller: String;
  Code: Integer;
begin
  if CurStep = ssInstall then
  begin
    Uninstaller := LegacyUninstaller();
    if Uninstaller <> '' then
    begin
      // a running old copy would lock its files
      Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /IM InspectionStation.exe', '', SW_HIDE, ewWaitUntilTerminated, Code);
      Exec(RemoveQuotes(Uninstaller), '/VERYSILENT /SUPPRESSMSGBOXES /NORESTART', '', SW_HIDE, ewWaitUntilTerminated, Code);
    end;
  end;
end;
