<#
.SYNOPSIS
  Create a desktop shortcut that opens the AudtoText window.

.DESCRIPTION
  Double-click the shortcut to open the window, or drop audio files onto it:
  Windows passes dropped files to the app, which queues them (or hands them
  to the window that is already open).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\make_shortcut.ps1 `
      -Python D:\transcribe_workspace\env\pythonw.exe `
      -ModelCacheDir D:\transcribe_workspace\models
#>
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$ModelCacheDir = "",
    [string]$Name = "AudtoText"
)
$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$gui = Join-Path $repo "audtotext_gui.py"
$icon = Join-Path $repo "assets\audtotext.ico"
$Python = (Resolve-Path $Python).Path
foreach ($path in @($gui, $icon)) {
    if (-not (Test-Path $path)) { throw "Not found: $path" }
}

$arguments = "`"$gui`""
if ($ModelCacheDir) {
    # No trailing backslash: it would escape the closing quote in argv.
    $cache = (Resolve-Path $ModelCacheDir).Path.TrimEnd('\')
    $arguments += " --model-cache-dir `"$cache`""
}

$link = Join-Path ([Environment]::GetFolderPath("Desktop")) "$Name.lnk"
$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($link)
$shortcut.TargetPath = $Python
$shortcut.Arguments = $arguments
$shortcut.WorkingDirectory = $repo
$shortcut.IconLocation = "$icon,0"
$shortcut.Description = "Drop audio files to transcribe them with AudtoText"
$shortcut.Save()
Write-Output "Created $link"
Write-Output "  target:    $Python $arguments"
