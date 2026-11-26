<#
.SYNOPSIS
    Registers (or unregisters) the Genshin Helper Native Messaging Host (com.genshinhelper.host)
    for local dev / Cargo target testing (directly targeting GenshinHelperHost.exe and GenshinHelper.exe).

.DESCRIPTION
    - Resolves GenshinHelperHost.exe and GenshinHelper.exe via:
      1. -HostExe / -ServerExe (or $env:GH_HOST_EXE / $env:GH_SERVER_EXE)
      2. Cargo target directory ($env:CARGO_TARGET_DIR or `cargo metadata` target_directory)\$BuildType\
      3. dist\GenshinHelper\
    - Derives the pinned Extension ID from extension/manifest.json via .NET SHA256.
    - Removes any legacy native_host_dev.cmd wrapper and writes com.genshinhelper.host.json
      pointing directly to GenshinHelperHost.exe in %LOCALAPPDATA%\GenshinHelper\dev_native_host\.
    - Registers HKCU NativeMessagingHosts keys for Chrome, Microsoft Edge, and Brave (no Admin needed).
    - Optional -WithElevatedTask (requires Administrator): creates the GenshinHelper_Elevated
      Scheduled Task pointing to GenshinHelper.exe --minimized with -WorkingDirectory $RootDir.
    - Runs a pure-.NET 4-byte LE stdio {"cmd":"ping"} smoke check against GenshinHelperHost.exe.
    - Optional -Uninstall: removes HKCU keys, dev_native_host files, and GenshinHelper_Elevated task.

.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/register_dev_native_host.ps1
.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/register_dev_native_host.ps1 -BuildType debug
.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/register_dev_native_host.ps1 -WithElevatedTask
.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/register_dev_native_host.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$WithElevatedTask,
    [switch]$Uninstall,
    [string]$HostExe,
    [string]$ServerExe,
    [ValidateSet("release", "debug")]
    [string]$BuildType = "release"
)

$ErrorActionPreference = "Stop"

$HostName = "com.genshinhelper.host"
$TaskName = "GenshinHelper_Elevated"
$RootDir = Split-Path -Parent $PSScriptRoot
$ExtManifestPath = Join-Path $RootDir "extension\manifest.json"
$RustManifestPath = Join-Path $RootDir "rust\Cargo.toml"

$DevHostDir = Join-Path $env:LOCALAPPDATA "GenshinHelper\dev_native_host"
$LegacyWrapperCmd = Join-Path $DevHostDir "native_host_dev.cmd"
$ManifestJsonPath = Join-Path $DevHostDir "$HostName.json"

$RegPaths = @(
    "HKCU:\Software\Google\Chrome\NativeMessagingHosts\$HostName",
    "HKCU:\Software\Microsoft\Edge\NativeMessagingHosts\$HostName",
    "HKCU:\Software\BraveSoftware\Brave-Browser\NativeMessagingHosts\$HostName"
)

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-CargoTargetDir {
    if ($env:CARGO_TARGET_DIR -and $env:CARGO_TARGET_DIR.Trim()) {
        return $env:CARGO_TARGET_DIR.Trim()
    }
    $cargoCmd = Get-Command "cargo" -ErrorAction SilentlyContinue
    if ($cargoCmd -and (Test-Path -LiteralPath $RustManifestPath)) {
        $metaJson = & $cargoCmd.Source metadata --manifest-path $RustManifestPath --no-deps --format-version 1 2>$null
        if ($LASTEXITCODE -eq 0 -and $metaJson) {
            $meta = $metaJson | ConvertFrom-Json
            if ($meta.target_directory) {
                return [string]$meta.target_directory
            }
        }
    }
    return $null
}

function Resolve-BinaryPath {
    param(
        [string]$ParamValue,
        [string]$EnvValue,
        [string]$ExeName,
        [string]$TargetDir,
        [string]$BuildProfile,
        [string]$RepoRoot
    )

    if ($ParamValue -and $ParamValue.Trim()) {
        $full = [System.IO.Path]::GetFullPath($ParamValue.Trim())
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            throw "Specified $ExeName does not exist: $full"
        }
        return $full
    }
    if ($EnvValue -and $EnvValue.Trim()) {
        $full = [System.IO.Path]::GetFullPath($EnvValue.Trim())
        if (-not (Test-Path -LiteralPath $full -PathType Leaf)) {
            throw "Environment-specified $ExeName does not exist: $full"
        }
        return $full
    }

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($TargetDir) {
        $candidates.Add((Join-Path (Join-Path $TargetDir $BuildProfile) $ExeName))
    }
    $candidates.Add((Join-Path $RepoRoot "dist\GenshinHelper\$ExeName"))

    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c -PathType Leaf) {
            return [System.IO.Path]::GetFullPath($c)
        }
    }
    return $null
}

if ($Uninstall) {
    Write-Host "[Uninstall] Removing dev Native Messaging Host registration..." -ForegroundColor Yellow
    foreach ($rp in $RegPaths) {
        if (Test-Path $rp) {
            Remove-Item -Path $rp -Force
            Write-Host "  Removed registry key: $rp"
        }
    }
    if (Test-Path $DevHostDir) {
        Remove-Item -Path $DevHostDir -Recurse -Force
        Write-Host "  Removed directory: $DevHostDir"
    }
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        if (Test-IsAdmin) {
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Host "  Removed Scheduled Task: $TaskName"
        } else {
            Write-Warning "Scheduled Task '$TaskName' still exists; re-run with Administrator privileges to remove it."
        }
    }
    Write-Host "[OK] Dev Native Messaging Host uninstalled." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path -LiteralPath $ExtManifestPath -PathType Leaf)) {
    throw "Missing extension/manifest.json at: $ExtManifestPath"
}

$CargoTargetDir = Resolve-CargoTargetDir
$ResolvedHostExe = Resolve-BinaryPath `
    -ParamValue $HostExe `
    -EnvValue $env:GH_HOST_EXE `
    -ExeName "GenshinHelperHost.exe" `
    -TargetDir $CargoTargetDir `
    -BuildProfile $BuildType `
    -RepoRoot $RootDir

if (-not $ResolvedHostExe) {
    throw "Unable to locate GenshinHelperHost.exe (checked -HostExe, GH_HOST_EXE, Cargo target '$BuildType', and dist\GenshinHelper). Build first with: cargo build --manifest-path rust/Cargo.toml --release -p gh-host -p gh-app"
}

$ResolvedServerExe = Resolve-BinaryPath `
    -ParamValue $ServerExe `
    -EnvValue $env:GH_SERVER_EXE `
    -ExeName "GenshinHelper.exe" `
    -TargetDir $CargoTargetDir `
    -BuildProfile $BuildType `
    -RepoRoot $RootDir

# Derive pinned Extension ID from extension/manifest.json public key using pure .NET
$ExtManifest = Get-Content -LiteralPath $ExtManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $ExtManifest.key) {
    throw "Missing 'key' field in $ExtManifestPath"
}
$DerBytes = [System.Convert]::FromBase64String([string]$ExtManifest.key)
$Sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $HashBytes = $Sha256.ComputeHash($DerBytes)
} finally {
    $Sha256.Dispose()
}
$ExtIdSb = New-Object System.Text.StringBuilder
for ($i = 0; $i -lt 16; $i++) {
    $b = $HashBytes[$i]
    [void]$ExtIdSb.Append([char](97 + ($b -shr 4)))
    [void]$ExtIdSb.Append([char](97 + ($b -band 15)))
}
$ExtId = $ExtIdSb.ToString()
if ($ExtId.Length -ne 32) {
    throw "Failed to derive pinned Extension ID from $ExtManifestPath (got '$ExtId')"
}

New-Item -ItemType Directory -Path $DevHostDir -Force | Out-Null

# Delete stale Python .cmd wrapper if present
if (Test-Path -LiteralPath $LegacyWrapperCmd) {
    Remove-Item -LiteralPath $LegacyWrapperCmd -Force
    Write-Host "  Removed stale wrapper: $LegacyWrapperCmd" -ForegroundColor DarkGray
}

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$ManifestObj = [ordered]@{
    name = $HostName
    description = "Genshin Helper Dev Native Messaging Host (Wake & Control Desktop Sync Server)"
    path = $ResolvedHostExe
    type = "stdio"
    allowed_origins = @(
        "chrome-extension://$ExtId/"
    )
}
$ManifestJson = ($ManifestObj | ConvertTo-Json -Depth 4) + "`r`n"
[System.IO.File]::WriteAllText($ManifestJsonPath, $ManifestJson, $Utf8NoBom)

foreach ($rp in $RegPaths) {
    New-Item -Path $rp -Force | Out-Null
    Set-ItemProperty -Path $rp -Name "(Default)" -Value $ManifestJsonPath
    Write-Host "  Registered: $rp" -ForegroundColor Cyan
}

if ($WithElevatedTask) {
    if (-not $ResolvedServerExe) {
        throw "Unable to locate GenshinHelper.exe for -WithElevatedTask (checked -ServerExe, GH_SERVER_EXE, Cargo target '$BuildType', and dist\GenshinHelper)."
    }
    if (-not (Test-IsAdmin)) {
        Write-Warning "Skipping Elevated Scheduled Task '$TaskName': please re-run this script as Administrator with -WithElevatedTask."
    } else {
        $Action = New-ScheduledTaskAction -Execute $ResolvedServerExe -Argument "--minimized" -WorkingDirectory $RootDir
        $Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
        $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 0)
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Principal $Principal -Settings $Settings -Force | Out-Null
        Write-Host "  Registered Elevated Scheduled Task: $TaskName -> `"$ResolvedServerExe`" --minimized (cwd: $RootDir)" -ForegroundColor Green
    }
}

# Smoke-test 4-byte LE Native Messaging stdio ping against GenshinHelperHost.exe via pure .NET
$Psi = New-Object System.Diagnostics.ProcessStartInfo
$Psi.FileName = $ResolvedHostExe
$Psi.UseShellExecute = $false
$Psi.RedirectStandardInput = $true
$Psi.RedirectStandardOutput = $true
$Psi.RedirectStandardError = $true
$Psi.CreateNoWindow = $true

$Proc = New-Object System.Diagnostics.Process
$Proc.StartInfo = $Psi
if (-not $Proc.Start()) {
    throw "Failed to start Native Host executable: $ResolvedHostExe"
}

try {
    $PayloadBytes = $Utf8NoBom.GetBytes('{"cmd":"ping"}')
    $Writer = New-Object System.IO.BinaryWriter($Proc.StandardInput.BaseStream, $Utf8NoBom, $true)
    try {
        $Writer.Write([uint32]$PayloadBytes.Length)
        $Writer.Write($PayloadBytes)
        $Writer.Flush()
    } finally {
        $Writer.Dispose()
    }
    $Proc.StandardInput.Close()

    $Reader = New-Object System.IO.BinaryReader($Proc.StandardOutput.BaseStream, $Utf8NoBom, $true)
    try {
        $LenBytes = $Reader.ReadBytes(4)
        if ($LenBytes.Length -ne 4) {
            $ErrText = $Proc.StandardError.ReadToEnd()
            throw "Short Native Messaging length header ($($LenBytes.Length) bytes). stderr=$ErrText"
        }
        $RespLen = [BitConverter]::ToUInt32($LenBytes, 0)
        if ($RespLen -eq 0 -or $RespLen -gt 1048576) {
            throw "Invalid Native Messaging response length: $RespLen"
        }
        $RespBytes = $Reader.ReadBytes([int]$RespLen)
        if ($RespBytes.Length -ne [int]$RespLen) {
            $ErrText = $Proc.StandardError.ReadToEnd()
            throw "Truncated Native Messaging payload ($($RespBytes.Length)/$RespLen bytes). stderr=$ErrText"
        }
    } finally {
        $Reader.Dispose()
    }

    if (-not $Proc.WaitForExit(5000)) {
        $Proc.Kill()
        throw "Native Host ping process timed out after 5000 ms."
    }
    if ($Proc.ExitCode -ne 0) {
        $ErrText = $Proc.StandardError.ReadToEnd()
        throw "Native Host ping exited with code $($Proc.ExitCode). stderr=$ErrText"
    }

    $RespJson = $Utf8NoBom.GetString($RespBytes)
    $RespObj = $RespJson | ConvertFrom-Json
    if ($RespObj.ok -ne $true) {
        throw "Native Host ping returned non-ok response: $RespJson"
    }
    Write-Host "  NATIVE_HOST_PING_OK: $RespJson" -ForegroundColor Green
} finally {
    $Proc.Dispose()
}

Write-Host "[OK] Dev Native Messaging Host ready (Extension ID: $ExtId)." -ForegroundColor Green
Write-Host "     Manifest  : $ManifestJsonPath"
Write-Host "     Host EXE  : $ResolvedHostExe"
if ($ResolvedServerExe) {
    Write-Host "     Server EXE: $ResolvedServerExe"
} else {
    Write-Host "     Server EXE: (not found; build gh-app or pass -ServerExe before -WithElevatedTask)" -ForegroundColor Yellow
}
