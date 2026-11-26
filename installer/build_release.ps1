<#
.SYNOPSIS
    Builds Genshin Helper pure-Rust Windows binaries (gh-app + gh-host) and Setup Installer (Inno Setup 6) with SHA-256 checksums.
#>
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if ($env:PYTHON_EXE -and (Test-Path $env:PYTHON_EXE)) {
    $PyExe = $env:PYTHON_EXE
} elseif (Test-Path (Join-Path $RootDir ".venv\Scripts\python.exe")) {
    $PyExe = Join-Path $RootDir ".venv\Scripts\python.exe"
} else {
    $PyCmd = Get-Command "python" -ErrorAction SilentlyContinue
    if (-not $PyCmd) {
        throw "Missing Python executable (set PYTHON_EXE or create .venv)"
    }
    $PyExe = $PyCmd.Source
}

Write-Host "[1/5] Verifying Extension Key <-> Native Host ID <-> App/Cargo/Installer Version consistency..." -ForegroundColor Cyan
$AppVersion = (& $PyExe -c @"
import base64, hashlib, json, re

m = json.load(open('extension/manifest.json', encoding='utf-8'))
ver = str(m.get('version', '')).strip()
assert ver, 'Missing version in extension/manifest.json'

cargo_txt = open('rust/Cargo.toml', encoding='utf-8').read()
cargo_match = re.search(r'(?ms)^\[workspace\.package\]\s.*?^version\s*=\s*"([^"]+)"', cargo_txt)
assert cargo_match and cargo_match.group(1) == ver, (
    f'rust/Cargo.toml workspace version {cargo_match.group(1) if cargo_match else None} != {ver}'
)

lib_txt = open('rust/crates/gh-core/src/lib.rs', encoding='utf-8').read()
app_match = re.search(r'pub\s+const\s+APP_VERSION\s*:\s*&str\s*=\s*"([^"]+)"', lib_txt)
host_match = re.search(r'pub\s+const\s+HOST_VERSION\s*:\s*&str\s*=\s*"([^"]+)"', lib_txt)
assert app_match and app_match.group(1) == ver, (
    f'gh-core APP_VERSION {app_match.group(1) if app_match else None} != {ver}'
)
assert host_match and host_match.group(1) == ver, (
    f'gh-core HOST_VERSION {host_match.group(1) if host_match else None} != {ver}'
)

der = base64.b64decode(m['key'])
ext_id = ''.join(chr(97 + (b >> 4)) + chr(97 + (b & 15)) for b in hashlib.sha256(der).digest()[:16])
nh = json.load(open('installer/native_host_manifest.json', encoding='utf-8'))
expected_origin = f'chrome-extension://{ext_id}/'
assert expected_origin in nh['allowed_origins'], f'{expected_origin} not in {nh}'

iss = open('installer/GenshinHelper.iss', encoding='utf-8').read()
assert ext_id in iss, f'{ext_id} missing from GenshinHelper.iss'
match = re.search(r'#define\s+MyAppVersion\s+"([^"]+)"', iss)
assert match and match.group(1) == ver, f'Inno version {match.group(1) if match else None} != {ver}'
print(ver)
"@).Trim()
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "  Version & pinned Extension ID verified: v$AppVersion" -ForegroundColor Green

Write-Host "[2/5] Checking Rust toolchain (cargo & rustc) and resolving TargetDir..." -ForegroundColor Cyan
$CargoCmd = Get-Command "cargo" -ErrorAction SilentlyContinue
$RustcCmd = Get-Command "rustc" -ErrorAction SilentlyContinue
if (-not $CargoCmd -or -not $RustcCmd) {
    throw "Missing Rust toolchain (cargo and rustc must be on PATH)."
}
$RustManifest = Join-Path $RootDir "rust\Cargo.toml"
if ($env:CARGO_TARGET_DIR -and $env:CARGO_TARGET_DIR.Trim()) {
    $TargetDir = $env:CARGO_TARGET_DIR.Trim()
} else {
    $MetaJson = & $CargoCmd.Source metadata --manifest-path $RustManifest --no-deps --format-version 1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $TargetDir = [string](($MetaJson | ConvertFrom-Json).target_directory)
}
if (-not $TargetDir) {
    throw "Failed to resolve Cargo target_directory."
}
Write-Host "  Cargo target directory: $TargetDir" -ForegroundColor DarkGray

Write-Host "[3/5] Building release binaries (gh-app + gh-host) with IP-clean path remapping & +crt-static..." -ForegroundColor Cyan
$CargoHome = if ($env:CARGO_HOME -and $env:CARGO_HOME.Trim()) { $env:CARGO_HOME.Trim() } else { Join-Path $env:USERPROFILE ".cargo" }
$RustupHome = if ($env:RUSTUP_HOME -and $env:RUSTUP_HOME.Trim()) { $env:RUSTUP_HOME.Trim() } else { Join-Path $env:USERPROFILE ".rustup" }

$FlagList = New-Object System.Collections.Generic.List[string]
$FlagList.Add("--remap-path-prefix=$RootDir=/src")
$FlagList.Add("--remap-path-prefix=$CargoHome=/cargo")
$FlagList.Add("--remap-path-prefix=$RustupHome=/rustup")
if ($env:USERPROFILE -and $env:USERPROFILE.Trim()) {
    $FlagList.Add("--remap-path-prefix=$($env:USERPROFILE.Trim())=/user")
}
$FlagList.Add("-C")
$FlagList.Add("link-arg=/PDBALTPATH:%_PDB%")
$FlagList.Add("-C")
$FlagList.Add("target-feature=+crt-static")

$PrevEncodedRustflags = $env:CARGO_ENCODED_RUSTFLAGS
try {
    $env:CARGO_ENCODED_RUSTFLAGS = $FlagList -join ([char]0x1F)
    & $CargoCmd.Source build --manifest-path $RustManifest --release -p gh-app -p gh-host
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    if ($null -eq $PrevEncodedRustflags) {
        Remove-Item Env:\CARGO_ENCODED_RUSTFLAGS -ErrorAction SilentlyContinue
    } else {
        $env:CARGO_ENCODED_RUSTFLAGS = $PrevEncodedRustflags
    }
}

if (Test-Path (Join-Path $RootDir "rust\target")) {
    throw "Unexpected rust\target directory created; Cargo target-dir hygiene violated."
}

$ReleaseDir = Join-Path $TargetDir "release"
$AppExe = Join-Path $ReleaseDir "GenshinHelper.exe"
$HostExe = Join-Path $ReleaseDir "GenshinHelperHost.exe"
if (-not (Test-Path -LiteralPath $AppExe -PathType Leaf)) {
    throw "Missing built binary: $AppExe"
}
if (-not (Test-Path -LiteralPath $HostExe -PathType Leaf)) {
    throw "Missing built binary: $HostExe"
}

Write-Host "[4/5] Staging dist\GenshinHelper, bundling Cargo licenses, and auditing..." -ForegroundColor Cyan
$DistAppDir = Join-Path $RootDir "dist\GenshinHelper"
if (Test-Path -LiteralPath $DistAppDir) {
    Remove-Item -LiteralPath $DistAppDir -Recurse -Force
}
New-Item -ItemType Directory -Path $DistAppDir -Force | Out-Null

Copy-Item -LiteralPath $AppExe, $HostExe, (Join-Path $RootDir "LICENSE"), (Join-Path $RootDir "THIRD_PARTY_NOTICES.md") -Destination $DistAppDir -Force

& $PyExe scripts/collect_licenses.py --cargo-manifest rust/Cargo.toml --output dist/GenshinHelper/THIRD_PARTY_LICENSES.md --copy-dir dist/GenshinHelper/licenses --strict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ForbiddenInternal = Get-ChildItem -LiteralPath $DistAppDir -Recurse -Directory -Filter "_internal"
if ($ForbiddenInternal) {
    throw "Unexpected legacy _internal directory found in dist\GenshinHelper"
}
$ForbiddenFiles = Get-ChildItem -LiteralPath $DistAppDir -Recurse -File | Where-Object {
    ($_.Extension.ToLowerInvariant() -in @(".dll", ".pyd", ".pdb", ".npz")) -or
    ($_.Name.ToLowerInvariant() -eq "paimon_master.png")
}
if ($ForbiddenFiles) {
    $Names = ($ForbiddenFiles | ForEach-Object { $_.FullName }) -join ", "
    throw "Forbidden artifact file(s) found in dist\GenshinHelper: $Names"
}

& $PyExe scripts/audit_release.py --strict-ip --artifact dist/GenshinHelper
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[5/5] Locating Inno Setup 6 (ISCC.exe)..." -ForegroundColor Cyan
$RegIscc = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
    "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1"
) | ForEach-Object {
    if (Test-Path $_) {
        $loc = (Get-ItemProperty -Path $_ -ErrorAction SilentlyContinue).InstallLocation
        if ($loc) { Join-Path $loc "ISCC.exe" }
    }
}

$IsccCandidates = @(
    (Get-Command "iscc.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
) + $RegIscc | Where-Object { $_ -and (Test-Path $_) }

$Iscc = $IsccCandidates | Select-Object -First 1
if (-not $Iscc) {
    throw "ISCC.exe (Inno Setup 6) not found. Install Inno Setup 6 to build the Setup executable."
}

& $Iscc (Join-Path $PSScriptRoot "GenshinHelper.iss")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$SetupExe = Join-Path $RootDir "dist\installer\GenshinHelper-Setup-v$AppVersion.exe"
if (-not (Test-Path $SetupExe)) {
    throw "Expected installer output missing: $SetupExe"
}

$Hash = (Get-FileHash -Path $SetupExe -Algorithm SHA256).Hash.ToLowerInvariant()
$Leaf = Split-Path -Leaf $SetupExe
"$Hash  $Leaf" | Out-File -FilePath "$SetupExe.sha256" -Encoding utf8
"$Hash  $Leaf" | Out-File -FilePath (Join-Path $RootDir "dist\installer\SHA256SUMS.txt") -Encoding utf8
Write-Host "BUILD COMPLETE: $SetupExe" -ForegroundColor Green
Write-Host "SHA-256: $Hash" -ForegroundColor Green
