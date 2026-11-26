//! UIPI elevation checks, HoYoPlay registry lookup, borderless launch, and
//! `SetWindowDisplayAffinity` helpers (`src/game_settings.py`).

use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use std::io::Read;
use std::mem::size_of;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{
    CloseHandle, ERROR_ACCESS_DENIED, GetLastError, HANDLE, HWND,
};
use windows::Win32::Security::{
    GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation,
};
use windows::Win32::System::Registry::{
    HKEY, HKEY_CURRENT_USER, KEY_READ, REG_EXPAND_SZ, REG_SZ, REG_VALUE_TYPE, RegCloseKey,
    RegOpenKeyExW, RegQueryValueExW,
};
use windows::Win32::System::Threading::{
    GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::Shell::ShellExecuteW;
use windows::Win32::UI::WindowsAndMessaging::{
    GetSystemMetrics, GetWindowDisplayAffinity, GetWindowThreadProcessId, IsWindow, SM_CXSCREEN,
    SM_CYSCREEN, SW_SHOWNORMAL, SetWindowDisplayAffinity, WINDOW_DISPLAY_AFFINITY,
};
use windows::core::{HRESULT, PCWSTR, w};

pub use gh_core::ui_layout::{
    AspectLabel, HdrClippingReport, UiLayout, build_borderless_launch_args, compute_ui_layout,
    detect_hdr_clipping,
};

const DETACHED_PROCESS: u32 = 0x0000_0008;
const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const SCHTASKS_TIMEOUT: Duration = Duration::from_secs(3);

/// Ordered HoYoPlay HKCU registry subkeys and candidate executable names (`src/game_settings.py`).
pub const HOYOPLAY_REGISTRY_CANDIDATES: &[(&str, &[&str])] = &[
    (
        r"Software\Cognosphere\HYP\1_0\hk4e_global",
        &["GenshinImpact.exe", "YuanShen.exe"],
    ),
    (
        r"Software\Cognosphere\HYP\1_0\hk4e_cn",
        &["YuanShen.exe", "GenshinImpact.exe"],
    ),
    (
        r"Software\miHoYo\HYP\1_0\hk4e_cn",
        &["YuanShen.exe", "GenshinImpact.exe"],
    ),
];

struct HandleGuard(HANDLE);

impl Drop for HandleGuard {
    fn drop(&mut self) {
        if !self.0.is_invalid() {
            // SAFETY: `self.0` is a valid owned Win32 kernel handle.
            unsafe {
                let _ = CloseHandle(self.0);
            }
            self.0 = HANDLE::default();
        }
    }
}

struct HkeyGuard(HKEY);

impl Drop for HkeyGuard {
    fn drop(&mut self) {
        if !self.0.is_invalid() {
            // SAFETY: `self.0` is a valid opened Win32 registry key handle.
            unsafe {
                let _ = RegCloseKey(self.0);
            }
            self.0 = HKEY::default();
        }
    }
}

/// UIPI elevation comparison between the helper process and target game window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UipiStatus {
    pub self_elevated: bool,
    pub game_elevated: Option<bool>,
    pub uipi_blocked: bool,
    pub needs_elevation: bool,
    pub mismatch: bool,
    pub reason: String,
}

/// Borderless launch command & spawn outcome (`src/game_settings.py`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BorderlessLaunchResult {
    pub ok: bool,
    pub dry_run: bool,
    pub exe_path: Option<String>,
    pub cmd: Vec<String>,
    pub pid: Option<u32>,
    pub error: Option<String>,
}

fn is_access_denied_error(err: &windows::core::Error) -> bool {
    // SAFETY: `GetLastError` is safe to call from any thread.
    let last = unsafe { GetLastError() };
    err.code() == HRESULT::from_win32(ERROR_ACCESS_DENIED.0) || last == ERROR_ACCESS_DENIED
}

fn query_token_elevated(h_process: HANDLE) -> Result<bool, windows::core::Error> {
    let mut h_token = HANDLE::default();
    // SAFETY: `h_process` is a valid process handle and `h_token` is initialized on success.
    unsafe { OpenProcessToken(h_process, TOKEN_QUERY, &mut h_token)? };
    let _token_guard = HandleGuard(h_token);

    let mut elev = TOKEN_ELEVATION::default();
    let mut ret_len = 0u32;
    // SAFETY: `elev` buffer matches `size_of::<TOKEN_ELEVATION>()` for `TokenElevation`.
    unsafe {
        GetTokenInformation(
            h_token,
            TokenElevation,
            Some((&mut elev as *mut TOKEN_ELEVATION).cast::<c_void>()),
            size_of::<TOKEN_ELEVATION>() as u32,
            &mut ret_len,
        )?;
    }
    Ok(elev.TokenIsElevated != 0)
}

/// Returns `true` if the current helper process holds an elevated Administrator token.
pub fn is_process_elevated() -> bool {
    // SAFETY: `GetCurrentProcess` returns a pseudo-handle that does not need closing.
    let current = unsafe { GetCurrentProcess() };
    query_token_elevated(current).unwrap_or(false)
}

/// Probes whether the process owning `hwnd` is elevated.
/// Infers `Some(true)` when `OpenProcess`/`OpenProcessToken` fails with `ERROR_ACCESS_DENIED`
/// from a non-elevated caller (UIPI boundary).
pub fn is_hwnd_elevated(hwnd: isize) -> Option<bool> {
    if hwnd <= 0 {
        return None;
    }
    let target = HWND(hwnd as *mut c_void);
    // SAFETY: `IsWindow` safely validates arbitrary HWND values.
    if unsafe { !IsWindow(Some(target)).as_bool() } {
        return None;
    }

    let mut pid = 0u32;
    // SAFETY: `target` was validated by `IsWindow`.
    unsafe {
        GetWindowThreadProcessId(target, Some(&mut pid));
    }
    if pid == 0 {
        return None;
    }

    // SAFETY: Opening target process with minimal `PROCESS_QUERY_LIMITED_INFORMATION`.
    let h_proc = match unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) } {
        Ok(h) => h,
        Err(err) => {
            if is_access_denied_error(&err) && !is_process_elevated() {
                return Some(true);
            }
            return None;
        }
    };
    let _proc_guard = HandleGuard(h_proc);

    match query_token_elevated(h_proc) {
        Ok(elev) => Some(elev),
        Err(err) => {
            if is_access_denied_error(&err) && !is_process_elevated() {
                Some(true)
            } else {
                None
            }
        }
    }
}

/// Pure UIPI status evaluator matching exact `src/game_settings.py` reason strings.
pub fn evaluate_uipi_status(
    self_elevated: bool,
    game_hwnd: Option<isize>,
    game_elevated: Option<bool>,
) -> UipiStatus {
    let has_hwnd = game_hwnd.is_some_and(|h| h > 0);
    let uipi_blocked = game_elevated == Some(true) && !self_elevated;
    let reason = if uipi_blocked {
        "Game process is elevated (Administrator) while helper is standard user; UIPI may block hotkeys."
            .to_owned()
    } else if !has_hwnd {
        "No game window handle provided.".to_owned()
    } else if game_elevated.is_none() {
        "Unable to determine game process elevation.".to_owned()
    } else {
        "OK".to_owned()
    };

    UipiStatus {
        self_elevated,
        game_elevated,
        uipi_blocked,
        needs_elevation: uipi_blocked,
        mismatch: uipi_blocked,
        reason,
    }
}

/// Checks whether UIPI blocks low-level hotkeys against `game_hwnd`.
pub fn check_uipi_status(game_hwnd: Option<isize>) -> UipiStatus {
    let self_elevated = is_process_elevated();
    let game_elevated = game_hwnd.and_then(is_hwnd_elevated);
    evaluate_uipi_status(self_elevated, game_hwnd, game_elevated)
}

/// Quotes a single Windows command-line argument per MSVCRT `CommandLineToArgvW` rules.
pub fn quote_win32_arg(arg: &str) -> String {
    if !arg.is_empty() && !arg.chars().any(|c| matches!(c, ' ' | '\t' | '\n' | '\r' | '"')) {
        return arg.to_owned();
    }
    let mut out = String::with_capacity(arg.len() + 2);
    out.push('"');
    let mut backslashes = 0usize;
    for ch in arg.chars() {
        if ch == '\\' {
            backslashes += 1;
        } else if ch == '"' {
            for _ in 0..(backslashes * 2 + 1) {
                out.push('\\');
            }
            out.push('"');
            backslashes = 0;
        } else {
            for _ in 0..backslashes {
                out.push('\\');
            }
            backslashes = 0;
            out.push(ch);
        }
    }
    for _ in 0..(backslashes * 2) {
        out.push('\\');
    }
    out.push('"');
    out
}

/// Returns `true` when existing CLI args can be faithfully satisfied by the fixed
/// `GenshinHelper_Elevated` scheduled task (which launches `"<app>\GenshinHelper.exe" --minimized`).
/// Any custom flag (e.g. `--port`, `--config`, `--lan-port`) forces the `runas` path so args are preserved.
pub fn can_use_elevated_task_for_args<S: AsRef<str>>(existing_args: &[S]) -> bool {
    existing_args
        .iter()
        .all(|a| a.as_ref().eq_ignore_ascii_case("--minimized"))
}

fn decode_schtasks_xml_bytes(raw: &[u8]) -> String {
    if raw.len() >= 2 && raw[0] == 0xFF && raw[1] == 0xFE {
        let units: Vec<u16> = raw[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    if raw.len() >= 4 && raw[1] == 0 && raw[3] == 0 {
        let units: Vec<u16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c))
            .collect();
        return String::from_utf16_lossy(&units);
    }
    String::from_utf8_lossy(raw).into_owned()
}

fn extract_xml_tag_value<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)?;
    Some(xml[start..start + end].trim())
}

fn unescape_xml_entities(input: &str) -> String {
    input
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

fn normalize_windows_path_for_compare(path: &str) -> String {
    let stripped = path
        .trim()
        .trim_matches('"')
        .strip_prefix(r"\\?\")
        .unwrap_or_else(|| path.trim().trim_matches('"'));
    stripped
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_ascii_lowercase()
}

/// Checks whether a `schtasks /query /xml` payload defines a `HighestAvailable` task whose
/// `<Command>` points to `current_exe`. Prevents a dev or portable binary from accidentally
/// waking an older installed binary via `GenshinHelper_Elevated`.
pub fn scheduled_task_xml_matches_elevated_exe(xml_bytes: &[u8], current_exe: &Path) -> bool {
    let xml = decode_schtasks_xml_bytes(xml_bytes);
    let Some(run_level) = extract_xml_tag_value(&xml, "RunLevel") else {
        return false;
    };
    if !run_level.eq_ignore_ascii_case("HighestAvailable") {
        return false;
    }
    let Some(raw_cmd) = extract_xml_tag_value(&xml, "Command") else {
        return false;
    };
    let unescaped = unescape_xml_entities(raw_cmd);
    let cleaned_cmd = unescaped.trim().trim_matches('"').trim();
    if cleaned_cmd.is_empty() {
        return false;
    }

    if let (Ok(canon_task), Ok(canon_cur)) = (
        std::fs::canonicalize(cleaned_cmd),
        std::fs::canonicalize(current_exe),
    ) {
        return normalize_windows_path_for_compare(&canon_task.to_string_lossy())
            == normalize_windows_path_for_compare(&canon_cur.to_string_lossy());
    }

    normalize_windows_path_for_compare(cleaned_cmd)
        == normalize_windows_path_for_compare(&current_exe.to_string_lossy())
}

fn run_schtasks_bounded(args: &[&str], capture_stdout: bool) -> Option<Vec<u8>> {
    let mut cmd = Command::new("schtasks.exe");
    cmd.args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .stdout(if capture_stdout {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .creation_flags(CREATE_NO_WINDOW);

    let mut child = cmd.spawn().ok()?;
    let reader_handle = child.stdout.take().map(|mut out| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = out.read_to_end(&mut buf);
            buf
        })
    });

    let deadline = Instant::now() + SCHTASKS_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = reader_handle
                    .and_then(|h| h.join().ok())
                    .unwrap_or_default();
                return status.success().then_some(output);
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => return None,
        }
    }
}

/// Attempts a zero-UAC elevated relaunch via the installer-provisioned `GenshinHelper_Elevated`
/// scheduled task, only if the task runs at `HighestAvailable` and targets `current_exe`.
fn try_relaunch_via_elevated_task(current_exe: &Path, existing_args: &[String]) -> bool {
    if !can_use_elevated_task_for_args(existing_args) {
        return false;
    }
    let Some(xml_bytes) = run_schtasks_bounded(
        &["/query", "/tn", gh_core::ELEVATED_TASK_NAME, "/xml"],
        true,
    ) else {
        return false;
    };
    if !scheduled_task_xml_matches_elevated_exe(&xml_bytes, current_exe) {
        return false;
    }
    run_schtasks_bounded(&["/run", "/tn", gh_core::ELEVATED_TASK_NAME], false).is_some()
}

/// Relaunches the current executable elevated.
/// First tries the zero-UAC `GenshinHelper_Elevated` scheduled task when launching the exact
/// installed executable with default arguments (e.g. Desktop / Start Menu shortcuts), then
/// falls back to UAC (`ShellExecuteW("runas")`) preserving all CLI arguments and `extra_args`.
/// Returns `true` if Windows accepted the elevation request.
pub fn relaunch_current_exe_elevated(extra_args: &[&str]) -> bool {
    let Ok(exe) = std::env::current_exe() else {
        return false;
    };
    let raw_args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(|a| a.to_string_lossy().into_owned())
        .collect();

    if try_relaunch_via_elevated_task(&exe, &raw_args) {
        return true;
    }

    let exe_w: Vec<u16> = exe
        .as_os_str()
        .to_string_lossy()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();

    let mut quoted_args: Vec<String> = raw_args.iter().map(|a| quote_win32_arg(a)).collect();
    for &extra in extra_args {
        quoted_args.push(quote_win32_arg(extra));
    }
    let params_str = quoted_args.join(" ");
    let params_w: Option<Vec<u16>> = if params_str.is_empty() {
        None
    } else {
        Some(params_str.encode_utf16().chain(std::iter::once(0)).collect())
    };
    let cwd_w: Option<Vec<u16>> = std::env::current_dir().ok().map(|p| {
        p.as_os_str()
            .to_string_lossy()
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect()
    });

    // SAFETY: `exe_w`, optional `params_w`, and optional `cwd_w` are NUL-terminated for `ShellExecuteW`.
    let res = unsafe {
        ShellExecuteW(
            None,
            w!("runas"),
            PCWSTR(exe_w.as_ptr()),
            params_w.as_ref().map_or(PCWSTR::null(), |v| PCWSTR(v.as_ptr())),
            cwd_w.as_ref().map_or(PCWSTR::null(), |v| PCWSTR(v.as_ptr())),
            SW_SHOWNORMAL,
        )
    };
    res.0 as isize > 32
}

fn read_hkcu_string_value(subkey: &str, value_name: &str) -> Option<String> {
    let subkey_w: Vec<u16> = subkey.encode_utf16().chain(std::iter::once(0)).collect();
    let val_w: Vec<u16> = value_name.encode_utf16().chain(std::iter::once(0)).collect();

    let mut hkey = HKEY::default();
    // SAFETY: `subkey_w` is NUL-terminated; `hkey` is closed by `HkeyGuard`.
    let open_res = unsafe {
        RegOpenKeyExW(
            HKEY_CURRENT_USER,
            PCWSTR(subkey_w.as_ptr()),
            Some(0),
            KEY_READ,
            &mut hkey,
        )
    };
    if open_res.is_err() {
        return None;
    }
    let _guard = HkeyGuard(hkey);

    let mut reg_type = REG_VALUE_TYPE::default();
    let mut byte_len = 0u32;
    // SAFETY: First query with `None` data buffer obtains exact required byte length.
    let size_res = unsafe {
        RegQueryValueExW(
            hkey,
            PCWSTR(val_w.as_ptr()),
            None,
            Some(&mut reg_type),
            None,
            Some(&mut byte_len),
        )
    };
    if size_res.is_err() || byte_len == 0 {
        return None;
    }
    if reg_type != REG_SZ && reg_type != REG_EXPAND_SZ {
        return None;
    }

    let u16_len = (byte_len as usize).div_ceil(2);
    let mut buf = vec![0u16; u16_len];
    let mut actual_bytes = (buf.len() * 2) as u32;
    // SAFETY: `buf` has capacity `actual_bytes` bytes.
    let read_res = unsafe {
        RegQueryValueExW(
            hkey,
            PCWSTR(val_w.as_ptr()),
            None,
            Some(&mut reg_type),
            Some(buf.as_mut_ptr().cast::<u8>()),
            Some(&mut actual_bytes),
        )
    };
    if read_res.is_err() {
        return None;
    }

    let valid_u16 = (actual_bytes as usize) / 2;
    let slice = &buf[..valid_u16.min(buf.len())];
    let trimmed_len = slice.iter().position(|&c| c == 0).unwrap_or(slice.len());
    let text = String::from_utf16_lossy(&slice[..trimmed_len])
        .trim()
        .to_owned();
    if text.is_empty() { None } else { Some(text) }
}

/// Testable HoYoPlay executable resolver using a caller-provided registry lookup closure.
pub fn find_genshin_executable_with<F>(mut read_install_path: F) -> Option<PathBuf>
where
    F: FnMut(&str) -> Option<String>,
{
    for (subkey, exe_names) in HOYOPLAY_REGISTRY_CANDIDATES {
        if let Some(install_dir) = read_install_path(subkey) {
            let trimmed = install_dir.trim();
            if trimmed.is_empty() {
                continue;
            }
            let base = Path::new(trimmed);
            for exe_name in *exe_names {
                let candidate = base.join(exe_name);
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// Discovers the Genshin Impact executable path from HoYoPlay HKCU registry entries.
pub fn find_genshin_executable() -> Option<PathBuf> {
    find_genshin_executable_with(|subkey| read_hkcu_string_value(subkey, "GameInstallPath"))
}

/// Launches (or dry-runs) Genshin Impact in Unity borderless windowed mode.
pub fn launch_genshin_borderless(
    width: Option<u32>,
    height: Option<u32>,
    exe_path: Option<&str>,
    dry_run: bool,
) -> BorderlessLaunchResult {
    let resolved_exe = exe_path
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| find_genshin_executable().map(|p| p.to_string_lossy().into_owned()));

    let Some(exe) = resolved_exe else {
        return BorderlessLaunchResult {
            ok: false,
            dry_run,
            exe_path: None,
            cmd: Vec::new(),
            pid: None,
            error: Some("Genshin Impact executable not found in HoYoPlay registry.".to_owned()),
        };
    };

    // SAFETY: `GetSystemMetrics` with `SM_CXSCREEN` / `SM_CYSCREEN` is safe on any thread.
    let w = width.unwrap_or_else(|| unsafe { GetSystemMetrics(SM_CXSCREEN).max(1) as u32 });
    let h = height.unwrap_or_else(|| unsafe { GetSystemMetrics(SM_CYSCREEN).max(1) as u32 });

    let cmd = match build_borderless_launch_args(w, h, Some(&exe)) {
        Ok(args) => args,
        Err(err) => {
            return BorderlessLaunchResult {
                ok: false,
                dry_run,
                exe_path: Some(exe),
                cmd: Vec::new(),
                pid: None,
                error: Some(err.to_string()),
            };
        }
    };

    if dry_run {
        return BorderlessLaunchResult {
            ok: true,
            dry_run: true,
            exe_path: Some(exe),
            cmd,
            pid: None,
            error: None,
        };
    }

    let mut command = Command::new(&exe);
    if cmd.len() > 1 {
        command.args(&cmd[1..]);
    }
    if let Some(parent) = Path::new(&exe).parent()
        && !parent.as_os_str().is_empty()
    {
        command.current_dir(parent);
    }
    command.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);

    match command.spawn() {
        Ok(child) => BorderlessLaunchResult {
            ok: true,
            dry_run: false,
            exe_path: Some(exe),
            cmd,
            pid: Some(child.id()),
            error: None,
        },
        Err(err) => BorderlessLaunchResult {
            ok: false,
            dry_run: false,
            exe_path: Some(exe),
            cmd,
            pid: None,
            error: Some(err.to_string()),
        },
    }
}

/// Applies `WDA_EXCLUDEFROMCAPTURE` (`0x11`) or `WDA_NONE` (`0x00`) to `hwnd`
/// so GDI `BitBlt` never captures our own Mini-HUD overlay unless requested.
pub fn refresh_exclude_from_capture(hwnd: isize, exclude: bool) -> bool {
    if hwnd <= 0 {
        return false;
    }
    let target = HWND(hwnd as *mut c_void);
    // SAFETY: `IsWindow` safely checks validity before setting affinity.
    if unsafe { !IsWindow(Some(target)).as_bool() } {
        return false;
    }
    let affinity = if exclude {
        gh_core::WDA_EXCLUDEFROMCAPTURE
    } else {
        gh_core::WDA_NONE
    };
    // SAFETY: `target` is a valid HWND and `affinity` is a valid `WDA_*` flag.
    unsafe { SetWindowDisplayAffinity(target, WINDOW_DISPLAY_AFFINITY(affinity)).is_ok() }
}

/// Queries the current Win32 `GetWindowDisplayAffinity` value for `hwnd`.
pub fn get_window_display_affinity(hwnd: isize) -> Option<u32> {
    if hwnd <= 0 {
        return None;
    }
    let target = HWND(hwnd as *mut c_void);
    // SAFETY: `IsWindow` validates `target`.
    if unsafe { !IsWindow(Some(target)).as_bool() } {
        return None;
    }
    let mut affinity = 0u32;
    // SAFETY: `target` is valid and `affinity` points to a stack `u32`.
    unsafe { GetWindowDisplayAffinity(target, &mut affinity).ok()? };
    Some(affinity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null_mut;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DestroyWindow, WS_EX_TOOLWINDOW, WS_POPUP,
    };

    static TEMP_SEQ: AtomicUsize = AtomicUsize::new(0);

    fn make_temp_dir(tag: &str) -> PathBuf {
        let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "gh_win32_gs_{tag}_{}_{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn uipi_checks_handle_invalid_hwnd_and_all_reason_branches() {
        let _self_elev = is_process_elevated();
        assert_eq!(is_hwnd_elevated(0), None);
        assert_eq!(is_hwnd_elevated(-42), None);

        let none_status = check_uipi_status(None);
        assert!(!none_status.uipi_blocked);
        assert!(!none_status.needs_elevation);
        assert_eq!(none_status.reason, "No game window handle provided.");

        let zero_status = check_uipi_status(Some(0));
        assert!(!zero_status.uipi_blocked);
        assert_eq!(zero_status.reason, "No game window handle provided.");

        let mismatch = evaluate_uipi_status(false, Some(1234), Some(true));
        assert!(mismatch.uipi_blocked);
        assert!(mismatch.needs_elevation);
        assert!(mismatch.mismatch);
        assert!(mismatch.reason.contains("elevated (Administrator)"));

        let matching = evaluate_uipi_status(false, Some(1234), Some(false));
        assert!(!matching.uipi_blocked);
        assert!(!matching.needs_elevation);
        assert_eq!(matching.reason, "OK");

        let unknown = evaluate_uipi_status(false, Some(1234), None);
        assert!(!unknown.uipi_blocked);
        assert_eq!(unknown.reason, "Unable to determine game process elevation.");
    }

    #[test]
    fn find_genshin_executable_with_follows_priority_and_exe_order() {
        let base = make_temp_dir("hoyoplay");
        let global_dir = base.join("global");
        let cn_dir = base.join("cn");
        std::fs::create_dir_all(&global_dir).expect("global dir");
        std::fs::create_dir_all(&cn_dir).expect("cn dir");

        let global_exe = global_dir.join("GenshinImpact.exe");
        let cn_exe = cn_dir.join("YuanShen.exe");
        std::fs::write(&cn_exe, b"stub").expect("write cn");

        // When only hk4e_cn has a valid exe, fallback finds YuanShen.exe.
        let found_cn = find_genshin_executable_with(|subkey| match subkey {
            s if s.ends_with("hk4e_global") => Some(global_dir.to_string_lossy().into_owned()),
            s if s.ends_with("hk4e_cn") => Some(cn_dir.to_string_lossy().into_owned()),
            _ => None,
        });
        assert_eq!(found_cn.as_deref(), Some(cn_exe.as_path()));

        // When hk4e_global exists too, global wins priority.
        std::fs::write(&global_exe, b"stub").expect("write global");
        let found_global = find_genshin_executable_with(|subkey| match subkey {
            s if s.ends_with("hk4e_global") => Some(global_dir.to_string_lossy().into_owned()),
            s if s.ends_with("hk4e_cn") => Some(cn_dir.to_string_lossy().into_owned()),
            _ => None,
        });
        assert_eq!(found_global.as_deref(), Some(global_exe.as_path()));

        // Missing files return None cleanly.
        let found_none = find_genshin_executable_with(|_| Some(base.join("missing").to_string_lossy().into_owned()));
        assert_eq!(found_none, None);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn launch_genshin_borderless_dry_run_and_reexports_work() {
        let layout = compute_ui_layout(1920, 1080, false).expect("reexported compute_ui_layout");
        assert_eq!(layout.aspect_label, AspectLabel::Ratio16x9);

        let fake_exe = r"D:\FakeGame\GenshinImpact.exe";
        let res = launch_genshin_borderless(Some(1920), Some(1080), Some(fake_exe), true);
        assert!(res.ok);
        assert!(res.dry_run);
        assert_eq!(res.exe_path.as_deref(), Some(fake_exe));
        assert_eq!(res.pid, None);
        assert_eq!(res.error, None);
        assert_eq!(
            res.cmd,
            vec![
                fake_exe,
                "-popupwindow",
                "-screen-fullscreen",
                "0",
                "-screen-width",
                "1920",
                "-screen-height",
                "1080",
            ]
        );

        let invalid_dims = launch_genshin_borderless(Some(0), Some(1080), Some(fake_exe), true);
        assert!(!invalid_dims.ok);
        assert!(invalid_dims.error.is_some());
    }

    #[test]
    fn refresh_exclude_from_capture_toggles_wda_on_real_window_and_rejects_zero() {
        assert!(!refresh_exclude_from_capture(0, true));
        assert!(!refresh_exclude_from_capture(-1, false));
        assert_eq!(get_window_display_affinity(0), None);

        let class_w: Vec<u16> = "STATIC\0".encode_utf16().collect();
        let title_w: Vec<u16> = "gh_wda_test\0".encode_utf16().collect();
        // SAFETY: Creating a hidden top-level tool window owned by the current process for WDA test.
        let hmod = unsafe { GetModuleHandleW(None).expect("GetModuleHandleW") };
        let hwnd = unsafe {
            CreateWindowExW(
                WS_EX_TOOLWINDOW,
                PCWSTR(class_w.as_ptr()),
                PCWSTR(title_w.as_ptr()),
                WS_POPUP,
                0,
                0,
                16,
                16,
                None,
                None,
                Some(hmod.into()),
                Some(null_mut()),
            )
            .expect("CreateWindowExW")
        };
        let raw_hwnd = hwnd.0 as isize;

        assert!(refresh_exclude_from_capture(raw_hwnd, true));
        assert_eq!(
            get_window_display_affinity(raw_hwnd),
            Some(gh_core::WDA_EXCLUDEFROMCAPTURE)
        );

        assert!(refresh_exclude_from_capture(raw_hwnd, false));
        assert_eq!(
            get_window_display_affinity(raw_hwnd),
            Some(gh_core::WDA_NONE)
        );

        // SAFETY: Destroying the test window created above.
        unsafe {
            let _ = DestroyWindow(hwnd);
        }
    }

    #[test]
    fn quote_win32_arg_matches_msvcrt_rules() {
        assert_eq!(quote_win32_arg(""), r#""""#);
        assert_eq!(quote_win32_arg("--no-elevate"), "--no-elevate");
        assert_eq!(
            quote_win32_arg(r"C:\App Dir\cfg.json"),
            r#""C:\App Dir\cfg.json""#
        );
        assert_eq!(quote_win32_arg("tab\targ"), "\"tab\targ\"");
        assert_eq!(quote_win32_arg(r#"a"b"#), r#""a\"b""#);
        assert_eq!(quote_win32_arg(r#"a\"b"#), r#""a\\\"b""#);
        assert_eq!(quote_win32_arg(r"C:\My Folder\"), r#""C:\My Folder\\""#);
    }

    #[test]
    fn elevated_task_relaunch_guards_args_run_level_and_exe_match() {
        assert!(can_use_elevated_task_for_args::<&str>(&[]));
        assert!(can_use_elevated_task_for_args(&["--minimized"]));
        assert!(can_use_elevated_task_for_args(&["--MINIMIZED"]));
        assert!(!can_use_elevated_task_for_args(&["--port", "9999"]));
        assert!(!can_use_elevated_task_for_args(&["--minimized", "--no-mutex"]));

        let installed = Path::new(r"C:\Program Files\Genshin Helper\GenshinHelper.exe");
        let dev_bin = Path::new(r"X:\CargoTarget\Genshin-Helper\release\GenshinHelper.exe");

        let xml_ok = br#"<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2">
  <Principals><Principal><RunLevel>HighestAvailable</RunLevel></Principal></Principals>
  <Actions><Exec>
    <Command>"C:\Program Files\Genshin Helper\GenshinHelper.exe"</Command>
    <Arguments>--minimized</Arguments>
  </Exec></Actions>
</Task>"#;
        assert!(scheduled_task_xml_matches_elevated_exe(xml_ok, installed));
        assert!(!scheduled_task_xml_matches_elevated_exe(xml_ok, dev_bin));

        // UTF-16LE with BOM + entity-escaped ampersand in path
        let utf16_str = concat!(
            "<?xml version=\"1.0\" encoding=\"UTF-16\"?>",
            "<Task><Principals><Principal><RunLevel>HighestAvailable</RunLevel></Principal></Principals>",
            "<Actions><Exec><Command>&quot;C:\\Apps &amp; Tools\\GenshinHelper.exe&quot;</Command></Exec></Actions></Task>"
        );
        let mut utf16_bytes = vec![0xFF, 0xFE];
        for u in utf16_str.encode_utf16() {
            utf16_bytes.extend_from_slice(&u.to_le_bytes());
        }
        assert!(scheduled_task_xml_matches_elevated_exe(
            &utf16_bytes,
            Path::new(r"c:\apps & tools\genshinhelper.exe"),
        ));

        // Non-elevated LeastPrivilege task must be rejected
        let xml_least = br#"<Task>
  <Principals><Principal><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Actions><Exec><Command>"C:\Program Files\Genshin Helper\GenshinHelper.exe"</Command></Exec></Actions>
</Task>"#;
        assert!(!scheduled_task_xml_matches_elevated_exe(xml_least, installed));
    }
}
