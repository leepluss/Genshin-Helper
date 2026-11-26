//! `timeBeginPeriod(1)` high-resolution timer guard and `SingleInstanceMutex` (`main.py`).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;
use windows::Win32::Foundation::{CloseHandle, ERROR_ALREADY_EXISTS, GetLastError, HANDLE};
use windows::Win32::Media::{TIMERR_NOERROR, timeBeginPeriod, timeEndPeriod};
use windows::Win32::System::Threading::CreateMutexW;
use windows::core::PCWSTR;

/// Idempotent RAII guard around `winmm::timeBeginPeriod(1)` / `timeEndPeriod(1)`
/// matching `main.py::_HighResTimer`.
#[derive(Debug, Default)]
pub struct HighResTimer {
    active: bool,
}

impl HighResTimer {
    pub const fn new() -> Self {
        Self { active: false }
    }

    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Toggles 1 ms timer resolution idempotently so `timeBeginPeriod` / `timeEndPeriod`
    /// refcounts always remain balanced.
    pub fn set(&mut self, on: bool) {
        if on == self.active {
            return;
        }
        if on {
            // SAFETY: Plain WinMM multimedia timer resolution call with constant period 1 ms.
            let res = unsafe { timeBeginPeriod(1) };
            if res == TIMERR_NOERROR {
                self.active = true;
            }
        } else {
            // SAFETY: Paired only when `self.active` is true (balanced with `timeBeginPeriod(1)`).
            unsafe {
                timeEndPeriod(1);
            }
            self.active = false;
        }
    }
}

impl Drop for HighResTimer {
    fn drop(&mut self) {
        self.set(false);
    }
}

/// Env flag set by the black-box sandbox (`tests/blackbox_contracts/sandbox.py`). When on, the
/// app still finds windows and validates hotkey bindings but never touches the user's desktop:
/// no foreground/focus changes (`WindowSwitcher`) and no global `WH_*_LL` input hooks.
pub const DESKTOP_SANDBOX_ENV: &str = "GH_DESKTOP_SANDBOX";

/// Returns true when `GH_DESKTOP_SANDBOX` is set to a truthy value (`1`/`true`/`yes`/`on`).
pub fn is_desktop_sandbox_env() -> bool {
    desktop_sandbox_flag(std::env::var(DESKTOP_SANDBOX_ENV).ok().as_deref())
}

/// Pure parser for the `GH_DESKTOP_SANDBOX` flag value.
pub fn desktop_sandbox_flag(raw: Option<&str>) -> bool {
    raw.map(|s| s.trim().to_ascii_lowercase())
        .is_some_and(|s| matches!(s.as_str(), "1" | "true" | "yes" | "on"))
}

/// Resolves the single-instance mutex name, honoring `GH_SINGLE_INSTANCE_MUTEX`
/// for sandboxed black-box tests (`tests/blackbox_contracts/sandbox.py`).
pub fn resolve_mutex_name() -> String {
    let env_val = std::env::var("GH_SINGLE_INSTANCE_MUTEX").ok();
    resolve_mutex_name_from(env_val.as_deref())
}

/// Pure helper for resolving mutex name from an optional env override.
pub fn resolve_mutex_name_from(env_override: Option<&str>) -> String {
    if let Some(raw) = env_override {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            return trimmed.to_owned();
        }
    }
    gh_core::SINGLE_INSTANCE_MUTEX.to_owned()
}

/// RAII named Win32 mutex guard (`Local\GenshinHelper_SingleInstance_v4` by default).
#[derive(Debug)]
pub struct SingleInstanceMutex {
    handle: HANDLE,
    name: String,
}

// SAFETY: Created with `bInitialOwner = false`, so the mutex has no thread affinity;
// `HANDLE` is an opaque kernel object handle closed exactly once on `Drop`.
unsafe impl Send for SingleInstanceMutex {}
unsafe impl Sync for SingleInstanceMutex {}

impl SingleInstanceMutex {
    /// Non-blocking single attempt to create and claim the named mutex.
    /// Returns `Ok(None)` when another instance already owns the mutex (`ERROR_ALREADY_EXISTS`).
    pub fn try_acquire(name: &str) -> std::io::Result<Option<Self>> {
        if name.trim().is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "mutex name must not be empty",
            ));
        }

        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: `wide` is NUL-terminated for the duration of `CreateMutexW`.
        let handle = unsafe { CreateMutexW(None, false, PCWSTR(wide.as_ptr())) }
            .map_err(|e| std::io::Error::from_raw_os_error(e.code().0))?;

        // SAFETY: `GetLastError` is read immediately after `CreateMutexW` on the same thread.
        let last_err = unsafe { GetLastError() };
        if last_err == ERROR_ALREADY_EXISTS {
            // SAFETY: Valid handle returned by `CreateMutexW` for existing mutex; close immediately.
            unsafe {
                let _ = CloseHandle(handle);
            }
            return Ok(None);
        }

        Ok(Some(Self {
            handle,
            name: name.to_owned(),
        }))
    }

    /// Attempts to acquire the named mutex up to 4 times (450 ms apart) plus a final try,
    /// invoking `on_contended_first` once on the initial contention (`main.py:365-392`).
    pub fn acquire_with_takeover<F>(
        name: &str,
        retry_delay: Duration,
        mut on_contended_first: F,
    ) -> std::io::Result<Option<Self>>
    where
        F: FnMut(),
    {
        for attempt in 0..4 {
            if let Some(guard) = Self::try_acquire(name)? {
                return Ok(Some(guard));
            }
            if attempt == 0 {
                on_contended_first();
            }
            if !retry_delay.is_zero() {
                std::thread::sleep(retry_delay);
            }
        }
        Self::try_acquire(name)
    }

    /// Resolves `GH_SINGLE_INSTANCE_MUTEX` (or default) and performs graceful `POST /shutdown`
    /// takeover on `127.0.0.1:{port}` before retrying mutex acquisition (`main.py:365-392`).
    pub fn acquire_for_port(port: u16) -> std::io::Result<Option<Self>> {
        let name = resolve_mutex_name();
        Self::acquire_with_takeover(&name, Duration::from_millis(450), || {
            send_shutdown_post(port);
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn is_primary(&self) -> bool {
        true
    }
}

impl Drop for SingleInstanceMutex {
    fn drop(&mut self) {
        if !self.handle.is_invalid() {
            // SAFETY: `self.handle` is uniquely owned by this struct and closed once on drop.
            unsafe {
                let _ = CloseHandle(self.handle);
            }
            self.handle = HANDLE::default();
        }
    }
}

fn send_shutdown_post(port: u16) {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let timeout = Duration::from_millis(1500);
    if let Ok(mut stream) = TcpStream::connect_timeout(&addr, timeout) {
        let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
        let _ = stream.set_write_timeout(Some(timeout));
        let req = format!(
            "POST /shutdown HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
        );
        let _ = stream.write_all(req.as_bytes());
        let _ = stream.flush();
        let mut drain_buf = [0u8; 256];
        let _ = stream.read(&mut drain_buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static MUTEX_SEQ: AtomicUsize = AtomicUsize::new(0);

    fn unique_test_mutex(tag: &str) -> String {
        let seq = MUTEX_SEQ.fetch_add(1, Ordering::Relaxed);
        format!(r"Local\gh_win32_test_{tag}_{}_{seq}", std::process::id())
    }

    #[test]
    fn high_res_timer_is_idempotent_and_cleans_up_on_drop() {
        let mut timer = HighResTimer::new();
        assert!(!timer.is_active());

        timer.set(true);
        assert!(timer.is_active());
        // Calling set(true) again must be a no-op (balanced refcount).
        timer.set(true);
        assert!(timer.is_active());

        timer.set(false);
        assert!(!timer.is_active());
        timer.set(false);
        assert!(!timer.is_active());

        // Drop while active must not panic and resets active state cleanly.
        let mut active_timer = HighResTimer::new();
        active_timer.set(true);
        assert!(active_timer.is_active());
        drop(active_timer);
    }

    #[test]
    fn desktop_sandbox_flag_accepts_only_truthy_values() {
        for on in ["1", "true", " TRUE ", "yes", "on"] {
            assert!(desktop_sandbox_flag(Some(on)), "{on:?} must enable sandbox");
        }
        for off in ["", "0", "false", "off", "no", "2"] {
            assert!(!desktop_sandbox_flag(Some(off)), "{off:?} must not enable sandbox");
        }
        assert!(!desktop_sandbox_flag(None));
    }

    #[test]
    fn resolve_mutex_name_honors_override_and_default() {
        assert_eq!(resolve_mutex_name_from(None), gh_core::SINGLE_INSTANCE_MUTEX);
        assert_eq!(
            resolve_mutex_name_from(Some("   ")),
            gh_core::SINGLE_INSTANCE_MUTEX
        );
        assert_eq!(
            resolve_mutex_name_from(Some(r" Local\CustomTestMutex ")),
            r"Local\CustomTestMutex"
        );
    }

    #[test]
    fn single_instance_mutex_detects_contention_and_releases_on_drop() {
        let name = unique_test_mutex("contention");

        let first = SingleInstanceMutex::try_acquire(&name)
            .expect("first acquire")
            .expect("primary acquired");
        assert!(first.is_primary());
        assert_eq!(first.name(), name);

        // Second attempt while first is alive must return None.
        let second = SingleInstanceMutex::try_acquire(&name).expect("second acquire");
        assert!(second.is_none(), "second acquire must report contention");

        drop(first);

        // Re-acquiring after drop succeeds.
        let reacquired = SingleInstanceMutex::try_acquire(&name)
            .expect("reacquire after drop")
            .expect("mutex released on drop");
        assert!(reacquired.is_primary());
    }

    #[test]
    fn acquire_with_takeover_invokes_shutdown_once_and_recovers() {
        let name = unique_test_mutex("takeover");
        let first = SingleInstanceMutex::try_acquire(&name)
            .expect("first")
            .expect("primary");
        let mut first_opt = Some(first);

        let mut shutdown_calls = 0usize;
        let acquired = SingleInstanceMutex::acquire_with_takeover(&name, Duration::ZERO, || {
            shutdown_calls += 1;
            // Simulate existing process gracefully closing its mutex upon shutdown signal.
            drop(first_opt.take());
        })
        .expect("acquire_with_takeover")
        .expect("acquired after takeover");

        assert_eq!(shutdown_calls, 1);
        assert_eq!(acquired.name(), name);
    }

    #[test]
    fn send_shutdown_post_sends_valid_http_post_request() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().expect("local addr").port();

        let server_thread = std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 512];
            let n = conn.read(&mut buf).expect("read");
            String::from_utf8_lossy(&buf[..n]).to_string()
        });

        send_shutdown_post(port);
        let req = server_thread.join().expect("join server");
        assert!(req.starts_with("POST /shutdown HTTP/1.1\r\n"));
        assert!(req.ends_with("\r\n\r\n{}"));
    }
}
