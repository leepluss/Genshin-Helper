//! Pure `WH_KEYBOARD_LL` + `WH_MOUSE_LL` hotkey listener (`src/hotkey.py`).
//!
//! Invariant: Read-only against the game window — strictly zero synthetic input injection,
//! suppress trigger VK only (never modifiers), gated by foreground game/HUD window.

use crate::hotkey_conflicts::{MODIFIER_VKS, hotkey_to_vk_combo, lookup_mouse_alias};
use gh_core::protocol::format_hotkey_display_name;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;
use windows::Win32::Foundation::{HINSTANCE, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, GetKeyState, VK_CONTROL, VK_MENU, VK_PACKET, VK_SHIFT,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageW, GetMessageW, HHOOK, KBDLLHOOKSTRUCT, LLKHF_ALTDOWN,
    MSLLHOOKSTRUCT, MSG, PM_NOREMOVE, PeekMessageW, PostThreadMessageW, SetWindowsHookExW,
    TranslateMessage, UnhookWindowsHookEx, WH_KEYBOARD_LL, WH_MOUSE_LL, WM_KEYDOWN, WM_KEYUP,
    WM_MBUTTONDOWN, WM_QUIT, WM_RBUTTONDOWN, WM_SYSKEYDOWN, WM_SYSKEYUP, WM_XBUTTONDOWN,
};

pub type HotkeyCallback = Arc<dyn Fn() + Send + Sync>;
pub type ShouldSuppressFn = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// Canonical 9 `AppConfig` hotkey action names in registration order.
pub const HOTKEY_ACTION_NAMES: [&str; 9] = [
    "sync",
    "debug",
    "switch",
    "nav_mark",
    "nav_undo",
    "nav_guide",
    "nav_next",
    "nav_layout",
    "nav_toggle",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedHotkey {
    pub is_mouse: bool,
    pub mouse_btn: Option<String>,
    pub modifiers: BTreeSet<&'static str>,
    pub main_key: String,
    /// Sorted modifier order (`alt`, `ctrl`, `shift`) matching `src/hotkey.py:146`.
    pub clean_str: String,
}

/// Parses a hotkey string using `UnifiedHotkeyManager._parse_hotkey` semantics (`src/hotkey.py:125-147`).
pub fn parse_hotkey(hotkey: &str) -> ParsedHotkey {
    let raw = hotkey.trim().to_ascii_lowercase();
    if raw.is_empty() {
        return ParsedHotkey {
            is_mouse: false,
            mouse_btn: None,
            modifiers: BTreeSet::new(),
            main_key: String::new(),
            clean_str: String::new(),
        };
    }

    let mut modifiers = BTreeSet::new();
    let mut main_key = String::new();
    let mut is_mouse = false;
    let mut mouse_btn: Option<String> = None;

    for tok in raw
        .split(['+', '-'])
        .flat_map(|p| p.split_whitespace())
        .filter(|p| !p.is_empty())
    {
        match tok {
            "alt" | "menu" => {
                modifiers.insert("alt");
            }
            "ctrl" | "control" => {
                modifiers.insert("ctrl");
            }
            "shift" => {
                modifiers.insert("shift");
            }
            other => {
                if let Some(btn) = lookup_mouse_alias(other) {
                    is_mouse = true;
                    mouse_btn = Some(btn.to_owned());
                    main_key = btn.to_owned();
                } else {
                    main_key = other.to_owned();
                }
            }
        }
    }

    let mut parts: Vec<&str> = modifiers.iter().copied().collect();
    if !main_key.is_empty() {
        parts.push(&main_key);
    }
    let clean_str = parts.join("+");

    ParsedHotkey {
        is_mouse,
        mouse_btn,
        modifiers,
        main_key,
        clean_str,
    }
}

/// Human-friendly display formatter delegating to `gh_core::protocol::format_hotkey_display_name`.
pub fn format_display_name(hotkey_str: &str) -> String {
    format_hotkey_display_name(hotkey_str)
}

#[derive(Clone)]
struct BindingEntry {
    name: String,
    hotkey_str: String,
    callback: HotkeyCallback,
    parsed: ParsedHotkey,
    vk: Option<u32>,
    use_ll_kb: bool,
    suppress: bool,
    enabled: bool,
}

#[derive(Clone)]
pub struct BindingSnapshot {
    pub name: String,
    pub callback: HotkeyCallback,
    pub is_mouse: bool,
    pub mouse_btn: Option<String>,
    pub has_alt: bool,
    pub has_ctrl: bool,
    pub has_shift: bool,
    pub vk: Option<u32>,
    pub use_ll_kb: bool,
    pub suppress: bool,
    pub enabled: bool,
}

impl From<&BindingEntry> for BindingSnapshot {
    fn from(b: &BindingEntry) -> Self {
        Self {
            name: b.name.clone(),
            callback: Arc::clone(&b.callback),
            is_mouse: b.parsed.is_mouse,
            mouse_btn: b.parsed.mouse_btn.clone(),
            has_alt: b.parsed.modifiers.contains("alt"),
            has_ctrl: b.parsed.modifiers.contains("ctrl"),
            has_shift: b.parsed.modifiers.contains("shift"),
            vk: b.vk,
            use_ll_kb: b.use_ll_kb,
            suppress: b.suppress,
            enabled: b.enabled,
        }
    }
}

struct SharedHookState {
    bindings: RwLock<Vec<BindingSnapshot>>,
    should_suppress: RwLock<Option<ShouldSuppressFn>>,
    callback_tx: Mutex<Option<Sender<HotkeyCallback>>>,
}

impl SharedHookState {
    fn dispatch_callback(&self, cb: HotkeyCallback) {
        if let Ok(guard) = self.callback_tx.lock()
            && let Some(tx) = guard.as_ref()
            && tx.send(Arc::clone(&cb)).is_ok()
        {
            return;
        }
        // Fallback if worker hasn't started / already stopped
        let _ = std::thread::Builder::new()
            .name("gh-hotkey-fallback".to_owned())
            .spawn(move || {
                let _ = catch_unwind(AssertUnwindSafe(|| cb()));
            });
    }
}

static ACTIVE_KB_STATE: OnceLock<RwLock<Option<Arc<SharedHookState>>>> = OnceLock::new();
static ACTIVE_MOUSE_STATE: OnceLock<RwLock<Option<Arc<SharedHookState>>>> = OnceLock::new();
// Raw `HHOOK` of the live LL hooks. Lock-free on purpose: `ll_*_proc` runs for EVERY
// system-wide key/mouse event (incl. mouse moves), and Windows stalls all input until the proc
// returns, so the hot path must never wait on a lock held by an install/uninstall/rebind.
static ACTIVE_KB_HHOOK: AtomicIsize = AtomicIsize::new(0);
static ACTIVE_MOUSE_HHOOK: AtomicIsize = AtomicIsize::new(0);

fn kb_state_slot() -> &'static RwLock<Option<Arc<SharedHookState>>> {
    ACTIVE_KB_STATE.get_or_init(|| RwLock::new(None))
}

fn mouse_state_slot() -> &'static RwLock<Option<Arc<SharedHookState>>> {
    ACTIVE_MOUSE_STATE.get_or_init(|| RwLock::new(None))
}

fn kb_hhook_slot() -> &'static AtomicIsize {
    &ACTIVE_KB_HHOOK
}

fn mouse_hhook_slot() -> &'static AtomicIsize {
    &ACTIVE_MOUSE_HHOOK
}

/// Pure decision outcome for `WH_KEYBOARD_LL` trigger-only suppression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KbHookAction {
    PassToNextHook,
    SuppressTrigger,
}

#[derive(Clone)]
pub struct KbDecision {
    pub action: KbHookAction,
    pub callbacks: Vec<HotkeyCallback>,
}

fn is_modifier_vk(vk: u32) -> bool {
    MODIFIER_VKS.contains(&vk)
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_kb_event(
    bindings: &[BindingSnapshot],
    suppressed_vks: &mut HashSet<u32>,
    vk: u32,
    is_down: bool,
    is_up: bool,
    alt_pressed: bool,
    ctrl_pressed: bool,
    shift_pressed: bool,
    should_suppress_fn: Option<&ShouldSuppressFn>,
) -> KbDecision {
    if vk == u32::from(VK_PACKET.0) || is_modifier_vk(vk) {
        return KbDecision {
            action: KbHookAction::PassToNextHook,
            callbacks: Vec::new(),
        };
    }

    if is_down {
        if suppressed_vks.contains(&vk) {
            return KbDecision {
                action: KbHookAction::SuppressTrigger,
                callbacks: Vec::new(),
            };
        }

        let mut callbacks = Vec::new();
        let mut suppress_this = false;

        for b in bindings {
            if !b.enabled || !b.use_ll_kb || b.vk != Some(vk) {
                continue;
            }
            if b.has_alt != alt_pressed || b.has_ctrl != ctrl_pressed || b.has_shift != shift_pressed
            {
                continue;
            }
            callbacks.push(Arc::clone(&b.callback));
            if b.suppress {
                let allowed = match should_suppress_fn {
                    Some(f) => {
                        let name = b.name.clone();
                        catch_unwind(AssertUnwindSafe(|| f(&name))).unwrap_or(false)
                    }
                    None => true,
                };
                if allowed {
                    suppress_this = true;
                }
            }
        }

        if suppress_this {
            suppressed_vks.insert(vk);
            return KbDecision {
                action: KbHookAction::SuppressTrigger,
                callbacks,
            };
        }

        return KbDecision {
            action: KbHookAction::PassToNextHook,
            callbacks,
        };
    }

    if is_up && suppressed_vks.remove(&vk) {
        return KbDecision {
            action: KbHookAction::SuppressTrigger,
            callbacks: Vec::new(),
        };
    }

    KbDecision {
        action: KbHookAction::PassToNextHook,
        callbacks: Vec::new(),
    }
}

fn is_vk_down(vk: i32) -> bool {
    // SAFETY: Plain Win32 async/thread key-state queries.
    unsafe { GetAsyncKeyState(vk) < 0 || GetKeyState(vk) < 0 }
}

thread_local! {
    static HOOK_SUPPRESSED_VKS: std::cell::RefCell<HashSet<u32>> = std::cell::RefCell::new(HashSet::new());
}

unsafe extern "system" fn ll_keyboard_proc(n_code: i32, w_param: WPARAM, l_param: LPARAM) -> LRESULT {
    let hhook_raw = kb_hhook_slot().load(Ordering::Acquire);
    let hhook = if hhook_raw == 0 {
        None
    } else {
        Some(HHOOK(hhook_raw as *mut std::ffi::c_void))
    };

    if n_code >= 0 && l_param.0 != 0 {
        // SAFETY: When `n_code >= 0` in `WH_KEYBOARD_LL`, `l_param` points to `KBDLLHOOKSTRUCT`.
        let kb = unsafe { &*(l_param.0 as *const KBDLLHOOKSTRUCT) };
        let msg = w_param.0 as u32;
        let is_down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
        let is_up = msg == WM_KEYUP || msg == WM_SYSKEYUP;

        let alt_pressed =
            (kb.flags.0 & LLKHF_ALTDOWN.0 != 0) || is_vk_down(i32::from(VK_MENU.0));
        let ctrl_pressed = is_vk_down(i32::from(VK_CONTROL.0));
        let shift_pressed = is_vk_down(i32::from(VK_SHIFT.0));

        let state_opt = kb_state_slot()
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(Arc::clone));

        if let Some(state) = state_opt {
            let bindings = state
                .bindings
                .read()
                .map(|g| g.clone())
                .unwrap_or_default();
            let suppress_fn = state
                .should_suppress
                .read()
                .ok()
                .and_then(|g| g.as_ref().map(Arc::clone));

            let decision = HOOK_SUPPRESSED_VKS.with(|cell| {
                let mut suppressed = cell.borrow_mut();
                evaluate_kb_event(
                    &bindings,
                    &mut suppressed,
                    kb.vkCode,
                    is_down,
                    is_up,
                    alt_pressed,
                    ctrl_pressed,
                    shift_pressed,
                    suppress_fn.as_ref(),
                )
            });

            for cb in decision.callbacks {
                state.dispatch_callback(cb);
            }

            if decision.action == KbHookAction::SuppressTrigger {
                return LRESULT(1);
            }
        }
    }

    // SAFETY: Forwarding unhandled/pass-through events to the next hook in chain.
    unsafe { CallNextHookEx(hhook, n_code, w_param, l_param) }
}

unsafe extern "system" fn ll_mouse_proc(n_code: i32, w_param: WPARAM, l_param: LPARAM) -> LRESULT {
    let hhook_raw = mouse_hhook_slot().load(Ordering::Acquire);
    let hhook = if hhook_raw == 0 {
        None
    } else {
        Some(HHOOK(hhook_raw as *mut std::ffi::c_void))
    };

    if n_code >= 0 && l_param.0 != 0 {
        let msg = w_param.0 as u32;
        let btn: Option<&'static str> = match msg {
            WM_MBUTTONDOWN => Some("middle"),
            WM_RBUTTONDOWN => Some("right"),
            WM_XBUTTONDOWN => {
                // SAFETY: `l_param` points to `MSLLHOOKSTRUCT` in `WH_MOUSE_LL`.
                let ms = unsafe { &*(l_param.0 as *const MSLLHOOKSTRUCT) };
                let xbutton = (ms.mouseData >> 16) & 0xFFFF;
                match xbutton {
                    1 => Some("mouse4"),
                    2 => Some("mouse5"),
                    _ => None,
                }
            }
            _ => None,
        };

        if let Some(clicked_btn) = btn
            && let Some(state) = mouse_state_slot()
                .read()
                .ok()
                .and_then(|g| g.as_ref().map(Arc::clone))
        {
            let alt_pressed = is_vk_down(i32::from(VK_MENU.0));
            let ctrl_pressed = is_vk_down(i32::from(VK_CONTROL.0));
            let shift_pressed = is_vk_down(i32::from(VK_SHIFT.0));

            let bindings = state
                .bindings
                .read()
                .map(|g| g.clone())
                .unwrap_or_default();
            for b in &bindings {
                if b.enabled
                    && b.is_mouse
                    && b.mouse_btn.as_deref() == Some(clicked_btn)
                    && b.has_alt == alt_pressed
                    && b.has_ctrl == ctrl_pressed
                    && b.has_shift == shift_pressed
                {
                    state.dispatch_callback(Arc::clone(&b.callback));
                }
            }
        }
    }

    // Mouse hook NEVER suppresses mouse clicks; always passes to next hook.
    // SAFETY: Forwarding mouse event to next hook.
    unsafe { CallNextHookEx(hhook, n_code, w_param, l_param) }
}

struct HookThreadRuntime {
    hhook_raw: isize,
    thread_id: u32,
    join_handle: JoinHandle<()>,
}

fn start_hook_thread(
    is_keyboard: bool,
    shared: Arc<SharedHookState>,
) -> Option<HookThreadRuntime> {
    if is_keyboard {
        if let Ok(mut slot) = kb_state_slot().write() {
            *slot = Some(Arc::clone(&shared));
        }
    } else if let Ok(mut slot) = mouse_state_slot().write() {
        *slot = Some(Arc::clone(&shared));
    }

    let (tx, rx) = mpsc::channel::<Result<(isize, u32), ()>>();
    let thread_name = if is_keyboard {
        "gh-kb-hook"
    } else {
        "gh-mouse-hook"
    };

    let join_handle = std::thread::Builder::new()
        .name(thread_name.to_owned())
        .spawn(move || {
            // SAFETY: Obtaining module handle & thread ID for dedicated LL hook message pump.
            let tid = unsafe { GetCurrentThreadId() };
            let hmod = unsafe { GetModuleHandleW(None) }
                .ok()
                .map(|m| HINSTANCE(m.0));
            let hook_id = if is_keyboard {
                WH_KEYBOARD_LL
            } else {
                WH_MOUSE_LL
            };
            let proc = if is_keyboard {
                ll_keyboard_proc as unsafe extern "system" fn(i32, WPARAM, LPARAM) -> LRESULT
            } else {
                ll_mouse_proc as unsafe extern "system" fn(i32, WPARAM, LPARAM) -> LRESULT
            };

            // SAFETY: Installing global low-level hook on this dedicated Win32 message-pump thread.
            let hhook = match unsafe { SetWindowsHookExW(hook_id, Some(proc), hmod, 0) } {
                Ok(h) if !h.is_invalid() => h,
                _ => {
                    let _ = tx.send(Err(()));
                    return;
                }
            };
            let raw = hhook.0 as isize;
            if is_keyboard {
                kb_hhook_slot().store(raw, Ordering::Release);
            } else {
                mouse_hhook_slot().store(raw, Ordering::Release);
            }

            let mut msg = MSG::default();
            // SAFETY: Force Win32 thread message queue creation before signaling readiness so an
            // immediate `PostThreadMessageW(WM_QUIT)` cannot fail with `ERROR_INVALID_THREAD_ID`.
            unsafe {
                let _ = PeekMessageW(&mut msg, None, 0, 0, PM_NOREMOVE);
            }

            let _ = tx.send(Ok((raw, tid)));

            loop {
                // SAFETY: Pumping Win32 messages until `WM_QUIT` (`GetMessageW` returns 0 or -1).
                let ret = unsafe { GetMessageW(&mut msg, None, 0, 0) };
                if ret.0 <= 0 {
                    break;
                }
                // SAFETY: Standard Win32 message dispatch.
                unsafe {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }

            // SAFETY: Ensure hook is unhooked before thread terminates and clear TLS suppressed set.
            unsafe {
                let _ = UnhookWindowsHookEx(hhook);
            }
            // Clear the slot only if it still holds this hook (a newer hook may already own it).
            if is_keyboard {
                HOOK_SUPPRESSED_VKS.with(|c| c.borrow_mut().clear());
                let _ = kb_hhook_slot().compare_exchange(raw, 0, Ordering::AcqRel, Ordering::Acquire);
            } else {
                let _ = mouse_hhook_slot().compare_exchange(raw, 0, Ordering::AcqRel, Ordering::Acquire);
            }
        })
        .ok()?;

    match rx.recv_timeout(Duration::from_secs(1)) {
        Ok(Ok((hhook_raw, thread_id))) => Some(HookThreadRuntime {
            hhook_raw,
            thread_id,
            join_handle,
        }),
        _ => {
            let _ = join_handle.join();
            None
        }
    }
}

fn clear_hook_state_slot(is_keyboard: bool, shared: &Arc<SharedHookState>) {
    if is_keyboard {
        if let Ok(mut slot) = kb_state_slot().write()
            && slot.as_ref().is_some_and(|s| Arc::ptr_eq(s, shared))
        {
            *slot = None;
        }
    } else if let Ok(mut slot) = mouse_state_slot().write()
        && slot.as_ref().is_some_and(|s| Arc::ptr_eq(s, shared))
    {
        *slot = None;
    }
}

fn join_hook_thread(runtime: Option<HookThreadRuntime>) {
    if let Some(rt) = runtime {
        // SAFETY: Unhooking handle and posting `WM_QUIT` to break `GetMessageW` cleanly.
        unsafe {
            let _ = UnhookWindowsHookEx(HHOOK(rt.hhook_raw as *mut std::ffi::c_void));
            let _ = PostThreadMessageW(rt.thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
        }
        if rt.join_handle.thread().id() != std::thread::current().id() {
            let _ = rt.join_handle.join();
        }
    }
}

struct InnerManager {
    bindings: BTreeMap<String, BindingEntry>,
    binding_order: Vec<String>,
    suppress_triggers: bool,
    kb_hook: Option<HookThreadRuntime>,
    mouse_hook: Option<HookThreadRuntime>,
    callback_worker: Option<JoinHandle<()>>,
}

/// Multi-binding low-level Win32 keyboard & mouse hotkey manager (`src/hotkey.py:83-636`).
///
/// Supports up to all 9 `AppConfig` actions with lazy hook installation, trigger-only VK
/// suppression gated on foreground game/HUD windows, and live rebinding.
pub struct UnifiedHotkeyManager {
    shared: Arc<SharedHookState>,
    inner: Mutex<InnerManager>,
    is_running: AtomicBool,
    /// `false` under `GH_DESKTOP_SANDBOX=1`: bindings are parsed, validated and reported as
    /// usual, but no global `WH_KEYBOARD_LL` / `WH_MOUSE_LL` hook is installed, so black-box
    /// test instances never see or delay the user's real keyboard/mouse input.
    install_os_hooks: bool,
}

impl Default for UnifiedHotkeyManager {
    fn default() -> Self {
        Self::new(true, None)
    }
}

impl UnifiedHotkeyManager {
    /// Installs real OS hooks unless `GH_DESKTOP_SANDBOX=1` (black-box sandbox) is set.
    pub fn new(suppress_triggers: bool, should_suppress: Option<ShouldSuppressFn>) -> Self {
        Self::with_os_hooks(
            suppress_triggers,
            should_suppress,
            !crate::runtime::is_desktop_sandbox_env(),
        )
    }

    /// Explicit constructor; `install_os_hooks = false` keeps all binding bookkeeping but never
    /// calls `SetWindowsHookExW` (desktop-safe tests / sandbox).
    pub fn with_os_hooks(
        suppress_triggers: bool,
        should_suppress: Option<ShouldSuppressFn>,
        install_os_hooks: bool,
    ) -> Self {
        Self {
            shared: Arc::new(SharedHookState {
                bindings: RwLock::new(Vec::new()),
                should_suppress: RwLock::new(should_suppress),
                callback_tx: Mutex::new(None),
            }),
            inner: Mutex::new(InnerManager {
                bindings: BTreeMap::new(),
                binding_order: Vec::new(),
                suppress_triggers,
                kb_hook: None,
                mouse_hook: None,
                callback_worker: None,
            }),
            is_running: AtomicBool::new(false),
            install_os_hooks,
        }
    }

    pub fn with_sync_binding(
        hotkey_str: &str,
        callback: HotkeyCallback,
        suppress_triggers: bool,
        should_suppress: Option<ShouldSuppressFn>,
    ) -> Self {
        let mgr = Self::new(suppress_triggers, should_suppress);
        mgr.register("sync", hotkey_str, callback, true, None);
        mgr
    }

    fn refresh_snapshot_locked(&self, inner: &InnerManager) {
        let snaps: Vec<BindingSnapshot> = inner
            .binding_order
            .iter()
            .filter_map(|name| inner.bindings.get(name).map(BindingSnapshot::from))
            .collect();
        if let Ok(mut g) = self.shared.bindings.write() {
            *g = snaps;
        }
    }

    fn ensure_callback_worker_locked(&self, inner: &mut InnerManager) {
        if inner.callback_worker.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel::<HotkeyCallback>();
        if let Ok(mut g) = self.shared.callback_tx.lock() {
            *g = Some(tx);
        }
        if let Ok(handle) = std::thread::Builder::new()
            .name("gh-hotkey-dispatch".to_owned())
            .spawn(move || {
                while let Ok(cb) = rx.recv() {
                    let _ = catch_unwind(AssertUnwindSafe(|| cb()));
                }
            })
        {
            inner.callback_worker = Some(handle);
        }
    }

    fn sync_kb_hook_state_locked(&self, inner: &mut InnerManager) -> Option<HookThreadRuntime> {
        let running = self.is_running.load(Ordering::Acquire);
        let need_kb = running
            && inner.bindings.values().any(|b| {
                b.enabled && !b.parsed.is_mouse && b.use_ll_kb && b.vk.is_some()
            });
        if need_kb && inner.kb_hook.is_none() && self.install_os_hooks {
            self.ensure_callback_worker_locked(inner);
            inner.kb_hook = start_hook_thread(true, Arc::clone(&self.shared));
            None
        } else if !need_kb && inner.kb_hook.is_some() {
            clear_hook_state_slot(true, &self.shared);
            inner.kb_hook.take()
        } else {
            None
        }
    }

    fn sync_mouse_hook_state_locked(&self, inner: &mut InnerManager) -> Option<HookThreadRuntime> {
        let running = self.is_running.load(Ordering::Acquire);
        let need_mouse = running
            && inner
                .bindings
                .values()
                .any(|b| b.enabled && b.parsed.is_mouse);
        if need_mouse && inner.mouse_hook.is_none() && self.install_os_hooks {
            self.ensure_callback_worker_locked(inner);
            inner.mouse_hook = start_hook_thread(false, Arc::clone(&self.shared));
            None
        } else if !need_mouse && inner.mouse_hook.is_some() {
            clear_hook_state_slot(false, &self.shared);
            inner.mouse_hook.take()
        } else {
            None
        }
    }

    pub fn register(
        &self,
        name: &str,
        hotkey_str: &str,
        callback: HotkeyCallback,
        enabled: bool,
        suppress_override: Option<bool>,
    ) {
        let clean_key = hotkey_str.trim().to_ascii_lowercase();
        let parsed = parse_hotkey(&clean_key);
        let vk = if parsed.is_mouse {
            None
        } else {
            hotkey_to_vk_combo(&clean_key).map(|(vk, _)| vk)
        };

        let (kb_stop, mouse_stop) = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            let suppress = suppress_override.unwrap_or(inner.suppress_triggers);
            let use_ll_kb = enabled && !parsed.is_mouse && vk.is_some();

            if !inner.bindings.contains_key(name) {
                inner.binding_order.push(name.to_owned());
            }
            inner.bindings.insert(
                name.to_owned(),
                BindingEntry {
                    name: name.to_owned(),
                    hotkey_str: clean_key,
                    callback,
                    parsed,
                    vk,
                    use_ll_kb,
                    suppress,
                    enabled,
                },
            );
            self.refresh_snapshot_locked(&inner);

            if self.is_running.load(Ordering::Acquire) {
                (
                    self.sync_kb_hook_state_locked(&mut inner),
                    self.sync_mouse_hook_state_locked(&mut inner),
                )
            } else {
                (None, None)
            }
        };
        join_hook_thread(kb_stop);
        join_hook_thread(mouse_stop);
    }

    pub fn start(&self) -> bool {
        let (ok, kb_stop, mouse_stop) = {
            let Ok(mut inner) = self.inner.lock() else {
                return false;
            };
            self.is_running.store(true, Ordering::Release);
            for b in inner.bindings.values_mut() {
                b.use_ll_kb = b.enabled && !b.parsed.is_mouse && b.vk.is_some();
            }
            self.refresh_snapshot_locked(&inner);
            let kb_stop = self.sync_kb_hook_state_locked(&mut inner);
            let mouse_stop = self.sync_mouse_hook_state_locked(&mut inner);

            let needs_kb = inner
                .bindings
                .values()
                .any(|b| b.enabled && !b.parsed.is_mouse && b.vk.is_some());
            let needs_mouse = inner
                .bindings
                .values()
                .any(|b| b.enabled && b.parsed.is_mouse);
            let ok = !self.install_os_hooks
                || ((!needs_kb || inner.kb_hook.is_some())
                    && (!needs_mouse || inner.mouse_hook.is_some()));
            (ok, kb_stop, mouse_stop)
        };
        join_hook_thread(kb_stop);
        join_hook_thread(mouse_stop);
        ok
    }

    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Release);
        let (kb_rt, mouse_rt, worker) = {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            for b in inner.bindings.values_mut() {
                b.use_ll_kb = false;
            }
            self.refresh_snapshot_locked(&inner);

            clear_hook_state_slot(true, &self.shared);
            clear_hook_state_slot(false, &self.shared);
            let kb_rt = inner.kb_hook.take();
            let mouse_rt = inner.mouse_hook.take();

            if let Ok(mut tx_slot) = self.shared.callback_tx.lock() {
                *tx_slot = None;
            }
            let worker = inner.callback_worker.take();
            (kb_rt, mouse_rt, worker)
        };

        join_hook_thread(kb_rt);
        join_hook_thread(mouse_rt);

        if let Some(worker) = worker
            && worker.thread().id() != std::thread::current().id()
        {
            let _ = worker.join();
        }
    }

    pub fn set_enabled(&self, name: &str, enabled: bool) -> bool {
        let (kb_stop, mouse_stop) = {
            let Ok(mut inner) = self.inner.lock() else {
                return false;
            };
            let Some(entry) = inner.bindings.get_mut(name) else {
                return false;
            };
            entry.enabled = enabled;
            entry.use_ll_kb = enabled && !entry.parsed.is_mouse && entry.vk.is_some();
            self.refresh_snapshot_locked(&inner);

            if self.is_running.load(Ordering::Acquire) {
                (
                    self.sync_kb_hook_state_locked(&mut inner),
                    self.sync_mouse_hook_state_locked(&mut inner),
                )
            } else {
                (None, None)
            }
        };
        join_hook_thread(kb_stop);
        join_hook_thread(mouse_stop);
        true
    }

    pub fn set_suppress_triggers(&self, enabled: bool) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        inner.suppress_triggers = enabled;
        for b in inner.bindings.values_mut() {
            b.suppress = enabled;
        }
        self.refresh_snapshot_locked(&inner);
    }

    pub fn set_should_suppress(&self, callback: Option<ShouldSuppressFn>) {
        if let Ok(mut g) = self.shared.should_suppress.write() {
            *g = callback;
        }
    }

    pub fn update_binding(&self, name: &str, new_hotkey_str: &str) -> bool {
        let clean = new_hotkey_str.trim().to_ascii_lowercase();
        if clean.is_empty() {
            return false;
        }
        let parsed = parse_hotkey(&clean);
        let vk = if parsed.is_mouse {
            None
        } else {
            hotkey_to_vk_combo(&clean).map(|(vk, _)| vk)
        };
        if !parsed.is_mouse && vk.is_none() {
            return false;
        }

        let (kb_stop, mouse_stop) = {
            let Ok(mut inner) = self.inner.lock() else {
                return false;
            };
            let Some(entry) = inner.bindings.get_mut(name) else {
                return false;
            };
            entry.hotkey_str = clean;
            entry.parsed = parsed;
            entry.vk = vk;
            entry.use_ll_kb = entry.enabled && !entry.parsed.is_mouse && entry.vk.is_some();
            self.refresh_snapshot_locked(&inner);

            if self.is_running.load(Ordering::Acquire) {
                (
                    self.sync_kb_hook_state_locked(&mut inner),
                    self.sync_mouse_hook_state_locked(&mut inner),
                )
            } else {
                (None, None)
            }
        };
        join_hook_thread(kb_stop);
        join_hook_thread(mouse_stop);
        true
    }

    pub fn update_hotkey(&self, new_hotkey_str: &str) -> bool {
        let target = {
            let Ok(inner) = self.inner.lock() else {
                return false;
            };
            if inner.bindings.contains_key("sync") {
                Some("sync".to_owned())
            } else {
                inner.binding_order.first().cloned()
            }
        };
        match target {
            Some(name) => self.update_binding(&name, new_hotkey_str),
            None => false,
        }
    }

    pub fn get_display_name(&self, name: &str) -> String {
        let Ok(inner) = self.inner.lock() else {
            return "NONE".to_owned();
        };
        if let Some(b) = inner.bindings.get(name) {
            return format_display_name(&b.hotkey_str);
        }
        if let Some(first_name) = inner.binding_order.first()
            && let Some(b) = inner.bindings.get(first_name)
        {
            return format_display_name(&b.hotkey_str);
        }
        "NONE".to_owned()
    }

    pub fn get_hotkey_str(&self, name: &str) -> String {
        let Ok(inner) = self.inner.lock() else {
            return String::new();
        };
        inner
            .bindings
            .get(name)
            .map(|b| b.hotkey_str.clone())
            .unwrap_or_default()
    }

    pub fn is_kb_hook_active(&self) -> bool {
        self.inner.lock().is_ok_and(|g| g.kb_hook.is_some())
    }

    pub fn is_mouse_hook_active(&self) -> bool {
        self.inner.lock().is_ok_and(|g| g.mouse_hook.is_some())
    }

    pub fn binding_enabled(&self, name: &str) -> Option<bool> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.bindings.get(name).map(|b| b.enabled))
    }

    pub fn binding_suppress(&self, name: &str) -> Option<bool> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.bindings.get(name).map(|b| b.suppress))
    }
}

impl Drop for UnifiedHotkeyManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn parse_hotkey_and_display_name_parity() {
        let p = parse_hotkey("Shift+Alt+K");
        assert!(!p.is_mouse);
        assert_eq!(p.main_key, "k");
        assert_eq!(p.clean_str, "alt+shift+k");

        let pm = parse_hotkey("alt+xbutton1");
        assert!(pm.is_mouse);
        assert_eq!(pm.mouse_btn.as_deref(), Some("mouse4"));
        assert_eq!(pm.clean_str, "alt+mouse4");

        assert_eq!(
            format_display_name("alt+mouse4"),
            "Alt + Mouse 4 (Chuột hông)"
        );
        assert_eq!(format_display_name("middle"), "Chuột giữa (Con lăn)");
        assert_eq!(format_display_name("right"), "Chuột phải");
        assert_eq!(format_display_name(""), "");
    }

    #[test]
    fn sandbox_manager_tracks_bindings_without_installing_os_hooks() {
        let mgr = UnifiedHotkeyManager::with_os_hooks(true, None, false);
        mgr.register("sync", "f9", Arc::new(|| {}), true, None);
        mgr.register("switch", "mouse5", Arc::new(|| {}), true, None);

        assert!(mgr.start(), "start must succeed without OS hooks");
        assert!(!mgr.is_kb_hook_active());
        assert!(!mgr.is_mouse_hook_active());

        // Bookkeeping & rebinding behave exactly like the hooked manager.
        assert!(mgr.set_enabled("switch", false));
        assert!(mgr.update_binding("sync", "alt+shift+k"));
        assert_eq!(mgr.get_display_name("sync"), "Alt + Shift + K");
        assert!(mgr.update_hotkey("mouse4"));
        assert_eq!(mgr.get_hotkey_str("sync"), "mouse4");
        assert!(!mgr.is_kb_hook_active());
        assert!(!mgr.is_mouse_hook_active());
        mgr.stop();
    }

    #[test]
    fn hook_lifecycle_lazy_enable_disable_and_live_rebind() {
        // Real OS hooks on purpose (hook lifecycle); runs exclusively via nextest `desktop-hooks`.
        let mgr = UnifiedHotkeyManager::with_os_hooks(true, Some(Arc::new(|_| false)), true);
        mgr.register("sync", "f9", Arc::new(|| {}), true, None);
        mgr.register("switch", "mouse5", Arc::new(|| {}), false, None);

        assert!(mgr.start());
        assert!(mgr.is_kb_hook_active());
        assert!(
            !mgr.is_mouse_hook_active(),
            "mouse hook must stay uninstalled while switch binding is disabled"
        );
        assert_eq!(mgr.get_display_name("sync"), "F9");
        assert_eq!(mgr.get_display_name("missing_action"), "F9");

        // Enabling mouse binding installs WH_MOUSE_LL; disabling removes it.
        assert!(mgr.set_enabled("switch", true));
        assert!(mgr.is_mouse_hook_active());
        assert!(mgr.set_enabled("switch", false));
        assert!(!mgr.is_mouse_hook_active());

        // Toggling suppression & live rebind keeps KB hook alive
        mgr.set_suppress_triggers(false);
        assert_eq!(mgr.binding_suppress("sync"), Some(false));
        mgr.set_suppress_triggers(true);
        assert_eq!(mgr.binding_suppress("sync"), Some(true));

        assert!(mgr.update_binding("sync", "alt+shift+k"));
        assert_eq!(mgr.get_hotkey_str("sync"), "alt+shift+k");
        assert_eq!(mgr.get_display_name("sync"), "Alt + Shift + K");
        assert!(mgr.is_kb_hook_active());

        // Switching sync to mouse removes KB hook and installs mouse hook
        assert!(mgr.update_hotkey("mouse4"));
        assert!(!mgr.is_kb_hook_active());
        assert!(mgr.is_mouse_hook_active());

        mgr.stop();
        assert!(!mgr.is_kb_hook_active());
        assert!(!mgr.is_mouse_hook_active());
    }

    #[test]
    fn trigger_only_vk_suppression_logic_respects_modifiers_repeats_and_gating() {
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_cb = Arc::clone(&hits);
        let bindings = vec![BindingSnapshot {
            name: "nav_mark".to_owned(),
            callback: Arc::new(move || {
                hits_cb.fetch_add(1, Ordering::Relaxed);
            }),
            is_mouse: false,
            mouse_btn: None,
            has_alt: true,
            has_ctrl: false,
            has_shift: false,
            vk: Some(0x4B), // 'K'
            use_ll_kb: true,
            suppress: true,
            enabled: true,
        }];
        let mut suppressed = HashSet::new();

        // 1. VK_PACKET and modifier VKs are always passed through and never added to suppressed_vks
        let pkt = evaluate_kb_event(
            &bindings,
            &mut suppressed,
            u32::from(VK_PACKET.0),
            true,
            false,
            true,
            false,
            false,
            None,
        );
        assert_eq!(pkt.action, KbHookAction::PassToNextHook);
        assert!(pkt.callbacks.is_empty());

        for &mod_vk in &MODIFIER_VKS {
            let m = evaluate_kb_event(
                &bindings,
                &mut suppressed,
                mod_vk,
                true,
                false,
                true,
                false,
                false,
                None,
            );
            assert_eq!(m.action, KbHookAction::PassToNextHook);
            assert!(!suppressed.contains(&mod_vk));
        }

        // 2. When should_suppress returns false, callback fires WITHOUT suppressing trigger key
        let gate_off: ShouldSuppressFn = Arc::new(|_| false);
        let pass_d = evaluate_kb_event(
            &bindings,
            &mut suppressed,
            0x4B,
            true,
            false,
            true,
            false,
            false,
            Some(&gate_off),
        );
        assert_eq!(pass_d.action, KbHookAction::PassToNextHook);
        assert_eq!(pass_d.callbacks.len(), 1);
        pass_d.callbacks[0]();
        assert_eq!(hits.load(Ordering::Relaxed), 1);
        assert!(!suppressed.contains(&0x4B));

        // 3. When should_suppress returns true, initial down fires callback & suppresses trigger VK
        let gate_on: ShouldSuppressFn = Arc::new(|_| true);
        let sup_d = evaluate_kb_event(
            &bindings,
            &mut suppressed,
            0x4B,
            true,
            false,
            true,
            false,
            false,
            Some(&gate_on),
        );
        assert_eq!(sup_d.action, KbHookAction::SuppressTrigger);
        assert_eq!(sup_d.callbacks.len(), 1);
        assert!(suppressed.contains(&0x4B));

        // 4. Auto-repeat keydown while VK is suppressed is suppressed WITHOUT re-firing callback
        let rep = evaluate_kb_event(
            &bindings,
            &mut suppressed,
            0x4B,
            true,
            false,
            true,
            false,
            false,
            Some(&gate_on),
        );
        assert_eq!(rep.action, KbHookAction::SuppressTrigger);
        assert!(rep.callbacks.is_empty());

        // 5. Matching keyup consumes suppressed VK and suppresses orphaned keyup
        let up = evaluate_kb_event(
            &bindings,
            &mut suppressed,
            0x4B,
            false,
            true,
            true,
            false,
            false,
            Some(&gate_on),
        );
        assert_eq!(up.action, KbHookAction::SuppressTrigger);
        assert!(!suppressed.contains(&0x4B));

        // Subsequent unrelated keyup passes through
        let up2 = evaluate_kb_event(
            &bindings,
            &mut suppressed,
            0x4B,
            false,
            true,
            true,
            false,
            false,
            Some(&gate_on),
        );
        assert_eq!(up2.action, KbHookAction::PassToNextHook);
    }

    #[test]
    fn all_9_action_bindings_register_cleanly() {
        let mgr = UnifiedHotkeyManager::with_os_hooks(true, None, true);
        let defaults = [
            ("sync", "alt+f11", true),
            ("debug", "alt+f12", true),
            ("switch", "alt+q", false),
            ("nav_mark", "alt+k", true),
            ("nav_undo", "alt+shift+k", true),
            ("nav_guide", "alt+h", true),
            ("nav_next", "alt+n", true),
            ("nav_layout", "alt+shift+h", true),
            ("nav_toggle", "alt+x", true),
        ];
        for (idx, &(name, hk, enabled)) in defaults.iter().enumerate() {
            assert_eq!(name, HOTKEY_ACTION_NAMES[idx]);
            mgr.register(name, hk, Arc::new(|| {}), enabled, None);
            assert_eq!(mgr.get_hotkey_str(name), hk);
            assert_eq!(mgr.binding_enabled(name), Some(enabled));
        }
    }

    #[test]
    fn stop_does_not_deadlock_when_in_flight_callback_reads_manager() {
        let mgr = Arc::new(UnifiedHotkeyManager::with_os_hooks(true, None, true));
        let entered = Arc::new(AtomicBool::new(false));
        let proceed = Arc::new(AtomicBool::new(false));

        let mgr_cb = Arc::clone(&mgr);
        let entered_cb = Arc::clone(&entered);
        let proceed_cb = Arc::clone(&proceed);
        mgr.register(
            "sync",
            "f9",
            Arc::new(move || {
                entered_cb.store(true, Ordering::Release);
                while !proceed_cb.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                let _ = mgr_cb.get_display_name("sync");
                let _ = mgr_cb.set_enabled("sync", false);
            }),
            true,
            None,
        );
        assert!(mgr.start());

        // Dispatch the callback on gh-hotkey-dispatch, wait until it is executing, then invoke
        // stop() on another thread so stop() reaches worker.join() before releasing the callback.
        let snap_cb = mgr.shared.bindings.read().expect("bindings")[0]
            .callback
            .clone();
        mgr.shared.dispatch_callback(snap_cb);
        while !entered.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
        let mgr_stop = Arc::clone(&mgr);
        let stopper = std::thread::spawn(move || {
            mgr_stop.stop();
        });
        std::thread::sleep(Duration::from_millis(20));
        proceed.store(true, Ordering::Release);
        stopper.join().expect("stop() must not deadlock with re-entrant callback");
        assert!(!mgr.is_kb_hook_active());
    }
}
