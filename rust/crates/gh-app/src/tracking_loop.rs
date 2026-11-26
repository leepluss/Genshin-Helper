//! 60 Hz heading / 20 Hz full minimap tracking loop, Map (M) FSM, HUD sync, LAN push, and HDR warning.

use crate::mmap_view::ViewFrame;
use crate::state::{AppState, WINDOW_TITLES, json_f64, json_i64, json_truthy};
use gh_core::frame::{FrameBuf, RectI32};
use gh_core::i18n::{get_language, t};
use gh_core::multimap::MultimapDb;
use gh_core::nav_controller::NavState;
use gh_core::ui_layout::compute_ui_layout;
use gh_cv::locator::{DEFAULT_AREA_NAMES, LocateResult, MapLocator};
use gh_cv::map_state::{
    CLOSE_PROBE_DEFAULT_THRESHOLD, MAX_RETRIES, MapScreenStateMachine, MapState, MapStepResult,
};
use gh_cv::minimap_tracker::{MinimapPose, TrackerState};
use gh_hud::layered::query_monitor_work_area;
use gh_hud::mmap_overlay::MMapOverlayMode;
use gh_hud::toast::ToastKind;
use gh_net::pin_guide::HoYoLABLabelCatalog;
use gh_win32::capture::WindowFinder;
use gh_win32::game_settings::check_uipi_status;
use gh_win32::runtime::HighResTimer;
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tokio::sync::watch;

pub const MINIMAP_TICK_S: f64 = 1.0 / 60.0;
/// Tick period while the M-Map is open with route or single-pin guidance (60 fps target).
pub const MMAP_GUIDE_TICK_S: f64 = 1.0 / 60.0;
/// With nothing to point at for this long, the M-Map overlay shows the text banner instead
/// (the first registration usually lands well within it after the map opens).
pub const MMAP_FALLBACK_DELAY: Duration = Duration::from_millis(700);
/// A longer gap between M-Map ticks restarts that grace period (map reopened, game refocused).
const MMAP_FALLBACK_GAP: Duration = Duration::from_millis(300);
pub const MINIMAP_FULL_STEP_S: f64 = 0.05;
pub const WINDOW_LOOKUP_CACHE_S: f64 = 0.25;
pub const HDR_CLIP_STREAK_WARN: usize = 8;

#[inline]
fn angle_diff_deg(a: Option<f64>, b: Option<f64>) -> f64 {
    match (a, b) {
        (None, None) => 0.0,
        (None, _) | (_, None) => 180.0,
        (Some(va), Some(vb)) => (((va - vb + 180.0).rem_euclid(360.0)) - 180.0).abs(),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct MmBcastState {
    map_id: i32,
    area_id: Option<i32>,
    lat: f64,
    lng: f64,
    heading: Option<f64>,
    camera_heading: Option<f64>,
    is_underground: bool,
    floor_id: Option<i32>,
}

fn resolve_minimap_area(
    maps_db: &MultimapDb,
    locator: Option<&MapLocator>,
    map_id: i32,
    world_x: f64,
    world_y: f64,
) -> (Option<i32>, Option<String>) {
    if map_id != 2 {
        return (None, None);
    }
    let Some((aid, db_name)) = maps_db.nearest_area_for_pos(map_id, world_x, world_y) else {
        return (None, None);
    };
    let name = db_name
        .or_else(|| locator.and_then(|l| l.area_names_map().get(&aid).cloned()))
        .or_else(|| {
            DEFAULT_AREA_NAMES
                .iter()
                .find(|&&(id, _)| id == aid)
                .map(|&(_, n)| n.to_owned())
        });
    (Some(aid), name)
}

#[derive(Debug, Clone, PartialEq)]
struct HudSig {
    state: NavState,
    target_id: Option<String>,
    dist_tenth: Option<i64>,
    delta_tenth: Option<i64>,
    floor_mismatch: bool,
    route_target_name: String,
    route_step: Option<usize>,
    route_total: Option<usize>,
    route_target_map_id: Option<i32>,
    route_section_label: String,
    route_stop_kind: String,
    remaining_count: usize,
    show_spot_card: bool,
    rect: (i32, i32, i32, i32),
    position: String,
    lang: String,
    sync_label: String,
    guide_label: String,
    desc: String,
    image_url: String,
}

#[derive(Debug, Clone)]
enum TickMode {
    MinimapFast,
    Minimap,
    MapM,
    MapMPaused,
    IdleClosed,
}

struct TickOutput {
    mode: TickMode,
    full_frame: Option<Arc<FrameBuf>>,
    step_res: Option<MapStepResult>,
    pose: Option<MinimapPose>,
    /// M-Map view tracking result (route guidance only).
    view: Option<ViewFrame>,
    /// True when the M-Map frame was captured via overlay-blind per-window WGC (`CreateForWindow`),
    /// making it CV-safe to set `mmap_overlay` to `WDA_NONE` for OBS/ShadowPlay recording.
    overlay_capture_safe: bool,
}

struct PendingVerify {
    ticks: usize,
    res: LocateResult,
}

struct TickParams {
    game_hwnd: isize,
    cl: i32,
    ct: i32,
    cw: i32,
    ch: i32,
    is_closed: bool,
    do_auto_sync: bool,
    do_minimap: bool,
    fast_ok: bool,
    /// Track the M-Map view for route guidance; `guide_map` = the route's map (slider-scale
    /// fallback until a view is registered).
    guide_view: bool,
    guide_map: i32,
    /// User wants HUDs visible in screen recordings AND M-Map route guidance is active.
    want_window_capture: bool,
}

/// Captured M-Map frame → state machine step (+ view tracking when guiding).
fn step_map_frame(
    state: &AppState,
    frame: FrameBuf,
    p: &TickParams,
    overlay_capture_safe: bool,
) -> TickOutput {
    let step = state
        .state_machine
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .step_frame(Some(frame.as_view()));
    let frame = Arc::new(frame);
    let view = match state.mmap_view.as_ref() {
        Some(engine) if p.guide_view && step.state != MapState::Closed => {
            // Mouse hotspot in client px; the game cursor is masked out of the motion estimate.
            let cursor = WindowFinder::cursor_screen_pos()
                .map(|(x, y)| (f64::from(x - p.cl), f64::from(y - p.ct)))
                .filter(|&(x, y)| x >= 0.0 && y >= 0.0 && x < f64::from(p.cw) && y < f64::from(p.ch));
            Some(engine.step(Arc::clone(&frame), step.check.slider_res.as_ref(), cursor, p.guide_map))
        }
        _ => None,
    };
    TickOutput {
        mode: TickMode::MapM,
        full_frame: Some(frame),
        step_res: Some(step),
        pose: None,
        view,
        overlay_capture_safe,
    }
}

fn run_tick_worker(state: &AppState, p: TickParams) -> TickOutput {
    let TickParams {
        cl,
        ct,
        cw,
        ch,
        is_closed,
        do_auto_sync,
        do_minimap,
        fast_ok,
        ..
    } = p;
    if fast_ok {
        let fast_rect_opt = {
            let mut tr_guard = state.tracker.lock().unwrap_or_else(|e| e.into_inner());
            tr_guard
                .as_mut()
                .and_then(|(tr, _)| tr.minimap_capture_rect(cw as u32, ch as u32, false).ok())
        };
        if let Some(rx_rect) = fast_rect_opt
            && rx_rect.width > 0
            && rx_rect.height > 0
        {
            let mm_fast = state.capturer.capture_region_bgra(
                cl + rx_rect.x,
                ct + rx_rect.y,
                rx_rect.width,
                rx_rect.height,
            );
            let now = state.start_instant.elapsed().as_secs_f64();
            let fast_pose = {
                let mut tr_guard = state.tracker.lock().unwrap_or_else(|e| e.into_inner());
                tr_guard.as_mut().and_then(|(tr, last_ts)| {
                    let ts = if now < *last_ts { *last_ts } else { now };
                    *last_ts = ts;
                    tr.step_headings_only(mm_fast.as_ref().map(|f| f.as_view()), Some(ts))
                })
            };
            if let Some(pose) = fast_pose {
                return TickOutput {
                    mode: TickMode::MinimapFast,
                    full_frame: None,
                    step_res: None,
                    pose: Some(pose),
                    view: None,
                    overlay_capture_safe: false,
                };
            }
        }
    }

    if is_closed {
        let probe_hit = {
            let mut sm = state
                .state_machine
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if sm.has_anchor_templates() {
                let pr = MapScreenStateMachine::get_close_probe_rect(cl, ct, cw, ch);
                if pr.width > 0 && pr.height > 0 {
                    let roi = state
                        .capturer
                        .capture_region(pr.x, pr.y, pr.width, pr.height);
                    sm.is_close_x_probe_frame(
                        roi.as_ref().map(|f| f.as_view()),
                        ch as usize,
                        CLOSE_PROBE_DEFAULT_THRESHOLD,
                    )
                    .0
                } else {
                    false
                }
            } else {
                let pr = MapScreenStateMachine::get_slider_probe_rect(cl, ct, cw, ch);
                if pr.width > 0 && pr.height > 0 {
                    let roi = state
                        .capturer
                        .capture_region(pr.x, pr.y, pr.width, pr.height);
                    MapScreenStateMachine::is_slider_probe_frame(
                        roi.as_ref().map(|f| f.as_view()),
                        cw as usize,
                        ch as usize,
                    )
                    .0
                } else {
                    false
                }
            }
        };

        if probe_hit {
            state.pause_minimap_tracker("map_m_probe");
            if do_auto_sync {
                let (frame_opt, overlay_capture_safe) = if p.want_window_capture && p.game_hwnd != 0
                {
                    match state
                        .capturer
                        .capture_window_client_outcome(p.game_hwnd, cl, ct, cw, ch)
                    {
                        gh_win32::wgc_hdr::WgcWindowCaptureOutcome::Frame(wf) => (Some(wf), true),
                        gh_win32::wgc_hdr::WgcWindowCaptureOutcome::WarmingUp
                        | gh_win32::wgc_hdr::WgcWindowCaptureOutcome::Failed => {
                            (state.capturer.capture_region(cl, ct, cw, ch), false)
                        }
                    }
                } else {
                    (state.capturer.capture_region(cl, ct, cw, ch), false)
                };
                let Some(frame) = frame_opt else {
                    return TickOutput {
                        mode: TickMode::MapM,
                        full_frame: None,
                        step_res: None,
                        pose: None,
                        view: None,
                        overlay_capture_safe: false,
                    };
                };
                return step_map_frame(state, frame, &p, overlay_capture_safe);
            }
            return TickOutput {
                mode: TickMode::MapMPaused,
                full_frame: None,
                step_res: None,
                pose: None,
                view: None,
                overlay_capture_safe: false,
            };
        }

        if do_minimap && let Ok(layout) = compute_ui_layout(cw as u32, ch as u32, false) {
            let mm = layout.minimap_rect;
            let pm = layout.paimon_maybe_rect;
            let cal_pad_x = (48.0 * layout.scale_ui).round() as i32;
            let cal_pad_y = (28.0 * layout.scale_ui).round() as i32;
            let hud_w = cw.min((mm.x + mm.width + cal_pad_x).max(pm.x + pm.width));
            let hud_h = ch.min((mm.y + mm.height + cal_pad_y).max(pm.y + pm.height));
            let hud_bgra = if hud_w > 0 && hud_h > 0 {
                state.capturer.capture_region_bgra(cl, ct, hud_w, hud_h)
            } else {
                None
            };

            let rec_opt = state
                .minimap_recorder
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();

            let now = state.start_instant.elapsed().as_secs_f64();
            let mut prev_ts_for_ring = 0.0_f64;
            let pose_opt = {
                let mut tr_guard = state.tracker.lock().unwrap_or_else(|e| e.into_inner());
                tr_guard.as_mut().and_then(|(tr, last_ts)| {
                    prev_ts_for_ring = *last_ts;
                    let ts = if now < *last_ts { *last_ts } else { now };
                    *last_ts = ts;
                    let pose = tr
                        .step_from_hud(
                            hud_bgra.as_ref().map(|f| f.as_view()),
                            cw as u32,
                            ch as u32,
                            Some(ts),
                            false,
                            None,
                        )
                        .ok()?;
                    if let Some(ref rec) = rec_opt
                        && rec.active()
                    {
                        let mut meta = Map::new();
                        meta.insert("state".to_owned(), json!(pose.state.as_str()));
                        meta.insert("match".to_owned(), json!(pose.match_method));
                        meta.insert("map_id".to_owned(), json!(pose.map_id));
                        meta.insert("world_x".to_owned(), json!(pose.world_x));
                        meta.insert("world_y".to_owned(), json!(pose.world_y));
                        meta.insert("span_world".to_owned(), json!(pose.span_world));
                        meta.insert("map_rotation_deg".to_owned(), json!(pose.map_rotation_deg));
                        meta.insert("underground".to_owned(), json!(pose.is_underground));
                        meta.insert("camera".to_owned(), json!(pose.camera_heading));
                        meta.insert("body".to_owned(), json!(pose.body_heading));
                        meta.insert("view".to_owned(), json!(pose.view_heading));
                        meta.insert("cal_dx_px".to_owned(), json!(pose.cal_dx_px));
                        meta.insert("cal_dy_px".to_owned(), json!(pose.cal_dy_px));
                        rec.add(hud_bgra.as_ref(), cw, ch, ts, Some(meta));
                    }
                    Some((pose, ts))
                })
            };

            if let Some((pose, ts)) = pose_opt {
                // Keep a 4 Hz (or immediate non-tracking) RAM ring of minimap crops for Alt+F12.
                if let Some(hud_frame) = hud_bgra
                    && (pose.state != TrackerState::Tracking
                        || (ts * 4.0).floor() != (prev_ts_for_ring * 4.0).floor())
                {
                    state.minimap_ring.push(
                        ts,
                        hud_frame,
                        json!({
                            "state": pose.state.as_str(),
                            "match_method": pose.match_method,
                            "map_id": pose.map_id,
                            "world_x": pose.world_x,
                            "world_y": pose.world_y,
                            "confidence": (pose.confidence * 1000.0).round() / 1000.0,
                            "is_underground": pose.is_underground,
                            "floor_name": pose.floor_name,
                            "stationary": pose.stationary,
                        }),
                    );
                }
                return TickOutput {
                    mode: TickMode::Minimap,
                    full_frame: None,
                    step_res: None,
                    pose: Some(pose),
                    view: None,
                    overlay_capture_safe: false,
                };
            }
        }

        return TickOutput {
            mode: TickMode::IdleClosed,
            full_frame: None,
            step_res: None,
            pose: None,
            view: None,
            overlay_capture_safe: false,
        };
    }

    state.pause_minimap_tracker("map_m_open");
    let (frame_opt, overlay_capture_safe) = if p.want_window_capture && p.game_hwnd != 0 {
        match state
            .capturer
            .capture_window_client_outcome(p.game_hwnd, cl, ct, cw, ch)
        {
            gh_win32::wgc_hdr::WgcWindowCaptureOutcome::Frame(wf) => (Some(wf), true),
            gh_win32::wgc_hdr::WgcWindowCaptureOutcome::WarmingUp
            | gh_win32::wgc_hdr::WgcWindowCaptureOutcome::Failed => {
                if state.mmap_overlay.capture_visible() {
                    // Zero-frame CV safety guard: switch overlay back to WDA_EXCLUDEFROMCAPTURE and
                    // skip this single transition frame rather than feeding overlay pixels into CV.
                    state.mmap_overlay.set_capture_visible(false);
                    (None, false)
                } else {
                    (state.capturer.capture_region(cl, ct, cw, ch), false)
                }
            }
        }
    } else if state.mmap_overlay.capture_visible() {
        state.mmap_overlay.set_capture_visible(false);
        (None, false)
    } else {
        (state.capturer.capture_region(cl, ct, cw, ch), false)
    };
    let Some(frame) = frame_opt else {
        return TickOutput {
            mode: TickMode::MapM,
            full_frame: None,
            step_res: None,
            pose: None,
            view: None,
            overlay_capture_safe: false,
        };
    };
    step_map_frame(state, frame, &p, overlay_capture_safe)
}

async fn sleep_or_shutdown(dur: Duration, shutdown_rx: &mut watch::Receiver<bool>) -> bool {
    if *shutdown_rx.borrow() {
        return true;
    }
    if dur.is_zero() {
        tokio::task::yield_now().await;
        return *shutdown_rx.borrow();
    }
    tokio::select! {
        _ = tokio::time::sleep(dur) => *shutdown_rx.borrow(),
        _ = shutdown_rx.changed() => *shutdown_rx.borrow(),
    }
}

#[allow(clippy::too_many_arguments)]
fn update_nav_hud_if_changed(
    state: &AppState,
    nav_hud_visible: &mut bool,
    last_hud_sig: &mut Option<HudSig>,
    guidance: gh_core::nav_controller::NavGuidance,
    game_rect: RectI32,
    position: &str,
    cur_lang: String,
    sync_label: String,
    guide_label: String,
) {
    let sig = HudSig {
        state: guidance.state,
        target_id: guidance.target.as_ref().map(|t| t.id.clone()),
        dist_tenth: Some((guidance.distance_m * 10.0).round() as i64),
        delta_tenth: guidance.delta_deg.map(|d| (d * 10.0).round() as i64),
        floor_mismatch: guidance.floor_mismatch,
        route_target_name: guidance.route_target_name.clone(),
        route_step: guidance.route_step,
        route_total: guidance.route_total,
        route_target_map_id: guidance.route_target_map_id,
        route_section_label: guidance.route_section_label.clone(),
        route_stop_kind: guidance.route_stop_kind.clone(),
        remaining_count: guidance.remaining_count,
        show_spot_card: guidance.show_spot_card,
        rect: (game_rect.x, game_rect.y, game_rect.width, game_rect.height),
        position: position.to_owned(),
        lang: cur_lang.clone(),
        sync_label: sync_label.clone(),
        guide_label: guide_label.clone(),
        desc: guidance
            .target
            .as_ref()
            .map(|t| t.description.clone())
            .unwrap_or_default(),
        image_url: guidance
            .target
            .as_ref()
            .map(|t| t.image_url.clone())
            .unwrap_or_default(),
    };

    if last_hud_sig.as_ref() != Some(&sig) {
        let wa = query_monitor_work_area(
            game_rect.x + game_rect.width / 2,
            game_rect.y + game_rect.height / 2,
        );
        state.nav_hud.update(
            guidance,
            Some(game_rect),
            Some(wa),
            None,
            Some(position),
            Some(&cur_lang),
            Some(&sync_label),
            0.0,
            None,
            Some(&guide_label),
        );
        *nav_hud_visible = true;
        *last_hud_sig = Some(sig);
    }
}

enum TrackCmd {
    Tick {
        state: Arc<AppState>,
        params: TickParams,
        reply: tokio::sync::oneshot::Sender<TickOutput>,
    },
    LocatePlayer {
        locator: Arc<gh_cv::locator::MapLocator>,
        frame: Arc<FrameBuf>,
        slider: Option<gh_cv::zoom_slider::ZoomSliderResult>,
        reply: tokio::sync::oneshot::Sender<Option<LocateResult>>,
    },
    ReleaseScratch,
}

struct TrackingWorker {
    tx: Option<std::sync::mpsc::SyncSender<TrackCmd>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl TrackingWorker {
    fn spawn() -> Option<Self> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<TrackCmd>(2);
        let handle = std::thread::Builder::new()
            .name("gh-track-worker".to_owned())
            .spawn(move || {
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        TrackCmd::Tick {
                            state,
                            params,
                            reply,
                        } => {
                            let out = run_tick_worker(&state, params);
                            let _ = reply.send(out);
                        }
                        TrackCmd::LocatePlayer {
                            locator,
                            frame,
                            slider,
                            reply,
                        } => {
                            let res = locator.locate_player_frame(
                                frame.as_view(),
                                false,
                                slider.as_ref(),
                            );
                            gh_cv::locator::release_thread_scratch();
                            let _ = reply.send(res);
                        }
                        TrackCmd::ReleaseScratch => {
                            gh_cv::locator::release_thread_scratch();
                        }
                    }
                }
                gh_cv::locator::release_thread_scratch();
            })
            .ok()?;
        Some(Self {
            tx: Some(tx),
            handle: Some(handle),
        })
    }

    async fn run_tick(&self, state: Arc<AppState>, params: TickParams) -> Option<TickOutput> {
        let tx = self.tx.as_ref()?;
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        if tx
            .send(TrackCmd::Tick {
                state,
                params,
                reply: reply_tx,
            })
            .is_err()
        {
            return None;
        }
        reply_rx.await.ok()
    }

    async fn locate_player(
        &self,
        locator: Arc<gh_cv::locator::MapLocator>,
        frame: Arc<FrameBuf>,
        slider: Option<gh_cv::zoom_slider::ZoomSliderResult>,
    ) -> Option<LocateResult> {
        let tx = self.tx.as_ref()?;
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        if tx
            .send(TrackCmd::LocatePlayer {
                locator,
                frame,
                slider,
                reply: reply_tx,
            })
            .is_err()
        {
            return None;
        }
        reply_rx.await.ok().flatten()
    }

    fn release_scratch(&self) {
        if let Some(ref tx) = self.tx {
            let _ = tx.try_send(TrackCmd::ReleaseScratch);
        }
    }
}

impl Drop for TrackingWorker {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

pub async fn run_tracking_loop(state: Arc<AppState>, mut shutdown_rx: watch::Receiver<bool>) {
    tracing::info!(
        "Tracking loop running (v5.4 Hybrid Minimap + 3-Anchor Map M FSM). Waiting for Genshin Impact window..."
    );

    let worker = TrackingWorker::spawn();
    let mut hires_timer = HighResTimer::new();
    let mut window_attached = false;
    let mut last_res: Option<(i32, i32)> = None;
    let mut pending_verify: Option<PendingVerify> = None;
    let mut uipi_checked_hwnd: Option<isize> = None;
    let mut nav_hud_visible = false;
    let mut last_hud_sig: Option<HudSig> = None;
    let mut last_mm_bcast_ts: Option<Instant> = None;
    let mut last_mm_bcast_state: Option<MmBcastState> = None;
    let mut last_full_step_mono: Option<Instant> = None;
    let mut last_pose_tracking = false;
    let mut cached_win_info: Option<(isize, i32, i32, i32, i32)> = None;
    let mut cached_win_mono: Option<Instant> = None;

    let mut rate_win_start: Option<Instant> = None;
    let mut rate_minimap_full: usize = 0;
    let mut rate_minimap_fast: usize = 0;
    let mut rate_work_ms: Vec<f64> = Vec::new();

    let mut gate_reject_streak: usize = 0;
    // (since, last tick) of the current stretch without a drawable M-Map guide scene.
    let mut mmap_no_scene: Option<(Instant, Instant)> = None;
    let mut last_nav_target_id: Option<String> = None;
    let mut hdr_clip_streak: usize = 0;
    let mut hdr_warned = false;
    let mut last_wgc_win_status = String::new();

    let reset_rate_window = |start: &mut Option<Instant>,
                             full: &mut usize,
                             fast: &mut usize,
                             work: &mut Vec<f64>| {
        *start = None;
        *full = 0;
        *fast = 0;
        work.clear();
    };

    let hide_nav_hud = |visible: &mut bool, sig: &mut Option<HudSig>| {
        if *visible || state.nav_hud.is_visible() {
            state.nav_hud.hide();
            *visible = false;
        }
        *sig = None;
    };

    let hide_mmap_overlay = || {
        state.mmap_overlay.hide();
        state.mmap_overlay.set_capture_visible(false);
        state.capturer.release_window_capture();
        if let Some(engine) = state.mmap_view.as_ref() {
            engine.reset();
        }
        if let Some(ref w) = worker {
            w.release_scratch();
        }
    };

    while state.server.is_running() && !*shutdown_rx.borrow() {
        if !state.engine_active.load(Ordering::Acquire) {
            if window_attached {
                window_attached = false;
                last_res = None;
            }
            state
                .state_machine
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .reset();
            pending_verify = None;
            uipi_checked_hwnd = None;
            last_mm_bcast_state = None;
            state.pause_minimap_tracker("standby");
            hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
            state.hide_radar();
            hide_mmap_overlay();
            hires_timer.set(false);
            reset_rate_window(
                &mut rate_win_start,
                &mut rate_minimap_full,
                &mut rate_minimap_fast,
                &mut rate_work_ms,
            );
            if sleep_or_shutdown(Duration::from_millis(350), &mut shutdown_rx).await {
                break;
            }
            continue;
        }

        let tick_start = Instant::now();
        let win_info = if let (Some(cached), Some(ts)) = (cached_win_info, cached_win_mono)
            && tick_start.duration_since(ts).as_secs_f64() < WINDOW_LOOKUP_CACHE_S
        {
            Some(cached)
        } else {
            let found = WindowFinder::get_game_window(&WINDOW_TITLES, false);
            cached_win_info = found;
            cached_win_mono = Some(tick_start);
            found
        };

        let Some((hwnd, left, top, width, height)) = win_info else {
            if window_attached {
                tracing::warn!("Genshin Impact window lost. Waiting for window...");
                window_attached = false;
                {
                    let mut gs = state.game_state.lock().unwrap_or_else(|e| e.into_inner());
                    gs.hwnd = None;
                    gs.rect = None;
                    gs.uipi_status = None;
                }
                last_res = None;
                uipi_checked_hwnd = None;
                if let Some(tray) = state
                    .tray
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                {
                    tray.set_uipi_warning(false);
                }
            }
            state
                .state_machine
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .reset();
            pending_verify = None;
            last_mm_bcast_state = None;
            state.pause_minimap_tracker("window_lost");
            hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
            state.hide_radar();
            // Game closed (e.g. Alt+F4 with M-Map open): the M-Map overlay windows are
            // independent topmost windows and would otherwise linger over the desktop.
            hide_mmap_overlay();
            reset_rate_window(
                &mut rate_win_start,
                &mut rate_minimap_full,
                &mut rate_minimap_fast,
                &mut rate_work_ms,
            );
            if sleep_or_shutdown(Duration::from_millis(800), &mut shutdown_rx).await {
                break;
            }
            continue;
        };

        let game_rect = RectI32 {
            x: left,
            y: top,
            width,
            height,
        };
        {
            let mut gs = state.game_state.lock().unwrap_or_else(|e| e.into_inner());
            gs.hwnd = Some(hwnd);
            gs.rect = Some(game_rect);
        }

        if width < 800 || height < 500 {
            state
                .state_machine
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .reset();
            pending_verify = None;
            state.pause_minimap_tracker("window_small");
            hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
            state.hide_radar();
            hide_mmap_overlay();
            reset_rate_window(
                &mut rate_win_start,
                &mut rate_minimap_full,
                &mut rate_minimap_fast,
                &mut rate_work_ms,
            );
            if sleep_or_shutdown(Duration::from_millis(500), &mut shutdown_rx).await {
                break;
            }
            continue;
        }

        if !window_attached || last_res != Some((width, height)) {
            tracing::info!("Attached to Genshin Impact window ({width}x{height})");
            window_attached = true;
            last_res = Some((width, height));
        }

        if hwnd != 0 && uipi_checked_hwnd != Some(hwnd) {
            uipi_checked_hwnd = Some(hwnd);
            if let Ok(uipi) = tokio::task::spawn_blocking(move || check_uipi_status(Some(hwnd))).await
            {
                let mismatch = uipi.mismatch;
                {
                    let mut gs = state.game_state.lock().unwrap_or_else(|e| e.into_inner());
                    gs.uipi_status = Some(uipi.clone());
                }
                if let Some(tray) = state
                    .tray
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                {
                    tray.set_uipi_warning(mismatch);
                }
                if mismatch {
                    tracing::warn!("[UIPI] {}", uipi.reason);
                    state.hud_toast.show_toast(
                        &t("uipi.warning.title", &[]),
                        &t("uipi.warning.short", &[]),
                        ToastKind::Warning,
                        5500,
                        Some(game_rect),
                    );
                }
            }
        }

        let cfg_snap = state.config.read().expect("config lock poisoned").clone();
        let has_tracker = state
            .tracker
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        let minimap_active = has_tracker && cfg_snap.minimap_tracking;
        let overlay_mode = MMapOverlayMode::from_str_lossy(&cfg_snap.mmap_overlay_mode);
        let route_needs_mmap = overlay_mode != MMapOverlayMode::Off && state.has_mmap_guide_target();
        let want_mmap_probe = cfg_snap.auto_sync_m || route_needs_mmap;
        if !want_mmap_probe && !minimap_active {
            state
                .state_machine
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .reset();
            pending_verify = None;
            state.pause_minimap_tracker("disabled");
            hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
            state.hide_radar();
            hide_mmap_overlay();
            reset_rate_window(
                &mut rate_win_start,
                &mut rate_minimap_full,
                &mut rate_minimap_fast,
                &mut rate_work_ms,
            );
            if sleep_or_shutdown(Duration::from_millis(500), &mut shutdown_rx).await {
                break;
            }
            continue;
        }

        let is_fg = WindowFinder::is_window_foreground(hwnd);
        let sm_state = state
            .state_machine
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .state();
        if !is_fg && sm_state != MapState::Settling && pending_verify.is_none() {
            state.pause_minimap_tracker("background");
            hires_timer.set(false);
            hide_mmap_overlay();
            reset_rate_window(
                &mut rate_win_start,
                &mut rate_minimap_full,
                &mut rate_minimap_fast,
                &mut rate_work_ms,
            );
            let mut hud_hwnds: Vec<isize> = state.nav_hud.hwnd().into_iter().collect();
            hud_hwnds.extend(state.mmap_overlay.hwnds());
            if state.is_radar_active()
                && let Ok(guard) = state.radar_map.lock()
                && let Some(ref ctrl) = *guard
                && let Some(rh) = ctrl.raw_hwnd()
            {
                hud_hwnds.push(rh);
            }
            let keep_hud = (nav_hud_visible || state.is_radar_active())
                && WindowFinder::is_game_visible_unfocused(
                    hwnd,
                    (left, top, width, height),
                    &hud_hwnds,
                    true,
                );
            if !keep_hud {
                hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
                state.hide_radar();
            }
            if sleep_or_shutdown(Duration::from_millis(250), &mut shutdown_rx).await {
                break;
            }
            continue;
        }

        if !want_mmap_probe && sm_state != MapState::Closed {
            state
                .state_machine
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .reset();
            pending_verify = None;
            hide_mmap_overlay();
        }

        let in_closed_state = state
            .state_machine
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .state()
            == MapState::Closed;
        let recorder_active = state
            .minimap_recorder
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|r| r.active());
        let heading_fast_ok = should_run_heading_fast(
            in_closed_state,
            minimap_active,
            last_pose_tracking,
            last_full_step_mono,
            tick_start,
            recorder_active,
        );
        let do_auto_sync = want_mmap_probe;
        let guide_map = if route_needs_mmap {
            state.mmap_guide_map_id()
        } else {
            2
        };

        let tick_params = TickParams {
            game_hwnd: hwnd,
            cl: left,
            ct: top,
            cw: width,
            ch: height,
            is_closed: in_closed_state,
            do_auto_sync,
            do_minimap: minimap_active,
            fast_ok: heading_fast_ok,
            guide_view: route_needs_mmap,
            guide_map,
            want_window_capture: cfg_snap.nav_hud_capture_visible && route_needs_mmap && hwnd != 0,
        };
        let tick_out = if let Some(ref w) = worker {
            match w.run_tick(Arc::clone(&state), tick_params).await {
                Some(out) => out,
                None => continue,
            }
        } else {
            let st_for_worker = Arc::clone(&state);
            match tokio::task::spawn_blocking(move || run_tick_worker(&st_for_worker, tick_params))
                .await
            {
                Ok(out) => out,
                Err(_) => continue,
            }
        };

        match tick_out.mode {
            TickMode::Minimap => {
                last_full_step_mono = Some(Instant::now());
                if rate_win_start.is_none() {
                    rate_win_start = Some(tick_start);
                }
                rate_minimap_full += 1;
                rate_work_ms.push(tick_start.elapsed().as_secs_f64() * 1000.0);
            }
            TickMode::MinimapFast => {
                if rate_win_start.is_none() {
                    rate_win_start = Some(tick_start);
                }
                rate_minimap_fast += 1;
                rate_work_ms.push(tick_start.elapsed().as_secs_f64() * 1000.0);
            }
            _ => {
                reset_rate_window(
                    &mut rate_win_start,
                    &mut rate_minimap_full,
                    &mut rate_minimap_fast,
                    &mut rate_work_ms,
                );
            }
        }

        if let Some(win_t0) = rate_win_start {
            let elapsed_s = win_t0.elapsed().as_secs_f64();
            if elapsed_s >= 10.0 {
                let n_ticks = rate_minimap_full + rate_minimap_fast;
                if n_ticks > 0 && !rate_work_ms.is_empty() {
                    rate_work_ms.sort_by(f64::total_cmp);
                    let med = rate_work_ms[rate_work_ms.len() / 2];
                    let p90_idx = ((rate_work_ms.len() as f64) * 0.9)
                        .floor()
                        .min((rate_work_ms.len() - 1) as f64)
                        as usize;
                    let p90 = rate_work_ms[p90_idx];
                    tracing::info!(
                        "[Loop] minimap {:.1} Hz (heading fast {:.0}%, full {:.1} Hz), tick work med {med:.1} ms / p90 {p90:.1} ms",
                        n_ticks as f64 / elapsed_s,
                        (rate_minimap_fast as f64 / n_ticks as f64) * 100.0,
                        rate_minimap_full as f64 / elapsed_s,
                    );
                }
                reset_rate_window(
                    &mut rate_win_start,
                    &mut rate_minimap_full,
                    &mut rate_minimap_fast,
                    &mut rate_work_ms,
                );
            }
        }

        match tick_out.mode {
            TickMode::Minimap | TickMode::MinimapFast => {
                last_pose_tracking = tick_out
                    .pose
                    .as_ref()
                    .is_some_and(|p| p.state == TrackerState::Tracking);
            }
            _ => {
                last_pose_tracking = false;
            }
        }

        match tick_out.mode {
            TickMode::MapMPaused => {
                hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
                state.hide_radar();
                hide_mmap_overlay();
            }
            TickMode::MapM => {
                let cur_sm_state = state
                    .state_machine
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .state();
                if cur_sm_state != MapState::Closed {
                    hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
                    state.hide_radar();
                }
                if cur_sm_state == MapState::Closed || !route_needs_mmap {
                    hide_mmap_overlay();
                }
                if let (Some(full_frame), Some(step_res)) = (tick_out.full_frame, tick_out.step_res)
                {
                    let frame_arc = full_frame;
                    let slider_hint = step_res.check.slider_res.clone();
                    if cfg_snap.auto_sync_m && step_res.should_auto_sync {
                        pending_verify = None;
                        if let Some(ref loc) = state.locator {
                            let locate_res = if let Some(ref w) = worker {
                                w.locate_player(
                                    Arc::clone(loc),
                                    Arc::clone(&frame_arc),
                                    slider_hint.clone(),
                                )
                                .await
                            } else {
                                let loc_clone = Arc::clone(loc);
                                let frame_for_loc = Arc::clone(&frame_arc);
                                let sh_clone = slider_hint.clone();
                                tokio::task::spawn_blocking(move || {
                                    let res = loc_clone.locate_player_frame(
                                        frame_for_loc.as_view(),
                                        false,
                                        sh_clone.as_ref(),
                                    );
                                    gh_cv::locator::release_thread_scratch();
                                    res
                                })
                                .await
                                .ok()
                                .flatten()
                            };

                            match locate_res {
                                None => {
                                    let mut sm = state
                                        .state_machine
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner());
                                    if sm.retries() == 0 {
                                        sm.request_retry();
                                    }
                                }
                                Some(res) if res.has_world_coords => {
                                    let dx_norm = (res.pixel_x - width as f64 * 0.5).abs()
                                        / (width as f64 * 0.5).max(1.0);
                                    let dy_norm = (res.pixel_y - height as f64 * 0.5).abs()
                                        / (height as f64 * 0.5).max(1.0);
                                    let inliers = res.inliers_count.unwrap_or(0);
                                    if dx_norm > 0.65 || dy_norm > 0.65 {
                                        tracing::info!(
                                            "[Auto-Sync M] Skipped: player arrow off-center ({dx_norm:.2}, {dy_norm:.2}) — user is panning."
                                        );
                                    } else if res.map_id == Some(2) && inliers < 2 {
                                        let mut sm = state
                                            .state_machine
                                            .lock()
                                            .unwrap_or_else(|e| e.into_inner());
                                        if sm.retries() < MAX_RETRIES {
                                            sm.request_retry();
                                        }
                                    } else {
                                        let mid = res.map_id.unwrap_or(2);
                                        let lang = get_language();
                                        let loc_map = HoYoLABLabelCatalog::get_map_name(
                                            mid,
                                            Some(&lang),
                                            res.map_name.as_deref().unwrap_or(""),
                                        );
                                        let mut bcast = match res.to_json_value() {
                                            Value::Object(m) => m,
                                            _ => Map::new(),
                                        };
                                        bcast.insert(
                                            "type".to_owned(),
                                            Value::String("SYNC_POSITION_ONLY".to_owned()),
                                        );
                                        bcast.insert("map_name".to_owned(), Value::String(loc_map));
                                        state.server.broadcast(Value::Object(bcast), false, false);
                                        state.apply_map_m_anchor(&res);
                                        if inliers <= 3 {
                                            pending_verify = Some(PendingVerify { ticks: 4, res });
                                        }
                                    }
                                }
                                Some(_) => {
                                    state
                                        .state_machine
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .request_retry();
                                }
                            }
                        }
                    } else if cfg_snap.auto_sync_m
                        && let Some(mut pv) = pending_verify.take()
                        && step_res.state == MapState::SyncedIdle
                    {
                        pv.ticks = pv.ticks.saturating_sub(1);
                        if pv.ticks > 0 {
                            pending_verify = Some(pv);
                        } else if let Some(ref loc) = state.locator {
                            let v_res_opt = if let Some(ref w) = worker {
                                w.locate_player(
                                    Arc::clone(loc),
                                    Arc::clone(&frame_arc),
                                    slider_hint.clone(),
                                )
                                .await
                            } else {
                                let loc_clone = Arc::clone(loc);
                                let frame_for_verify = Arc::clone(&frame_arc);
                                let sh_clone = slider_hint.clone();
                                tokio::task::spawn_blocking(move || {
                                    let res = loc_clone.locate_player_frame(
                                        frame_for_verify.as_view(),
                                        false,
                                        sh_clone.as_ref(),
                                    );
                                    gh_cv::locator::release_thread_scratch();
                                    res
                                })
                                .await
                                .ok()
                                .flatten()
                            };

                            if let Some(v_res) = v_res_opt
                                && v_res.has_world_coords
                                && let (
                                    Some(vwx),
                                    Some(vwy),
                                    Some(pwx),
                                    Some(pwy),
                                ) = (
                                    v_res.world_x,
                                    v_res.world_y,
                                    pv.res.world_x,
                                    pv.res.world_y,
                                )
                            {
                                let dx_norm = (v_res.pixel_x - width as f64 * 0.5).abs()
                                    / (width as f64 * 0.5).max(1.0);
                                let dy_norm = (v_res.pixel_y - height as f64 * 0.5).abs()
                                    / (height as f64 * 0.5).max(1.0);
                                if dx_norm <= 0.65 && dy_norm <= 0.65 {
                                    let dist = (vwx - pwx).hypot(vwy - pwy);
                                    let same_layer = v_res.is_underground == pv.res.is_underground
                                        && v_res.group_id == pv.res.group_id;
                                    let v_inl = v_res.inliers_count.unwrap_or(0);
                                    let p_inl = pv.res.inliers_count.unwrap_or(0);
                                    if (dist > 60.0 && v_inl > p_inl)
                                        || (!same_layer && v_inl >= p_inl)
                                    {
                                        let mid = v_res.map_id.unwrap_or(2);
                                        let lang = get_language();
                                        let loc_map = HoYoLABLabelCatalog::get_map_name(
                                            mid,
                                            Some(&lang),
                                            v_res.map_name.as_deref().unwrap_or(""),
                                        );
                                        let mut bcast = match v_res.to_json_value() {
                                            Value::Object(m) => m,
                                            _ => Map::new(),
                                        };
                                        bcast.insert(
                                            "type".to_owned(),
                                            Value::String("SYNC_POSITION_ONLY".to_owned()),
                                        );
                                        bcast.insert("map_name".to_owned(), Value::String(loc_map));
                                        state.server.broadcast(Value::Object(bcast), false, false);
                                        state.apply_map_m_anchor(&v_res);
                                    }
                                }
                            }
                        }
                    }

                    if cur_sm_state != MapState::Closed && route_needs_mmap {
                        let lang = get_language();
                        let drawer_open = tick_out.view.is_some_and(|v| v.drawer_open);
                        // Draw only on a confident view; the drawer covers the right third.
                        let guide = tick_out
                            .view
                            .filter(|v| !v.drawer_open)
                            .and_then(|v| v.transform)
                            .and_then(|t| {
                                state.build_mmap_guide_scene(game_rect, t, overlay_mode, &lang)
                            });
                        let scene = if guide.is_some() {
                            mmap_no_scene = None;
                            guide
                        } else {
                            // Nothing to point at: after a grace period, a text banner.
                            let now = Instant::now();
                            let since = match mmap_no_scene {
                                Some((since, last)) if now - last < MMAP_FALLBACK_GAP => since,
                                _ => now,
                            };
                            mmap_no_scene = Some((since, now));
                            if now - since >= MMAP_FALLBACK_DELAY {
                                state.build_mmap_fallback_scene(
                                    game_rect,
                                    overlay_mode,
                                    &lang,
                                    drawer_open,
                                )
                            } else {
                                None
                            }
                        };
                        let want_mmap_cap_vis =
                            cfg_snap.nav_hud_capture_visible && tick_out.overlay_capture_safe;
                        if cfg_snap.nav_hud_capture_visible {
                            let wgc_st = state.capturer.window_capture_status();
                            if wgc_st != last_wgc_win_status {
                                if wgc_st.contains("failed") {
                                    tracing::warn!(
                                        "[MMap-OBS] Per-window WGC capture unavailable ({wgc_st}); keeping M-Map overlay excluded from capture"
                                    );
                                } else if wgc_st.starts_with("active(") {
                                    tracing::info!(
                                        "[MMap-OBS] Per-window WGC capture ready ({wgc_st}); M-Map overlay visible to OBS/recorders"
                                    );
                                }
                                last_wgc_win_status = wgc_st;
                            }
                        } else if !last_wgc_win_status.is_empty() {
                            last_wgc_win_status.clear();
                        }
                        state.mmap_overlay.set_capture_visible(want_mmap_cap_vis);
                        match scene {
                            Some(sc) => state.mmap_overlay.update(sc),
                            None => state.mmap_overlay.hide(),
                        }
                        if let Some(line) =
                            state.mmap_view.as_ref().and_then(|e| e.take_stats_line())
                        {
                            tracing::info!("{line}");
                        }
                    }
                }
            }
            TickMode::Minimap | TickMode::MinimapFast => {
                hide_mmap_overlay();
                if let Some(ref pose) = tick_out.pose {
                    if pose.hdr_clipped {
                        hdr_clip_streak += 1;
                        if hdr_clip_streak >= HDR_CLIP_STREAK_WARN && !hdr_warned {
                            hdr_warned = true;
                            tracing::warn!("[HDR] {}", t("hdr.warning.short", &[]));
                            state.hud_toast.show_toast(
                                &t("hdr.warning.title", &[]),
                                &t("hdr.warning.short", &[]),
                                ToastKind::Warning,
                                5500,
                                Some(game_rect),
                            );
                        }
                    } else {
                        hdr_clip_streak = 0;
                    }

                    let pose_valid = pose.state == TrackerState::Tracking
                        && pose.map_id.is_some()
                        && pose.lat.is_some()
                        && pose.lng.is_some();
                    let should_bc_pos = pose.should_broadcast_web;
                    let allow_heading_only = pose.stationary
                        && matches!(
                            pose.match_method.as_str(),
                            "stationary_skip" | "heading_fast"
                        );

                    if pose_valid && (should_bc_pos || allow_heading_only) {
                        let prev_last = state
                            .game_state
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .last_pos
                            .clone();

                        let heading_val = pose.body_heading.or(pose.view_heading);
                        let cam_val = pose.camera_heading.or(pose.view_heading);

                        let (eff_mid, eff_lat, eff_lng, eff_wx, eff_wy, eff_ug, eff_fid, eff_fname) =
                            if !should_bc_pos
                                && allow_heading_only
                                && let Some(ref pl) = prev_last
                                && let (Some(plat), Some(plng)) =
                                    (json_f64(pl.get("lat")), json_f64(pl.get("lng")))
                            {
                                (
                                    json_i64(pl.get("map_id"))
                                        .map(|i| i as i32)
                                        .unwrap_or_else(|| pose.map_id.unwrap_or(2)),
                                    plat,
                                    plng,
                                    json_f64(pl.get("world_x")).unwrap_or(plng),
                                    json_f64(pl.get("world_y")).unwrap_or(plat),
                                    json_truthy(pl.get("is_underground"), pose.is_underground),
                                    pl.get("floor_id")
                                        .and_then(|v| json_i64(Some(v)))
                                        .map(|i| i as i32)
                                        .or(pose.floor_id),
                                    pl.get("floor_name")
                                        .and_then(Value::as_str)
                                        .map(str::to_owned)
                                        .unwrap_or_else(|| pose.floor_name.clone()),
                                )
                            } else {
                                let mid = pose.map_id.unwrap_or(2);
                                let lat = pose.lat.unwrap_or(0.0);
                                let lng = pose.lng.unwrap_or(0.0);
                                (
                                    mid,
                                    lat,
                                    lng,
                                    pose.world_x.unwrap_or(lng),
                                    pose.world_y.unwrap_or(lat),
                                    pose.is_underground,
                                    pose.floor_id,
                                    pose.floor_name.clone(),
                                )
                            };

                        let (eff_area_id, eff_area_name) = if !should_bc_pos
                            && allow_heading_only
                            && let Some(ref pl) = prev_last
                            && let Some(aid) = json_i64(pl.get("area_id")).map(|i| i as i32)
                        {
                            let aname = pl
                                .get("area_name")
                                .and_then(Value::as_str)
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                                .map(str::to_owned)
                                .or_else(|| {
                                    DEFAULT_AREA_NAMES
                                        .iter()
                                        .find(|&&(id, _)| id == aid)
                                        .map(|&(_, n)| n.to_owned())
                                });
                            (Some(aid), aname)
                        } else {
                            resolve_minimap_area(
                                &state.maps_db,
                                state.locator.as_deref(),
                                eff_mid,
                                eff_wx,
                                eff_wy,
                            )
                        };
                        let eff_group_id = if eff_ug {
                            eff_fid
                                .and_then(|fid| state.maps_db.resolve_floor_group_id(eff_mid, fid))
                                .or_else(|| {
                                    if !should_bc_pos && allow_heading_only {
                                        prev_last
                                            .as_ref()
                                            .and_then(|pl| json_i64(pl.get("group_id")))
                                            .map(|i| i as i32)
                                    } else {
                                        None
                                    }
                                })
                        } else {
                            None
                        };

                        let pos_info = json!({
                            "map_id": eff_mid,
                            "area_id": eff_area_id,
                            "area_name": eff_area_name,
                            "lat": eff_lat,
                            "lng": eff_lng,
                            "world_x": eff_wx,
                            "world_y": eff_wy,
                            "heading": heading_val,
                            "camera_heading": cam_val,
                            "view_heading": pose.view_heading,
                            "is_underground": eff_ug,
                            "group_id": eff_group_id,
                            "floor_id": eff_fid,
                            "floor_name": eff_fname,
                        });
                        {
                            let mut gs = state.game_state.lock().unwrap_or_else(|e| e.into_inner());
                            gs.last_pos = Some(pos_info);
                        }

                        if state.has_active_route_doc() {
                            let floor_id_str = eff_fid.map(|f| f.to_string());
                            let adv = {
                                let mut nav = state
                                    .nav_controller
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                nav.advance_route_on_position_on_floor(
                                    eff_mid,
                                    eff_lat,
                                    eff_lng,
                                    floor_id_str.as_deref(),
                                )
                            };
                            if adv.is_some() {
                                let st = state
                                    .nav_controller
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .route_status_json();
                                state.push_route_state_update(&st);
                            }
                        }

                        let now_mono = Instant::now();
                        if last_mm_bcast_ts
                            .is_none_or(|t0| now_mono.duration_since(t0).as_secs_f64() >= 0.05)
                        {
                            let should_send = if !should_bc_pos && allow_heading_only {
                                last_mm_bcast_state.as_ref().is_some_and(|prev| {
                                    angle_diff_deg(heading_val, prev.heading) >= 2.0
                                        || angle_diff_deg(cam_val, prev.camera_heading) >= 2.0
                                })
                            } else if let Some(ref prev) = last_mm_bcast_state {
                                prev.map_id != eff_mid
                                    || prev.area_id != eff_area_id
                                    || prev.is_underground != eff_ug
                                    || prev.floor_id != eff_fid
                                    || (eff_lat - prev.lat).hypot(eff_lng - prev.lng) >= 0.8
                                    || angle_diff_deg(heading_val, prev.heading) >= 2.0
                                    || angle_diff_deg(cam_val, prev.camera_heading) >= 2.0
                            } else {
                                true
                            };

                            if should_send {
                                last_mm_bcast_ts = Some(now_mono);
                                last_mm_bcast_state = Some(MmBcastState {
                                    map_id: eff_mid,
                                    area_id: eff_area_id,
                                    lat: eff_lat,
                                    lng: eff_lng,
                                    heading: heading_val,
                                    camera_heading: cam_val,
                                    is_underground: eff_ug,
                                    floor_id: eff_fid,
                                });
                                let conf_r = (pose.confidence * 1000.0).round() / 1000.0;
                                let lang = get_language();
                                let eff_map_name = HoYoLABLabelCatalog::get_map_name(
                                    eff_mid,
                                    Some(&lang),
                                    state.maps_db.map_name(eff_mid).unwrap_or(""),
                                );
                                state.server.broadcast_ephemeral(json!({
                                    "type": "MINIMAP_POSITION",
                                    "source": "minimap",
                                    "has_world_coords": true,
                                    "map_id": eff_mid,
                                    "map_name": eff_map_name,
                                    "area_id": eff_area_id,
                                    "area_name": eff_area_name,
                                    "world_x": eff_wx,
                                    "world_y": eff_wy,
                                    "lat": eff_lat,
                                    "lng": eff_lng,
                                    "heading": heading_val,
                                    "camera_heading": cam_val,
                                    "body_heading": pose.body_heading,
                                    "view_heading": pose.view_heading,
                                    "minimap_mode": pose.minimap_mode.as_str(),
                                    "zoom_regime": pose.zoom_regime.as_str(),
                                    "is_underground": eff_ug,
                                    "group_id": eff_group_id,
                                    "floor_id": eff_fid,
                                    "floor_short": eff_fname,
                                    "floor_name": eff_fname,
                                    "confidence": conf_r,
                                    "hdr_clipped": pose.hdr_clipped,
                                    "state": pose.state.as_str(),
                                }));
                            }
                        }

                        state.push_lan_snapshot(false);
                    }

                    // Navigation Mini-HUD + Native Floating Live Radar Map update
                    let radar_active = state.is_radar_active();
                    if !cfg_snap.nav_hud_enabled && !radar_active {
                        hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
                    } else if matches!(
                        pose.match_method.as_str(),
                        "gate_rejected" | "gate_hold" | "paimon_pause" | "loading_pause" | "paused"
                    ) {
                        gate_reject_streak += 1;
                        if gate_reject_streak >= 3 {
                            hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
                            state.hide_radar();
                        }
                    } else {
                        gate_reject_streak = 0;
                        let fallback_pos = state
                            .game_state
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .last_pos
                            .clone();
                        let use_fallback = pose.lat.is_none()
                            && fallback_pos
                                .as_ref()
                                .and_then(|v| json_f64(v.get("lat")))
                                .is_some();

                        let eff_map_id = pose
                            .map_id
                            .or_else(|| {
                                fallback_pos
                                    .as_ref()
                                    .and_then(|v| json_i64(v.get("map_id")))
                                    .map(|i| i as i32)
                            })
                            .unwrap_or(2);
                        let eff_lat = pose
                            .lat
                            .or_else(|| fallback_pos.as_ref().and_then(|v| json_f64(v.get("lat"))))
                            .unwrap_or(0.0);
                        let eff_lng = pose
                            .lng
                            .or_else(|| fallback_pos.as_ref().and_then(|v| json_f64(v.get("lng"))))
                            .unwrap_or(0.0);
                        let eff_heading = pose.view_heading.or(pose.camera_heading);
                        let eff_is_ug = if use_fallback {
                            fallback_pos
                                .as_ref()
                                .map(|v| json_truthy(v.get("is_underground"), pose.is_underground))
                                .unwrap_or(pose.is_underground)
                        } else {
                            pose.is_underground
                        };
                        let eff_floor_id = if use_fallback {
                            fallback_pos
                                .as_ref()
                                .and_then(|v| json_i64(v.get("floor_id")))
                                .map(|i| i as i32)
                                .or(pose.floor_id)
                        } else {
                            pose.floor_id
                        };
                        let mut eff_floor_name = if use_fallback {
                            fallback_pos
                                .as_ref()
                                .and_then(|v| v.get("floor_name").and_then(Value::as_str))
                                .unwrap_or(&pose.floor_name)
                                .trim()
                                .to_owned()
                        } else {
                            pose.floor_name.trim().to_owned()
                        };
                        if eff_is_ug
                            && (eff_floor_name.is_empty()
                                || eff_floor_name.eq_ignore_ascii_case("G")
                                || eff_floor_name.chars().all(|c| c.is_ascii_digit()))
                            && let Some(fid) = eff_floor_id
                            && let Some(ref loc) = state.locator
                        {
                            let resolved_lbl = loc.resolve_floor_label(eff_map_id, fid);
                            if !resolved_lbl.is_empty() {
                                eff_floor_name = resolved_lbl;
                            }
                        }

                        if cfg_snap.nav_hud_enabled {
                            let fid_str = eff_floor_id.map(|f| f.to_string());
                            let (mut guidance, nearby_snap) = {
                                let mut nav = state
                                    .nav_controller
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                let g = nav.compute_guidance(
                                    eff_lat,
                                    eff_lng,
                                    eff_heading,
                                    eff_map_id,
                                    fid_str.as_deref(),
                                    eff_is_ug,
                                    pose.state.as_str(),
                                    &eff_floor_name,
                                );
                                let nb = nav.nearby_snapshot(eff_lat, eff_lng, eff_map_id, 6);
                                (g, nb)
                            };

                            {
                                let tgt = guidance.target.as_ref();
                                let snap = json!({
                                    "state": guidance.state.as_str(),
                                    "target_id": tgt.map(|t| t.id.as_str()),
                                    "target_name": tgt.map(|t| t.display_name()),
                                    "target_category": tgt.map(|t| t.category.as_str()),
                                    "target_lat": tgt.map(|t| t.lat),
                                    "target_lng": tgt.map(|t| t.lng),
                                    "distance_m": guidance.distance_m,
                                    "bearing_deg": guidance.bearing_deg,
                                    "delta_deg": guidance.delta_deg,
                                    "route_mode": guidance.route_mode.as_str(),
                                    "floor_mismatch": guidance.floor_mismatch,
                                    "remaining_count": guidance.remaining_count,
                                    "player": {
                                        "lat": eff_lat,
                                        "lng": eff_lng,
                                        "map_id": eff_map_id,
                                        "floor_id": eff_floor_id,
                                        "floor_name": eff_floor_name,
                                        "is_underground": eff_is_ug,
                                    },
                                    "heading": {
                                        "view": pose.view_heading,
                                        "camera": pose.camera_heading,
                                        "body": pose.body_heading,
                                        "used": eff_heading,
                                    },
                                    "hud_visible": nav_hud_visible,
                                    "nearby": nearby_snap,
                                });
                                *state
                                    .nav_state_snapshot
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner()) = Some(snap);
                            }

                            let cur_target_id = guidance.target.as_ref().map(|t| t.id.clone());
                            if cur_target_id != last_nav_target_id {
                                last_nav_target_id = cur_target_id;
                                state.ensure_guide(guidance.target.as_ref());
                            }

                            let cur_lang = get_language();
                            if guidance.state == NavState::Idle
                                && guidance.target.is_none()
                                && !guidance.is_route_mode
                            {
                                let idle_hint = if !state.server.has_active_client(5.0) {
                                    if cur_lang == "vi" {
                                        "Mở / F5 HoYoLAB Map để kết nối"
                                    } else {
                                        "Open / F5 HoYoLAB Map to connect"
                                    }
                                } else if cur_lang == "vi" {
                                    "Bật bộ lọc rương/thần đồng trên HoYoLAB"
                                } else {
                                    "Enable pin filters on HoYoLAB Map"
                                };
                                guidance.delta_deg = None;
                                guidance.route_target_name = idle_hint.to_owned();
                            }

                            let sync_label = state.hotkey_mgr.get_display_name("nav_mark");
                            let guide_label = state.hotkey_mgr.get_display_name("nav_guide");
                            update_nav_hud_if_changed(
                                &state,
                                &mut nav_hud_visible,
                                &mut last_hud_sig,
                                guidance,
                                game_rect,
                                &cfg_snap.nav_hud_position,
                                cur_lang,
                                sync_label,
                                guide_label,
                            );
                        } else {
                            hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
                        }

                        if radar_active && (pose.lat.is_some() || use_fallback) {
                            state.push_radar_frame(
                                game_rect,
                                eff_map_id,
                                eff_lng,
                                eff_lat,
                                pose.body_heading.or(eff_heading),
                                pose.camera_heading.or(eff_heading),
                                eff_is_ug,
                                eff_floor_id,
                                eff_floor_name,
                                cfg_snap.nav_hud_capture_visible,
                            );
                        }
                    }
                } else {
                    hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
                    state.hide_radar();
                }
            }
            TickMode::IdleClosed => {
                hide_mmap_overlay();
                state.hide_radar();
                if cfg_snap.nav_hud_enabled && state.has_active_route_doc() {
                    let guidance = {
                        let mut nav = state
                            .nav_controller
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        nav.compute_guidance(0.0, 0.0, None, 2, None, false, "UNLOCALIZED", "")
                    };
                    let cur_lang = get_language();
                    let sync_label = state.hotkey_mgr.get_display_name("nav_mark");
                    let guide_label = state.hotkey_mgr.get_display_name("nav_guide");
                    update_nav_hud_if_changed(
                        &state,
                        &mut nav_hud_visible,
                        &mut last_hud_sig,
                        guidance,
                        game_rect,
                        &cfg_snap.nav_hud_position,
                        cur_lang,
                        sync_label,
                        guide_label,
                    );
                } else {
                    hide_nav_hud(&mut nav_hud_visible, &mut last_hud_sig);
                }
            }
        }

        let cur_state = state
            .state_machine
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .state();
        let live_minimap = cur_state == MapState::Closed && minimap_active;
        let guiding_mmap =
            route_needs_mmap && cur_state != MapState::Closed && state.mmap_view.is_some();
        hires_timer.set(live_minimap || guiding_mmap);

        let sleep_dur = if guiding_mmap {
            let rem = MMAP_GUIDE_TICK_S - tick_start.elapsed().as_secs_f64();
            if rem > 0.001 {
                Duration::from_secs_f64(rem)
            } else {
                Duration::ZERO
            }
        } else if cur_state == MapState::Settling {
            Duration::from_millis(45)
        } else if live_minimap {
            let elapsed = tick_start.elapsed().as_secs_f64();
            let rem = MINIMAP_TICK_S - elapsed;
            if rem > 0.001 {
                Duration::from_secs_f64(rem)
            } else {
                Duration::ZERO
            }
        } else if cur_state == MapState::Closed {
            Duration::from_millis(75)
        } else if route_needs_mmap {
            Duration::from_millis(70)
        } else {
            Duration::from_millis(110)
        };

        if sleep_or_shutdown(sleep_dur, &mut shutdown_rx).await {
            break;
        }
    }

    hires_timer.set(false);
    state.mmap_overlay.hide();
    state.nav_hud.hide();
    state.hide_radar();
}

fn should_run_heading_fast(
    in_closed_state: bool,
    minimap_active: bool,
    last_pose_tracking: bool,
    last_full_step_mono: Option<Instant>,
    tick_start: Instant,
    recorder_active: bool,
) -> bool {
    in_closed_state
        && minimap_active
        && last_pose_tracking
        && last_full_step_mono.is_some_and(|t0| {
            tick_start.saturating_duration_since(t0).as_secs_f64() < MINIMAP_FULL_STEP_S
        })
        && !recorder_active
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_fast_schedule_uses_completion_time_not_tick_start() {
        let base = Instant::now();
        let full_start = base;
        let completed_at = base + Duration::from_millis(80);
        let next_tick = completed_at + Duration::from_millis(16);

        // Stamping tick_start starves 60 Hz heading after any slow (>42ms) full step.
        assert!(!should_run_heading_fast(
            true,
            true,
            true,
            Some(full_start),
            next_tick,
            false,
        ));

        // Stamping completion time runs fast heading ticks inside the budget,
        // then yields to the next full step after MINIMAP_FULL_STEP_S elapses.
        assert!(should_run_heading_fast(
            true,
            true,
            true,
            Some(completed_at),
            next_tick,
            false,
        ));
        assert!(!should_run_heading_fast(
            true,
            true,
            true,
            Some(completed_at),
            completed_at + Duration::from_millis(55),
            false,
        ));
    }

    #[test]
    fn resolve_minimap_area_handles_teyvat_fallback_and_submaps() {
        let sample = r#"{
            "version": "4.0.0",
            "maps": {
                "2": {
                    "id": 2,
                    "name": "Teyvat",
                    "areas": [
                        {"id": 1, "name": "Mondstadt"}
                    ],
                    "waypoints": [
                        {"id": 1, "label_id": 2, "is_primary": true, "x_pos": 100.0, "y_pos": 100.0, "area_id": 1},
                        {"id": 2, "label_id": 2, "is_primary": true, "x_pos": -4000.0, "y_pos": 2500.0, "area_id": 11}
                    ]
                },
                "9": {
                    "id": 9,
                    "name": "The Chasm",
                    "areas": [{"id": 6, "name": "Chasm"}],
                    "waypoints": [
                        {"id": 3, "label_id": 2, "is_primary": true, "x_pos": 0.0, "y_pos": 0.0, "area_id": 6}
                    ]
                }
            }
        }"#;
        let db = MultimapDb::from_json_str(sample).expect("parse db");
        assert_eq!(
            resolve_minimap_area(&db, None, 2, 120.0, 90.0),
            (Some(1), Some("Mondstadt".to_owned()))
        );
        // Missing from `areas` table -> falls back to DEFAULT_AREA_NAMES ("Natlan").
        assert_eq!(
            resolve_minimap_area(&db, None, 2, -3950.0, 2480.0),
            (Some(11), Some("Natlan".to_owned()))
        );
        // Non-Teyvat submaps never emit Map-2 area switches.
        assert_eq!(resolve_minimap_area(&db, None, 9, 0.0, 0.0), (None, None));
        assert_eq!(
            resolve_minimap_area(&db, None, 2, f64::NAN, 0.0),
            (None, None)
        );
    }
}
