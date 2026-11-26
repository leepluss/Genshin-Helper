//! Anchor + track measurement for the v5.5 M-Map route guidance, on a real drag recording.
//!
//! Phase A (expensive, cached in `CARGO_TARGET_TMPDIR/e5_cache_<stem>_<fps>fps.json`): decode the
//! video at `GH_MMAP_FPS` (default 30), and per frame measure the zoom slider, the frame-to-frame
//! motion (`ShiftEstimator`, zoom-compensated from the slider) and a full registration
//! (`locate_view_opts`, map hint, no terrain-only fallback), each timed.
//! Phase B (cheap): replay [`ViewTracker`] over the recorded motions while feeding registrations
//! the way the live app would — an async worker that starts on the newest frame whenever it is
//! free and delivers after its measured latency — for a sweep of tracker settings.
//!
//! Ground truth: strong registrations (≥ 6 inliers, not terrain-only, right map) on frames the
//! simulated worker did NOT feed to the tracker (holdout). "Wrong" = drawn transform more than
//! 40 px (@1440p) away from the holdout, or drawn on the wrong map.
//!
//! Run:
//! `GH_MMAP_VIDEO="<file.mp4>" cargo nextest run --manifest-path rust/Cargo.toml -p gh-cv --profile cv-heavy --cargo-profile cv-bench --run-ignored all -E "test(e5_)" --no-capture`
//! Optional: `GH_MMAP_LABELS`, `GH_MMAP_FPS`, `GH_MMAP_RECOMPUTE=1` (everything) or
//! `GH_MMAP_REMOTION=1` (slider/cursor/motion only, registrations reused from the cache),
//! `GH_MMAP_CURSOR_MASK=0` (disable cursor masking), `GH_MMAP_VOTES=1` (translation-vote
//! registration, own cache file), `GH_FFMPEG`, `GH_FFPROBE`.
//!
//! The recording has the game cursor burnt in; the live app knows it from `GetCursorPos`, so the
//! harness locates it by colour (white tip + yellow diamond) to feed the same mask.

use gh_cv::locator::{LocatorPaths, MapLocator, ViewHint, ViewOptions, ViewRegistration};
use gh_cv::view_tracker::{
    RegistrationVerdict, ShiftEstimator, SliderScale, TrackSource, TrackerConfig, ViewMotion, ViewTracker,
    ViewTransform, slider_zoom_ratio,
};
use gh_cv::zoom_slider::ZoomSliderDetector;
use ndarray::ArrayView3;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, VecDeque};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const WRONG_PX_1440: f64 = 40.0;
const GT_MIN_INLIERS: usize = 6;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("..")
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).ok().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| default.to_owned())
}

fn probe_dims(ffprobe: &str, video: &Path) -> (usize, usize) {
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x"])
        .arg(video)
        .output()
        .expect("run ffprobe (set GH_FFPROBE)");
    let txt = String::from_utf8_lossy(&out.stdout);
    let mut it = txt.trim().split('x').map(|v| v.trim().parse::<usize>().expect("ffprobe dims"));
    (it.next().expect("width"), it.next().expect("height"))
}

struct Segment {
    from: f64,
    to: f64,
    map_id: i32,
}

fn load_labels(path: &Path) -> Vec<Segment> {
    let txt = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("labels file {}", path.display()));
    let v: Value = serde_json::from_str(&txt).expect("labels json");
    v["segments"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| Segment {
            from: s["from"].as_f64().unwrap_or(0.0),
            to: s["to"].as_f64().unwrap_or(0.0),
            map_id: s["map_id"].as_i64().unwrap_or(-1) as i32,
        })
        .collect()
}

fn label_at(segs: &[Segment], t: f64) -> Option<i32> {
    segs.iter().find(|s| t >= s.from && t < s.to).map(|s| s.map_id)
}

fn finite_or(v: f64, d: f64) -> f64 {
    if v.is_finite() { v } else { d }
}

/// NaN-free mirror of [`ViewRegistration`] for the JSON cache.
#[derive(Serialize, Deserialize, Clone)]
struct RegRec {
    map_id: i32,
    scale: f64,
    tx: f64,
    ty: f64,
    inliers: usize,
    rmse: f64,
    terrain_score: f64,
    terrain_only: bool,
    zoom_t: Option<f64>,
    ug: bool,
    cand: usize,
}

impl RegRec {
    fn from_reg(r: &ViewRegistration) -> Self {
        Self {
            map_id: r.map_id,
            scale: r.scale,
            tx: r.tx,
            ty: r.ty,
            inliers: r.inliers_count,
            rmse: finite_or(r.rmse, 1e9),
            terrain_score: finite_or(r.terrain_score, 0.0),
            terrain_only: r.terrain_only,
            zoom_t: r.zoom_ratio_t.filter(|v| v.is_finite()),
            ug: r.is_underground,
            cand: r.candidates_count,
        }
    }

    fn to_reg(&self) -> ViewRegistration {
        ViewRegistration {
            map_id: self.map_id,
            scale: self.scale,
            tx: self.tx,
            ty: self.ty,
            inliers_count: self.inliers,
            rmse: self.rmse,
            terrain_score: self.terrain_score,
            terrain_only: self.terrain_only,
            zoom_ratio_t: self.zoom_t,
            is_underground: self.ug,
            candidates_count: self.cand,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
struct MotionRec {
    r: f64,
    ex: f64,
    ey: f64,
    peak: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct FrameRec {
    t: f64,
    label: Option<i32>,
    hint: Option<i32>,
    slider_t: Option<f64>,
    slider_scale: Option<f64>,
    slider_ms: f64,
    #[serde(default)]
    cursor: Option<(f64, f64)>,
    motion: Option<MotionRec>,
    motion_ms: f64,
    reg: Option<RegRec>,
    reg_ms: f64,
}

#[derive(Serialize, Deserialize)]
struct Cache {
    video: String,
    fps: f64,
    w: usize,
    h: usize,
    frames: Vec<FrameRec>,
}

/// `GH_MMAP_VOTES=1`: register with [`ViewOptions::translation_votes`] (separate cache file).
fn use_votes() -> bool {
    std::env::var("GH_MMAP_VOTES").is_ok_and(|v| v == "1")
}

fn percentile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * q).round() as usize]
}

/// Hotspot (tip) of Genshin's cursor in a full-resolution BGR frame: the yellow diamond with the
/// white tip up-left of it. Test-only stand-in for `GetCursorPos`.
fn detect_cursor(img: ArrayView3<'_, u8>) -> Option<(f64, f64)> {
    let (h, w, _) = img.dim();
    let px = img.as_slice()?;
    let at = |x: usize, y: usize| {
        let i = (y * w + x) * 3;
        (px[i], px[i + 1], px[i + 2])
    };
    let yellow = |x: usize, y: usize| {
        let (b, g, r) = at(x, y);
        r >= 225 && (140..=232).contains(&g) && b <= 80
    };
    let white = |x: usize, y: usize| {
        let (b, g, r) = at(x, y);
        r >= 215 && g >= 215 && b >= 190
    };
    let k = h as f64 / 1440.0;
    const CELL: usize = 32;
    let (ncx, ncy) = (w / CELL + 1, h / CELL + 1);
    let mut counts = vec![0u32; ncx * ncy];
    // Only where the motion patch looks (plus the mask margin): the quest-tracker icons along the
    // top edge also carry a yellow crown + white body and would otherwise win when the real cursor
    // is motion-blurred.
    let (sx0, sx1) = ((w as f64 * 0.155) as usize, (w as f64 * 0.995) as usize);
    let (sy0, sy1) = ((h as f64 * 0.125) as usize, (h as f64 * 0.875) as usize);
    for y in (sy0..sy1).step_by(2) {
        for x in (sx0..sx1).step_by(2) {
            if yellow(x, y) {
                counts[(y / CELL) * ncx + x / CELL] += 1;
            }
        }
    }
    let min_yellow = (40.0 * k * k).max(8.0) as usize;
    let max_yellow = (400.0 * k * k) as usize;
    let min_white = (25.0 * k * k).max(6.0) as usize;
    let mut best: Option<(usize, (f64, f64))> = None;
    for cy in 0..ncy {
        for cx in 0..ncx {
            if counts[cy * ncx + cx] < 3 {
                continue;
            }
            let (x0, y0) = ((cx * CELL).saturating_sub(16), (cy * CELL).saturating_sub(16));
            let (x1, y1) = ((cx * CELL + CELL + 16).min(w), (cy * CELL + CELL + 16).min(h));
            let (mut n, mut sx, mut sy) = (0usize, 0f64, 0f64);
            for y in y0..y1 {
                for x in x0..x1 {
                    if yellow(x, y) {
                        n += 1;
                        sx += x as f64;
                        sy += y as f64;
                    }
                }
            }
            if n < min_yellow || n > max_yellow {
                continue;
            }
            let (mx, my) = (sx / n as f64, sy / n as f64);
            let r = (26.0 * k) as isize;
            let e = (4.0 * k) as isize;
            let mut nw = 0usize;
            for dy in -r..=e {
                for dx in -r..=e {
                    let (x, y) = (mx as isize + dx, my as isize + dy);
                    if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && white(x as usize, y as usize) {
                        nw += 1;
                    }
                }
            }
            if nw >= min_white && best.is_none_or(|(bn, _)| n > bn) {
                best = Some((n, (mx - 11.0 * k, my - 11.0 * k)));
            }
        }
    }
    best.map(|(_, p)| p)
}

fn measure(video: &Path, segs: &[Segment], fps: f64, w: usize, h: usize, prior: Option<&Cache>) -> Cache {
    let root = repo_root();
    let loc = MapLocator::with_paths(
        LocatorPaths {
            db_path: Some(root.join("data").join("multimap_v4.json")),
            terrain_npz_path: Some(root.join("data").join("terrain_n2_v4.npz")),
            cave_masks_npz_path: Some(root.join("data").join("cave_masks_v4.npz")),
            ui_anchors_npz_path: Some(root.join("data").join("ui_anchors_v4.npz")),
            legacy_underground_db_path: Some(root.join("data").join("underground_db.json")),
        },
        None,
    );
    let mut child = Command::new(env_or("GH_FFMPEG", "ffmpeg"))
        .args(["-v", "error", "-i"])
        .arg(video)
        .args(["-vf", &format!("fps={fps}"), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg (set GH_FFMPEG)");
    let mut stdout = child.stdout.take().expect("ffmpeg stdout");
    let mut buf = vec![0u8; w * h * 3];
    let mut est = ShiftEstimator::new();
    let mask_cursor = env_or("GH_MMAP_CURSOR_MASK", "1") != "0";
    let mut last_scale: Option<SliderScale> = None;
    let mut last_label: Option<i32> = None;
    let mut frames = Vec::new();
    let start = Instant::now();
    while stdout.read_exact(&mut buf).is_ok() {
        let idx = frames.len();
        let t = idx as f64 / fps;
        if idx > 0 && idx.is_multiple_of(150) {
            println!(
                "  ... {idx} frames (t={t:.0}s), {:.0} ms/frame wall incl. decode",
                start.elapsed().as_secs_f64() * 1000.0 / idx as f64
            );
        }
        let img = ArrayView3::from_shape((h, w, 3), &buf[..]).expect("frame shape");
        let label = label_at(segs, t);
        if label.is_some() {
            last_label = label;
        }
        let hint = label.or(last_label);

        let t0 = Instant::now();
        let slider = ZoomSliderDetector::detect_slider(img).filter(|s| s.found && s.ratio_t.is_finite());
        let slider_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let map_for_scale = hint.unwrap_or(2);
        let slider_scale = ZoomSliderDetector::get_map_scale(slider.as_ref(), map_for_scale, w);
        // Same rule as `ViewTracker::on_frame`: ratios only between readings of one map.
        let reading = slider_scale.map(|scale| SliderScale { map_id: map_for_scale, scale });
        let zoom_r = slider_zoom_ratio(last_scale, reading);
        if reading.is_some() {
            last_scale = reading;
        }
        let cursor = if mask_cursor { detect_cursor(img) } else { None };
        let t0 = Instant::now();
        let motion = est.push_frame(img, zoom_r, cursor);
        let motion_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let (reg, reg_ms) = match prior.and_then(|p| p.frames.get(idx)) {
            Some(pf) => (pf.reg.clone(), pf.reg_ms),
            None => {
                let t0 = Instant::now();
                let opts = ViewOptions { allow_terrain_only: false, translation_votes: use_votes() };
                let r = loc.locate_view_with(img, slider.as_ref(), hint.map_or(ViewHint::None, ViewHint::Map), opts);
                (r.as_ref().map(RegRec::from_reg), t0.elapsed().as_secs_f64() * 1000.0)
            }
        };

        frames.push(FrameRec {
            t,
            label,
            hint,
            slider_t: slider.as_ref().map(|s| s.ratio_t),
            slider_scale,
            slider_ms,
            cursor,
            motion: motion.map(|m| MotionRec { r: m.r, ex: m.ex, ey: m.ey, peak: m.peak }),
            motion_ms,
            reg,
            reg_ms,
        });
    }
    let _ = child.wait();
    Cache { video: video.display().to_string(), fps, w, h, frames }
}

#[derive(Clone, Copy, Debug)]
enum Schedule {
    /// `workers` workers; a free one starts on the newest frame, result ready after its measured
    /// latency (divided by `speed`).
    Async { speed: f64, workers: usize },
    /// Every `cadence`-th frame, delivered `delay` frames later.
    Fixed { cadence: usize, delay: usize },
}

#[derive(Default, Clone)]
struct SegAcc {
    frames: usize,
    covered: usize,
    anchored: usize,
    gt: usize,
    gt_covered: usize,
    wrong: usize,
    wrong_map: usize,
    err: Vec<f64>,
    switches: usize,
    switch_px: Vec<f64>,
}

struct ReplayOut {
    segs: BTreeMap<String, SegAcc>,
    verdicts: BTreeMap<String, usize>,
    /// Per frame: (covered, wrong, anchored this frame).
    per_frame: Vec<(bool, bool, Option<f64>)>,
    fed: usize,
}

fn verdict_name(v: RegistrationVerdict) -> &'static str {
    match v {
        RegistrationVerdict::Agrees => "agrees",
        RegistrationVerdict::Restored => "restored",
        RegistrationVerdict::Acquired => "acquired",
        RegistrationVerdict::Switched => "switched",
        RegistrationVerdict::CandidateStarted => "candidate_started",
        RegistrationVerdict::CandidateReinforced => "candidate_reinforced",
        RegistrationVerdict::Stale => "stale",
        RegistrationVerdict::NotConfident => "not_confident",
    }
}

fn strong_reg(f: &FrameRec) -> Option<ViewTransform> {
    let r = f.reg.as_ref()?;
    let reg = r.to_reg();
    let ok = reg.is_confident() && !r.terrain_only && r.inliers >= GT_MIN_INLIERS;
    ok.then(|| ViewTransform::from_registration(&reg))
}

/// Holdout ground truth per frame: a strong registration on the labelled map that is not
/// contradicted by every strong registration of the previous `GT_CONSENSUS_FRAMES` frames
/// (carried forward by the recorded motions). Frames without such neighbours are kept.
fn ground_truth(cache: &Cache) -> Vec<Option<ViewTransform>> {
    const GT_CONSENSUS_FRAMES: usize = 6;
    let (w, h) = (cache.w, cache.h);
    let fr = &cache.frames;
    (0..fr.len())
        .map(|i| {
            let gt = strong_reg(&fr[i]).filter(|g| fr[i].label == Some(g.map_id))?;
            let mut neighbours = 0usize;
            let mut agreeing = 0usize;
            'nb: for j in i.saturating_sub(GT_CONSENSUS_FRAMES)..i {
                let Some(mut t) = strong_reg(&fr[j]).filter(|t| t.map_id == gt.map_id) else { continue };
                for f in &fr[j + 1..=i] {
                    let Some(m) = f.motion else { continue 'nb };
                    t = t.apply_motion(&ViewMotion { r: m.r, ex: m.ex, ey: m.ey, peak: m.peak });
                }
                neighbours += 1;
                if t.disagreement_px_1440(&gt, w, h) <= WRONG_PX_1440 {
                    agreeing += 1;
                }
            }
            (neighbours == 0 || agreeing > 0).then_some(gt)
        })
        .collect()
}

fn replay(cache: &Cache, gt_all: &[Option<ViewTransform>], cfg: TrackerConfig, sched: Schedule) -> ReplayOut {
    let (w, h, fps) = (cache.w, cache.h, cache.fps);
    let mut tr = ViewTracker::new(cfg);
    let mut pending: VecDeque<(usize, u64, ViewRegistration)> = VecDeque::new();
    let n_workers = match sched {
        Schedule::Async { workers, .. } => workers.max(1),
        Schedule::Fixed { .. } => 1,
    };
    let mut worker_free_at = vec![0.0f64; n_workers];
    let mut fed = vec![false; cache.frames.len()];
    let mut out = ReplayOut { segs: BTreeMap::new(), verdicts: BTreeMap::new(), per_frame: Vec::new(), fed: 0 };
    for (i, f) in cache.frames.iter().enumerate() {
        let motion = f.motion.map(|m| ViewMotion { r: m.r, ex: m.ex, ey: m.ey, peak: m.peak });
        let seq = tr.on_motion(motion, w, h);
        // Worker start (registration of this very frame).
        let start = match sched {
            Schedule::Async { speed, .. } => {
                let k = (0..n_workers)
                    .min_by(|&a, &b| worker_free_at[a].total_cmp(&worker_free_at[b]))
                    .unwrap_or(0);
                if f.t + 1e-9 >= worker_free_at[k] && tr.frame_is_stable() {
                    let done = f.t + f.reg_ms / 1000.0 / speed;
                    worker_free_at[k] = done;
                    Some(((done * fps).ceil() as usize).max(i))
                } else {
                    None
                }
            }
            Schedule::Fixed { cadence, delay } => i.is_multiple_of(cadence.max(1)).then_some(i + delay),
        };
        if let Some(ready) = start {
            fed[i] = true;
            out.fed += 1;
            if let Some(r) = &f.reg {
                let at = pending.partition_point(|p| p.0 <= ready);
                pending.insert(at, (ready, seq, r.to_reg()));
            }
        }
        let mut switch_px = None;
        while pending.front().is_some_and(|p| p.0 <= i) {
            let (_, s, r) = pending.pop_front().expect("front");
            let before = tr.output().transform;
            let v = tr.on_registration(s, &r);
            *out.verdicts.entry(verdict_name(v).to_owned()).or_default() += 1;
            if v == RegistrationVerdict::Switched
                && let (Some(b), Some(a)) = (before, tr.output().transform)
            {
                switch_px = Some(b.disagreement_px_1440(&a, w, h).min(1e6));
            }
        }
        let o = tr.output();
        let key = f.label.map_or("unlabelled".to_owned(), |m| format!("map_{m:02}"));
        let acc = out.segs.entry(key).or_default();
        acc.frames += 1;
        let mut wrong = false;
        if let Some(cur) = o.transform {
            acc.covered += 1;
            if o.source == Some(TrackSource::Anchored) {
                acc.anchored += 1;
            }
            if f.label.is_some_and(|l| l != cur.map_id) {
                acc.wrong_map += 1;
                wrong = true;
            }
        }
        if !fed[i]
            && let Some(gt) = gt_all[i]
        {
            acc.gt += 1;
            if let Some(cur) = o.transform {
                acc.gt_covered += 1;
                let e = cur.disagreement_px_1440(&gt, w, h);
                acc.err.push(e.min(1e6));
                if e > WRONG_PX_1440 && !wrong {
                    acc.wrong += 1;
                    wrong = true;
                }
            }
        }
        if let Some(px) = switch_px {
            acc.switches += 1;
            acc.switch_px.push(px);
        }
        out.per_frame.push((o.transform.is_some(), wrong, switch_px));
    }
    out
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 }
}

fn totals(out: &ReplayOut, filter: impl Fn(&str) -> bool) -> SegAcc {
    let mut t = SegAcc::default();
    for (k, a) in &out.segs {
        if !filter(k) {
            continue;
        }
        t.frames += a.frames;
        t.covered += a.covered;
        t.anchored += a.anchored;
        t.gt += a.gt;
        t.gt_covered += a.gt_covered;
        t.wrong += a.wrong;
        t.wrong_map += a.wrong_map;
        t.err.extend_from_slice(&a.err);
        t.switches += a.switches;
        t.switch_px.extend_from_slice(&a.switch_px);
    }
    t
}

fn seg_json(a: &SegAcc) -> Value {
    json!({
        "frames": a.frames, "coverage_pct": pct(a.covered, a.frames), "anchored_frames": a.anchored,
        "gt_frames": a.gt, "gt_covered_pct": pct(a.gt_covered, a.gt),
        "wrong_vs_gt": a.wrong, "wrong_map": a.wrong_map,
        "wrong_pct_of_covered": pct(a.wrong + a.wrong_map, a.covered),
        "err_p50": percentile(&a.err, 0.5), "err_p95": percentile(&a.err, 0.95), "err_max": percentile(&a.err, 1.0),
        "switches": a.switches, "switch_px": a.switch_px,
    })
}

#[test]
#[ignore]
fn e5_view_tracker_on_drag_recording() {
    let Some(video) = std::env::var_os("GH_MMAP_VIDEO").map(PathBuf::from) else {
        println!("GH_MMAP_VIDEO not set — nothing to measure");
        return;
    };
    let root = repo_root();
    let stem = video.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let labels_path = std::env::var_os("GH_MMAP_LABELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("scratch").join("mmap_rec").join(format!("{stem}.labels.json")));
    let segs = load_labels(&labels_path);
    let fps: f64 = env_or("GH_MMAP_FPS", "30").parse().expect("GH_MMAP_FPS");
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let variant = if use_votes() { "_votes" } else { "" };
    let cache_path = tmp.join(format!("e5_cache_{stem}_{fps}fps{variant}.json"));
    println!("registration: translation votes {}", if use_votes() { "ON" } else { "off" });
    let recompute = std::env::var("GH_MMAP_RECOMPUTE").is_ok_and(|v| v == "1");
    let remotion = std::env::var("GH_MMAP_REMOTION").is_ok_and(|v| v == "1");

    let cached: Option<Cache> = std::fs::read_to_string(&cache_path)
        .ok()
        .filter(|_| !recompute)
        .map(|txt| serde_json::from_str(&txt).expect("cache json"));
    let cache: Cache = match cached {
        Some(c) if !remotion => {
            println!("phase A: cached {}", cache_path.display());
            c
        }
        prior => {
            let (w, h) = probe_dims(&env_or("GH_FFPROBE", "ffprobe"), &video);
            println!(
                "phase A: measuring {} — {w}x{h} @ {fps} fps{}",
                video.display(),
                if prior.is_some() { " (registrations reused from cache)" } else { "" }
            );
            let c = measure(&video, &segs, fps, w, h, prior.as_ref());
            std::fs::write(&cache_path, serde_json::to_string(&c).expect("json")).expect("write cache");
            println!("phase A cached -> {}", cache_path.display());
            c
        }
    };
    let n = cache.frames.len();
    assert!(n > 0, "no frames");
    // Re-apply labels (allows editing the labels file without re-measuring).
    let mut cache = cache;
    for f in &mut cache.frames {
        f.label = label_at(&segs, f.t);
    }

    // ---- real-time cost ----
    let col = |g: fn(&FrameRec) -> f64| cache.frames.iter().map(g).collect::<Vec<f64>>();
    let slider_ms = col(|f| f.slider_ms);
    let motion_ms = col(|f| f.motion_ms);
    let per_frame_ms = col(|f| f.slider_ms + f.motion_ms);
    let reg_ms = col(|f| f.reg_ms);
    println!("\n===== e5 anchor+track ({n} frames @ {fps} fps, {}x{}) =====", cache.w, cache.h);
    println!(
        "per-frame (slider + motion) ms: p50 {:.1} p95 {:.1} max {:.1} | slider p50 {:.1} p95 {:.1} | motion p50 {:.1} p95 {:.1}",
        percentile(&per_frame_ms, 0.5), percentile(&per_frame_ms, 0.95), percentile(&per_frame_ms, 1.0),
        percentile(&slider_ms, 0.5), percentile(&slider_ms, 0.95), percentile(&motion_ms, 0.5), percentile(&motion_ms, 0.95),
    );
    println!(
        "registration (worker) ms: p50 {:.0} p95 {:.0} max {:.0}",
        percentile(&reg_ms, 0.5), percentile(&reg_ms, 0.95), percentile(&reg_ms, 1.0)
    );
    let peaks: Vec<f64> = cache.frames.iter().filter_map(|f| f.motion.map(|m| f64::from(m.peak))).collect();
    println!(
        "motion peak p05 {:.3} p50 {:.3}; zoom-compensated frames {}",
        percentile(&peaks, 0.05), percentile(&peaks, 0.5),
        cache.frames.iter().filter(|f| f.motion.is_some_and(|m| m.r != 1.0)).count()
    );
    println!(
        "cursor located on {:.1}% of frames",
        pct(cache.frames.iter().filter(|f| f.cursor.is_some()).count(), n)
    );
    let raw_conf: usize = cache.frames.iter().filter(|f| f.reg.as_ref().is_some_and(|r| r.to_reg().is_confident())).count();
    println!("raw confident registrations: {:.1}% of frames", pct(raw_conf, n));

    // ---- sweep ----
    let base = TrackerConfig::default();
    let asyncs = Schedule::Async { speed: 1.0, workers: 1 };
    let mut sweep: Vec<(String, TrackerConfig, Schedule)> = vec![
        ("base async 1 worker".into(), base, asyncs),
        ("async 2 workers".into(), base, Schedule::Async { speed: 1.0, workers: 2 }),
        ("async 3 workers".into(), base, Schedule::Async { speed: 1.0, workers: 3 }),
        ("async x0.5 speed".into(), base, Schedule::Async { speed: 0.5, workers: 1 }),
        ("fixed c3 d3".into(), base, Schedule::Fixed { cadence: 3, delay: 3 }),
        ("fixed c1 d0 (ideal)".into(), base, Schedule::Fixed { cadence: 1, delay: 0 }),
    ];
    for p in [0.06f32, 0.10, 0.12] {
        sweep.push((format!("min_peak {p}"), TrackerConfig { min_peak: p, ..base }, asyncs));
    }
    for p in [0.03f32, 0.05] {
        sweep.push((format!("weak_peak {p}"), TrackerConfig { weak_peak: p, ..base }, asyncs));
    }
    for a in [25.0, 60.0] {
        sweep.push((format!("agree {a}"), TrackerConfig { agree_px_1440: a, ..base }, asyncs));
    }
    for (a, sw) in [(3u32, 6u32), (5, 6), (4, 4), (4, 8)] {
        sweep.push((
            format!("acquire {a} switch {sw}"),
            TrackerConfig { acquire_points: a, switch_points: sw, ..base },
            asyncs,
        ));
    }
    sweep.push(("same_view_min_inliers 0".into(), TrackerConfig { same_view_min_inliers: 0, ..base }, asyncs));
    for d in [24.0, 96.0] {
        sweep.push((format!("distinct {d}px"), TrackerConfig { distinct_view_px_1440: d, ..base }, asyncs));
    }

    let gt_all = ground_truth(&cache);
    let gt_strong = cache.frames.iter().filter(|f| strong_reg(f).is_some_and(|g| f.label == Some(g.map_id))).count();
    println!(
        "holdout ground truth: {} strong registrations on the labelled map, {} kept after the neighbour-consensus check",
        gt_strong,
        gt_all.iter().filter(|g| g.is_some()).count()
    );
    let is_labelled = |k: &str| k != "unlabelled";
    println!("\n{:<22} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6} {:>7} {:>5}",
             "config", "cov all", "cov 2", "cov 34", "cov 36", "cov 40", "cov 7", "wrong%", "wrongN", "wMap", "errP95", "sw");
    let mut sweep_json = serde_json::Map::new();
    let mut base_out = None;
    for (name, cfg, sched) in &sweep {
        let out = replay(&cache, &gt_all, *cfg, *sched);
        let all = totals(&out, is_labelled);
        let cov = |k: &str| out.segs.get(k).map_or(f64::NAN, |a| pct(a.covered, a.frames));
        println!(
            "{name:<22} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% | {:>5.2}% {:>6} {:>6} {:>7.1} {:>5}",
            pct(all.covered, all.frames), cov("map_02"), cov("map_34"), cov("map_36"), cov("map_40"), cov("map_07"),
            pct(all.wrong + all.wrong_map, all.covered), all.wrong, all.wrong_map, percentile(&all.err, 0.95), all.switches,
        );
        let mut segj = serde_json::Map::new();
        for (k, a) in &out.segs {
            segj.insert(k.clone(), seg_json(a));
        }
        segj.insert("ALL_LABELLED".into(), seg_json(&all));
        sweep_json.insert(name.clone(), json!({"config": format!("{cfg:?} {sched:?}"), "fed": out.fed,
                                               "verdicts": out.verdicts, "segments": segj}));
        if base_out.is_none() {
            base_out = Some(out);
        }
    }

    // ---- base detail ----
    let out = base_out.expect("base");
    println!("\n--- base async: per segment ---");
    for (k, a) in &out.segs {
        println!(
            "{k:<11} frames {:>5} | coverage {:>5.1}% | anchored {:>4} | gt {:>4} (covered {:>5.1}%) | wrong {:>3} wMap {:>3} | err p50 {:>5.1} p95 {:>6.1} max {:>8.1} | switches {:>2}",
            a.frames, pct(a.covered, a.frames), a.anchored, a.gt, pct(a.gt_covered, a.gt), a.wrong, a.wrong_map,
            percentile(&a.err, 0.5), percentile(&a.err, 0.95), percentile(&a.err, 1.0), a.switches,
        );
    }
    println!("verdicts: {:?} (fed {} frames)", out.verdicts, out.fed);
    println!("timeline (1 char = 1 s): # fully covered · 0-9 coverage tenths · X wrong · - unlabelled");
    let per_sec = fps.round().max(1.0) as usize;
    let mut line = String::new();
    for (si, chunk) in out.per_frame.chunks(per_sec).enumerate() {
        let t0 = si * per_sec;
        let labelled = cache.frames[t0..t0 + chunk.len()].iter().any(|f| f.label.is_some());
        let cov = chunk.iter().filter(|c| c.0).count();
        let ch = if chunk.iter().any(|c| c.1) {
            'X'
        } else if !labelled {
            '-'
        } else if cov == chunk.len() {
            '#'
        } else {
            char::from_digit((10 * cov / chunk.len()) as u32, 10).unwrap_or('?')
        };
        line.push(ch);
    }
    for (li, c) in line.as_bytes().chunks(60).enumerate() {
        println!("{:>4}s {}", li * 60, String::from_utf8_lossy(c));
    }
    let detail: Vec<Value> = out
        .per_frame
        .iter()
        .zip(&cache.frames)
        .map(|(p, f)| json!({"t": f.t, "label": f.label, "cov": p.0, "wrong": p.1, "switch_px": p.2,
                             "peak": f.motion.map(|m| m.peak), "r": f.motion.map(|m| m.r),
                             "reg_map": f.reg.as_ref().map(|r| r.map_id), "inl": f.reg.as_ref().map(|r| r.inliers)}))
        .collect();
    let report = json!({"video": cache.video, "fps": fps, "frames": n, "sweep": sweep_json, "base_detail": detail});
    let rp = tmp.join(format!("e5_{stem}.json"));
    std::fs::write(&rp, serde_json::to_string_pretty(&report).expect("json")).expect("write report");
    println!("JSON report: {}", rp.display());
}

/// Quick A/B of `ViewOptions::translation_votes` on a time range of the recording:
/// `GH_MMAP_PROBE="119.3-124"` (seconds). Prints per-frame inliers / latency of both variants and
/// how far apart their transforms are.
#[test]
#[ignore]
fn e5_votes_probe() {
    let (Some(video), Ok(range)) = (std::env::var_os("GH_MMAP_VIDEO").map(PathBuf::from), std::env::var("GH_MMAP_PROBE"))
    else {
        println!("GH_MMAP_VIDEO / GH_MMAP_PROBE not set — nothing to probe");
        return;
    };
    let (from, to) = range.split_once('-').map(|(a, b)| (a.parse::<f64>().expect("from"), b.parse::<f64>().expect("to"))).expect("from-to");
    let root = repo_root();
    let stem = video.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let segs = load_labels(&root.join("scratch").join("mmap_rec").join(format!("{stem}.labels.json")));
    let (w, h) = probe_dims(&env_or("GH_FFPROBE", "ffprobe"), &video);
    let loc = MapLocator::with_paths(
        LocatorPaths {
            db_path: Some(root.join("data").join("multimap_v4.json")),
            terrain_npz_path: Some(root.join("data").join("terrain_n2_v4.npz")),
            cave_masks_npz_path: Some(root.join("data").join("cave_masks_v4.npz")),
            ui_anchors_npz_path: Some(root.join("data").join("ui_anchors_v4.npz")),
            legacy_underground_db_path: Some(root.join("data").join("underground_db.json")),
        },
        None,
    );
    let fps = 30.0;
    let mut child = Command::new(env_or("GH_FFMPEG", "ffmpeg"))
        .args(["-v", "error", "-i"])
        .arg(&video)
        .args(["-vf", &format!("fps={fps},trim=start={from}:end={to},setpts=PTS-STARTPTS"), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg");
    let mut stdout = child.stdout.take().expect("stdout");
    let mut buf = vec![0u8; w * h * 3];
    let (mut n, mut conf_a, mut conf_b, mut strong_a, mut strong_b) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut ms_a, mut ms_b) = (Vec::new(), Vec::new());
    while stdout.read_exact(&mut buf).is_ok() {
        let t = from + n as f64 / fps;
        n += 1;
        let img = ArrayView3::from_shape((h, w, 3), &buf[..]).expect("frame");
        let hint = label_at(&segs, t).map_or(ViewHint::Map(2), ViewHint::Map);
        let slider = ZoomSliderDetector::detect_slider(img).filter(|s| s.found && s.ratio_t.is_finite());
        let run = |votes: bool| {
            let opts = ViewOptions { allow_terrain_only: false, translation_votes: votes };
            let t0 = Instant::now();
            let r = loc.locate_view_with(img, slider.as_ref(), hint, opts);
            (r, t0.elapsed().as_secs_f64() * 1000.0)
        };
        let (a, ta) = run(false);
        let (b, tb) = run(true);
        ms_a.push(ta);
        ms_b.push(tb);
        let ca = a.as_ref().is_some_and(ViewRegistration::is_confident);
        let cb = b.as_ref().is_some_and(ViewRegistration::is_confident);
        conf_a += usize::from(ca);
        conf_b += usize::from(cb);
        strong_a += usize::from(a.as_ref().is_some_and(|r| r.inliers_count >= 6));
        strong_b += usize::from(b.as_ref().is_some_and(|r| r.inliers_count >= 6));
        let dis = match (&a, &b) {
            (Some(x), Some(y)) => format!(
                "{:.0}",
                ViewTransform::from_registration(x).disagreement_px_1440(&ViewTransform::from_registration(y), w, h)
            ),
            _ => "-".into(),
        };
        let brief = |r: &Option<ViewRegistration>| {
            r.as_ref().map_or("none".to_owned(), |r| format!("{:>2} inl {:>8.0} {:>8.0} s{:.3} terr{:.2}", r.inliers_count, r.tx, r.ty, r.scale, r.terrain_score))
        };
        println!("t={t:7.2} | base {} {:>4.0}ms | votes {} {:>4.0}ms | apart {dis}px", brief(&a), ta, brief(&b), tb);
    }
    let _ = child.wait();
    println!(
        "\n{n} frames: confident base {conf_a} vs votes {conf_b}; >=6 inliers base {strong_a} vs votes {strong_b}; ms p50 base {:.0} votes {:.0}, p95 base {:.0} votes {:.0}",
        percentile(&ms_a, 0.5), percentile(&ms_b, 0.5), percentile(&ms_a, 0.95), percentile(&ms_b, 0.95)
    );
}
