//! Real-drag measurement for the v5.5 M-Map route guidance: runs `MapLocator::locate_view` on a
//! screen recording of the M-Map being dragged (decoded on the fly by ffmpeg, nothing written
//! to disk). Recordings have no absolute ground truth, so two algorithm-independent checks:
//!   * map correctness vs hand-labelled segments (`<video stem>.labels.json`);
//!   * drag consistency: the pixel shift between consecutive frames implied by the two
//!     registrations must match the shift measured directly on the images (phase correlation).
//!
//! Run:
//! `GH_MMAP_VIDEO="<file.mp4>" cargo nextest run --manifest-path rust/Cargo.toml -p gh-cv --profile cv-heavy --cargo-profile cv-bench --run-ignored all -E "test(e4_)" --no-capture`
//! Optional: `GH_MMAP_LABELS` (default `scratch/mmap_rec/<video stem>.labels.json`),
//! `GH_MMAP_FPS` (default 6), `GH_FFMPEG` / `GH_FFPROBE` (default from PATH).

use gh_cv::locator::{LocatorPaths, MapLocator, ViewHint, ViewRegistration};
use ndarray::Array3;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const OK_PX_1440: f64 = 12.0;
const BAD_PX_1440: f64 = 40.0;
/// Downscale factor and central crop (in downscaled px) used for phase correlation; excludes
/// the quest panel, zoom slider, top bar, right buttons and the bottom-right region label.
const PC_DOWN: usize = 4;
const PC_X0: usize = 112;
const PC_Y0: usize = 52;
const PC_W: usize = 512;
const PC_H: usize = 256;
const PC_MIN_PEAK: f32 = 0.06;

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
    let Ok(txt) = std::fs::read_to_string(path) else {
        println!("(no labels file at {} — map correctness not scored)", path.display());
        return Vec::new();
    };
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

/// Downscaled, Hann-windowed grayscale crop of the map area.
fn pc_patch(img: &Array3<u8>) -> Vec<f32> {
    let (h, w, _) = img.dim();
    let mut out = vec![0f32; PC_W * PC_H];
    let n = (PC_DOWN * PC_DOWN) as f32;
    for py in 0..PC_H {
        for px in 0..PC_W {
            let (x0, y0) = ((PC_X0 + px) * PC_DOWN, (PC_Y0 + py) * PC_DOWN);
            let mut acc = 0f32;
            for dy in 0..PC_DOWN {
                for dx in 0..PC_DOWN {
                    let (x, y) = ((x0 + dx).min(w - 1), (y0 + dy).min(h - 1));
                    let (b, g, r) = (img[[y, x, 0]] as f32, img[[y, x, 1]] as f32, img[[y, x, 2]] as f32);
                    acc += 0.114 * b + 0.587 * g + 0.299 * r;
                }
            }
            let wx = 0.5 - 0.5 * (std::f32::consts::TAU * px as f32 / (PC_W - 1) as f32).cos();
            let wy = 0.5 - 0.5 * (std::f32::consts::TAU * py as f32 / (PC_H - 1) as f32).cos();
            out[py * PC_W + px] = acc / n * wx * wy;
        }
    }
    let mean = out.iter().sum::<f32>() / out.len() as f32;
    out.iter_mut().for_each(|v| *v -= mean);
    out
}

fn fft2(data: &mut [Complex<f32>], planner: &mut FftPlanner<f32>, inverse: bool) {
    let (row, col) = if inverse {
        (planner.plan_fft_inverse(PC_W), planner.plan_fft_inverse(PC_H))
    } else {
        (planner.plan_fft_forward(PC_W), planner.plan_fft_forward(PC_H))
    };
    for r in data.chunks_mut(PC_W) {
        row.process(r);
    }
    let mut colbuf = vec![Complex::new(0f32, 0f32); PC_H];
    for c in 0..PC_W {
        for r in 0..PC_H {
            colbuf[r] = data[r * PC_W + c];
        }
        col.process(&mut colbuf);
        for r in 0..PC_H {
            data[r * PC_W + c] = colbuf[r];
        }
    }
}

/// Content displacement (frame A -> frame B) in FULL-resolution px, plus peak strength.
fn phase_shift(a: &[f32], b: &[f32], planner: &mut FftPlanner<f32>) -> ((f64, f64), f32) {
    let mut fa: Vec<Complex<f32>> = a.iter().map(|&v| Complex::new(v, 0.0)).collect();
    let mut fb: Vec<Complex<f32>> = b.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fft2(&mut fa, planner, false);
    fft2(&mut fb, planner, false);
    let mut r: Vec<Complex<f32>> = fa
        .iter()
        .zip(&fb)
        .map(|(x, y)| {
            let p = x.conj() * y;
            let m = p.norm();
            if m > 1e-9 { p / m } else { Complex::new(0.0, 0.0) }
        })
        .collect();
    fft2(&mut r, planner, true);
    let n = (PC_W * PC_H) as f32;
    let (mut best, mut bi) = (f32::MIN, 0usize);
    for (i, v) in r.iter().enumerate() {
        let re = v.re / n;
        if re > best {
            best = re;
            bi = i;
        }
    }
    let (mut dx, mut dy) = ((bi % PC_W) as f64, (bi / PC_W) as f64);
    if dx > (PC_W / 2) as f64 {
        dx -= PC_W as f64;
    }
    if dy > (PC_H / 2) as f64 {
        dy -= PC_H as f64;
    }
    ((dx * PC_DOWN as f64, dy * PC_DOWN as f64), best)
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * q).round() as usize]
}

/// Previous frame: phase-correlation patch, confident registrations per mode, segment label.
type PrevFrame = (Vec<f32>, [Option<ViewRegistration>; 2], Option<i32>);

#[derive(Default)]
struct SegStats {
    frames: usize,
    confident: usize,
    right_map: usize,
    wrong_map: usize,
    underground: usize,
    ms: Vec<f64>,
    cons_err: Vec<f64>,
    cons_bad: usize,
}

impl SegStats {
    fn json(&mut self) -> Value {
        let pct = |n: usize, d: usize| if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 };
        let cons_n = self.cons_err.len();
        json!({
            "frames": self.frames,
            "confident_pct": pct(self.confident, self.frames),
            "right_map_pct_of_frames": pct(self.right_map, self.frames),
            "wrong_map_confident": self.wrong_map,
            "underground_pct": pct(self.underground, self.frames),
            "ms_p50": percentile(&mut self.ms.clone(), 0.5),
            "ms_p95": percentile(&mut self.ms.clone(), 0.95),
            "consistency_pairs": cons_n,
            "consistency_err_p50": percentile(&mut self.cons_err.clone(), 0.5),
            "consistency_err_p95": percentile(&mut self.cons_err.clone(), 0.95),
            "consistency_ok_pct": pct(self.cons_err.iter().filter(|&&e| e <= OK_PX_1440).count(), cons_n),
            "consistency_bad_gt40": self.cons_bad,
        })
    }
}

fn reg_brief(r: Option<&ViewRegistration>) -> Value {
    r.map_or(Value::Null, |r| {
        json!({"map": r.map_id, "s": r.scale, "tx": r.tx, "ty": r.ty, "inl": r.inliers_count,
               "terr": r.terrain_score, "t_only": r.terrain_only, "conf": r.is_confident(),
               "ug": r.is_underground, "zoom": r.zoom_ratio_t, "cand": r.candidates_count})
    })
}

#[test]
#[ignore]
fn e4_view_registration_on_drag_recording() {
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
    let fps: f64 = env_or("GH_MMAP_FPS", "6").parse().expect("GH_MMAP_FPS");
    let ffmpeg = env_or("GH_FFMPEG", "ffmpeg");
    let (w, h) = probe_dims(&env_or("GH_FFPROBE", "ffprobe"), &video);
    println!("video {} — {w}x{h}, sampling {fps} fps", video.display());

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

    let mut child = Command::new(&ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(&video)
        .args(["-vf", &format!("fps={fps}"), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg (set GH_FFMPEG)");
    let mut stdout = child.stdout.take().expect("ffmpeg stdout");
    let mut buf = vec![0u8; w * h * 3];

    let modes = [("no_hint", false), ("map_hint", true)];
    let mut stats: BTreeMap<(String, String), SegStats> = BTreeMap::new();
    let mut timeline: BTreeMap<String, String> = modes.iter().map(|(m, _)| ((*m).to_owned(), String::new())).collect();
    let mut prev: Option<PrevFrame> = None;
    let mut planner = FftPlanner::<f32>::new();
    let mut frames_json = Vec::new();
    let mut idx = 0usize;
    let run_start = Instant::now();

    while stdout.read_exact(&mut buf).is_ok() {
        let t = idx as f64 / fps;
        idx += 1;
        if idx.is_multiple_of(60) {
            println!(
                "  ... {idx} frames (t={t:.0}s), {:.0} ms/frame wall incl. decode",
                run_start.elapsed().as_secs_f64() * 1000.0 / idx as f64
            );
        }
        let img = Array3::from_shape_vec((h, w, 3), buf.clone()).expect("frame shape");
        let label = label_at(&segs, t);
        let seg_key = label.map_or("unlabelled".to_owned(), |m| format!("map_{m}"));
        let patch = pc_patch(&img);

        let mut regs: [Option<ViewRegistration>; 2] = [None, None];
        let mut row = json!({"t": t, "label": label});
        for (mi, (mode, use_hint)) in modes.iter().enumerate() {
            let hint = match (use_hint, label) {
                (true, Some(m)) => ViewHint::Map(m),
                _ => ViewHint::None,
            };
            let t0 = Instant::now();
            let r = loc.locate_view(img.view(), None, hint);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let conf = r.as_ref().is_some_and(ViewRegistration::is_confident);
            let st = stats.entry(((*mode).to_owned(), seg_key.clone())).or_default();
            st.frames += 1;
            st.ms.push(ms);
            if r.as_ref().is_some_and(|r| r.is_underground) {
                st.underground += 1;
            }
            let ch = match (conf, label, r.as_ref()) {
                (false, _, _) => '.',
                (true, None, _) => 'o',
                (true, Some(lm), Some(r)) if r.map_id == lm => '#',
                _ => 'X',
            };
            if conf {
                st.confident += 1;
                match (label, r.as_ref()) {
                    (Some(lm), Some(r)) if r.map_id == lm => st.right_map += 1,
                    (Some(_), Some(_)) => st.wrong_map += 1,
                    _ => {}
                }
            }
            timeline.get_mut(*mode).expect("mode").push(ch);
            row[*mode] = json!({"ms": ms, "reg": reg_brief(r.as_ref())});
            regs[mi] = if conf { r } else { None };
        }

        if let Some((ref ppatch, ref pregs, plabel)) = prev {
            let ((dx, dy), peak) = phase_shift(ppatch, &patch, &mut planner);
            row["phase"] = json!({"dx": dx, "dy": dy, "peak": peak});
            for (mi, (mode, _)) in modes.iter().enumerate() {
                if let (Some(a), Some(b)) = (&pregs[mi], &regs[mi])
                    && a.map_id == b.map_id
                    && (a.scale / b.scale - 1.0).abs() < 0.01
                    && peak >= PC_MIN_PEAK
                    && plabel == label
                {
                    let s = 0.5 * (a.scale + b.scale);
                    let (px, py) = ((a.tx - b.tx) / s, (a.ty - b.ty) / s);
                    let err = (px - dx).hypot(py - dy) * 1440.0 / h as f64;
                    let st = stats.get_mut(&((*mode).to_owned(), seg_key.clone())).expect("stats");
                    st.cons_err.push(err);
                    if err > BAD_PX_1440 {
                        st.cons_bad += 1;
                    }
                    row[format!("{mode}_consistency_err")] = json!(err);
                }
            }
        }
        prev = Some((patch, regs, label));
        frames_json.push(row);
    }
    let _ = child.wait();
    assert!(idx > 0, "ffmpeg produced no frames");

    println!("\n===== e4 drag-recording report ({idx} frames @ {fps} fps) =====");
    println!("timeline legend: # confident & right map · X confident & WRONG map · o confident (unlabelled gap) · . nothing drawn");
    let per_sec = fps.round().max(1.0) as usize;
    for (mode, tl) in &timeline {
        println!("--- {mode} ---");
        let chars: Vec<char> = tl.chars().collect();
        for (li, chunk) in chars.chunks(per_sec * 20).enumerate() {
            let s: String = chunk.chunks(per_sec).map(|c| c.iter().collect::<String>()).collect::<Vec<_>>().join(" ");
            println!("{:>4}s {s}", li * 20);
        }
    }
    let mut summary = serde_json::Map::new();
    for ((mode, seg), st) in stats.iter_mut() {
        let j = st.json();
        println!(
            "{mode:<8} {seg:<11} frames {:>4} | confident {:>5.1}% | right map {:>5.1}% | WRONG-map {:>3} | consistency ok {:>5.1}% of {:>3} pairs, p95 {:>6.1}px, >40px {:>3} | ug {:>5.1}% | ms p50 {:>5.1} p95 {:>6.1}",
            j["frames"], j["confident_pct"].as_f64().unwrap_or(0.0), j["right_map_pct_of_frames"].as_f64().unwrap_or(0.0),
            j["wrong_map_confident"], j["consistency_ok_pct"].as_f64().unwrap_or(0.0), j["consistency_pairs"],
            j["consistency_err_p95"].as_f64().unwrap_or(f64::NAN), j["consistency_bad_gt40"],
            j["underground_pct"].as_f64().unwrap_or(0.0), j["ms_p50"].as_f64().unwrap_or(0.0), j["ms_p95"].as_f64().unwrap_or(0.0),
        );
        summary.insert(format!("{mode}/{seg}"), j);
    }
    let report = json!({"video": video.display().to_string(), "fps": fps, "frames": idx,
                        "summary": summary, "timeline": timeline, "frames_detail": frames_json});
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("e4_{stem}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&report).expect("json")).expect("write report");
    println!("JSON report: {}", out.display());
}
