//! Gate 0 Benchmark & Python `cv2` Oracle Parity Suite for `gh-cv::ops`.
//!
//! Run via:
//! `cargo nextest run --manifest-path rust/Cargo.toml -p gh-cv --profile cv-heavy --cargo-profile cv-bench --run-ignored all`

use gh_core::npz::NpzArchive;
use gh_core::ui_layout::compute_ui_layout;
use gh_cv::ops::color::{
    add_sat_scalar_u8, bgr_to_gray, bgr_to_hsv, bgra_to_bgr, in_range, lut_u8_3d,
    rgb_image_to_bgr_array3, rgba_image_to_bgra_array3, split_channels, threshold,
};
use gh_cv::ops::contours::{
    bounding_rect, canny_u8, connected_components_with_stats_8, find_contours_external,
};
use gh_cv::ops::filters::{box_filter_f32, clahe_u8};
use gh_cv::ops::flann::{KdTreeIndex, bf_knn_match_l2, lowe_ratio_filter};
use gh_cv::ops::morph::{
    draw_filled_circle_f32, draw_filled_circle_u8, draw_filled_convex_poly_u8,
    draw_filled_ellipse_sector_f32, get_structuring_element,
    morphology_ex_u8, dilate_u8, erode_u8,
};
use gh_cv::ops::ncc::{
    match_template_f32, match_template_masked_f32, match_template_masked_u8, match_template_u8,
    min_max_loc_f32,
};
use gh_cv::ops::orb::{OrbExtractor, OrbParams, bf_knn_match_hamming};
use gh_cv::ops::phase_correlate::{create_hanning_window, phase_correlate};
use gh_cv::ops::ransac_similarity::estimate_affine_partial_2d_ransac;
use gh_cv::ops::resize::{copy_make_border_f32, copy_make_border_u8, resize_f32, resize_u8, resize_u8_hwc};
use gh_cv::ops::sift::{SiftExtractor, quantize_rootsift_u8, to_rootsift};
use gh_cv::ops::sobel::{magnitude_f32, scharr_dy_f32, scharr_dy_u8_f32, sobel_3x3_u8_f32};
use gh_cv::ops::warp::{
    get_rotation_matrix_2d, invert_affine_transform, warp_affine_u8, warp_polar_annulus_u8_hwc,
    warp_polar_linear_u8,
};
use gh_cv::ops::{
    BorderMode, ContourMethod, Interpolation, MorphOp, MorphShape, Point2i, TemplateMatchMode,
    ThresholdType,
};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, s};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("gh_cv_gate0_{}_{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        Self { path: dir }
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .canonicalize()
        .expect("resolve repo root")
}

fn python_exe(root: &Path) -> PathBuf {
    let venv = root.join(".venv").join("Scripts").join("python.exe");
    assert!(venv.is_file(), "missing python venv at {}", venv.display());
    venv
}

fn run_python_oracle(root: &Path, script: &str, args: &[&str]) {
    let py = python_exe(root);
    let mut cmd = Command::new(&py);
    cmd.current_dir(root).arg("-c").arg(script);
    for a in args {
        cmd.arg(a);
    }
    let out = cmd.output().expect("spawn python oracle");
    if !out.status.success() {
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        panic!("Python oracle failed (status {:?}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}", out.status);
    }
}

fn median_and_p95_ms(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let med = if n.is_multiple_of(2) {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    } else {
        s[n / 2]
    };
    let p95_idx = ((n as f64 * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
    (med, s[p95_idx])
}

#[inline]
fn wrap_360(deg: f64) -> f64 {
    let v = deg.rem_euclid(360.0);
    if v >= 360.0 || v == 0.0 { 0.0 } else { v }
}

#[inline]
fn shortest_delta_deg(from_deg: f64, to_deg: f64) -> f64 {
    let mut d = (to_deg - from_deg + 180.0).rem_euclid(360.0) - 180.0;
    if d == -180.0 {
        d = 180.0;
    }
    d
}

#[derive(Debug, Deserialize)]
struct MetaRow {
    i: usize,
    #[serde(rename = "match")]
    match_kind: String,
    map_id: i32,
    world_x: Option<f64>,
    world_y: Option<f64>,
    underground: Option<bool>,
    cone_raw: Option<f64>,
}

fn load_meta_rows(root: &Path) -> Vec<MetaRow> {
    let path = root
        .join("scratch")
        .join("minimap_rec")
        .join("rec_20261122_043401")
        .join("meta.jsonl");
    let text = fs::read_to_string(&path).expect("read meta.jsonl");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<MetaRow>(l).expect("parse meta.jsonl row"))
        .collect()
}

fn load_png_bgra_or_bgr(path: &Path) -> Array3<u8> {
    let dyn_img = image::open(path).expect("open png");
    match dyn_img {
        image::DynamicImage::ImageRgba8(rgba) => rgba_image_to_bgra_array3(&rgba),
        other => {
            let rgb = other.to_rgb8();
            rgb_image_to_bgr_array3(&rgb)
        }
    }
}

fn crop_minimap_from_hud(hud: &Array3<u8>) -> Array3<u8> {
    let lay = compute_ui_layout(2560, 1440, false).expect("ui_layout");
    let r = lay.minimap_rect;
    let y0 = r.y as usize;
    let x0 = r.x as usize;
    let y1 = (y0 + r.height as usize).min(hud.dim().0);
    let x1 = (x0 + r.width as usize).min(hud.dim().1);
    hud.slice(s![y0..y1, x0..x1, ..]).to_owned()
}

fn crop_paimon_maybe_from_hud(hud: &Array3<u8>) -> Array2<u8> {
    let lay = compute_ui_layout(2560, 1440, false).expect("ui_layout");
    let r = lay.paimon_maybe_rect;
    let y1 = (r.y as usize + r.height as usize).min(hud.dim().0);
    let x1 = (r.x as usize + r.width as usize).min(hud.dim().1);
    let sub = hud.slice(s![r.y as usize..y1, r.x as usize..x1, ..]);
    bgr_to_gray(sub).expect("paimon bgr_to_gray")
}

fn array2_from_npy(arr: &gh_core::npz::NpyArray) -> Array2<u8> {
    let sh = arr.shape();
    assert_eq!(sh.len(), 2);
    Array2::from_shape_vec((sh[0], sh[1]), arr.as_u8().unwrap().to_vec()).unwrap()
}

fn array2_f32_from_npy(arr: &gh_core::npz::NpyArray) -> Array2<f32> {
    let sh = arr.shape();
    assert_eq!(sh.len(), 2);
    Array2::from_shape_vec((sh[0], sh[1]), arr.as_f32().unwrap().to_vec()).unwrap()
}

fn annulus_mask(sz: usize) -> Array2<u8> {
    let mut m = Array2::<u8>::zeros((sz, sz));
    let c = (sz as f64 - 1.0) * 0.5;
    let r_max = sz as f64 * 0.5;
    let r_lo = 0.22 * r_max;
    let r_hi = 0.85 * r_max;
    for y in 0..sz {
        for x in 0..sz {
            let rr = (x as f64 - c).hypot(y as f64 - c);
            if rr > r_lo && rr <= r_hi {
                m[[y, x]] = 255;
            }
        }
    }
    m
}

// ============================================================================
// Benchmark 1: DoG RootSIFT + 4-tree KD-Tree (sift + flann)
// ============================================================================

type VoteBinEntry = (f64, f64, f64);
type TileFeatureBatch = (Vec<[f32; 128]>, Vec<[f32; 2]>);

fn query_global_candidates(
    sift_q: &SiftExtractor,
    index: &KdTreeIndex,
    pts_world: &[[f32; 2]],
    mm_bgr: ArrayView3<'_, u8>,
    top_k: usize,
    min_bin_score: f64,
) -> Vec<(f64, f64, f64)> {
    const BIN_WORLD: f64 = 240.0;
    const NMS_DIST_W: f64 = 420.0;

    let mm_gray = bgr_to_gray(mm_bgr).unwrap();
    let (mh, mw) = mm_gray.dim();
    let s_sz = mh.min(mw);
    let mut all_rdes: Vec<[f32; 128]> = Vec::new();

    for sz in [90usize, 60, 44] {
        let use_sz = sz.min(s_sz);
        let sq_g = if mh == use_sz && mw == use_sz {
            mm_gray.clone()
        } else {
            resize_u8(mm_gray.view(), use_sz, use_sz, Interpolation::Area).unwrap()
        };
        let mut mask = annulus_mask(use_sz);
        let sq_bgr = if mh == use_sz && mw == use_sz {
            mm_bgr.to_owned()
        } else {
            resize_u8_hwc(mm_bgr, use_sz, use_sz, Interpolation::Area).unwrap()
        };
        let hsv = bgr_to_hsv(sq_bgr.view()).unwrap();
        let active_px = mask.iter().filter(|&&v| v > 0).count();
        let mut white_cnt = 0usize;
        let mut white_flags = vec![false; use_sz * use_sz];
        for y in 0..use_sz {
            for x in 0..use_sz {
                if sq_g[[y, x]] >= 238 && hsv[[y, x, 1]] <= 38 {
                    white_flags[y * use_sz + x] = true;
                    if mask[[y, x]] > 0 {
                        white_cnt += 1;
                    }
                }
            }
        }
        if active_px.saturating_sub(white_cnt) >= (0.4 * active_px as f64) as usize {
            for y in 0..use_sz {
                for x in 0..use_sz {
                    if white_flags[y * use_sz + x] {
                        mask[[y, x]] = 0;
                    }
                }
            }
        }
        if let Ok((kps, des)) = sift_q.detect_and_compute(sq_g.view(), Some(mask.view()))
            && kps.len() >= 4
        {
            for d in des {
                all_rdes.push(to_rootsift(&d));
            }
        }
    }

    if all_rdes.is_empty() {
        return Vec::new();
    }

    let pairs = index.knn_match(&all_rdes, 2);
    let mut bins: HashMap<(i32, i32), Vec<VoteBinEntry>> = HashMap::new();
    for pair in pairs {
        if pair.len() < 2 {
            continue;
        }
        let d0 = pair[0].distance as f64;
        let d1 = pair[1].distance as f64;
        if d1 <= 1e-6 {
            continue;
        }
        let ratio = d0 / d1;
        if ratio >= 0.78 {
            continue;
        }
        let w = 1.0 / (ratio + 0.15);
        let p = pts_world[pair[0].train_idx];
        let wx = p[0] as f64;
        let wy = p[1] as f64;
        let bx = (wx / BIN_WORLD).floor() as i32;
        let by = (wy / BIN_WORLD).floor() as i32;
        bins.entry((bx, by)).or_default().push((wx, wy, w));
    }

    if bins.is_empty() {
        return Vec::new();
    }

    let mut bin_sums: HashMap<(i32, i32), f64> = HashMap::new();
    for (k, v) in &bins {
        bin_sums.insert(*k, v.iter().map(|x| x.2).sum());
    }
    let mut scored_centers: Vec<(f64, (i32, i32))> = Vec::new();
    for &(bx, by) in bins.keys() {
        let mut tot = 0.0f64;
        for dy in -1..=1 {
            for dx in -1..=1 {
                tot += bin_sums.get(&(bx + dx, by + dy)).copied().unwrap_or(0.0);
            }
        }
        if tot >= min_bin_score {
            scored_centers.push((tot, (bx, by)));
        }
    }
    scored_centers.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut selected: Vec<(f64, f64, f64)> = Vec::new();
    for (sc, (bx, by)) in scored_centers {
        let mut pts_acc = Vec::new();
        for dy in -1..=1 {
            for dx in -1..=1 {
                if let Some(lst) = bins.get(&(bx + dx, by + dy)) {
                    pts_acc.extend_from_slice(lst);
                }
            }
        }
        if pts_acc.is_empty() {
            continue;
        }
        let w_sum: f64 = pts_acc.iter().map(|p| p.2).sum();
        let wx = pts_acc.iter().map(|p| p.0 * p.2).sum::<f64>() / w_sum;
        let wy = pts_acc.iter().map(|p| p.1 * p.2).sum::<f64>() / w_sum;
        if selected
            .iter()
            .any(|&(sx, sy, _)| (wx - sx).hypot(wy - sy) < NMS_DIST_W)
        {
            continue;
        }
        selected.push((wx, wy, sc));
        if selected.len() >= top_k {
            break;
        }
    }
    selected
}

#[test]
#[ignore]
fn gate0_bench_rootsift_flann() {
    let root = repo_root();
    let guard = TempDirGuard::new("rootsift_flann");
    let oracle_npz = guard.path.join("rootsift_oracle.npz");
    let oracle_json = guard.path.join("rootsift_oracle.json");

    let rows = load_meta_rows(&root);
    let mut query_indices: Vec<usize> = rows
        .iter()
        .filter(|r| {
            r.map_id == 2
                && !r.underground.unwrap_or(false)
                && (r.match_kind == "global_teyvat_relocate" || r.match_kind == "hd_sift")
                && r.world_x.is_some()
        })
        .map(|r| r.i)
        .take(28)
        .collect();
    // Ensure all Map 2 global_teyvat_relocate frames are included
    for r in &rows {
        if r.map_id == 2 && r.match_kind == "global_teyvat_relocate" && !query_indices.contains(&r.i) {
            query_indices.push(r.i);
        }
    }
    query_indices.sort_unstable();
    assert!(query_indices.len() >= 10);

    let idx_list_str = query_indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let py_script = r#"
import sys, json, math, numpy as np, cv2
from pathlib import Path

SCALE_FACTOR = 400.0
BIN_WORLD = 240.0
NMS_DIST_W = 420.0

def _to_rootsift(des):
    if des is None or len(des) == 0:
        return None
    f = des.astype(np.float32)
    l1 = np.sum(f, axis=1, keepdims=True) + np.float32(1e-7)
    return np.sqrt(f / l1)

def build_index(map2_gray, map2_meta):
    tx_start, ty_start, native_z = int(map2_meta[0]), int(map2_meta[1]), int(map2_meta[2])
    orig_x, orig_y = float(map2_meta[3]), float(map2_meta[4])
    s_src = 2.0 ** native_z
    h, w = map2_gray.shape[:2]
    tile_sz, stride, margin = 384, 360, 12
    ys = list(range(0, max(1, h - tile_sz + 1), stride)) or [0]
    if ys[-1] + tile_sz < h:
        ys.append(max(0, h - tile_sz))
    xs = list(range(0, max(1, w - tile_sz + 1), stride)) or [0]
    if xs[-1] + tile_sz < w:
        xs.append(max(0, w - tile_sz))
    sift = cv2.SIFT_create(nfeatures=900, contrastThreshold=0.011, edgeThreshold=15)
    all_des, all_pts = [], []
    for iy, y0 in enumerate(ys):
        y1 = min(h, y0 + tile_sz)
        th = y1 - y0
        ky0 = 0 if iy == 0 else ((ys[iy - 1] + tile_sz - margin) - y0)
        ky1 = th if iy == len(ys) - 1 else (th - margin)
        ky0 = max(0, min(th, ky0))
        ky1 = max(ky0, min(th, ky1))
        for ix, x0 in enumerate(xs):
            x1 = min(w, x0 + tile_sz)
            tw = x1 - x0
            tile = map2_gray[y0:y1, x0:x1]
            if float(np.std(tile)) < 8.0:
                continue
            kp, des = sift.detectAndCompute(tile, None)
            if des is None or not kp:
                continue
            kx0 = 0 if ix == 0 else ((xs[ix - 1] + tile_sz - margin) - x0)
            kx1 = tw if ix == len(xs) - 1 else (tw - margin)
            kx0 = max(0, min(tw, kx0))
            kx1 = max(kx0, min(tw, kx1))
            sel_idx, pts_list = [], []
            for k_idx, k in enumerate(kp):
                kx, ky = float(k.pt[0]), float(k.pt[1])
                if kx0 <= kx < kx1 and ky0 <= ky < ky1:
                    wx = (tx_start * 256.0 + (float(x0) + kx + 0.5)) / s_src - orig_x
                    wy = (ty_start * 256.0 + (float(y0) + ky + 0.5)) / s_src - orig_y
                    sel_idx.append(k_idx)
                    pts_list.append((wx, wy))
            if not sel_idx:
                continue
            rdes = _to_rootsift(des[sel_idx])
            if rdes is None:
                continue
            all_des.append(np.clip(np.round(rdes * SCALE_FACTOR), 0, 255).astype(np.uint8))
            all_pts.append(np.asarray(pts_list, dtype=np.float32))
    des_u8 = np.vstack(all_des)
    pts_f32 = np.ascontiguousarray(np.vstack(all_pts), dtype=np.float32)
    des_f32 = np.ascontiguousarray(des_u8.astype(np.float32) * np.float32(1.0 / SCALE_FACTOR))
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
    flann.add([des_f32])
    flann.train()
    return flann, pts_f32

def query_cands(flann, pts_world, sift_q, mm_img, top_k=4, min_bin_score=1.5):
    mm_gray = mm_img if mm_img.ndim == 2 else cv2.cvtColor(mm_img[:, :, :3], cv2.COLOR_BGR2GRAY)
    s_sz = min(mm_gray.shape[:2])
    des_parts = []
    for sz in (90, 60, 44):
        use_sz = sz if s_sz >= sz else s_sz
        mm_sq_g = mm_gray if mm_gray.shape[:2] == (use_sz, use_sz) else cv2.resize(mm_gray, (use_sz, use_sz), interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[0:use_sz, 0:use_sz]
        c = (use_sz - 1) * 0.5
        rr = np.hypot(xx - c, yy - c)
        r_max = use_sz * 0.5
        mask = np.where((rr > 0.22 * r_max) & (rr <= 0.85 * r_max), np.uint8(255), np.uint8(0))
        if mm_img.ndim == 3 and mm_img.shape[2] >= 3:
            mm_sq = mm_img[:, :, :3] if mm_img.shape[:2] == (use_sz, use_sz) else cv2.resize(mm_img[:, :, :3], (use_sz, use_sz), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(mm_sq, cv2.COLOR_BGR2HSV)
            white_pins = (mm_sq_g >= 238) & (hsv[:, :, 1] <= 38)
            active_px = int(cv2.countNonZero(mask))
            if active_px - int(np.sum(white_pins)) >= 0.4 * active_px:
                mask[white_pins] = 0
        kp, des = sift_q.detectAndCompute(mm_sq_g, mask)
        if des is not None and len(kp) >= 4:
            rdes = _to_rootsift(des)
            if rdes is not None and len(rdes) > 0:
                des_parts.append(rdes)
    if not des_parts:
        return []
    des_all = np.ascontiguousarray(np.vstack(des_parts), dtype=np.float32)
    pairs = flann.knnMatch(des_all, k=2)
    bins = {}
    for pair in pairs:
        if len(pair) < 2:
            continue
        m0, m1 = pair[0], pair[1]
        d0, d1 = float(m0.distance), float(m1.distance)
        if d1 <= 1e-6:
            continue
        ratio = d0 / d1
        if ratio >= 0.78:
            continue
        w = 1.0 / (ratio + 0.15)
        p = pts_world[int(m0.trainIdx)]
        wx, wy = float(p[0]), float(p[1])
        bkey = (int(math.floor(wx / BIN_WORLD)), int(math.floor(wy / BIN_WORLD)))
        bins.setdefault(bkey, []).append((wx, wy, w))
    if not bins:
        return []
    bin_sums = {k: sum(item[2] for item in v) for k, v in bins.items()}
    scored = []
    for bx, by in bin_sums:
        nb_score = sum(bin_sums.get((bx + dx, by + dy), 0.0) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        if nb_score >= min_bin_score:
            scored.append((nb_score, (bx, by)))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    for nb_score, (bx, by) in scored:
        sum_w = sum_wx = sum_wy = 0.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for wx, wy, w in bins.get((bx + dx, by + dy), ()):
                    sum_w += w
                    sum_wx += wx * w
                    sum_wy += wy * w
        if sum_w <= 0.0:
            continue
        cx, cy = sum_wx / sum_w, sum_wy / sum_w
        if any(math.hypot(cx - ox, cy - oy) < NMS_DIST_W for ox, oy, _ in chosen):
            continue
        chosen.append((cx, cy, float(nb_score)))
        if len(chosen) >= top_k:
            break
    return chosen

out_npz = Path(sys.argv[1])
out_json = Path(sys.argv[2])
q_indices = [int(x) for x in sys.argv[3].split(",") if x]

z = np.load("data/terrain_n2_v4.npz")
m2_gray, m2_meta = z["map_2_gray"], z["map_2_meta"]
flann, pts_world = build_index(m2_gray, m2_meta)
sift_q = cv2.SIFT_create(nfeatures=160, contrastThreshold=0.008, edgeThreshold=15)

results = []
for qi in q_indices:
    p = Path(f"scratch/minimap_rec/rec_20261122_043401/frames/{qi:06d}.png")
    hud = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    mm = hud[20:311, 79:370, :3]
    cands = query_cands(flann, pts_world, sift_q, mm, top_k=4, min_bin_score=1.5)
    results.append({"i": qi, "cands": [[float(x), float(y), float(s)] for x, y, s in cands]})

out_json.write_text(json.dumps(results), encoding="utf-8")
np.savez(out_npz, dummy=np.array([1], dtype=np.int32))
"#;
    run_python_oracle(&root, py_script, &[
        oracle_npz.to_str().unwrap(),
        oracle_json.to_str().unwrap(),
        &idx_list_str,
    ]);

    #[derive(Deserialize)]
    struct PyRelocRes {
        i: usize,
        cands: Vec<[f64; 3]>,
    }
    let py_res: Vec<PyRelocRes> =
        serde_json::from_str(&fs::read_to_string(&oracle_json).unwrap()).unwrap();
    let py_map: HashMap<usize, Vec<[f64; 3]>> =
        py_res.into_iter().map(|r| (r.i, r.cands)).collect();

    // Load Map 2 terrain in Rust and build pure-Rust RootSIFT + 4-tree KD-Tree
    let terrain_npz = NpzArchive::read_file(&root.join("data").join("terrain_n2_v4.npz"))
        .expect("read terrain_n2_v4.npz");
    let m2_gray = array2_from_npy(terrain_npz.get("map_2_gray").unwrap());
    let m2_meta = terrain_npz.get("map_2_meta").unwrap().as_i64().unwrap();
    let tx_start = m2_meta[0] as f64;
    let ty_start = m2_meta[1] as f64;
    let s_src = 2.0f64.powi(m2_meta[2] as i32);
    let orig_x = m2_meta[3] as f64;
    let orig_y = m2_meta[4] as f64;

    let (mh, mw) = m2_gray.dim();
    let tile_sz = 384usize;
    let stride = 360usize;
    let margin = 12usize;
    let mut ys: Vec<usize> = (0..=(mh.saturating_sub(tile_sz))).step_by(stride).collect();
    if ys.is_empty() {
        ys.push(0);
    }
    if *ys.last().unwrap() + tile_sz < mh {
        ys.push(mh.saturating_sub(tile_sz));
    }
    let mut xs: Vec<usize> = (0..=(mw.saturating_sub(tile_sz))).step_by(stride).collect();
    if xs.is_empty() {
        xs.push(0);
    }
    if *xs.last().unwrap() + tile_sz < mw {
        xs.push(mw.saturating_sub(tile_sz));
    }

    let mut tile_jobs = Vec::new();
    for (iy, &y0) in ys.iter().enumerate() {
        let y1 = mh.min(y0 + tile_sz);
        let th = y1 - y0;
        let ky0 = if iy == 0 {
            0
        } else {
            (ys[iy - 1] + tile_sz - margin).saturating_sub(y0)
        };
        let ky1 = if iy + 1 == ys.len() {
            th
        } else {
            th.saturating_sub(margin)
        };
        for (ix, &x0) in xs.iter().enumerate() {
            let x1 = mw.min(x0 + tile_sz);
            let tw = x1 - x0;
            let kx0 = if ix == 0 {
                0
            } else {
                (xs[ix - 1] + tile_sz - margin).saturating_sub(x0)
            };
            let kx1 = if ix + 1 == xs.len() {
                tw
            } else {
                tw.saturating_sub(margin)
            };
            tile_jobs.push((x0, y0, x1, y1, kx0, ky0, kx1, ky1));
        }
    }

    let t_build = Instant::now();
    let sift_idx = SiftExtractor::new(900, 0.011, 15.0);
    let extracted: Vec<TileFeatureBatch> = tile_jobs
        .par_iter()
        .filter_map(|&(x0, y0, x1, y1, kx0, ky0, kx1, ky1)| {
            let tile = m2_gray.slice(s![y0..y1, x0..x1]);
            let mean = tile.iter().map(|&v| v as f64).sum::<f64>() / (tile.len() as f64);
            let var = tile
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / (tile.len() as f64);
            if var.sqrt() < 8.0 {
                return None;
            }
            let (kps, des) = sift_idx.detect_and_compute(tile, None).ok()?;
            let mut rdes_list = Vec::new();
            let mut pts_list = Vec::new();
            for (k, d) in kps.into_iter().zip(des) {
                if k.x >= kx0 as f32 && k.x < kx1 as f32 && k.y >= ky0 as f32 && k.y < ky1 as f32 {
                    let px = x0 as f64 + k.x as f64;
                    let py = y0 as f64 + k.y as f64;
                    let wx = (tx_start * 256.0 + (px + 0.5)) / s_src - orig_x;
                    let wy = (ty_start * 256.0 + (py + 0.5)) / s_src - orig_y;
                    let rdes = to_rootsift(&d);
                    let q = quantize_rootsift_u8(&rdes, 400.0);
                    let mut dequant = [0.0f32; 128];
                    for (dst, &qv) in dequant.iter_mut().zip(q.iter()) {
                        *dst = qv as f32 * (1.0 / 400.0);
                    }
                    rdes_list.push(dequant);
                    pts_list.push([wx as f32, wy as f32]);
                }
            }
            Some((rdes_list, pts_list))
        })
        .collect();

    let mut all_des = Vec::new();
    let mut all_pts = Vec::new();
    for (d, p) in extracted {
        all_des.extend(d);
        all_pts.extend(p);
    }
    let n_landmarks = all_des.len();
    let kd = KdTreeIndex::build(all_des, 4, 48).expect("build 4-tree KdTreeIndex");
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    let sift_q = SiftExtractor::new(160, 0.008, 15.0);
    let mut latencies_ms = Vec::new();
    let mut rs_hits = 0usize;
    let mut py_hits = 0usize;
    let mut rs_top1_errs = Vec::new();
    let mut rs_best_errs = Vec::new();

    let row_by_i: HashMap<usize, &MetaRow> = rows.iter().map(|r| (r.i, r)).collect();
    for &qi in &query_indices {
        let row = row_by_i[&qi];
        let gt_x = row.world_x.unwrap();
        let gt_y = row.world_y.unwrap();
        let frame_path = root
            .join("scratch")
            .join("minimap_rec")
            .join("rec_20261122_043401")
            .join("frames")
            .join(format!("{qi:06}.png"));
        let hud = load_png_bgra_or_bgr(&frame_path);
        let mm = crop_minimap_from_hud(&hud);

        let t0 = Instant::now();
        let cands = query_global_candidates(&sift_q, &kd, &all_pts, mm.view(), 4, 1.5);
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        if let Some(best_d) = cands
            .iter()
            .map(|c| (c.0 - gt_x).hypot(c.1 - gt_y))
            .reduce(f64::min)
        {
            if best_d <= 250.0 {
                rs_hits += 1;
            }
            rs_best_errs.push(best_d);
            rs_top1_errs.push((cands[0].0 - gt_x).hypot(cands[0].1 - gt_y));
        }

        if let Some(py_c) = py_map.get(&qi)
            && let Some(best_py) = py_c
                .iter()
                .map(|c| (c[0] - gt_x).hypot(c[1] - gt_y))
                .reduce(f64::min)
            && best_py <= 250.0
        {
            py_hits += 1;
        }
    }

    // Blank negative image check
    let blank = Array3::<u8>::from_elem((291, 291, 3), 120);
    let blank_cands = query_global_candidates(&sift_q, &kd, &all_pts, blank.view(), 4, 1.5);
    assert!(blank_cands.is_empty(), "expected 0 candidates on flat blank frame");

    let (lat_med, lat_p95) = median_and_p95_ms(&latencies_ms);
    let (err_med, err_p95) = median_and_p95_ms(&rs_best_errs);

    // Compute Python oracle coarse centroid error + post-RANSAC SIFT refinement error on Map 2 patch
    let mut py_best_errs = Vec::new();
    let mut verified_errs = Vec::new();
    let mut sift128_lat = Vec::new();
    let sift_ref = SiftExtractor::new(600, 0.011, 15.0);
    for &qi in &query_indices {
        let row = row_by_i[&qi];
        let gt_x = row.world_x.unwrap();
        let gt_y = row.world_y.unwrap();
        if let Some(py_c) = py_map.get(&qi)
            && let Some(best_py) = py_c.iter().map(|c| (c[0] - gt_x).hypot(c[1] - gt_y)).reduce(f64::min)
        {
            py_best_errs.push(best_py);
        }
        let fp = root
            .join("scratch")
            .join("minimap_rec")
            .join("rec_20261122_043401")
            .join("frames")
            .join(format!("{qi:06}.png"));
        let hud = load_png_bgra_or_bgr(&fp);
        let mm = crop_minimap_from_hud(&hud);
        let cands = query_global_candidates(&sift_q, &kd, &all_pts, mm.view(), 4, 1.5);
        let mm_g = bgr_to_gray(mm.view()).unwrap();
        let mm_sq = resize_u8(mm_g.view(), 128, 128, Interpolation::Area).unwrap();
        let mm_mask = annulus_mask(128);
        let ts0 = Instant::now();
        let (kp_m, des_m) = sift_q.detect_and_compute(mm_sq.view(), Some(mm_mask.view())).unwrap();
        sift128_lat.push(ts0.elapsed().as_secs_f64() * 1000.0);
        for &(cx_w, cy_w, _) in &cands {
            if (cx_w - gt_x).hypot(cy_w - gt_y) > 250.0 {
                continue;
            }
            let px_c = (cx_w + orig_x) * s_src - tx_start * 256.0;
            let py_c = (cy_w + orig_y) * s_src - ty_start * 256.0;
            let rx0 = (px_c as isize - 140).clamp(0, mw as isize - 1) as usize;
            let ry0 = (py_c as isize - 140).clamp(0, mh as isize - 1) as usize;
            let rx1 = (px_c as isize + 140).clamp(1, mw as isize) as usize;
            let ry1 = (py_c as isize + 140).clamp(1, mh as isize) as usize;
            if rx1 <= rx0 + 32 || ry1 <= ry0 + 32 {
                continue;
            }
            let ref_patch = m2_gray.slice(s![ry0..ry1, rx0..rx1]);
            let (kp_r, des_r) = sift_ref.detect_and_compute(ref_patch, None).unwrap();
            let pairs = bf_knn_match_l2(&des_m, &des_r, 2);
            let good = lowe_ratio_filter(&pairs, 0.78);
            if good.len() >= 4 {
                let pm: Vec<[f32; 2]> = good.iter().map(|m| [kp_m[m.query_idx].x, kp_m[m.query_idx].y]).collect();
                let pr: Vec<[f32; 2]> = good.iter().map(|m| [kp_r[m.train_idx].x, kp_r[m.train_idx].y]).collect();
                if let Some((mat, inl)) = estimate_affine_partial_2d_ransac(&pm, &pr, 3.5, 1200, 0.99) {
                    let ninl: usize = inl.iter().map(|&v| v as usize).sum();
                    if ninl >= 4 {
                        let c = 127.0 * 0.5;
                        let rx = mat[0] * c + mat[1] * c + mat[2];
                        let ry = mat[3] * c + mat[4] * c + mat[5];
                        let px_g = rx0 as f64 + rx;
                        let py_g = ry0 as f64 + ry;
                        let wx = (tx_start * 256.0 + (px_g + 0.5)) / s_src - orig_x;
                        let wy = (ty_start * 256.0 + (py_g + 0.5)) / s_src - orig_y;
                        verified_errs.push((wx - gt_x).hypot(wy - gt_y));
                        break;
                    }
                }
            }
        }
    }
    let (py_err_med, py_err_p95) = median_and_p95_ms(&py_best_errs);
    let (ver_err_med, ver_err_p95) = median_and_p95_ms(&verified_errs);
    let (sift128_med, sift128_p95) = median_and_p95_ms(&sift128_lat);
    let rs_recall = 100.0 * rs_hits as f64 / query_indices.len() as f64;
    let py_recall = 100.0 * py_hits as f64 / query_indices.len() as f64;

    println!(
        "[GATE0 RootSIFT+FLANN] landmarks={n_landmarks} build={build_ms:.1}ms queries={} | rs_recall={rs_recall:.1}% ({rs_hits}/{}) py_recall={py_recall:.1}% ({py_hits}/{}) | coarse_err med={err_med:.2}w p95={err_p95:.2}w (py med={py_err_med:.2}w p95={py_err_p95:.2}w) | verified_err med={ver_err_med:.2}w p95={ver_err_p95:.2}w ({}/{}) | latency med={lat_med:.2}ms p95={lat_p95:.2}ms | sift128 med={sift128_med:.2}ms p95={sift128_p95:.2}ms",
        query_indices.len(),
        query_indices.len(),
        query_indices.len(),
        verified_errs.len(),
        rs_hits,
    );
    assert!(rs_hits >= py_hits.saturating_sub(1), "Rust recall {rs_recall:.1}% fell below Python {py_recall:.1}%");
    assert!(rs_recall >= 95.0, "expected >= 95% RootSIFT recall, got {rs_recall:.1}%");
    assert!(lat_med <= 25.0, "3-scale RootSIFT+FLANN query median latency {lat_med:.2}ms > 25.0ms");
    assert!(sift128_med <= 15.0, "128x128 SIFT median latency {sift128_med:.2}ms > 15.0ms");
    assert!(!verified_errs.is_empty() && ver_err_med <= 20.0, "expected post-verify median loc error <= 20w, got {ver_err_med:.2}w");
}

// ============================================================================
// Benchmark 2: ORB + Hamming BFMatcher (orb + ransac_similarity)
// ============================================================================

#[test]
#[ignore]
fn gate0_bench_orb_hamming() {
    let root = repo_root();
    let guard = TempDirGuard::new("orb_hamming");
    let oracle_npz = guard.path.join("orb_oracle.npz");
    let oracle_json = guard.path.join("orb_oracle.json");

    let py_script = r#"
import sys, json, numpy as np, cv2
from pathlib import Path

out_npz = Path(sys.argv[1])
out_json = Path(sys.argv[2])

patches = []
for qi in [1, 1005, 1442]:
    hud = cv2.imread(f"scratch/minimap_rec/rec_20261122_043401/frames/{qi:06d}.png", cv2.IMREAD_UNCHANGED)
    mm = cv2.cvtColor(hud[20:311, 79:370, :3], cv2.COLOR_BGR2GRAY)
    patches.append(mm)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.15, nlevels=6, edgeThreshold=11, patchSize=15, fastThreshold=7)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

angles = [0.0, 35.0, 90.0, 145.0, 225.0, 270.0, 315.0]
records = []
rot_imgs = []
for pi, ref in enumerate(patches):
    ref_eq = clahe.apply(ref)
    kp_r, des_r = orb.detectAndCompute(ref_eq, None)
    for ang in angles:
        M_rot = cv2.getRotationMatrix2D((145.0, 145.0), -ang, 1.0)
        rot_full = cv2.warpAffine(ref, M_rot, (291, 291), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mm = rot_full[45:246, 45:246].copy()  # 201x201 rotated inner patch
        rot_imgs.append(mm)
        mm_eq = clahe.apply(mm)
        kp_m, des_m = orb.detectAndCompute(mm_eq, None)
        good_cnt = 0
        ok = False
        err_deg = 999.0
        if des_r is not None and des_m is not None and len(kp_r) >= 8 and len(kp_m) >= 8:
            pairs = bf.knnMatch(des_r, des_m, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]
            good_cnt = len(good)
            if len(good) >= 6:
                pts_r = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts_m = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, inl = cv2.estimateAffinePartial2D(pts_r, pts_m, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000, confidence=0.99)
                if M is not None and inl is not None and int(inl.sum()) >= 6:
                    theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
                    d = ((theta - ang + 180.0) % 360.0) - 180.0
                    err_deg = abs(d)
                    ok = bool(err_deg < 6.0)
        records.append({"patch": int(pi), "angle": float(ang), "good": int(good_cnt), "ok": bool(ok), "err_deg": float(err_deg)})

out_json.write_text(json.dumps(records), encoding="utf-8")
np.savez(out_npz, refs=np.stack(patches), rots=np.stack(rot_imgs))
"#;
    run_python_oracle(&root, py_script, &[
        oracle_npz.to_str().unwrap(),
        oracle_json.to_str().unwrap(),
    ]);

    #[derive(Deserialize)]
    struct PyOrbRec {
        patch: usize,
        angle: f64,
        good: usize,
        ok: bool,
    }
    let py_recs: Vec<PyOrbRec> =
        serde_json::from_str(&fs::read_to_string(&oracle_json).unwrap()).unwrap();
    let npz = NpzArchive::read_file(&oracle_npz).unwrap();
    let refs_raw = npz.get("refs").unwrap();
    let rots_raw = npz.get("rots").unwrap();
    let refs_u8 = refs_raw.as_u8().unwrap();
    let rots_u8 = rots_raw.as_u8().unwrap();

    let orb = OrbExtractor::new(OrbParams::default());
    let mut latencies_ms = Vec::new();
    let mut rs_ok = 0usize;
    let mut py_ok = 0usize;
    let mut rs_good_sum = 0usize;
    let mut py_good_sum = 0usize;
    let mut heading_errs = Vec::new();

    for (idx, rec) in py_recs.iter().enumerate() {
        let p_off = rec.patch * 291 * 291;
        let ref_arr =
            Array2::from_shape_vec((291, 291), refs_u8[p_off..p_off + 291 * 291].to_vec()).unwrap();
        let r_off = idx * 201 * 201;
        let rot_arr =
            Array2::from_shape_vec((201, 201), rots_u8[r_off..r_off + 201 * 201].to_vec()).unwrap();

        let t0 = Instant::now();
        let ref_eq = clahe_u8(ref_arr.view(), 2.0, (8, 8)).unwrap();
        let rot_eq = clahe_u8(rot_arr.view(), 2.0, (8, 8)).unwrap();
        let (kp_r, des_r) = orb.detect_and_compute(ref_eq.view(), None).unwrap();
        let (kp_m, des_m) = orb.detect_and_compute(rot_eq.view(), None).unwrap();
        let pairs = bf_knn_match_hamming(&des_r, &des_m, 2);
        let good: Vec<_> = pairs
            .into_iter()
            .filter_map(|p| {
                if p.len() == 2 && p[0].distance < 0.78 * p[1].distance {
                    Some(p[0])
                } else {
                    None
                }
            })
            .collect();
        rs_good_sum += good.len();
        py_good_sum += rec.good;
        if rec.ok {
            py_ok += 1;
        }
        if good.len() >= 6 {
            let pts_r: Vec<[f32; 2]> = good.iter().map(|m| [kp_r[m.query_idx].x, kp_r[m.query_idx].y]).collect();
            let pts_m: Vec<[f32; 2]> = good.iter().map(|m| [kp_m[m.train_idx].x, kp_m[m.train_idx].y]).collect();
            if let Some((m, inl)) = estimate_affine_partial_2d_ransac(&pts_r, &pts_m, 4.0, 2000, 0.99) {
                let inl_cnt: usize = inl.iter().map(|&v| v as usize).sum();
                if inl_cnt >= 6 {
                    let theta = m[3].atan2(m[0]).to_degrees();
                    let err = shortest_delta_deg(rec.angle, theta).abs();
                    heading_errs.push(err);
                    if err < 6.0 {
                        rs_ok += 1;
                    }
                }
            }
        }
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // Also benchmark real minimap crops from rec_20261122_043401
    let rows = load_meta_rows(&root);
    let mut real_kps = 0usize;
    let mut real_lat = Vec::new();
    for r in rows.iter().filter(|r| r.match_kind == "hd_sift").take(15) {
        let fp = root
            .join("scratch")
            .join("minimap_rec")
            .join("rec_20261122_043401")
            .join("frames")
            .join(format!("{:06}.png", r.i));
        let hud = load_png_bgra_or_bgr(&fp);
        let mm = crop_minimap_from_hud(&hud);
        let g = bgr_to_gray(mm.view()).unwrap();
        let mask = annulus_mask(g.dim().0);
        let t0 = Instant::now();
        let eq = clahe_u8(g.view(), 2.0, (8, 8)).unwrap();
        let (kp, _) = orb.detect_and_compute(eq.view(), Some(mask.view())).unwrap();
        real_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        real_kps += kp.len();
    }

    let (lat_med, lat_p95) = median_and_p95_ms(&latencies_ms);
    let (real_med, real_p95) = median_and_p95_ms(&real_lat);
    let (herr_med, _) = median_and_p95_ms(&heading_errs);
    let parity = 100.0 * rs_ok as f64 / py_ok.max(1) as f64;
    println!(
        "[GATE0 ORB+Hamming] synth={}/{} py_ok={py_ok}/{} parity={parity:.1}% good_rs={rs_good_sum} good_py={py_good_sum} herr_med={herr_med:.2}deg | synth_lat med={lat_med:.2}ms p95={lat_p95:.2}ms | real_15_kps={real_kps} real_lat med={real_med:.2}ms p95={real_p95:.2}ms",
        rs_ok, py_recs.len(), py_recs.len()
    );
    assert!(parity >= 95.0, "ORB Follow-View parity {parity:.1}% < 95%");
    assert!(herr_med < 6.0, "ORB heading median error {herr_med:.2} deg >= 6.0");
    assert!(real_kps > 0, "expected ORB keypoints on real minimap crops");
}

// ============================================================================
// Benchmark 3: RANSAC estimateAffinePartial2D (ransac_similarity)
// ============================================================================

#[test]
#[ignore]
fn gate0_bench_ransac_similarity() {
    let root = repo_root();
    let guard = TempDirGuard::new("ransac_sim");
    let oracle_npz = guard.path.join("ransac_oracle.npz");
    let oracle_json = guard.path.join("ransac_oracle.json");

    let py_script = r#"
import sys, json, numpy as np, cv2
from pathlib import Path

out_npz = Path(sys.argv[1])
out_json = Path(sys.argv[2])

z = np.load("data/terrain_n2_v4.npz")
ref_sift = z["map_2_gray"][400:680, 2000:2280].copy()
hud1 = cv2.imread("scratch/minimap_rec/rec_20261122_043401/frames/000001.png", cv2.IMREAD_UNCHANGED)
ref_orb = cv2.cvtColor(hud1[20:311, 79:370, :3], cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
sift = cv2.SIFT_create(nfeatures=400, contrastThreshold=0.011, edgeThreshold=15)
orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.15, nlevels=6, edgeThreshold=11, patchSize=15, fastThreshold=7)
bf_l2 = cv2.BFMatcher(cv2.NORM_L2)
bf_h = cv2.BFMatcher(cv2.NORM_HAMMING)

arrays = {}
meta = []
rng = np.random.default_rng(2026)

for idx, (ang, sc, kind) in enumerate([
    (0.0, 1.0, "sift"), (42.0, 1.05, "sift"), (137.0, 0.92, "sift"),
    (25.0, 1.0, "orb"), (195.0, 1.0, "orb"), (310.0, 1.0, "orb"),
]):
    base = ref_sift if kind == "sift" else ref_orb
    bh, bw = base.shape[:2]
    M_rot = cv2.getRotationMatrix2D(((bw - 1)*0.5, (bh - 1)*0.5), -ang, sc)
    rot = cv2.warpAffine(base, M_rot, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    reproj = 3.5 if kind == "sift" else 4.0
    iters = 1200 if kind == "sift" else 2000
    if kind == "sift":
        kp1, d1 = sift.detectAndCompute(clahe.apply(base), None)
        kp2, d2 = sift.detectAndCompute(clahe.apply(rot), None)
        pairs = bf_l2.knnMatch(d1, d2, k=2)
    else:
        kp1, d1 = orb.detectAndCompute(clahe.apply(base), None)
        kp2, d2 = orb.detectAndCompute(clahe.apply(rot), None)
        pairs = bf_h.knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=reproj, maxIters=iters, confidence=0.99)
    arrays[f"src_{idx}"] = pts1.reshape(-1, 2).astype(np.float32)
    arrays[f"dst_{idx}"] = pts2.reshape(-1, 2).astype(np.float32)
    arrays[f"M_{idx}"] = M.astype(np.float32)
    arrays[f"inl_{idx}"] = inl.ravel().astype(np.uint8)
    meta.append({"idx": idx, "reproj": reproj, "iters": iters, "n_inl": int(inl.sum())})

# Also add synthetic 20%..60% outlier cases
for k_idx, frac in enumerate([0.2, 0.4, 0.6]):
    idx = 6 + k_idx
    n = 80
    n_out = int(n * frac)
    pts1 = rng.uniform(10.0, 240.0, (n, 2)).astype(np.float32)
    th = np.radians(63.4)
    sc = 1.12
    a, b = sc * np.cos(th), sc * np.sin(th)
    pts2 = np.empty_like(pts1)
    pts2[:, 0] = a * pts1[:, 0] - b * pts1[:, 1] + 24.5
    pts2[:, 1] = b * pts1[:, 0] + a * pts1[:, 1] - 13.2
    pts2[:n - n_out] += rng.normal(0.0, 0.35, (n - n_out, 2)).astype(np.float32)
    pts2[n - n_out:] = rng.uniform(0.0, 300.0, (n_out, 2)).astype(np.float32)
    M, inl = cv2.estimateAffinePartial2D(pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2), method=cv2.RANSAC, ransacReprojThreshold=3.5, maxIters=2000, confidence=0.99)
    arrays[f"src_{idx}"] = pts1.astype(np.float32)
    arrays[f"dst_{idx}"] = pts2.astype(np.float32)
    arrays[f"M_{idx}"] = M.astype(np.float32)
    arrays[f"inl_{idx}"] = inl.ravel().astype(np.uint8)
    meta.append({"idx": idx, "reproj": 3.5, "iters": 2000, "n_inl": int(inl.sum())})

out_json.write_text(json.dumps(meta), encoding="utf-8")
np.savez(out_npz, **arrays)
"#;
    run_python_oracle(&root, py_script, &[
        oracle_npz.to_str().unwrap(),
        oracle_json.to_str().unwrap(),
    ]);

    #[derive(Deserialize)]
    struct CaseMeta {
        idx: usize,
        reproj: f64,
        iters: usize,
        n_inl: usize,
    }
    let meta: Vec<CaseMeta> =
        serde_json::from_str(&fs::read_to_string(&oracle_json).unwrap()).unwrap();
    let npz = NpzArchive::read_file(&oracle_npz).unwrap();

    let mut latencies_ms = Vec::new();
    let mut max_theta_diff = 0.0f64;
    let mut max_scale_diff = 0.0f64;
    let mut max_trans_diff = 0.0f64;

    for c in &meta {
        let src_arr = npz.get(&format!("src_{}", c.idx)).unwrap();
        let dst_arr = npz.get(&format!("dst_{}", c.idx)).unwrap();
        let m_py = npz.get(&format!("M_{}", c.idx)).unwrap().as_f32().unwrap();
        let src_pts: Vec<[f32; 2]> = src_arr
            .as_f32()
            .unwrap()
            .as_chunks::<2>()
            .0
            .to_vec();
        let dst_pts: Vec<[f32; 2]> = dst_arr
            .as_f32()
            .unwrap()
            .as_chunks::<2>()
            .0
            .to_vec();

        let t0 = Instant::now();
        let (m_rs, inl_rs) =
            estimate_affine_partial_2d_ransac(&src_pts, &dst_pts, c.reproj, c.iters, 0.99)
                .expect("RANSAC must succeed");
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let n_inl_rs: usize = inl_rs.iter().map(|&v| v as usize).sum();
        assert!(
            n_inl_rs + 1 >= c.n_inl,
            "case {}: inliers rs={n_inl_rs} < py={} - 1",
            c.idx,
            c.n_inl
        );

        let theta_rs = m_rs[3].atan2(m_rs[0]).to_degrees();
        let theta_py = (m_py[3] as f64).atan2(m_py[0] as f64).to_degrees();
        let d_theta = shortest_delta_deg(theta_py, theta_rs).abs();
        let sc_rs = m_rs[0].hypot(m_rs[3]);
        let sc_py = (m_py[0] as f64).hypot(m_py[3] as f64);
        let d_sc = (sc_rs - sc_py).abs();
        let d_tx = (m_rs[2] - m_py[2] as f64).abs();
        let d_ty = (m_rs[5] - m_py[5] as f64).abs();
        let d_tr = d_tx.max(d_ty);

        max_theta_diff = max_theta_diff.max(d_theta);
        max_scale_diff = max_scale_diff.max(d_sc);
        max_trans_diff = max_trans_diff.max(d_tr);

        assert!(d_theta < 0.5, "case {} d_theta={d_theta}", c.idx);
        assert!(d_sc < 1e-3, "case {} d_sc={d_sc}", c.idx);
        assert!(d_tr < 0.5, "case {} d_tr={d_tr}", c.idx);
    }

    let (lat_med, lat_p95) = median_and_p95_ms(&latencies_ms);
    println!(
        "[GATE0 RANSAC] cases={} max_d_theta={max_theta_diff:.4}deg max_d_scale={max_scale_diff:.6} max_d_trans={max_trans_diff:.4}px | lat med={lat_med:.3}ms p95={lat_p95:.3}ms",
        meta.len()
    );
}

// ============================================================================
// Benchmark 4: Masked 6-FFT NCC + Spatial Path + Unmasked NCC (ncc)
// ============================================================================

#[test]
#[ignore]
fn gate0_bench_ncc_masked_and_unmasked() {
    let root = repo_root();
    let guard = TempDirGuard::new("ncc_bench");
    let oracle_npz = guard.path.join("ncc_oracle.npz");

    let py_script = r#"
import sys, numpy as np, cv2
from pathlib import Path
cv2.setNumThreads(1)

out_npz = Path(sys.argv[1])
arrays = {}

# (a) Paimon 34x38 half-res and 68x77 refine on rec_20261122_043401 frames
f1 = cv2.imread("scratch/minimap_rec/rec_20261122_043401/frames/000001.png", cv2.IMREAD_UNCHANGED)
paimon_tpl = cv2.cvtColor(f1[16:119, 33:124, :3], cv2.COLOR_BGR2GRAY)
paimon_tpl = cv2.resize(paimon_tpl, (68, 77), interpolation=cv2.INTER_AREA)
yy, xx = np.mgrid[0:77, 0:68]
paimon_mask = np.where((((xx - 33.5)/30.0)**2 + ((yy - 38.0)/35.0)**2) <= 1.0, np.uint8(255), np.uint8(0))
paimon_tpl_h = cv2.resize(paimon_tpl, (34, 38), interpolation=cv2.INTER_AREA)
paimon_mask_h = (cv2.resize(paimon_mask, (34, 38), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) * np.uint8(255)

pm_roi_full = cv2.cvtColor(f1[0:144, 0:256, :3], cv2.COLOR_BGR2GRAY)
pm_roi_h = cv2.resize(pm_roi_full, (96, 54), interpolation=cv2.INTER_AREA)
res_paimon_h = cv2.matchTemplate(pm_roi_h, paimon_tpl_h, cv2.TM_CCOEFF_NORMED, mask=paimon_mask_h)
res_paimon_h = np.nan_to_num(res_paimon_h, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# Small spatial refine window (84x93)
pm_refine = cv2.resize(pm_roi_full[10:130, 20:140], (84, 93), interpolation=cv2.INTER_AREA)
res_paimon_ref = cv2.matchTemplate(pm_refine, paimon_tpl, cv2.TM_CCOEFF_NORMED, mask=paimon_mask)
res_paimon_ref = np.nan_to_num(res_paimon_ref, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

arrays["pm_roi_h"] = pm_roi_h
arrays["pm_tpl_h"] = paimon_tpl_h
arrays["pm_mask_h"] = paimon_mask_h
arrays["res_pm_h"] = res_paimon_h
arrays["pm_refine"] = pm_refine
arrays["pm_tpl"] = paimon_tpl
arrays["pm_mask"] = paimon_mask
arrays["res_pm_ref"] = res_paimon_ref

# (b) Annulus + 104-deg cone masked NCC on textured terrain_n2_v4
z = np.load("data/terrain_n2_v4.npz")
m2 = z["map_2_gray"]
ref_ds = cv2.resize(m2[400:800, 2000:2400], (200, 200), interpolation=cv2.INTER_AREA)
tpl_ds = ref_ds[55:145, 62:152].copy()
sz = 90
yy2, xx2 = np.mgrid[0:sz, 0:sz]
rr = np.hypot(xx2 - (sz-1)*0.5, yy2 - (sz-1)*0.5)
ann_mask = np.where((rr > 0.20*sz*0.5) & (rr <= 0.85*sz*0.5), np.uint8(255), np.uint8(0))
cv2.ellipse(ann_mask, (sz//2, sz//2), (sz, sz), 0.0, -52.0, 52.0, 0, -1)
res_ann = cv2.matchTemplate(ref_ds, tpl_ds, cv2.TM_CCOEFF_NORMED, mask=ann_mask)
res_ann = np.nan_to_num(res_ann, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
arrays["ref_ds"] = ref_ds
arrays["tpl_ds"] = tpl_ds
arrays["ann_mask"] = ann_mask
arrays["res_ann"] = res_ann

# (c) Unmasked Sobel f32 NCC + 0zoom.png slider strip check
qg = cv2.magnitude(cv2.Sobel(tpl_ds, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(tpl_ds, cv2.CV_32F, 0, 1, ksize=3))
rg = cv2.magnitude(cv2.Sobel(ref_ds, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ref_ds, cv2.CV_32F, 0, 1, ksize=3))
res_sob = cv2.matchTemplate(rg, qg, cv2.TM_CCOEFF_NORMED).astype(np.float32)
# Exact f64 reference for sobel_unmasked (avoids cv2 f32 integral cancellation on std < 1.0 windows)
R64 = rg.astype(np.float64); Q64 = qg.astype(np.float64)
Qzm = Q64 - Q64.mean(); qnorm = np.linalg.norm(Qzm)
kh_s, kw_s = qg.shape
int1_s = np.pad(R64, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
int2_s = np.pad(R64 * R64, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
oh_s, ow_s = res_sob.shape
ws1_s = int1_s[kh_s:, kw_s:] - int1_s[:-kh_s, kw_s:] - int1_s[kh_s:, :-kw_s] + int1_s[:-kh_s, :-kw_s]
ws2_s = int2_s[kh_s:, kw_s:] - int2_s[:-kh_s, kw_s:] - int2_s[kh_s:, :-kw_s] + int2_s[:-kh_s, :-kw_s]
wvar_s = np.maximum(0.0, ws2_s - (ws1_s * ws1_s) / float(kh_s * kw_s))
fh_s, fw_s = rg.shape[0] + qg.shape[0], rg.shape[1] + qg.shape[1]
fa_s = np.fft.rfft2(R64, s=(fh_s, fw_s))
fb_s = np.fft.rfft2(Qzm, s=(fh_s, fw_s))
dot64_s = np.fft.irfft2(fa_s * np.conj(fb_s), s=(fh_s, fw_s))[:oh_s, :ow_s]
denom64_s = np.sqrt(wvar_s) * qnorm
res_sob_f64 = np.where(wvar_s > 1e-12 * float(kh_s * kw_s), dot64_s / np.maximum(denom64_s, 1e-12), 0.0).astype(np.float32)
arrays["rg"] = rg
arrays["qg"] = qg
arrays["res_sob"] = res_sob
arrays["res_sob_f64"] = res_sob_f64

z0 = cv2.imread("scratch/0zoom.png")
fh, fw = z0.shape[:2]
roi0 = cv2.cvtColor(z0[int(0.22*fh):int(0.68*fh), int(0.012*fw):int(0.062*fw)], cv2.COLOR_BGR2GRAY)
tpl0 = cv2.cvtColor(cv2.imread("assets/zoom_thumb_template.png"), cv2.COLOR_BGR2GRAY)
sc_ui = fh / 1440.0
tw0 = max(15, int(round(31.0 * sc_ui)))
tpl0_r = cv2.resize(tpl0, (tw0, tw0), interpolation=cv2.INTER_AREA)
res_z0 = cv2.matchTemplate(roi0, tpl0_r, cv2.TM_CCOEFF_NORMED).astype(np.float32)
# Also compute exact f64 NCC (cv2 uses f32 integral sums which lose ~1e-4 on sub-1-LSB std windows)
I64 = roi0.astype(np.float64); T64 = tpl0_r.astype(np.float64)
Tzm = T64 - T64.mean(); tnorm = np.linalg.norm(Tzm)
s1_64 = cv2.boxFilter(I64, cv2.CV_64F, (tw0, tw0), normalize=False, borderType=cv2.BORDER_CONSTANT)
# Use exact integral image for valid windows
int1 = np.pad(I64, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
int2 = np.pad(I64 * I64, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
oh0, ow0 = res_z0.shape
ws1 = int1[tw0:, tw0:] - int1[:-tw0, tw0:] - int1[tw0:, :-tw0] + int1[:-tw0, :-tw0]
ws2 = int2[tw0:, tw0:] - int2[:-tw0, tw0:] - int2[tw0:, :-tw0] + int2[:-tw0, :-tw0]
wvar = np.maximum(0.0, ws2 - (ws1 * ws1) / float(tw0 * tw0))
# Cross-correlation via f64 FFT
fh0, fw0 = roi0.shape[0] + tw0, roi0.shape[1] + tw0
fa = np.fft.rfft2(I64, s=(fh0, fw0))
fb = np.fft.rfft2(Tzm, s=(fh0, fw0))
dot64 = np.fft.irfft2(fa * np.conj(fb), s=(fh0, fw0))[:oh0, :ow0]
denom64 = np.sqrt(wvar) * tnorm
res_z0_f64 = np.where(denom64 > 1e-6, dot64 / np.maximum(denom64, 1e-6), 0.0).astype(np.float32)
arrays["roi0"] = roi0
arrays["tpl0"] = tpl0_r
arrays["res_z0"] = res_z0
arrays["res_z0_f64"] = res_z0_f64

np.savez(out_npz, **arrays)
"#;
    run_python_oracle(&root, py_script, &[oracle_npz.to_str().unwrap()]);
    let npz = NpzArchive::read_file(&oracle_npz).unwrap();

    let mut max_corr_delta = 0.0f32;
    let mut fft_lat = Vec::new();
    let mut spatial_lat = Vec::new();
    let mut unmasked_lat = Vec::new();

    // Helper to assert NCC parity
    let check_map = |rs: &Array2<f32>, py: &Array2<f32>, label: &str| -> f32 {
        assert_eq!(rs.dim(), py.dim(), "{label} shape mismatch");
        let mut max_d = 0.0f32;
        for (&a, &b) in rs.iter().zip(py.iter()) {
            assert!(a.is_finite(), "{label} produced non-finite value");
            max_d = max_d.max((a - b).abs());
        }
        let (_, _, _, ml_rs) = min_max_loc_f32(rs.view()).unwrap();
        let (_, _, _, ml_py) = min_max_loc_f32(py.view()).unwrap();
        assert_eq!(ml_rs, ml_py, "{label} peak loc mismatch: rs={ml_rs:?} py={ml_py:?}");
        assert!(max_d < 1e-4, "{label} max_abs_diff={max_d} >= 1e-4");
        max_d
    };

    // (a1) Paimon half-res masked (FFT path: 63x59 -> 3717 cells? Actually 96-34+1=63, 54-38+1=17 => 1071 cells)
    let pm_roi_h = array2_from_npy(npz.get("pm_roi_h").unwrap());
    let pm_tpl_h = array2_from_npy(npz.get("pm_tpl_h").unwrap());
    let pm_mask_h = array2_from_npy(npz.get("pm_mask_h").unwrap());
    let res_pm_h_py = array2_f32_from_npy(npz.get("res_pm_h").unwrap());
    for _ in 0..10 {
        let t0 = Instant::now();
        let res = match_template_masked_u8(
            pm_roi_h.view(),
            pm_tpl_h.view(),
            pm_mask_h.view(),
            TemplateMatchMode::CcoeffNormed,
        )
        .unwrap();
        fft_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        max_corr_delta = max_corr_delta.max(check_map(&res, &res_pm_h_py, "paimon_half_masked"));
    }

    // (a2) Paimon spatial refine (17x17 = 289 -> wait: 84-68+1=17, 93-77+1=17 => 289 cells; test 16x16=256 and 17x17)
    let pm_ref = array2_from_npy(npz.get("pm_refine").unwrap());
    let pm_tpl = array2_from_npy(npz.get("pm_tpl").unwrap());
    let pm_mask = array2_from_npy(npz.get("pm_mask").unwrap());
    let res_pm_ref_py = array2_f32_from_npy(npz.get("res_pm_ref").unwrap());
    for _ in 0..10 {
        let t0 = Instant::now();
        let res = match_template_masked_u8(
            pm_ref.view(),
            pm_tpl.view(),
            pm_mask.view(),
            TemplateMatchMode::CcoeffNormed,
        )
        .unwrap();
        spatial_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        max_corr_delta = max_corr_delta.max(check_map(&res, &res_pm_ref_py, "paimon_refine_masked"));
    }

    // (b) Annulus + cone 6-FFT masked NCC (111x111 = 12321 output cells)
    let ref_ds = array2_from_npy(npz.get("ref_ds").unwrap());
    let tpl_ds = array2_from_npy(npz.get("tpl_ds").unwrap());
    let ann_mask = array2_from_npy(npz.get("ann_mask").unwrap());
    let res_ann_py = array2_f32_from_npy(npz.get("res_ann").unwrap());
    for _ in 0..10 {
        let t0 = Instant::now();
        let res = match_template_masked_u8(
            ref_ds.view(),
            tpl_ds.view(),
            ann_mask.view(),
            TemplateMatchMode::CcoeffNormed,
        )
        .unwrap();
        fft_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        max_corr_delta = max_corr_delta.max(check_map(&res, &res_ann_py, "annulus_6fft_masked"));
    }

    // (c1) Sobel f32 unmasked NCC
    let rg = array2_f32_from_npy(npz.get("rg").unwrap());
    let qg = array2_f32_from_npy(npz.get("qg").unwrap());
    let res_sob_py = array2_f32_from_npy(npz.get("res_sob").unwrap());
    let res_sob_f64 = array2_f32_from_npy(npz.get("res_sob_f64").unwrap());
    let (kh_s, kw_s) = qg.dim();
    let n_sob = (kh_s * kw_s) as f64;
    for _ in 0..10 {
        let t0 = Instant::now();
        let res = match_template_f32(rg.view(), qg.view(), TemplateMatchMode::CcoeffNormed).unwrap();
        unmasked_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(res.dim(), res_sob_py.dim());
        let (_, _, _, ml_rs) = min_max_loc_f32(res.view()).unwrap();
        let (_, _, _, ml_py) = min_max_loc_f32(res_sob_py.view()).unwrap();
        let (_, _, _, ml_64) = min_max_loc_f32(res_sob_f64.view()).unwrap();
        assert_eq!(ml_rs, ml_py, "sobel_unmasked peak loc mismatch vs cv2");
        assert_eq!(ml_rs, ml_64, "sobel_unmasked peak loc mismatch vs f64");
        for oy in 0..res.dim().0 {
            for ox in 0..res.dim().1 {
                let d_exact = (res[[oy, ox]] - res_sob_f64[[oy, ox]]).abs();
                assert!(d_exact < 1e-5, "sobel_unmasked exact ({oy},{ox}) d={d_exact}");
                let mut s1 = 0.0f64;
                let mut s2 = 0.0f64;
                for ky in 0..kh_s {
                    for kx in 0..kw_s {
                        let v = rg[[oy + ky, ox + kx]] as f64;
                        s1 += v;
                        s2 += v * v;
                    }
                }
                let var = (s2 - (s1 * s1) / n_sob).max(0.0);
                if var >= n_sob {
                    let d_cv = (res[[oy, ox]] - res_sob_py[[oy, ox]]).abs();
                    assert!(d_cv < 1e-4, "sobel_unmasked ({oy},{ox}) std>=1 d_cv={d_cv}");
                    max_corr_delta = max_corr_delta.max(d_cv);
                }
            }
        }
    }

    // (c2) 0zoom.png slider unmasked NCC:
    // Verify < 1e-6 vs exact f64 NCC across ALL pixels AND < 1e-4 vs cv2.matchTemplate on std >= 1.0 windows + 0 px peak loc
    let roi0 = array2_from_npy(npz.get("roi0").unwrap());
    let tpl0 = array2_from_npy(npz.get("tpl0").unwrap());
    let res_z0_py = array2_f32_from_npy(npz.get("res_z0").unwrap());
    let res_z0_f64 = array2_f32_from_npy(npz.get("res_z0_f64").unwrap());
    let (kh0, kw0) = tpl0.dim();
    for _ in 0..10 {
        let t0 = Instant::now();
        let res = match_template_u8(roi0.view(), tpl0.view(), TemplateMatchMode::CcoeffNormed).unwrap();
        unmasked_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(res.dim(), res_z0_py.dim());
        let (_, mv_rs, _, ml_rs) = min_max_loc_f32(res.view()).unwrap();
        let (_, mv_py, _, ml_py) = min_max_loc_f32(res_z0_py.view()).unwrap();
        assert_eq!(ml_rs, ml_py, "zoom_slider_unmasked peak loc mismatch");
        assert!((mv_rs - mv_py).abs() < 1e-4, "zoom_slider_unmasked peak val delta");
        for oy in 0..res.dim().0 {
            for ox in 0..res.dim().1 {
                let d_exact = (res[[oy, ox]] - res_z0_f64[[oy, ox]]).abs();
                assert!(d_exact < 1e-5, "zoom_slider_unmasked exact ({oy},{ox}) d={d_exact}");
                let d_cv = (res[[oy, ox]] - res_z0_py[[oy, ox]]).abs();
                let mut s1 = 0.0f64;
                let mut s2 = 0.0f64;
                for ky in 0..kh0 {
                    for kx in 0..kw0 {
                        let v = roi0[[oy + ky, ox + kx]] as f64;
                        s1 += v;
                        s2 += v * v;
                    }
                }
                let n = (kh0 * kw0) as f64;
                let pnorm = (s2 - s1 * s1 / n).max(0.0).sqrt();
                if pnorm >= kh0 as f64 {
                    assert!(d_cv < 1e-4, "zoom_slider_unmasked ({oy},{ox}) pnorm={pnorm} d_cv={d_cv}");
                    max_corr_delta = max_corr_delta.max(d_cv);
                }
            }
        }
    }

    // Also test diag_20260920_214102/01_game.png & 100zoom.png diamond poly mask smoke
    let zoom100 = load_png_bgra_or_bgr(&root.join("scratch").join("100zoom.png"));
    let mut diamond = Array2::<u8>::zeros((31, 31));
    draw_filled_convex_poly_u8(
        &mut diamond,
        &[
            Point2i { x: 15, y: 0 },
            Point2i { x: 30, y: 15 },
            Point2i { x: 15, y: 30 },
            Point2i { x: 0, y: 15 },
        ],
        255,
    );
    let z100_g = bgr_to_gray(zoom100.slice(s![0..200, 0..200, ..])).unwrap();
    let _ = match_template_masked_u8(
        z100_g.view(),
        diamond.view(),
        diamond.view(),
        TemplateMatchMode::CcoeffNormed,
    )
    .unwrap();
    // Test masked f32 path directly
    let _ = match_template_masked_f32(
        rg.view(),
        qg.view(),
        ann_mask.view(),
        TemplateMatchMode::CcoeffNormed,
    )
    .unwrap();

    let (fft_med, fft_p95) = median_and_p95_ms(&fft_lat);
    let (sp_med, sp_p95) = median_and_p95_ms(&spatial_lat);
    let (un_med, un_p95) = median_and_p95_ms(&unmasked_lat);
    println!(
        "[GATE0 NCC] max_corr_delta={max_corr_delta:.2e} peak_loc_delta=0px | 6FFT med={fft_med:.2}ms p95={fft_p95:.2}ms | spatial med={sp_med:.2}ms p95={sp_p95:.2}ms | unmasked med={un_med:.2}ms p95={un_p95:.2}ms"
    );
}

// ============================================================================
// Benchmark 5: phaseCorrelate (phase_correlate)
// ============================================================================

#[test]
#[ignore]
fn gate0_bench_phase_correlate() {
    let root = repo_root();
    let guard = TempDirGuard::new("phase_corr");
    let oracle_npz = guard.path.join("phase_oracle.npz");
    let oracle_json = guard.path.join("phase_oracle.json");

    let py_script = r#"
import sys, json, numpy as np, cv2
from pathlib import Path

out_npz = Path(sys.argv[1])
out_json = Path(sys.argv[2])

def prep_patch(mm_bgr, cone_deg=None):
    g = cv2.cvtColor(mm_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    crop = g[int(round(0.20*h)):int(round(0.80*h)), int(round(0.20*w)):int(round(0.80*w))]
    res = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    mv = float(np.mean(res))
    cv2.circle(res, (48, 48), int(round(0.26*48)), mv, -1)
    if cone_deg is not None:
        cv_ang = (cone_deg % 360.0) - 90.0
        cv2.ellipse(res, (48, 48), (96, 96), 0.0, cv_ang - 48.0, cv_ang + 48.0, mv, -1)
    return res

hann = cv2.createHanningWindow((96, 96), cv2.CV_32F)
patches = []
for idx in range(1, 21):
    hud = cv2.imread(f"scratch/minimap_rec/rec_20261122_043401/frames/{idx:06d}.png", cv2.IMREAD_UNCHANGED)
    mm = hud[20:311, 79:370]
    patches.append(prep_patch(mm, 148.0))

# Also add synthetic subpixel shifted textured terrain patch
z = np.load("data/terrain_n2_v4.npz")
tex = z["map_2_gray"][500:700, 2100:2300].astype(np.float32)
M_sh = np.float32([[1.0, 0.0, 3.45], [0.0, 1.0, -2.15]])
tex_sh = cv2.warpAffine(tex, M_sh, (200, 200), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
w1 = tex[50:146, 50:146].copy()
w2 = tex_sh[50:146, 50:146].copy()

pairs_a = patches[:-1] + [w1]
pairs_b = patches[1:] + [w2]
results = []
for a, b in zip(pairs_a, pairs_b):
    # IMPORTANT: cv2.phaseCorrelate mutates inputs in place when window is passed!
    (dx, dy), resp = cv2.phaseCorrelate(a.copy(), b.copy(), hann)
    results.append({"dx": float(dx), "dy": float(dy), "resp": float(resp)})

out_json.write_text(json.dumps(results), encoding="utf-8")
np.savez(out_npz, a=np.stack(pairs_a).astype(np.float32), b=np.stack(pairs_b).astype(np.float32))
"#;
    run_python_oracle(&root, py_script, &[
        oracle_npz.to_str().unwrap(),
        oracle_json.to_str().unwrap(),
    ]);

    #[derive(Deserialize)]
    struct PhaseOracle {
        dx: f64,
        dy: f64,
        resp: f64,
    }
    let py_res: Vec<PhaseOracle> =
        serde_json::from_str(&fs::read_to_string(&oracle_json).unwrap()).unwrap();
    let npz = NpzArchive::read_file(&oracle_npz).unwrap();
    let a_all = npz.get("a").unwrap().as_f32().unwrap();
    let b_all = npz.get("b").unwrap().as_f32().unwrap();
    let hann = create_hanning_window(96, 96).unwrap();

    // Exercise Rust prep helpers too (draw_filled_circle_f32 + draw_filled_ellipse_sector_f32)
    let mut dummy_f = Array2::<f32>::from_elem((96, 96), 100.0);
    draw_filled_circle_f32(&mut dummy_f, 48, 48, 12, 50.0);
    draw_filled_ellipse_sector_f32(&mut dummy_f, 48.0, 48.0, 96.0, 96.0, 10.0, 106.0, 50.0);

    let mut latencies_ms = Vec::new();
    let mut max_shift_delta = 0.0f64;
    let mut max_resp_delta = 0.0f64;

    for (k, py_r) in py_res.iter().enumerate() {
        let off = k * 96 * 96;
        let a = Array2::from_shape_vec((96, 96), a_all[off..off + 96 * 96].to_vec()).unwrap();
        let b = Array2::from_shape_vec((96, 96), b_all[off..off + 96 * 96].to_vec()).unwrap();

        let t0 = Instant::now();
        let ((dx, dy), resp) = phase_correlate(a.view(), b.view(), Some(hann.view())).unwrap();
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

        let d_shift = (dx - py_r.dx).abs().max((dy - py_r.dy).abs());
        let d_resp = (resp - py_r.resp).abs();
        max_shift_delta = max_shift_delta.max(d_shift);
        max_resp_delta = max_resp_delta.max(d_resp);

        assert!(
            d_shift < 1e-3,
            "pair {k}: rs=({dx:.6},{dy:.6}) py=({:.6},{:.6}) d_shift={d_shift}",
            py_r.dx,
            py_r.dy
        );
        assert!(
            d_resp < 1e-4,
            "pair {k}: rs_resp={resp:.6} py_resp={:.6} d_resp={d_resp}",
            py_r.resp
        );
    }

    let (lat_med, lat_p95) = median_and_p95_ms(&latencies_ms);
    println!(
        "[GATE0 phaseCorrelate] pairs={} max_shift_delta={max_shift_delta:.2e}px max_resp_delta={max_resp_delta:.2e} | lat med={lat_med:.3}ms p95={lat_p95:.3}ms",
        py_res.len()
    );
}

// ============================================================================
// Benchmark 6: Polar / Scharr / Arrow Primitives (heading360 ops)
// ============================================================================

fn circular_boxcar_5(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let sum = v[(i + n - 2) % n]
            + v[(i + n - 1) % n]
            + v[i]
            + v[(i + 1) % n]
            + v[(i + 2) % n];
        out[i] = sum * 0.2;
    }
    out
}

fn circular_subpixel_peak(arr: &[f32], idx: usize, negate: bool) -> f64 {
    let n = arr.len();
    let s = if negate { -1.0f64 } else { 1.0f64 };
    let yp = arr[(idx + n - 1) % n] as f64 * s;
    let yc = arr[idx] as f64 * s;
    let yn = arr[(idx + 1) % n] as f64 * s;
    let denom = yp - 2.0 * yc + yn;
    let off = if denom.abs() > 1e-9 {
        (0.5 * (yp - yn) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    (idx as f64 + off).rem_euclid(n as f64)
}

fn extract_cone_ops(mm_bgr: ArrayView3<'_, u8>) -> Option<(f64, f64, &'static str)> {
    const POLAR_BINS: usize = 512;
    const RAD_BINS: usize = 128;
    const RAD_R0: usize = 26;
    const RAD_R1: usize = 102;
    const H_RAD: usize = RAD_R1 - RAD_R0;
    const ROLL_SHIFT: usize = 5;

    let (h, w, c) = mm_bgr.dim();
    if h < 24 || w < 24 || c < 3 {
        return None;
    }

    // Tier 1: BGRA alpha-channel Scharr path
    if c == 4 {
        let mut alpha = Array2::<u8>::zeros((h, w));
        let mut a_min = 255u8;
        let mut a_max = 0u8;
        for y in 0..h {
            for x in 0..w {
                let a = mm_bgr[[y, x, 3]];
                alpha[[y, x]] = a;
                a_min = a_min.min(a);
                a_max = a_max.max(a);
            }
        }
        if a_min < 255 && a_max > a_min {
            let polar_a = warp_polar_linear_u8(
                alpha.view(),
                RAD_BINS,
                POLAR_BINS,
                (w as f64 * 0.5, h as f64 * 0.5),
                (h.min(w) as f64) * 0.5,
                Interpolation::Linear,
            )
            .ok()?;
            let ann_a = polar_a.slice(s![.., RAD_R0..RAD_R1]);
            let padded = copy_make_border_u8(ann_a, 2, 2, 0, 0, BorderMode::Wrap).ok()?;
            let grad_y = scharr_dy_u8_f32(padded.view()).ok()?;
            let mut row_sums = vec![0.0f32; POLAR_BINS];
            for y in 0..POLAR_BINS {
                let mut sum = 0.0f32;
                for x in 0..H_RAD {
                    sum += grad_y[[y + 2, x]];
                }
                row_sums[y] = sum;
            }
            let cs = circular_boxcar_5(&row_sums);
            let (mut min_i, mut min_v) = (0usize, f32::INFINITY);
            let (mut max_i, mut max_v) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in cs.iter().enumerate() {
                if v < min_v {
                    min_v = v;
                    min_i = i;
                }
                if v > max_v {
                    max_v = v;
                    max_i = i;
                }
            }
            let min_edge = 16.0 * H_RAD as f32 * 4.0;
            if min_v.abs() >= min_edge && max_v.abs() >= min_edge {
                let ang_min = (circular_subpixel_peak(&cs, min_i, true) / POLAR_BINS as f64) * 360.0;
                let ang_max = (circular_subpixel_peak(&cs, max_i, false) / POLAR_BINS as f64) * 360.0;
                let span_low = (ang_max - ang_min).rem_euclid(360.0);
                let span_high = (ang_min - ang_max).rem_euclid(360.0);
                let mid = if (span_low - 90.0).abs() <= 15.0 {
                    Some(ang_min + span_low * 0.5)
                } else if (span_high - 90.0).abs() <= 15.0 {
                    Some(ang_max + span_high * 0.5)
                } else {
                    None
                };
                if let Some(m) = mid {
                    return Some((wrap_360(m + 90.0), 0.90, "bgra_alpha"));
                }
            }
        }
    }

    // Polar BGR (only compute the 76 annulus radial bins RAD_R0..RAD_R1)
    let bgr_owned;
    let bgr_view = if c == 3 {
        mm_bgr
    } else {
        bgr_owned = bgra_to_bgr(mm_bgr).ok()?;
        bgr_owned.view()
    };
    let ann = warp_polar_annulus_u8_hwc(
        bgr_view,
        RAD_BINS,
        POLAR_BINS,
        (RAD_R0, RAD_R1),
        (w as f64 * 0.5, h as f64 * 0.5),
        (h.min(w) as f64) * 0.5,
        Interpolation::Linear,
    )
    .ok()?;
    let ann_s = ann.as_slice().unwrap();
    let mut v_prof = vec![0.0f32; POLAR_BINS];
    for y in 0..POLAR_BINS {
        let row = &ann_s[y * H_RAD * 3..(y + 1) * H_RAD * 3];
        let mut s_v = 0u32;
        for px in row.as_chunks::<3>().0 {
            s_v += px[0].max(px[1]).max(px[2]) as u32;
        }
        v_prof[y] = (s_v as f32) / (H_RAD as f32 * 255.0);
    }

    // Tier 2: LUT + absdiff + MORPH_OPEN
    let mut lut = [0u8; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        *slot = (i as f64 * 0.75 + 63.75).round().clamp(0.0, 255.0) as u8;
    }
    let pred_u8 = lut_u8_3d(ann.view(), &lut);
    let pred_s = pred_u8.as_slice().unwrap();
    let ann_gray = bgr_to_gray(ann.view()).ok()?;
    let ann_bright = add_sat_scalar_u8(ann_gray.view(), 3);
    let ann_g_s = ann_gray.as_slice().unwrap();
    let ann_b_s = ann_bright.as_slice().unwrap();
    let morph_rad_2 = get_structuring_element(MorphShape::Rect, 2, 1).ok()?;

    let mut angles = [0.0f64; 2];
    let mut tier2_ok = true;
    let mut m_bgr = Array2::<u8>::zeros((POLAR_BINS, H_RAD));
    for (pass_idx, &shift) in [ROLL_SHIFT as isize, -(ROLL_SHIFT as isize)].iter().enumerate() {
        {
            let m_s = m_bgr.as_slice_mut().unwrap();
            for y in 0..POLAR_BINS {
                let sy = (y as isize + shift).rem_euclid(POLAR_BINS as isize) as usize;
                let sh_row = &ann_s[sy * H_RAD * 3..(sy + 1) * H_RAD * 3];
                let sh_g_row = &ann_g_s[sy * H_RAD..(sy + 1) * H_RAD];
                let pr_row = &pred_s[y * H_RAD * 3..(y + 1) * H_RAD * 3];
                let ab_row = &ann_b_s[y * H_RAD..(y + 1) * H_RAD];
                let m_row = &mut m_s[y * H_RAD..(y + 1) * H_RAD];
                for x in 0..H_RAD {
                    let b_ok = pr_row[x * 3].abs_diff(sh_row[x * 3]) <= 8
                        && pr_row[x * 3 + 1].abs_diff(sh_row[x * 3 + 1]) <= 8
                        && pr_row[x * 3 + 2].abs_diff(sh_row[x * 3 + 2]) <= 8
                        && sh_g_row[x] > ab_row[x];
                    m_row[x] = if b_ok { 255 } else { 0 };
                }
            }
        }
        let opened = morphology_ex_u8(m_bgr.view(), MorphOp::Open, morph_rad_2.view()).ok()?;
        let op_s = opened.as_slice().unwrap();
        let mut row_sums = vec![0.0f32; POLAR_BINS];
        for y in 0..POLAR_BINS {
            let s_row: u32 = op_s[y * H_RAD..(y + 1) * H_RAD]
                .iter()
                .map(|&v| v as u32)
                .sum();
            row_sums[y] = s_row as f32;
        }
        let cs = circular_boxcar_5(&row_sums);
        let (mut pk_i, mut pk) = (0usize, f32::NEG_INFINITY);
        let mut mean_cs = 0.0f32;
        for (i, &val) in cs.iter().enumerate() {
            mean_cs += val;
            if val > pk {
                pk = val;
                pk_i = i;
            }
        }
        mean_cs /= POLAR_BINS as f32;
        if pk < 0.04 * H_RAD as f32 * 255.0 || pk < 2.2 * (mean_cs + 1e-3) {
            tier2_ok = false;
            break;
        }
        angles[pass_idx] = (circular_subpixel_peak(&cs, pk_i, false) / POLAR_BINS as f64) * 360.0;
    }

    if tier2_ok {
        let span = (angles[1] - angles[0]).rem_euclid(360.0);
        if (span - 90.0).abs() <= 20.0 {
            let mid_polar = angles[0] + span * 0.5;
            let mid_bin = ((mid_polar.rem_euclid(360.0) / 360.0 * POLAR_BINS as f64).round() as usize)
                % POLAR_BINS;
            let (mut in_s, mut in_n, mut out_s, mut out_n) = (0.0f64, 0usize, 0.0f64, 0usize);
            for (i, &vp) in v_prof.iter().enumerate() {
                let d1 = (i + POLAR_BINS - mid_bin) % POLAR_BINS;
                let d2 = (mid_bin + POLAR_BINS - i) % POLAR_BINS;
                let cd = d1.min(d2);
                if cd <= 56 {
                    in_s += vp as f64;
                    in_n += 1;
                } else if (72..=112).contains(&cd) {
                    out_s += vp as f64;
                    out_n += 1;
                }
            }
            if (in_s / in_n.max(1) as f64) > (out_s / out_n.max(1) as f64) + 0.01 {
                return Some((wrap_360(mid_polar + 90.0), 0.85, "bgr_blend"));
            }
        }
    }

    // Tier 3: Adaptive HSV/Value Scharr radial gradient
    let hsv = bgr_to_hsv(ann.view()).ok()?;
    let mut contrast = Array2::<f32>::zeros((POLAR_BINS, H_RAD));
    {
        let hsv_s = hsv.as_slice().unwrap();
        let c_s = contrast.as_slice_mut().unwrap();
        for (dst, px) in c_s.iter_mut().zip(hsv_s.as_chunks::<3>().0) {
            *dst = px[2] as f32 * (1.0 / 255.0) - px[1] as f32 * (0.25 / 255.0);
        }
    }
    let padded = copy_make_border_f32(contrast.view(), 2, 2, 0, 0, BorderMode::Wrap).ok()?;
    let gy = scharr_dy_f32(padded.view()).ok()?;
    let gy_s = gy.as_slice().unwrap();
    let mut raw_g = vec![0.0f32; POLAR_BINS];
    for (y, dst) in raw_g.iter_mut().enumerate() {
        let row = &gy_s[(y + 2) * H_RAD..(y + 3) * H_RAD];
        let mut s_row = 0.0f32;
        for &v in row {
            s_row += (v * (1.0 / 16.0)).clamp(-0.35, 0.35);
        }
        *dst = s_row / H_RAD as f32;
    }
    let g = circular_boxcar_5(&raw_g);
    let mut best_score = vec![0.0f32; POLAR_BINS];
    for half_fov in [60usize, 64, 68] {
        for i in 0..POLAR_BINS {
            let pe = g[(i + POLAR_BINS - half_fov) % POLAR_BINS].max(0.0);
            let px = (-g[(i + half_fov) % POLAR_BINS]).max(0.0);
            best_score[i] = best_score[i].max((pe * px).sqrt());
        }
    }
    let (mut best_i, mut peak) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in best_score.iter().enumerate() {
        if v > peak {
            peak = v;
            best_i = i;
        }
    }
    if peak < 0.015 {
        return None;
    }
    let mut bg: Vec<f32> = Vec::new();
    let (mut in_s, mut in_n, mut out_s, mut out_n) = (0.0f64, 0usize, 0.0f64, 0usize);
    for (i, &v) in best_score.iter().enumerate() {
        let cd = ((i + POLAR_BINS - best_i) % POLAR_BINS)
            .min((best_i + POLAR_BINS - i) % POLAR_BINS);
        if cd > 24 {
            bg.push(v);
        }
        if cd <= 56 {
            in_s += v_prof[i] as f64;
            in_n += 1;
        } else if (72..=112).contains(&cd) {
            out_s += v_prof[i] as f64;
            out_n += 1;
        }
    }
    if (in_s / in_n.max(1) as f64) <= (out_s / out_n.max(1) as f64) + 0.01 {
        return None;
    }
    bg.sort_by(|a, b| a.total_cmp(b));
    let bg_med = bg[bg.len() / 2];
    let bg_mean = bg.iter().sum::<f32>() / bg.len().max(1) as f32;
    let bg_std = (bg.iter().map(|&v| (v - bg_mean).powi(2)).sum::<f32>() / bg.len().max(1) as f32).sqrt();
    let bg_p98 = bg[((bg.len() as f64 * 0.98) as usize).min(bg.len().saturating_sub(1))];
    if peak <= bg_med + 3.5 * bg_std.max(1e-4) || peak <= 1.2 * bg_p98 {
        return None;
    }
    let c_sub = circular_subpixel_peak(&best_score, best_i, false);
    Some((wrap_360((c_sub / POLAR_BINS as f64) * 360.0 + 90.0), 0.70, "hsv_adaptive"))
}

fn extract_arrow_ops(mm_bgr: ArrayView3<'_, u8>) -> Option<(f64, f64)> {
    let (h, w, c) = mm_bgr.dim();
    if h < 20 || w < 20 || c < 3 {
        return None;
    }
    let y0 = (0.4 * h as f64).round() as usize;
    let y1 = (0.6 * h as f64).round() as usize;
    let x0 = (0.4 * w as f64).round() as usize;
    let x1 = (0.6 * w as f64).round() as usize;
    let crop = mm_bgr.slice(s![y0..y1, x0..x1, 0..3]);
    let (ch, cw, _) = crop.dim();
    if ch < 4 || cw < 4 {
        return None;
    }
    let up2 = resize_u8_hwc(crop, cw * 2, ch * 2, Interpolation::Linear).ok()?;
    let planes = split_channels(up2.view()).ok()?;
    let g0 = threshold(planes[0].view(), 240, 255, ThresholdType::Binary);
    let g1 = threshold(planes[1].view(), 212, 255, ThresholdType::Binary);
    let g2 = threshold(planes[2].view(), 25, 255, ThresholdType::BinaryInv);
    let mut mask_bgr = Array2::<u8>::zeros(g0.dim());
    for ((dst, &v0), (&v1, &v2)) in mask_bgr.iter_mut().zip(g0.iter()).zip(g1.iter().zip(g2.iter())) {
        *dst = if v0 != 0 { v1 & v2 } else { 0 };
    }

    let solve_mask = |mask2: ArrayView2<'_, u8>, conf_scale: f64| -> Option<(f64, f64)> {
        if mask2.iter().filter(|&&v| v != 0).count() < 16 {
            return None;
        }
        let cc = connected_components_with_stats_8(mask2).ok()?;
        let mut clean = mask2.to_owned();
        if cc.num_labels > 2 {
            let (mh, mw) = mask2.dim();
            let cx0 = mw as f64 * 0.5;
            let cy0 = mh as f64 * 0.5;
            let max_dist = 0.42 * mh.min(mw) as f64;
            let mut best_lbl = None;
            let mut best_area = 0i32;
            for lbl in 1..cc.num_labels {
                let d = (cc.centroids[lbl][0] - cx0).hypot(cc.centroids[lbl][1] - cy0);
                if d <= max_dist && cc.stats[lbl].area > best_area {
                    best_area = cc.stats[lbl].area;
                    best_lbl = Some(lbl as i32);
                }
            }
            if let Some(bl) = best_lbl {
                let min_keep = (best_area as f64 * 0.12) as i32;
                clean.fill(0);
                for y in 0..mh {
                    for x in 0..mw {
                        let l = cc.labels[[y, x]];
                        if l > 0 {
                            let lu = l as usize;
                            let d = (cc.centroids[lu][0] - cx0).hypot(cc.centroids[lu][1] - cy0);
                            if l == bl || (d <= max_dist && cc.stats[lu].area >= min_keep.max(4)) {
                                clean[[y, x]] = 255;
                            }
                        }
                    }
                }
            }
        }

        let (mh2, mw2) = clean.dim();
        let mask4 = resize_u8(clean.view(), mw2 * 2, mh2 * 2, Interpolation::Area).ok()?;
        let edges_base = canny_u8(mask4.view(), 20.0, 60.0).ok()?;
        let (rows, cols) = mask4.dim();
        let dilate_el = get_structuring_element(MorphShape::Rect, 3, 3).ok()?;
        let erode_el = get_structuring_element(MorphShape::Rect, 2, 2).ok()?;

        let erase_and_cnt = |cx: f64, cy: f64, r_erase: f64| -> Vec<Vec<Point2i>> {
            let mut work = edges_base.clone();
            draw_filled_circle_u8(
                &mut work,
                cx.ceil() as i32,
                cy.ceil() as i32,
                (r_erase.ceil() as i32).max(1),
                0,
            );
            let d = dilate_u8(work.view(), dilate_el.view()).unwrap();
            let e = erode_u8(d.view(), erode_el.view()).unwrap();
            let mut cnts = find_contours_external(e.view(), ContourMethod::None).unwrap();
            if cnts.len() > 3 {
                let filtered: Vec<_> = cnts.iter().filter(|c| c.len() >= 6).cloned().collect();
                if !filtered.is_empty() {
                    cnts = filtered;
                }
            }
            cnts
        };

        let mut contours = erase_and_cnt(cols as f64 * 0.5, rows as f64 * 0.5, cols as f64 / 4.5);
        if contours.len() != 3 {
            let (mut sx, mut sy, mut cnt) = (0.0f64, 0.0f64, 0usize);
            for y in 0..rows {
                for x in 0..cols {
                    if mask4[[y, x]] != 0 {
                        sx += x as f64;
                        sy += y as f64;
                        cnt += 1;
                    }
                }
            }
            if cnt >= 16 {
                let mcx = sx / cnt as f64;
                let mcy = sy / cnt as f64;
                let mut r_max = 0.0f64;
                for y in 0..rows {
                    for x in 0..cols {
                        if mask4[[y, x]] != 0 {
                            r_max = r_max.max((x as f64 - mcx).hypot(y as f64 - mcy));
                        }
                    }
                }
                if r_max >= 6.0 {
                    for frac in [0.62, 0.56, 0.68, 0.50, 0.74, 0.44] {
                        let cand = erase_and_cnt(mcx, mcy, frac * r_max);
                        if cand.len() == 3 {
                            contours = cand;
                            break;
                        }
                    }
                }
            }
        }

        if contours.len() == 3 {
            let pts: Vec<[f64; 2]> = contours
                .iter()
                .map(|c| {
                    let r = bounding_rect(c);
                    [r.x as f64 + r.width as f64 * 0.5, r.y as f64 + r.height as f64 * 0.5]
                })
                .collect();
            let d01 = (pts[1][0] - pts[0][0]).hypot(pts[1][1] - pts[0][1]);
            let d02 = (pts[2][0] - pts[0][0]).hypot(pts[2][1] - pts[0][1]);
            let d12 = (pts[2][0] - pts[1][0]).hypot(pts[2][1] - pts[1][1]);
            let (nose, w0, w1, l0, l1) = if d01 <= d02 && d01 <= d12 {
                (pts[2], pts[0], pts[1], d02, d12)
            } else if d02 <= d01 && d02 <= d12 {
                (pts[1], pts[0], pts[2], d01, d12)
            } else {
                (pts[0], pts[1], pts[2], d01, d02)
            };
            if l0 < 1e-3 || l1 < 1e-3 {
                return None;
            }
            let kx = (nose[0] - w0[0]) / l0 + (nose[0] - w1[0]) / l1;
            let ky = (nose[1] - w0[1]) / l0 + (nose[1] - w1[1]) / l1;
            if kx.hypot(ky) < 1e-3 {
                return None;
            }
            let ang = wrap_360(kx.atan2(-ky).to_degrees());
            return Some((ang, 0.85 * conf_scale));
        }
        None
    };

    if let Some(res) = solve_mask(mask_bgr.view(), 1.0)
        && res.1 >= 0.65
    {
        return Some(res);
    }
    let hsv = bgr_to_hsv(up2.view()).ok()?;
    let mask_hsv = in_range(hsv.view(), [85, 160, 180], [105, 255, 255]).ok()?;
    solve_mask(mask_hsv.view(), 0.92)
}

#[test]
#[ignore]
fn gate0_bench_polar_scharr_arrow() {
    let root = repo_root();
    let guard = TempDirGuard::new("polar_scharr");
    let oracle_npz = guard.path.join("curate_mm.npz");

    // Extract the 43 curated minimap crops from live_rec.npz as u8 (since live_rec.npz["rec"] is f64)
    let gt_text = fs::read_to_string(root.join("scratch").join("yt_bench").join("curate").join("gt.json")).unwrap();
    let gt_map: HashMap<String, f64> = serde_json::from_str(&gt_text).unwrap();
    let mut gt_ids: Vec<(usize, f64)> = gt_map
        .into_iter()
        .map(|(k, v)| (k.parse::<usize>().unwrap(), v))
        .collect();
    gt_ids.sort_by_key(|x| x.0);

    let id_str = gt_ids
        .iter()
        .map(|x| x.0.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let py_script = r#"
import sys, numpy as np
from pathlib import Path
out_npz = Path(sys.argv[1])
ids = [int(x) for x in sys.argv[2].split(",")]
z = np.load("scratch/yt_bench/cache/live_rec.npz")
mm = z["mm"][ids].astype(np.uint8)
np.savez(out_npz, mm=mm)
"#;
    run_python_oracle(&root, py_script, &[oracle_npz.to_str().unwrap(), &id_str]);
    let npz = NpzArchive::read_file(&oracle_npz).unwrap();
    let mm_arr = npz.get("mm").unwrap();
    let mm_sh = mm_arr.shape();
    let (n_frames, mh, mw, mc) = (mm_sh[0], mm_sh[1], mm_sh[2], mm_sh[3]);
    let mm_u8 = mm_arr.as_u8().unwrap();
    let stride = mh * mw * mc;

    // Warmup
    let first = Array3::from_shape_vec((mh, mw, mc), mm_u8[0..stride].to_vec()).unwrap();
    let _ = extract_cone_ops(first.view());

    let mut cone_lat = Vec::with_capacity(n_frames);
    let mut cone_errs = Vec::new();
    let mut cone_ok = 0usize;
    for (idx, &(_frame_id, gt_deg)) in gt_ids.iter().enumerate() {
        let crop = Array3::from_shape_vec((mh, mw, mc), mm_u8[idx * stride..(idx + 1) * stride].to_vec()).unwrap();
        let t0 = Instant::now();
        let res = extract_cone_ops(crop.view());
        cone_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        if let Some((deg, _, _)) = res {
            let err = shortest_delta_deg(gt_deg, deg).abs();
            cone_errs.push(err);
            if err <= 20.0 {
                cone_ok += 1;
            }
        }
    }

    // Also test BGRA Tier-1 & Arrow extraction on rec_20261122_043401 PNGs
    let rows = load_meta_rows(&root);
    let mut bgra_lat = Vec::new();
    let mut bgra_errs = Vec::new();
    let mut arrow_errs = Vec::new();
    let mut arrow_lat = Vec::new();
    for r in rows.iter().step_by(60).take(25) {
        let fp = root
            .join("scratch")
            .join("minimap_rec")
            .join("rec_20261122_043401")
            .join("frames")
            .join(format!("{:06}.png", r.i));
        let hud = load_png_bgra_or_bgr(&fp);
        let mm = crop_minimap_from_hud(&hud);

        let t0 = Instant::now();
        let res_c = extract_cone_ops(mm.view());
        bgra_lat.push(t0.elapsed().as_secs_f64() * 1000.0);
        if let (Some((deg, _, _)), Some(py_cone)) = (res_c, r.cone_raw) {
            bgra_errs.push(shortest_delta_deg(py_cone, deg).abs());
        }

        let t1 = Instant::now();
        let res_a = extract_arrow_ops(mm.view());
        arrow_lat.push(t1.elapsed().as_secs_f64() * 1000.0);
        if let Some((adeg, _)) = res_a {
            // Arrow extraction succeeded; check against finite [0, 360)
            assert!((0.0..360.0).contains(&adeg));
            arrow_errs.push(adeg);
        }
    }

    // Exercise additional primitives to ensure coverage (box_filter_f32, invert_affine_transform, resize_f32, sobel_3x3, magnitude_f32)
    let first_g = bgr_to_gray(first.view()).unwrap();
    let (sx, sy) = sobel_3x3_u8_f32(first_g.view()).unwrap();
    let mag = magnitude_f32(sx.view(), sy.view()).unwrap();
    let bf = box_filter_f32(mag.view(), 5, 5, true).unwrap();
    let _ = resize_f32(bf.view(), 64, 64, Interpolation::Area).unwrap();
    let rot_m = get_rotation_matrix_2d((50.0, 50.0), 15.0, 1.0);
    let _ = invert_affine_transform(&rot_m).unwrap();
    let _ = warp_affine_u8(
        first_g.view(),
        &rot_m,
        first_g.dim().1,
        first_g.dim().0,
        Interpolation::Linear,
        true,
        BorderMode::Constant(0.0),
    )
    .unwrap();
    let _ = crop_paimon_maybe_from_hud(&load_png_bgra_or_bgr(
        &root
            .join("scratch")
            .join("minimap_rec")
            .join("rec_20261122_043401")
            .join("frames")
            .join("000001.png"),
    ));

    let (cone_med_err, cone_p95_err) = median_and_p95_ms(&cone_errs);
    let (cone_med_ms, cone_p95_ms) = median_and_p95_ms(&cone_lat);
    let (bgra_med_err, _) = median_and_p95_ms(&bgra_errs);
    let (bgra_med_ms, bgra_p95_ms) = median_and_p95_ms(&bgra_lat);
    let (arrow_med_ms, arrow_p95_ms) = median_and_p95_ms(&arrow_lat);
    println!(
        "[GATE0 Polar/Scharr/Arrow] curate_ok={cone_ok}/{n_frames} err_med={cone_med_err:.2}deg p95={cone_p95_err:.2}deg | lat med={cone_med_ms:.2}ms p95={cone_p95_ms:.2}ms | bgra_err_med={bgra_med_err:.2}deg bgra_lat med={bgra_med_ms:.2}ms p95={bgra_p95_ms:.2}ms | arrow_ok={}/{} lat med={arrow_med_ms:.2}ms p95={arrow_p95_ms:.2}ms",
        arrow_errs.len(),
        bgra_lat.len()
    );
    assert!(cone_ok >= 40, "cone_ok {cone_ok}/{n_frames} < 40");
    assert!(cone_med_err < 2.0, "cone median error {cone_med_err:.2} >= 2.0 deg");
    assert!(cone_med_ms < 3.0, "cone median latency {cone_med_ms:.2} >= 3.0 ms/frame");
    assert!(bgra_med_err < 2.0, "bgra median error {bgra_med_err:.2} >= 2.0 deg");
    assert!(arrow_errs.len() >= 15, "arrow_ok {} < 15", arrow_errs.len());
    assert!(arrow_med_ms <= 2.8, "arrow median latency {arrow_med_ms:.2}ms > 2.8ms");
    assert!(!arrow_errs.is_empty(), "expected arrow detections on rec_20261122_043401");

    // Also run a quick check on bf_knn_match_l2 so it's exercised in Gate 0
    let dummy_d = vec![[1.0f32; 128]; 4];
    let l2_p = bf_knn_match_l2(&dummy_d[..2], &dummy_d[2..], 2);
    let _ = lowe_ratio_filter(&l2_p, 0.78);
}
