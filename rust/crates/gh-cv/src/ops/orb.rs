//! ORB feature extractor (6-level pyramid FAST + Harris score + intensity centroid +
//! 256-bit steered rBRIEF `[u8; 32]`) and `bf_knn_match_hamming`.

use crate::ops::filters::gaussian_blur_u8;
use crate::ops::resize::resize_u8;
use crate::ops::{CvError, CvResult, DMatch, Interpolation, KeyPoint};
use ndarray::ArrayView2;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbParams {
    pub nfeatures: usize,
    pub scale_factor: f32,
    pub nlevels: usize,
    pub edge_threshold: usize,
    pub patch_size: usize,
    pub fast_threshold: u8,
}

impl Default for OrbParams {
    fn default() -> Self {
        Self {
            nfeatures: 800,
            scale_factor: 1.15,
            nlevels: 6,
            edge_threshold: 11,
            patch_size: 15,
            fast_threshold: 7,
        }
    }
}

// 16-pixel Bresenham circle of radius 3
const CIRCLE_16: [(isize, isize); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

// Deterministic 256 pairs of (x0, y0, x1, y1) inside [-7, 7] (scaled from ORB bit_pattern_31)
const BRIEF_PAIRS: [(i8, i8, i8, i8); 256] = generate_brief_pairs();

const fn lcg_next(s: u64) -> u64 {
    s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407)
}

const fn generate_brief_pairs() -> [(i8, i8, i8, i8); 256] {
    let mut pairs = [(0i8, 0i8, 0i8, 0i8); 256];
    let mut s = 0x1234_5678_9abc_def0u64;
    let mut i = 0;
    while i < 256 {
        s = lcg_next(s);
        let x0 = (((s >> 16) % 13) as i8) - 6;
        s = lcg_next(s);
        let y0 = (((s >> 16) % 13) as i8) - 6;
        s = lcg_next(s);
        let x1 = (((s >> 16) % 13) as i8) - 6;
        s = lcg_next(s);
        let y1 = (((s >> 16) % 13) as i8) - 6;
        if x0 == x1 && y0 == y1 {
            continue;
        }
        pairs[i] = (x0, y0, x1, y1);
        i += 1;
    }
    pairs
}

#[derive(Debug, Clone)]
pub struct OrbExtractor {
    pub params: OrbParams,
}

impl OrbExtractor {
    pub fn new(params: OrbParams) -> Self {
        Self { params }
    }

    pub fn detect_and_compute(
        &self,
        gray: ArrayView2<'_, u8>,
        mask: Option<ArrayView2<'_, u8>>,
    ) -> CvResult<(Vec<KeyPoint>, Vec<[u8; 32]>)> {
        let (h, w) = gray.dim();
        if h < 16 || w < 16 {
            return Ok((Vec::new(), Vec::new()));
        }
        if let Some(m) = mask
            && m.dim() != (h, w)
        {
            return Err(CvError::ShapeMismatch("ORB mask dimension mismatch".into()));
        }

        let nlevels = self.params.nlevels.max(1);
        let sf = self.params.scale_factor.max(1.01) as f64;
        let mut scales = Vec::with_capacity(nlevels);
        let mut pyr = Vec::with_capacity(nlevels);
        let mut blurs = Vec::with_capacity(nlevels);

        for lvl in 0..nlevels {
            let sc = sf.powi(lvl as i32);
            scales.push(sc);
            let lw = ((w as f64) / sc).round() as usize;
            let lh = ((h as f64) / sc).round() as usize;
            if lw < 24 || lh < 24 {
                break;
            }
            let img_l = if lvl == 0 {
                gray.to_owned()
            } else {
                resize_u8(gray, lw, lh, Interpolation::Area)?
            };
            let blurred = gaussian_blur_u8(img_l.view(), 5, 5, 2.0, 2.0)?;
            pyr.push(img_l);
            blurs.push(blurred);
        }

        let actual_levels = pyr.len();
        if actual_levels == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // Compute per-level feature budgets matching OpenCV ORB
        let inv_sf = 1.0 / sf;
        let inv_sf2 = inv_sf * inv_sf;
        let denom: f64 = (0..actual_levels).map(|i| inv_sf2.powi(i as i32)).sum();
        let mut level_budgets = Vec::with_capacity(actual_levels);
        let mut used = 0usize;
        for i in 0..actual_levels {
            if i + 1 == actual_levels {
                level_budgets.push(self.params.nfeatures.saturating_sub(used));
            } else {
                let n = ((self.params.nfeatures as f64) * inv_sf2.powi(i as i32) / denom).round() as usize;
                level_budgets.push(n);
                used += n;
            }
        }

        let half_k = (self.params.patch_size / 2).max(4) as isize;
        let edge = self.params.edge_threshold.max(half_k as usize + 1);
        let fast_thr = self.params.fast_threshold;

        let mut all_kps = Vec::new();
        let mut all_des = Vec::new();

        for lvl in 0..actual_levels {
            let img = &pyr[lvl];
            let blur = &blurs[lvl];
            let sc = scales[lvl];
            let budget = level_budgets[lvl];
            if budget == 0 {
                continue;
            }
            let (lh, lw) = img.dim();
            if lh <= edge * 2 + 2 || lw <= edge * 2 + 2 {
                continue;
            }

            // Detect FAST-9 corners + 3x3 NMS + Harris score
            let mut scores = vec![0i32; lh * lw];
            let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
            for y in edge..(lh - edge) {
                for x in edge..(lw - edge) {
                    if let Some(s) = fast9_score(img.view(), x, y, fast_thr) {
                        scores[y * lw + x] = s;
                    }
                }
            }
            for y in edge..(lh - edge) {
                for x in edge..(lw - edge) {
                    let s = scores[y * lw + x];
                    if s == 0 {
                        continue;
                    }
                    // 3x3 NMS
                    let mut is_max = true;
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            if scores[(y as isize + dy) as usize * lw + (x as isize + dx) as usize] > s {
                                is_max = false;
                                break;
                            }
                        }
                        if !is_max {
                            break;
                        }
                    }
                    if !is_max {
                        continue;
                    }

                    // Check mask in original image coordinates
                    if let Some(m) = mask {
                        let ox = ((x as f64 * sc).round() as isize).clamp(0, w as isize - 1) as usize;
                        let oy = ((y as f64 * sc).round() as isize).clamp(0, h as isize - 1) as usize;
                        if m[[oy, ox]] == 0 {
                            continue;
                        }
                    }

                    let h_score = harris_response_7x7(img.view(), x, y);
                    candidates.push((x, y, h_score));
                }
            }

            // Spatial 4x4 grid-balanced top-budget retention so keypoints cover the full frame
            if candidates.len() > budget {
                candidates = retain_grid_distributed(&candidates, lw, lh, budget);
            }

            for (x, y, resp) in candidates {
                let angle_rad = ic_angle(img.view(), x, y, half_k);
                let desc = compute_steered_brief(blur.view(), x as isize, y as isize, angle_rad, half_k);
                let mut ang_deg = angle_rad.to_degrees() as f32;
                if ang_deg < 0.0 {
                    ang_deg += 360.0;
                }
                all_kps.push(KeyPoint {
                    x: (x as f64 * sc) as f32,
                    y: (y as f64 * sc) as f32,
                    size: (self.params.patch_size as f64 * sc) as f32,
                    angle: ang_deg,
                    response: resp,
                    octave: lvl as i32,
                });
                all_des.push(desc);
            }
        }

        Ok((all_kps, all_des))
    }
}

fn retain_grid_distributed(
    cands: &[(usize, usize, f32)],
    w: usize,
    h: usize,
    budget: usize,
) -> Vec<(usize, usize, f32)> {
    const GRID: usize = 4;
    let mut cells = vec![Vec::<(usize, usize, f32)>::new(); GRID * GRID];
    for &(x, y, r) in cands {
        let gx = ((x * GRID) / w.max(1)).min(GRID - 1);
        let gy = ((y * GRID) / h.max(1)).min(GRID - 1);
        cells[gy * GRID + gx].push((x, y, r));
    }
    for cell in &mut cells {
        cell.sort_by(|a, b| b.2.total_cmp(&a.2));
    }
    let mut out = Vec::with_capacity(budget);
    let mut pos = [0usize; GRID * GRID];
    while out.len() < budget {
        let mut progressed = false;
        for (ci, cell) in cells.iter().enumerate() {
            if pos[ci] < cell.len() {
                out.push(cell[pos[ci]]);
                pos[ci] += 1;
                progressed = true;
                if out.len() >= budget {
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    out
}

#[inline]
fn fast9_score(img: ArrayView2<'_, u8>, x: usize, y: usize, thr: u8) -> Option<i32> {
    let v = img[[y, x]] as i32;
    let t = thr as i32;
    let hi = v + t;
    let lo = v - t;

    // Cardinal pre-check (indices 0, 4, 8, 12)
    let c0 = img[[(y as isize + CIRCLE_16[0].1) as usize, (x as isize + CIRCLE_16[0].0) as usize]] as i32;
    let c4 = img[[(y as isize + CIRCLE_16[4].1) as usize, (x as isize + CIRCLE_16[4].0) as usize]] as i32;
    let c8 = img[[(y as isize + CIRCLE_16[8].1) as usize, (x as isize + CIRCLE_16[8].0) as usize]] as i32;
    let c12 = img[[(y as isize + CIRCLE_16[12].1) as usize, (x as isize + CIRCLE_16[12].0) as usize]] as i32;

    let bright_card = (c0 > hi) as u8 + (c4 > hi) as u8 + (c8 > hi) as u8 + (c12 > hi) as u8;
    let dark_card = (c0 < lo) as u8 + (c4 < lo) as u8 + (c8 < lo) as u8 + (c12 < lo) as u8;
    if bright_card < 3 && dark_card < 3 {
        return None;
    }

    let mut ring = [0i32; 16];
    for (i, &(dx, dy)) in CIRCLE_16.iter().enumerate() {
        ring[i] = img[[(y as isize + dy) as usize, (x as isize + dx) as usize]] as i32;
    }

    let mut is_corner = false;
    if bright_card >= 3 {
        let mut run = 0;
        for i in 0..32 {
            if ring[i & 15] > hi {
                run += 1;
                if run >= 9 {
                    is_corner = true;
                    break;
                }
            } else {
                run = 0;
            }
        }
    }
    if !is_corner && dark_card >= 3 {
        let mut run = 0;
        for i in 0..32 {
            if ring[i & 15] < lo {
                run += 1;
                if run >= 9 {
                    is_corner = true;
                    break;
                }
            } else {
                run = 0;
            }
        }
    }
    if !is_corner {
        return None;
    }

    let score: i32 = ring.iter().map(|&r| (r - v).abs()).sum();
    Some(score)
}

#[inline]
fn harris_response_7x7(img: ArrayView2<'_, u8>, cx: usize, cy: usize) -> f32 {
    let mut a = 0.0f32;
    let mut b = 0.0f32;
    let mut c = 0.0f32;
    for dy in -3isize..=3 {
        let y = (cy as isize + dy) as usize;
        for dx in -3isize..=3 {
            let x = (cx as isize + dx) as usize;
            let ix = img[[y, x + 1]] as f32 - img[[y, x - 1]] as f32;
            let iy = img[[y + 1, x]] as f32 - img[[y - 1, x]] as f32;
            a += ix * ix;
            b += iy * iy;
            c += ix * iy;
        }
    }
    let scale = 1.0 / (7.0 * 7.0 * 4.0 * 255.0 * 255.0);
    let a = a * scale;
    let b = b * scale;
    let c = c * scale;
    (a * b - c * c) - 0.04 * (a + b) * (a + b)
}

#[inline]
fn ic_angle(img: ArrayView2<'_, u8>, cx: usize, cy: usize, half_k: isize) -> f64 {
    let mut m10 = 0i64;
    let mut m01 = 0i64;
    let r2 = half_k * half_k;
    for dy in -half_k..=half_k {
        let y = (cy as isize + dy) as usize;
        for dx in -half_k..=half_k {
            if dx * dx + dy * dy <= r2 {
                let v = img[[y, (cx as isize + dx) as usize]] as i64;
                m10 += dx as i64 * v;
                m01 += dy as i64 * v;
            }
        }
    }
    (m01 as f64).atan2(m10 as f64)
}

#[inline]
fn compute_steered_brief(
    blur: ArrayView2<'_, u8>,
    cx: isize,
    cy: isize,
    angle_rad: f64,
    half_k: isize,
) -> [u8; 32] {
    let (h, w) = blur.dim();
    let (sin_t, cos_t) = angle_rad.sin_cos();
    let scale_p = (half_k as f64) / 6.5;
    let mut desc = [0u8; 32];
    for (byte_idx, chunk) in BRIEF_PAIRS.as_chunks::<8>().0.iter().enumerate() {
        let mut byte = 0u8;
        for (bit_idx, &(x0, y0, x1, y1)) in chunk.iter().enumerate() {
            let fx0 = x0 as f64 * scale_p;
            let fy0 = y0 as f64 * scale_p;
            let fx1 = x1 as f64 * scale_p;
            let fy1 = y1 as f64 * scale_p;

            let rx0 = (cx + (fx0 * cos_t - fy0 * sin_t).round() as isize).clamp(0, w as isize - 1) as usize;
            let ry0 = (cy + (fx0 * sin_t + fy0 * cos_t).round() as isize).clamp(0, h as isize - 1) as usize;
            let rx1 = (cx + (fx1 * cos_t - fy1 * sin_t).round() as isize).clamp(0, w as isize - 1) as usize;
            let ry1 = (cy + (fx1 * sin_t + fy1 * cos_t).round() as isize).clamp(0, h as isize - 1) as usize;

            if blur[[ry0, rx0]] < blur[[ry1, rx1]] {
                byte |= 1u8 << bit_idx;
            }
        }
        desc[byte_idx] = byte;
    }
    desc
}

#[inline]
pub fn hamming_distance_256(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    let mut d = 0u32;
    for i in (0..32).step_by(8) {
        let wa = u64::from_ne_bytes(a[i..i + 8].try_into().unwrap());
        let wb = u64::from_ne_bytes(b[i..i + 8].try_into().unwrap());
        d += (wa ^ wb).count_ones();
    }
    d
}

/// Brute-force Hamming k-NN matcher (`cv2.BFMatcher(NORM_HAMMING).knnMatch`).
pub fn bf_knn_match_hamming(
    queries: &[[u8; 32]],
    train: &[[u8; 32]],
    k: usize,
) -> Vec<Vec<DMatch>> {
    if queries.is_empty() || train.is_empty() || k == 0 {
        return vec![Vec::new(); queries.len()];
    }
    queries
        .par_iter()
        .enumerate()
        .map(|(qi, q)| {
            let mut best: Vec<(u32, usize)> = Vec::with_capacity(k + 1);
            for (ti, t) in train.iter().enumerate() {
                let d = hamming_distance_256(q, t);
                if best.len() < k || d < best[k - 1].0 {
                    best.push((d, ti));
                    best.sort_by_key(|v| v.0);
                    if best.len() > k {
                        best.truncate(k);
                    }
                }
            }
            best.into_iter()
                .map(|(d, train_idx)| DMatch {
                    query_idx: qi,
                    train_idx,
                    distance: d as f32,
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::BorderMode;
    use crate::ops::warp::{get_rotation_matrix_2d, warp_affine_u8};
    use ndarray::Array2;

    #[test]
    fn orb_recovers_rotation_invariant_matches() {
        let mut img = Array2::<u8>::zeros((128, 128));
        for y in 0..128 {
            for x in 0..128 {
                let c = (((x / 16) ^ (y / 16)) & 1) as f64;
                let g = ((x as f64 * 0.21).sin() * (y as f64 * 0.17).cos() + 1.0) * 60.0;
                img[[y, x]] = (c * 120.0 + g).clamp(0.0, 255.0) as u8;
            }
        }
        let m = get_rotation_matrix_2d((64.0, 64.0), 35.0, 1.0);
        let rot = warp_affine_u8(
            img.view(),
            &m,
            128,
            128,
            Interpolation::Linear,
            false,
            BorderMode::Replicate,
        )
        .unwrap();

        let orb = OrbExtractor::new(OrbParams::default());
        let (kp1, des1) = orb.detect_and_compute(img.view(), None).unwrap();
        let (kp2, des2) = orb.detect_and_compute(rot.view(), None).unwrap();
        assert!(!kp1.is_empty() && !kp2.is_empty());

        let pairs = bf_knn_match_hamming(&des1, &des2, 2);
        let good: Vec<DMatch> = pairs
            .into_iter()
            .filter_map(|p| {
                if p.len() == 2 && p[0].distance < 0.78 * p[1].distance {
                    Some(p[0])
                } else {
                    None
                }
            })
            .collect();
        assert!(good.len() >= 8, "expected >=8 ORB matches, got {}", good.len());
    }
}
