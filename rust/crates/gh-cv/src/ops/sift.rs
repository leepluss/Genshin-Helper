//! DoG Scale-Space SIFT (`SiftExtractor` implementing `FeatureMatcher`),
//! RootSIFT (`L1 + sqrt`) and quantized `u8` helpers.

use crate::ops::color::frame_view_to_gray_array;
use crate::ops::filters::gaussian_blur_f32;
use crate::ops::resize::resize_f32;
use crate::ops::{CvError, CvResult, FeatureMatcher, Interpolation, KeyPoint};
use gh_core::frame::FrameView;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiftParams {
    pub nfeatures: usize,
    pub n_octave_layers: usize,
    pub contrast_threshold: f64,
    pub edge_threshold: f64,
    pub sigma: f64,
}

impl Default for SiftParams {
    fn default() -> Self {
        Self {
            nfeatures: 0,
            n_octave_layers: 3,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            sigma: 1.6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SiftExtractor {
    pub params: SiftParams,
}

impl SiftExtractor {
    pub fn new(
        nfeatures: usize,
        contrast_threshold: f64,
        edge_threshold: f64,
    ) -> Self {
        Self {
            params: SiftParams {
                nfeatures,
                n_octave_layers: 3,
                contrast_threshold,
                edge_threshold,
                sigma: 1.6,
            },
        }
    }

    pub fn detect_and_compute(
        &self,
        gray: ArrayView2<'_, u8>,
        mask: Option<ArrayView2<'_, u8>>,
    ) -> CvResult<(Vec<KeyPoint>, Vec<[f32; 128]>)> {
        let (h, w) = gray.dim();
        if h < 4 || w < 4 {
            return Ok((Vec::new(), Vec::new()));
        }
        if let Some(m) = mask
            && m.dim() != (h, w)
        {
            return Err(CvError::ShapeMismatch("SIFT mask dimension mismatch".into()));
        }

        // Convert to [0, 1] float32 and build -1 octave base image (2x upscaled, sigma=1.6)
        let mut base_0 = Array2::<f32>::zeros((h, w));
        let base_s = base_0.as_slice_mut().expect("contiguous");
        if let Some(gs) = gray.as_slice() {
            for (dst, &raw) in base_s.iter_mut().zip(gs.iter()) {
                *dst = raw as f32 * (1.0 / 255.0);
            }
        } else {
            for (dst, &raw) in base_s.iter_mut().zip(gray.iter()) {
                *dst = raw as f32 * (1.0 / 255.0);
            }
        }
        let up2 = resize_f32(base_0.view(), w * 2, h * 2, Interpolation::Linear)?;
        let sig_diff = (self.params.sigma * self.params.sigma - 1.0).max(0.01).sqrt();
        let k_blur = ((sig_diff * 6.0).ceil() as usize | 1).max(3);
        let init_img = gaussian_blur_f32(up2.view(), k_blur, k_blur, sig_diff, sig_diff)?;

        let min_dim = (w * 2).min(h * 2);
        let max_octaves = ((min_dim as f64).log2().floor() as isize - 2).clamp(1, 7) as usize;
        let n_oct_layers = self.params.n_octave_layers.max(1);
        let n_gauss_layers = n_oct_layers + 3;

        // Incremental blur sigmas matching Lowe / OpenCV
        let k = 2.0f64.powf(1.0 / n_oct_layers as f64);
        let mut sigmas = vec![0.0f64; n_gauss_layers];
        sigmas[0] = self.params.sigma;
        for (i, slot) in sigmas.iter_mut().enumerate().skip(1) {
            let sig_prev = self.params.sigma * k.powi((i - 1) as i32);
            let sig_total = sig_prev * k;
            *slot = (sig_total * sig_total - sig_prev * sig_prev).max(1e-6).sqrt();
        }

        let mut init_opt = Some(init_img);
        let mut gauss_pyr: Vec<Vec<Array2<f32>>> = Vec::with_capacity(max_octaves);
        for o in 0..max_octaves {
            let mut oct = Vec::with_capacity(n_gauss_layers);
            if o == 0 {
                oct.push(init_opt.take().unwrap());
            } else {
                let prev = &gauss_pyr[o - 1][n_oct_layers];
                let (ph, pw) = prev.dim();
                let nh = ph / 2;
                let nw = pw / 2;
                if nh < 8 || nw < 8 {
                    break;
                }
                let prev_s = prev.as_slice().expect("contiguous gauss layer");
                let mut down = Array2::<f32>::zeros((nh, nw));
                let down_s = down.as_slice_mut().expect("contiguous down");
                for y in 0..nh {
                    let prow = &prev_s[(y * 2) * pw..(y * 2) * pw + pw];
                    let drow = &mut down_s[y * nw..(y + 1) * nw];
                    for (x, dst) in drow.iter_mut().enumerate() {
                        *dst = prow[x * 2];
                    }
                }
                oct.push(down);
            }
            for &sig in sigmas.iter().skip(1) {
                let ks = ((sig * 6.0).ceil() as usize | 1).max(3);
                let prev_img = oct.last().unwrap();
                let blurred = gaussian_blur_f32(prev_img.view(), ks, ks, sig, sig)?;
                oct.push(blurred);
            }
            gauss_pyr.push(oct);
        }

        // Build DoG pyramid
        let mut dog_pyr: Vec<Vec<Array2<f32>>> = Vec::with_capacity(gauss_pyr.len());
        for oct in &gauss_pyr {
            let mut dogs = Vec::with_capacity(oct.len() - 1);
            for i in 0..(oct.len() - 1) {
                let a = oct[i + 1].as_slice().expect("contiguous a");
                let b = oct[i].as_slice().expect("contiguous b");
                let mut d = Array2::<f32>::zeros(oct[i].dim());
                let ds = d.as_slice_mut().expect("contiguous d");
                for ((dst, &va), &vb) in ds.iter_mut().zip(a.iter()).zip(b.iter()) {
                    *dst = va - vb;
                }
                dogs.push(d);
            }
            dog_pyr.push(dogs);
        }

        let prelim_thresh = (0.5 * self.params.contrast_threshold / n_oct_layers as f64) as f32;
        let edge_r = self.params.edge_threshold;
        let edge_thresh_ratio = ((edge_r + 1.0) * (edge_r + 1.0)) / edge_r;
        const BORDER: usize = 5;
        const SIFT_MAX_INTERP_STEPS: usize = 5;

        #[derive(Clone, Copy)]
        struct RawCandidate {
            o: usize,
            ul: usize,
            x: usize,
            y: usize,
            off_x: f64,
            off_y: f64,
            fx_img: f32,
            fy_img: f32,
            scale_in_oct: f64,
            kpt_size: f32,
            response: f32,
        }

        let mut candidates: Vec<RawCandidate> = Vec::new();
        for (o, dogs) in dog_pyr.iter().enumerate() {
            let oct_scale = 0.5f64 * (1usize << o) as f64;
            let (oh, ow) = dogs[0].dim();
            if oh <= BORDER * 2 || ow <= BORDER * 2 {
                continue;
            }
            let dog_slices: Vec<&[f32]> = dogs
                .iter()
                .map(|d| d.as_slice().expect("contiguous dog"))
                .collect();

            for layer_init in 1..=n_oct_layers {
                let prev = dog_slices[layer_init - 1];
                let curr = dog_slices[layer_init];
                let next = dog_slices[layer_init + 1];
                for y_init in BORDER..(oh - BORDER) {
                    let row_off = y_init * ow;
                    for x_init in BORDER..(ow - BORDER) {
                        let val = curr[row_off + x_init];
                        if val.abs() <= prelim_thresh {
                            continue;
                        }
                        if !is_extremum_26(prev, curr, next, ow, y_init, x_init, val) {
                            continue;
                        }

                        // Subpixel 3D Taylor refinement
                        let mut x = x_init as isize;
                        let mut y = y_init as isize;
                        let mut layer = layer_init as isize;
                        let mut converged = false;
                        let mut offset = [0.0f64; 3];
                        for _ in 0..SIFT_MAX_INTERP_STEPS {
                            if x < BORDER as isize
                                || x >= (ow - BORDER) as isize
                                || y < BORDER as isize
                                || y >= (oh - BORDER) as isize
                                || layer < 1
                                || layer > n_oct_layers as isize
                            {
                                break;
                            }
                            let ux = x as usize;
                            let uy = y as usize;
                            let ul = layer as usize;
                            let dp = dog_slices[ul - 1];
                            let dc = dog_slices[ul];
                            let dn = dog_slices[ul + 1];
                            let c_row = uy * ow + ux;
                            let u_row = (uy - 1) * ow + ux;
                            let d_row = (uy + 1) * ow + ux;

                            let dx = 0.5 * (dc[c_row + 1] - dc[c_row - 1]) as f64;
                            let dy = 0.5 * (dc[d_row] - dc[u_row]) as f64;
                            let ds = 0.5 * (dn[c_row] - dp[c_row]) as f64;
                            let v2 = 2.0 * dc[c_row] as f64;
                            let dxx = dc[c_row + 1] as f64 + dc[c_row - 1] as f64 - v2;
                            let dyy = dc[d_row] as f64 + dc[u_row] as f64 - v2;
                            let dss = dn[c_row] as f64 + dp[c_row] as f64 - v2;
                            let dxy = 0.25
                                * (dc[d_row + 1] as f64
                                    - dc[d_row - 1] as f64
                                    - dc[u_row + 1] as f64
                                    + dc[u_row - 1] as f64);
                            let dxs = 0.25
                                * (dn[c_row + 1] as f64
                                    - dn[c_row - 1] as f64
                                    - dp[c_row + 1] as f64
                                    + dp[c_row - 1] as f64);
                            let dys = 0.25
                                * (dn[d_row] as f64
                                    - dn[u_row] as f64
                                    - dp[d_row] as f64
                                    + dp[u_row] as f64);

                            let h_mat = [
                                [dxx, dxy, dxs],
                                [dxy, dyy, dys],
                                [dxs, dys, dss],
                            ];
                            let rhs = [-dx, -dy, -ds];
                            let Some(sol) = solve_3x3(&h_mat, &rhs) else {
                                break;
                            };
                            offset = sol;
                            if offset[0].abs() < 0.5
                                && offset[1].abs() < 0.5
                                && offset[2].abs() < 0.5
                            {
                                converged = true;
                                break;
                            }
                            if !offset[0].is_finite()
                                || !offset[1].is_finite()
                                || !offset[2].is_finite()
                                || offset[0].abs() > 1e4
                                || offset[1].abs() > 1e4
                                || offset[2].abs() > 1e4
                            {
                                break;
                            }
                            x += offset[0].round() as isize;
                            y += offset[1].round() as isize;
                            layer += offset[2].round() as isize;
                        }
                        if !converged {
                            continue;
                        }
                        if x < BORDER as isize
                            || x >= (ow - BORDER) as isize
                            || y < BORDER as isize
                            || y >= (oh - BORDER) as isize
                            || layer < 1
                            || layer > n_oct_layers as isize
                        {
                            continue;
                        }
                        let ux = x as usize;
                        let uy = y as usize;
                        let ul = layer as usize;
                        let dp = dog_slices[ul - 1];
                        let dc = dog_slices[ul];
                        let dn = dog_slices[ul + 1];
                        let c_row = uy * ow + ux;
                        let u_row = (uy - 1) * ow + ux;
                        let d_row = (uy + 1) * ow + ux;

                        let dx = 0.5 * (dc[c_row + 1] - dc[c_row - 1]) as f64;
                        let dy = 0.5 * (dc[d_row] - dc[u_row]) as f64;
                        let ds = 0.5 * (dn[c_row] - dp[c_row]) as f64;
                        let contr = dc[c_row] as f64
                            + 0.5 * (dx * offset[0] + dy * offset[1] + ds * offset[2]);
                        if contr.abs() * (n_oct_layers as f64) < self.params.contrast_threshold {
                            continue;
                        }

                        // Edge response check
                        let v2 = 2.0 * dc[c_row] as f64;
                        let dxx = dc[c_row + 1] as f64 + dc[c_row - 1] as f64 - v2;
                        let dyy = dc[d_row] as f64 + dc[u_row] as f64 - v2;
                        let dxy = 0.25
                            * (dc[d_row + 1] as f64
                                - dc[d_row - 1] as f64
                                - dc[u_row + 1] as f64
                                + dc[u_row - 1] as f64);
                        let tr = dxx + dyy;
                        let det = dxx * dyy - dxy * dxy;
                        if det <= 0.0 || tr * tr >= edge_thresh_ratio * det {
                            continue;
                        }

                        let fx_img = (x as f64 + offset[0]) * oct_scale;
                        let fy_img = (y as f64 + offset[1]) * oct_scale;
                        if let Some(m) = mask {
                            let mx = (fx_img.round() as isize).clamp(0, w as isize - 1) as usize;
                            let my = (fy_img.round() as isize).clamp(0, h as isize - 1) as usize;
                            if m[[my, mx]] == 0 {
                                continue;
                            }
                        }

                        let scale_in_oct = self.params.sigma
                            * 2.0f64.powf((layer as f64 + offset[2]) / n_oct_layers as f64);
                        candidates.push(RawCandidate {
                            o,
                            ul,
                            x: ux,
                            y: uy,
                            off_x: offset[0],
                            off_y: offset[1],
                            fx_img: fx_img as f32,
                            fy_img: fy_img as f32,
                            scale_in_oct,
                            kpt_size: (scale_in_oct * oct_scale * 2.0) as f32,
                            response: contr.abs() as f32,
                        });
                    }
                }
            }
        }

        // Prune weak DoG candidates BEFORE computing expensive orientation histograms & 128-D descriptors.
        // Keep +32 cushion so any candidate yielding 0 orientations cannot shortchange nfeatures.
        if self.params.nfeatures > 0 && candidates.len() > self.params.nfeatures + 32 {
            candidates.sort_by(|a, b| {
                b.response
                    .total_cmp(&a.response)
                    .then_with(|| (a.o, a.ul, a.y, a.x).cmp(&(b.o, b.ul, b.y, b.x)))
            });
            candidates.truncate(self.params.nfeatures + 32);
        }

        let process_candidate = |cand: &RawCandidate| -> Vec<(KeyPoint, [f32; 128])> {
            let gauss_img = &gauss_pyr[cand.o][cand.ul];
            let (gh, gw) = gauss_img.dim();
            let g_s = gauss_img.as_slice().expect("contiguous gauss_img");

            let hist = calc_orientation_hist(
                g_s,
                gh,
                gw,
                cand.x,
                cand.y,
                (3.0 * 1.5 * cand.scale_in_oct).round() as isize,
                1.5 * cand.scale_in_oct,
            );
            let max_h = hist.iter().copied().fold(0.0f64, f64::max);
            if max_h <= 0.0 {
                return Vec::new();
            }
            let mag_thr = max_h * 0.8;
            let mut out = Vec::with_capacity(2);
            for b in 0..36 {
                let prev_b = (b + 35) % 36;
                let next_b = (b + 1) % 36;
                if hist[b] > hist[prev_b] && hist[b] > hist[next_b] && hist[b] >= mag_thr {
                    let denom = hist[prev_b] - 2.0 * hist[b] + hist[next_b];
                    let sub = if denom.abs() > 1e-12 {
                        0.5 * (hist[prev_b] - hist[next_b]) / denom
                    } else {
                        0.0
                    };
                    let mut bin = b as f64 + sub;
                    if bin < 0.0 {
                        bin += 36.0;
                    } else if bin >= 36.0 {
                        bin -= 36.0;
                    }
                    let angle_deg = (360.0 - bin * 10.0) % 360.0;
                    let desc = calc_sift_descriptor(
                        g_s,
                        gh,
                        gw,
                        cand.x as f64 + cand.off_x,
                        cand.y as f64 + cand.off_y,
                        angle_deg,
                        cand.scale_in_oct,
                    );
                    out.push((
                        KeyPoint {
                            x: cand.fx_img,
                            y: cand.fy_img,
                            size: cand.kpt_size,
                            angle: angle_deg as f32,
                            response: cand.response,
                            octave: cand.o as i32 - 1,
                        },
                        desc,
                    ));
                }
            }
            out
        };

        let mut pairs: Vec<(KeyPoint, [f32; 128])> = if candidates.len() >= 16 {
            candidates
                .par_iter()
                .map(process_candidate)
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect()
        } else {
            candidates.iter().flat_map(process_candidate).collect()
        };

        // Retain top nfeatures by response if requested
        if self.params.nfeatures > 0 && pairs.len() > self.params.nfeatures {
            pairs.sort_by(|a, b| b.0.response.total_cmp(&a.0.response));
            pairs.truncate(self.params.nfeatures);
        }

        let mut keypoints = Vec::with_capacity(pairs.len());
        let mut descriptors = Vec::with_capacity(pairs.len());
        for (kp, desc) in pairs {
            keypoints.push(kp);
            descriptors.push(desc);
        }
        Ok((keypoints, descriptors))
    }
}

impl FeatureMatcher for SiftExtractor {
    fn extract_keypoints_and_descriptors(&self, gray: FrameView<'_>) -> Vec<([f32; 2], Vec<f32>)> {
        let Ok(arr) = frame_view_to_gray_array(gray) else {
            return Vec::new();
        };
        let Ok((kps, des)) = self.detect_and_compute(arr.view(), None) else {
            return Vec::new();
        };
        kps.into_iter()
            .zip(des)
            .map(|(k, d)| ([k.x, k.y], d.to_vec()))
            .collect()
    }
}

#[inline]
fn is_extremum_26(
    prev: &[f32],
    curr: &[f32],
    next: &[f32],
    ow: usize,
    y: usize,
    x: usize,
    val: f32,
) -> bool {
    let r_m1 = (y - 1) * ow + x;
    let r_0 = y * ow + x;
    let r_p1 = (y + 1) * ow + x;
    if val > 0.0 {
        for r in [r_m1, r_0, r_p1] {
            for dx in [-1isize, 0, 1] {
                let idx = (r as isize + dx) as usize;
                if val <= prev[idx] || val <= next[idx] {
                    return false;
                }
                if (r != r_0 || dx != 0) && val <= curr[idx] {
                    return false;
                }
            }
        }
        true
    } else {
        for r in [r_m1, r_0, r_p1] {
            for dx in [-1isize, 0, 1] {
                let idx = (r as isize + dx) as usize;
                if val >= prev[idx] || val >= next[idx] {
                    return false;
                }
                if (r != r_0 || dx != 0) && val >= curr[idx] {
                    return false;
                }
            }
        }
        true
    }
}

fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() <= 1e-14 {
        return None;
    }
    let inv = 1.0 / det;
    let x0 = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
        * inv;
    let x1 = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
        * inv;
    let x2 = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv;
    Some([x0, x1, x2])
}

fn calc_orientation_hist(
    img: &[f32],
    h: usize,
    w: usize,
    cx: usize,
    cy: usize,
    radius: isize,
    sigma: f64,
) -> [f64; 36] {
    let mut raw = [0.0f64; 36];
    let exp_scale = -1.0 / (2.0 * sigma * sigma);
    let r = radius.clamp(1, 48);
    let r_u = r as usize;
    let mut exp_lut = [0.0f64; 49];
    for (d, slot) in exp_lut.iter_mut().enumerate().take(r_u + 1) {
        let df = d as f64;
        *slot = (df * df * exp_scale).exp();
    }

    let cy_i = cy as isize;
    let cx_i = cx as isize;
    let y_min = (cy_i - r).max(1);
    let y_max = (cy_i + r).min(h as isize - 2);
    let x_min = (cx_i - r).max(1);
    let x_max = (cx_i + r).min(w as isize - 2);

    for py in y_min..=y_max {
        let dy_abs = (py - cy_i).unsigned_abs();
        let wy = exp_lut[dy_abs];
        let uy = py as usize;
        let row_m1 = &img[(uy - 1) * w..(uy - 1) * w + w];
        let row_0 = &img[uy * w..uy * w + w];
        let row_p1 = &img[(uy + 1) * w..(uy + 1) * w + w];
        for px in x_min..=x_max {
            let ux = px as usize;
            let gx = row_0[ux + 1] - row_0[ux - 1];
            let gy = row_m1[ux] - row_p1[ux];
            let mag = (gx * gx + gy * gy).sqrt() as f64;
            let wt = wy * exp_lut[(px - cx_i).unsigned_abs()];
            let mut ori = gy.atan2(gx).to_degrees() as f64;
            if ori < 0.0 {
                ori += 360.0;
            }
            let bin = ((ori * 0.1).round() as usize) % 36;
            raw[bin] += wt * mag;
        }
    }
    let mut smoothed = [0.0f64; 36];
    for i in 0..36 {
        let im2 = (i + 34) % 36;
        let im1 = (i + 35) % 36;
        let ip1 = (i + 1) % 36;
        let ip2 = (i + 2) % 36;
        smoothed[i] =
            (raw[im2] + raw[ip2]) / 16.0 + (raw[im1] + raw[ip1]) * 4.0 / 16.0 + raw[i] * 6.0 / 16.0;
    }
    smoothed
}

fn calc_sift_descriptor(
    img: &[f32],
    h: usize,
    w: usize,
    ptx: f64,
    pty: f64,
    ori_deg: f64,
    scl: f64,
) -> [f32; 128] {
    const D: usize = 4;
    const N_ORI: usize = 8;
    let angle_rad = (360.0 - ori_deg).to_radians();
    let cos_t = angle_rad.cos();
    let sin_t = angle_rad.sin();
    let hist_width = 3.0 * scl;
    let cos_rot = (cos_t / hist_width) as f32;
    let sin_rot = (sin_t / hist_width) as f32;
    let exp_scale = (-1.0 / (0.5 * (D as f64) * (D as f64))) as f32;
    let radius = ((hist_width * std::f64::consts::SQRT_2 * (D as f64 + 1.0) * 0.5).round() as isize)
        .min((h.max(w) / 2) as isize);

    let cx = ptx.round() as isize;
    let cy = pty.round() as isize;
    let mut hist = [0.0f64; D * D * N_ORI];

    let y_min = (cy - radius).max(1);
    let y_max = (cy + radius).min(h as isize - 2);
    let x_min = (cx - radius).max(1);
    let x_max = (cx + radius).min(w as isize - 2);
    let ori_base = (360.0 - ori_deg) as f32;
    let bins_per_deg = N_ORI as f32 / 360.0;
    let d_offset = (D as f32) * 0.5 - 0.5;
    let d_limit = D as f32;

    for py in y_min..=y_max {
        let i = (py - cy) as f32;
        let uy = py as usize;
        let row_m1 = &img[(uy - 1) * w..(uy - 1) * w + w];
        let row_0 = &img[uy * w..uy * w + w];
        let row_p1 = &img[(uy + 1) * w..(uy + 1) * w + w];
        let c_base = -i * sin_rot;
        let r_base = i * cos_rot;
        for px in x_min..=x_max {
            let j = (px - cx) as f32;
            let c_rot = j * cos_rot + c_base;
            let r_rot = j * sin_rot + r_base;
            let rbin = r_rot + d_offset;
            let cbin = c_rot + d_offset;
            if rbin <= -1.0 || rbin >= d_limit || cbin <= -1.0 || cbin >= d_limit {
                continue;
            }
            let ux = px as usize;
            let gx = row_0[ux + 1] - row_0[ux - 1];
            let gy = row_m1[ux] - row_p1[ux];
            let mag = (gx * gx + gy * gy).sqrt();
            let wt = (((c_rot * c_rot + r_rot * r_rot) * exp_scale).exp() * mag) as f64;

            let mut ori = gy.atan2(gx).to_degrees();
            if ori < 0.0 {
                ori += 360.0;
            }
            let obin = ((ori - ori_base) * bins_per_deg).rem_euclid(N_ORI as f32) as f64;

            let r0 = (rbin as f64).floor() as isize;
            let c0 = (cbin as f64).floor() as isize;
            let o0 = obin.floor() as usize;
            let dr = rbin as f64 - r0 as f64;
            let dc = cbin as f64 - c0 as f64;
            let do_f = obin - o0 as f64;

            for dr_i in 0..2isize {
                let rr = r0 + dr_i;
                if !(0..4).contains(&rr) {
                    continue;
                }
                let wr = if dr_i == 0 { 1.0 - dr } else { dr };
                for dc_i in 0..2isize {
                    let cc = c0 + dc_i;
                    if !(0..4).contains(&cc) {
                        continue;
                    }
                    let wc = if dc_i == 0 { 1.0 - dc } else { dc };
                    let w_rc = wt * wr * wc;
                    let base = ((rr as usize) * D + (cc as usize)) * N_ORI;
                    hist[base + (o0 & 7)] += w_rc * (1.0 - do_f);
                    hist[base + ((o0 + 1) & 7)] += w_rc * do_f;
                }
            }
        }
    }

    // L2 normalize, clamp <= 0.2, re-normalize, scale by 512 matching OpenCV SIFT
    let norm1 = hist.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let thr = norm1 * 0.2;
    let mut norm2_sq = 0.0f64;
    for v in &mut hist {
        if *v > thr {
            *v = thr;
        }
        norm2_sq += *v * *v;
    }
    let scale = if norm2_sq > 1e-18 {
        512.0 / norm2_sq.sqrt()
    } else {
        0.0
    };
    let mut out = [0.0f32; 128];
    for (dst, &v) in out.iter_mut().zip(hist.iter()) {
        *dst = (v * scale) as f32;
    }
    out
}

/// Converts a raw SIFT descriptor to RootSIFT (`L1`-normalize then elementwise `sqrt`).
pub fn to_rootsift(des: &[f32; 128]) -> [f32; 128] {
    let l1: f32 = des.iter().map(|v| v.max(0.0)).sum::<f32>() + 1e-7;
    let inv = 1.0 / l1;
    let mut out = [0.0f32; 128];
    for (dst, &v) in out.iter_mut().zip(des.iter()) {
        *dst = (v.max(0.0) * inv).sqrt();
    }
    out
}

/// Quantizes a RootSIFT descriptor to `u8` (`round(rdes * scale_factor).clamp(0, 255)`).
pub fn quantize_rootsift_u8(rdes: &[f32; 128], scale_factor: f32) -> [u8; 128] {
    let mut out = [0u8; 128];
    for (dst, &v) in out.iter_mut().zip(rdes.iter()) {
        *dst = (v * scale_factor).round().clamp(0.0, 255.0) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sift_detects_keypoints_on_textured_patch_and_rootsift_unit_l2() {
        let mut img = Array2::<u8>::zeros((96, 96));
        for y in 0..96 {
            for x in 0..96 {
                let checker = (((x / 12) + (y / 12)) % 2) as f64;
                let wave = ((x as f64 * 0.25).sin() * (y as f64 * 0.23).cos() + 1.0) * 0.5;
                img[[y, x]] = ((checker * 140.0 + wave * 100.0).clamp(0.0, 255.0)) as u8;
            }
        }
        let extractor = SiftExtractor::new(160, 0.008, 15.0);
        let (kps, des) = extractor.detect_and_compute(img.view(), None).unwrap();
        assert!(!kps.is_empty(), "expected SIFT keypoints on checkerboard");
        assert_eq!(kps.len(), des.len());
        let rdes = to_rootsift(&des[0]);
        let l2: f32 = rdes.iter().map(|v| v * v).sum();
        assert!((l2 - 1.0).abs() < 1e-3, "RootSIFT L2 norm^2={l2}");
        let q = quantize_rootsift_u8(&rdes, 400.0);
        assert!(q.iter().any(|&v| v > 0));
    }
}
