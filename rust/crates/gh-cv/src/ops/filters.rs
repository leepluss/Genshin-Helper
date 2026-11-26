//! Spatial filters: Gaussian blur (`u8` / `f32`), O(1) separable box filter (`f32`),
//! median blur (`u8`), and CLAHE (`u8`, bilinear tile CDF interpolation).

use crate::ops::{CvError, CvResult, border_reflect_101};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use rayon::prelude::*;

/// Computes a 1D normalized Gaussian kernel matching OpenCV `getGaussianKernel`.
pub fn gaussian_kernel_1d(ksize: usize, sigma: f64) -> CvResult<Vec<f64>> {
    if ksize == 0 || ksize.is_multiple_of(2) {
        return Err(CvError::InvalidArgument(format!(
            "gaussian ksize must be odd and >= 1, got {ksize}"
        )));
    }
    if sigma <= 0.0 {
        match ksize {
            1 => return Ok(vec![1.0]),
            3 => return Ok(vec![0.25, 0.5, 0.25]),
            5 => return Ok(vec![0.0625, 0.25, 0.375, 0.25, 0.0625]),
            7 => {
                return Ok(vec![
                    0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
                ]);
            }
            9 => {
                return Ok(vec![
                    0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375, 0.19921875, 0.1171875,
                    0.05078125, 0.015625,
                ]);
            }
            _ => {}
        }
    }
    let s = if sigma <= 0.0 {
        0.3 * (((ksize - 1) as f64 * 0.5) - 1.0) + 0.8
    } else {
        sigma
    };
    let half = (ksize / 2) as isize;
    let inv_2s2 = 1.0 / (2.0 * s * s);
    let mut k = Vec::with_capacity(ksize);
    let mut sum = 0.0f64;
    for i in -half..=half {
        let v = (-((i * i) as f64) * inv_2s2).exp();
        k.push(v);
        sum += v;
    }
    let inv = 1.0 / sum;
    for v in &mut k {
        *v *= inv;
    }
    Ok(k)
}

/// Separable 2D Gaussian blur on 2D `f32` array (`BORDER_REFLECT_101`).
pub fn gaussian_blur_f32(
    src: ArrayView2<'_, f32>,
    kx: usize,
    ky: usize,
    sigma_x: f64,
    sigma_y: f64,
) -> CvResult<Array2<f32>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("gaussian_blur_f32 empty".into()));
    }
    let kern_x: Vec<f32> = gaussian_kernel_1d(kx, sigma_x)?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let kern_y: Vec<f32> = gaussian_kernel_1d(ky, if sigma_y <= 0.0 { sigma_x } else { sigma_y })?
        .into_iter()
        .map(|v| v as f32)
        .collect();
    let rx = (kx / 2) as isize;
    let ry = (ky / 2) as isize;
    let rx_u = rx as usize;
    let ry_u = ry as usize;

    let src_contig;
    let src_s = if let Some(s) = src.as_slice() {
        s
    } else {
        src_contig = src.to_owned();
        src_contig.as_slice().unwrap()
    };

    let mut tmp = vec![0.0f32; h * w];
    let right_kx = &kern_x[rx_u + 1..];
    let kc_x = kern_x[rx_u];
    let blur_horiz_row = |y: usize, trow: &mut [f32]| {
        let srow = &src_s[y * w..(y + 1) * w];
        if w > kx {
            for (x, dst) in trow.iter_mut().enumerate().take(rx_u) {
                let mut acc = 0.0f32;
                for (i, &kv) in kern_x.iter().enumerate() {
                    acc += srow[border_reflect_101(x as isize + i as isize - rx, w)] * kv;
                }
                *dst = acc;
            }
            for x in rx_u..(w - rx_u) {
                let mut acc = srow[x] * kc_x;
                for (d, &kd) in right_kx.iter().enumerate() {
                    let dist = d + 1;
                    acc += (srow[x - dist] + srow[x + dist]) * kd;
                }
                trow[x] = acc;
            }
            for x in (w - rx_u)..w {
                let mut acc = 0.0f32;
                for (i, &kv) in kern_x.iter().enumerate() {
                    acc += srow[border_reflect_101(x as isize + i as isize - rx, w)] * kv;
                }
                trow[x] = acc;
            }
        } else {
            for (x, dst) in trow.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (i, &kv) in kern_x.iter().enumerate() {
                    acc += srow[border_reflect_101(x as isize + i as isize - rx, w)] * kv;
                }
                *dst = acc;
            }
        }
    };
    if h * w >= 16_384 {
        tmp.par_chunks_mut(w).enumerate().for_each(|(y, trow)| {
            blur_horiz_row(y, trow);
        });
    } else {
        for (y, trow) in tmp.chunks_exact_mut(w).enumerate() {
            blur_horiz_row(y, trow);
        }
    }

    let mut out = Array2::<f32>::zeros((h, w));
    let out_s = out.as_slice_mut().unwrap();
    let right_ky = &kern_y[ry_u + 1..];
    let kc_y = kern_y[ry_u];
    let blur_vert_row = |y: usize, out_row: &mut [f32]| {
        if y >= ry_u && y + ry_u < h {
            let r_mid = &tmp[y * w..(y + 1) * w];
            for (dst, &v) in out_row.iter_mut().zip(r_mid.iter()) {
                *dst = v * kc_y;
            }
            for (d, &kd) in right_ky.iter().enumerate() {
                let dist = d + 1;
                let r_up = &tmp[(y - dist) * w..(y - dist + 1) * w];
                let r_dn = &tmp[(y + dist) * w..(y + dist + 1) * w];
                for ((dst, &u), &b) in out_row.iter_mut().zip(r_up.iter()).zip(r_dn.iter()) {
                    *dst += (u + b) * kd;
                }
            }
        } else {
            for (j, &kv) in kern_y.iter().enumerate() {
                let sy = border_reflect_101(y as isize + j as isize - ry, h);
                let trow = &tmp[sy * w..(sy + 1) * w];
                for (dst, &v) in out_row.iter_mut().zip(trow.iter()) {
                    *dst += v * kv;
                }
            }
        }
    };
    if h * w >= 16_384 {
        out_s.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
            blur_vert_row(y, out_row);
        });
    } else {
        for (y, out_row) in out_s.chunks_exact_mut(w).enumerate() {
            blur_vert_row(y, out_row);
        }
    }
    Ok(out)
}

/// Separable 2D Gaussian blur on 2D `u8` array (`BORDER_REFLECT_101`).
pub fn gaussian_blur_u8(
    src: ArrayView2<'_, u8>,
    kx: usize,
    ky: usize,
    sigma_x: f64,
    sigma_y: f64,
) -> CvResult<Array2<u8>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("gaussian_blur_u8 empty".into()));
    }
    let kern_x = gaussian_kernel_1d(kx, sigma_x)?;
    let kern_y = gaussian_kernel_1d(ky, if sigma_y <= 0.0 { sigma_x } else { sigma_y })?;
    let rx = (kx / 2) as isize;
    let ry = (ky / 2) as isize;

    let mut tmp = vec![0.0f64; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for (i, &kv) in kern_x.iter().enumerate() {
                let sx = border_reflect_101(x as isize + i as isize - rx, w);
                acc += src[[y, sx]] as f64 * kv;
            }
            tmp[y * w + x] = acc;
        }
    }

    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for (j, &kv) in kern_y.iter().enumerate() {
                let sy = border_reflect_101(y as isize + j as isize - ry, h);
                acc += tmp[sy * w + x] * kv;
            }
            out[[y, x]] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

/// Separable 2D Gaussian blur on 3D `(H, W, C)` `u8` image (`BORDER_REFLECT_101`).
pub fn gaussian_blur_u8_hwc(
    src: ArrayView3<'_, u8>,
    kx: usize,
    ky: usize,
    sigma_x: f64,
    sigma_y: f64,
) -> CvResult<Array3<u8>> {
    let (h, w, c) = src.dim();
    if h == 0 || w == 0 || c == 0 {
        return Err(CvError::EmptyInput("gaussian_blur_u8_hwc empty".into()));
    }
    let kern_x = gaussian_kernel_1d(kx, sigma_x)?;
    let kern_y = gaussian_kernel_1d(ky, if sigma_y <= 0.0 { sigma_x } else { sigma_y })?;
    let rx = (kx / 2) as isize;
    let ry = (ky / 2) as isize;

    let mut tmp = vec![0.0f64; h * w * c];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * c;
            for (i, &kv) in kern_x.iter().enumerate() {
                let sx = border_reflect_101(x as isize + i as isize - rx, w);
                for ch in 0..c {
                    tmp[base + ch] += src[[y, sx, ch]] as f64 * kv;
                }
            }
        }
    }

    let mut out = Array3::<u8>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f64; 4];
            if c <= 4 {
                for (j, &kv) in kern_y.iter().enumerate() {
                    let sy = border_reflect_101(y as isize + j as isize - ry, h);
                    let base = (sy * w + x) * c;
                    for ch in 0..c {
                        acc[ch] += tmp[base + ch] * kv;
                    }
                }
                for (ch, &v) in acc.iter().take(c).enumerate() {
                    out[[y, x, ch]] = v.round().clamp(0.0, 255.0) as u8;
                }
            } else {
                for ch in 0..c {
                    let mut v = 0.0f64;
                    for (j, &kv) in kern_y.iter().enumerate() {
                        let sy = border_reflect_101(y as isize + j as isize - ry, h);
                        v += tmp[(sy * w + x) * c + ch] * kv;
                    }
                    out[[y, x, ch]] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(out)
}

/// O(1)-per-pixel separable sliding box filter on 2D `f32` (`BORDER_REFLECT_101`, anchor `(kw/2, kh/2)`).
pub fn box_filter_f32(
    src: ArrayView2<'_, f32>,
    kw: usize,
    kh: usize,
    normalize: bool,
) -> CvResult<Array2<f32>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 || kw == 0 || kh == 0 {
        return Err(CvError::EmptyInput("box_filter_f32 invalid dims".into()));
    }
    let ax = (kw / 2) as isize;
    let ay = (kh / 2) as isize;

    let mut tmp = vec![0.0f64; h * w];
    for y in 0..h {
        let mut sum = 0.0f64;
        for kx in 0..kw {
            let sx = border_reflect_101(kx as isize - ax, w);
            sum += src[[y, sx]] as f64;
        }
        tmp[y * w] = sum;
        for x in 1..w {
            let add_x = border_reflect_101(x as isize - ax + kw as isize - 1, w);
            let sub_x = border_reflect_101(x as isize - ax - 1, w);
            sum += src[[y, add_x]] as f64 - src[[y, sub_x]] as f64;
            tmp[y * w + x] = sum;
        }
    }

    let norm = if normalize {
        1.0 / ((kw * kh) as f64)
    } else {
        1.0
    };
    let mut out = Array2::<f32>::zeros((h, w));
    for x in 0..w {
        let mut sum = 0.0f64;
        for ky in 0..kh {
            let sy = border_reflect_101(ky as isize - ay, h);
            sum += tmp[sy * w + x];
        }
        out[[0, x]] = (sum * norm) as f32;
        for y in 1..h {
            let add_y = border_reflect_101(y as isize - ay + kh as isize - 1, h);
            let sub_y = border_reflect_101(y as isize - ay - 1, h);
            sum += tmp[add_y * w + x] - tmp[sub_y * w + x];
            out[[y, x]] = (sum * norm) as f32;
        }
    }
    Ok(out)
}

/// 2D median blur on `u8` (`BORDER_REPLICATE`, odd `ksize >= 1`).
pub fn median_blur_u8(src: ArrayView2<'_, u8>, ksize: usize) -> CvResult<Array2<u8>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("median_blur_u8 empty".into()));
    }
    if ksize == 0 || ksize.is_multiple_of(2) {
        return Err(CvError::InvalidArgument("median ksize must be odd >= 1".into()));
    }
    if ksize == 1 {
        return Ok(src.to_owned());
    }
    let r = (ksize / 2) as isize;
    let target = (ksize * ksize) / 2;
    let mut out = Array2::<u8>::zeros((h, w));
    let mut buf = Vec::with_capacity(ksize * ksize);
    for y in 0..h {
        for x in 0..w {
            buf.clear();
            for dy in -r..=r {
                let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                for dx in -r..=r {
                    let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    buf.push(src[[sy, sx]]);
                }
            }
            let (_, mid, _) = buf.select_nth_unstable(target);
            out[[y, x]] = *mid;
        }
    }
    Ok(out)
}

/// Contrast Limited Adaptive Histogram Equalization matching `cv2.createCLAHE(clipLimit, tileGridSize)`.
pub fn clahe_u8(
    src: ArrayView2<'_, u8>,
    clip_limit: f64,
    tile_grid: (usize, usize),
) -> CvResult<Array2<u8>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("clahe_u8 empty".into()));
    }
    let (tx_count, ty_count) = tile_grid;
    if tx_count == 0 || ty_count == 0 {
        return Err(CvError::InvalidArgument("clahe tile_grid must be >= 1".into()));
    }

    // Pad with BORDER_REFLECT_101 if dimensions are not divisible by tile_grid
    let pad_w = if w.is_multiple_of(tx_count) {
        0
    } else {
        tx_count - (w % tx_count)
    };
    let pad_h = if h.is_multiple_of(ty_count) {
        0
    } else {
        ty_count - (h % ty_count)
    };
    let pw = w + pad_w;
    let ph = h + pad_h;
    let tile_w = pw / tx_count;
    let tile_h = ph / ty_count;
    let tile_area = tile_w * tile_h;

    let clip = if clip_limit > 0.0 {
        ((clip_limit * tile_area as f64) / 256.0).floor().max(1.0) as usize
    } else {
        usize::MAX
    };

    let mut luts = vec![[0u8; 256]; ty_count * tx_count];
    for ty in 0..ty_count {
        for tx in 0..tx_count {
            let mut hist = [0usize; 256];
            let y0 = ty * tile_h;
            let x0 = tx * tile_w;
            for dy in 0..tile_h {
                let sy = border_reflect_101((y0 + dy) as isize, h);
                for dx in 0..tile_w {
                    let sx = border_reflect_101((x0 + dx) as isize, w);
                    hist[src[[sy, sx]] as usize] += 1;
                }
            }

            if clip < usize::MAX {
                let mut clipped = 0usize;
                for c in &mut hist {
                    if *c > clip {
                        clipped += *c - clip;
                        *c = clip;
                    }
                }
                let redist = clipped / 256;
                let mut residual = clipped % 256;
                for c in &mut hist {
                    *c += redist;
                }
                if let Some(step_raw) = 256usize.checked_div(residual) {
                    let step = step_raw.max(1);
                    let mut i = 0usize;
                    while i < 256 && residual > 0 {
                        hist[i] += 1;
                        residual -= 1;
                        i += step;
                    }
                    let mut k = 0usize;
                    while residual > 0 {
                        hist[k] += 1;
                        residual -= 1;
                        k = (k + 1) % 256;
                    }
                }
            }

            let lut = &mut luts[ty * tx_count + tx];
            let scale = 255.0 / (tile_area as f64);
            let mut cdf = 0usize;
            for (i, &c) in hist.iter().enumerate() {
                cdf += c;
                lut[i] = (cdf as f64 * scale).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    let inv_tw = 1.0 / (tile_w as f64);
    let inv_th = 1.0 / (tile_h as f64);
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        let tyf = (y as f64 + 0.5) * inv_th - 0.5;
        let ty1 = (tyf.floor() as isize).max(0) as usize;
        let ty2 = (ty1 + 1).min(ty_count - 1);
        let wy2 = if tyf < 0.0 || ty1 >= ty_count - 1 {
            0.0
        } else {
            tyf - ty1 as f64
        };
        let wy1 = 1.0 - wy2;

        for x in 0..w {
            let txf = (x as f64 + 0.5) * inv_tw - 0.5;
            let tx1 = (txf.floor() as isize).max(0) as usize;
            let tx2 = (tx1 + 1).min(tx_count - 1);
            let wx2 = if txf < 0.0 || tx1 >= tx_count - 1 {
                0.0
            } else {
                txf - tx1 as f64
            };
            let wx1 = 1.0 - wx2;

            let v = src[[y, x]] as usize;
            let p11 = luts[ty1 * tx_count + tx1][v] as f64;
            let p12 = luts[ty1 * tx_count + tx2][v] as f64;
            let p21 = luts[ty2 * tx_count + tx1][v] as f64;
            let p22 = luts[ty2 * tx_count + tx2][v] as f64;

            let top = wx1 * p11 + wx2 * p12;
            let bot = wx1 * p21 + wx2 * p22;
            out[[y, x]] = (wy1 * top + wy2 * bot).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn box_filter_constant_preserves_value() {
        let a = Array2::<f32>::from_elem((8, 10), 42.5);
        let b = box_filter_f32(a.view(), 5, 3, true).unwrap();
        for &v in b.iter() {
            assert!((v - 42.5).abs() < 1e-5);
        }
    }

    #[test]
    fn gaussian_blur_preserves_constant_and_small_kernels_match_opencv() {
        assert_eq!(gaussian_kernel_1d(1, 0.0).unwrap(), vec![1.0]);
        assert_eq!(gaussian_kernel_1d(3, 0.0).unwrap(), vec![0.25, 0.5, 0.25]);
        assert_eq!(
            gaussian_kernel_1d(5, 0.0).unwrap(),
            vec![0.0625, 0.25, 0.375, 0.25, 0.0625]
        );
        assert_eq!(
            gaussian_kernel_1d(7, 0.0).unwrap(),
            vec![0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
        );
        let a = Array2::<u8>::from_elem((12, 12), 100);
        let b = gaussian_blur_u8(a.view(), 5, 5, 0.0, 0.0).unwrap();
        assert!(b.iter().all(|&v| v == 100));
    }

    #[test]
    fn clahe_enhances_low_contrast_ramp() {
        let mut a = Array2::<u8>::zeros((16, 16));
        for y in 0..16 {
            for x in 0..16 {
                a[[y, x]] = (y * 16 + x) as u8;
            }
        }
        let c = clahe_u8(a.view(), 2.0, (4, 4)).unwrap();
        assert_eq!(c[[15, 15]], 255);
        assert!(c[[0, 0]] <= 32);
    }

    #[test]
    fn median_blur_removes_impulse() {
        let mut a = array![
            [10u8, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 250, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10]
        ];
        let m = median_blur_u8(a.view(), 3).unwrap();
        assert_eq!(m[[2, 2]], 10);
        a[[0, 0]] = 250;
        let m2 = median_blur_u8(a.view(), 3).unwrap();
        assert_eq!(m2[[0, 0]], 10);
    }

    #[test]
    fn gaussian_blur_f32_folded_matches_naive_separable() {
        for &(h, w, kx, ky, sig) in &[(19usize, 23usize, 5usize, 7usize, 1.3), (160, 141, 13, 11, 2.1)] {
            let mut src = Array2::<f32>::zeros((h, w));
            for y in 0..h {
                for x in 0..w {
                    src[[y, x]] = (((y * 29 + x * 47 + y * x * 3) % 256) as f32) / 255.0;
                }
            }
            let fast = gaussian_blur_f32(src.view(), kx, ky, sig, sig).unwrap();

            let k_x: Vec<f32> = gaussian_kernel_1d(kx, sig).unwrap().into_iter().map(|v| v as f32).collect();
            let k_y: Vec<f32> = gaussian_kernel_1d(ky, sig).unwrap().into_iter().map(|v| v as f32).collect();
            let rx = (kx / 2) as isize;
            let ry = (ky / 2) as isize;
            let mut tmp = Array2::<f32>::zeros((h, w));
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0f32;
                    for (i, &kv) in k_x.iter().enumerate() {
                        acc += src[[y, border_reflect_101(x as isize + i as isize - rx, w)]] * kv;
                    }
                    tmp[[y, x]] = acc;
                }
            }
            let mut max_d = 0.0f32;
            for y in 0..h {
                for x in 0..w {
                    let mut acc = 0.0f32;
                    for (j, &kv) in k_y.iter().enumerate() {
                        acc += tmp[[border_reflect_101(y as isize + j as isize - ry, h), x]] * kv;
                    }
                    max_d = max_d.max((fast[[y, x]] - acc).abs());
                }
            }
            assert!(max_d < 1e-5, "shape ({h},{w}) k=({kx},{ky}) max_d={max_d}");
        }
    }
}
