//! OpenCV-compatible 2D Hanning window (`sqrt(wy * wx)`) and `phase_correlate`
//! (2D FFT cross-power spectrum + 5x5 weighted centroid).

use crate::ops::ncc::{fft2d_c64, next_smooth_fft_len, plan_fft_pair_f64};
use crate::ops::{CvError, CvResult};
use ndarray::{Array2, ArrayView2};
use rustfft::num_complex::Complex;

/// Creates OpenCV's geometric-mean 2D Hanning window:
/// `w[y, x] = sqrt(0.5 * (1 - cos(2*pi*y/(h-1))) * 0.5 * (1 - cos(2*pi*x/(w-1))))`.
pub fn create_hanning_window(w: usize, h: usize) -> CvResult<Array2<f32>> {
    if w == 0 || h == 0 {
        return Err(CvError::EmptyInput("create_hanning_window zero dimension".into()));
    }
    let wx: Vec<f64> = (0..w)
        .map(|x| {
            if w == 1 {
                1.0
            } else {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * x as f64 / (w - 1) as f64).cos())
            }
        })
        .collect();
    let wy: Vec<f64> = (0..h)
        .map(|y| {
            if h == 1 {
                1.0
            } else {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * y as f64 / (h - 1) as f64).cos())
            }
        })
        .collect();

    let mut out = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        let r = wy[y];
        for x in 0..w {
            out[[y, x]] = (r * wx[x]).sqrt() as f32;
        }
    }
    Ok(out)
}

/// Computes sub-pixel phase shift `((dx, dy), response)` matching `cv2.phaseCorrelate(src1, src2, window)`.
/// Positive `(dx, dy)` means `src2` is `src1` shifted by `(+dx, +dy)`.
pub fn phase_correlate(
    src1: ArrayView2<'_, f32>,
    src2: ArrayView2<'_, f32>,
    window: Option<ArrayView2<'_, f32>>,
) -> CvResult<((f64, f64), f64)> {
    let (h, w) = src1.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("phase_correlate empty".into()));
    }
    if src2.dim() != (h, w) {
        return Err(CvError::ShapeMismatch("src1 and src2 dimensions differ".into()));
    }
    if let Some(win) = window
        && win.dim() != (h, w)
    {
        return Err(CvError::ShapeMismatch("window dimensions differ".into()));
    }

    let fh = next_smooth_fft_len(h);
    let fw = next_smooth_fft_len(w);
    let (row_fwd, row_inv) = plan_fft_pair_f64(fw);
    let (col_fwd, col_inv) = plan_fft_pair_f64(fh);

    let mut fa = vec![Complex::new(0.0, 0.0); fh * fw];
    let mut fb = vec![Complex::new(0.0, 0.0); fh * fw];
    let s1_contig;
    let s2_contig;
    let s1 = if let Some(s) = src1.as_slice() {
        s
    } else {
        s1_contig = src1.to_owned();
        s1_contig.as_slice().unwrap()
    };
    let s2 = if let Some(s) = src2.as_slice() {
        s
    } else {
        s2_contig = src2.to_owned();
        s2_contig.as_slice().unwrap()
    };
    if let Some(win) = window {
        let w_contig;
        let ws = if let Some(s) = win.as_slice() {
            s
        } else {
            w_contig = win.to_owned();
            w_contig.as_slice().unwrap()
        };
        for y in 0..h {
            let fa_r = &mut fa[y * fw..y * fw + w];
            let fb_r = &mut fb[y * fw..y * fw + w];
            let r1 = &s1[y * w..(y + 1) * w];
            let r2 = &s2[y * w..(y + 1) * w];
            let rw = &ws[y * w..(y + 1) * w];
            for x in 0..w {
                let wt = rw[x] as f64;
                fa_r[x].re = r1[x] as f64 * wt;
                fb_r[x].re = r2[x] as f64 * wt;
            }
        }
    } else {
        for y in 0..h {
            let fa_r = &mut fa[y * fw..y * fw + w];
            let fb_r = &mut fb[y * fw..y * fw + w];
            let r1 = &s1[y * w..(y + 1) * w];
            let r2 = &s2[y * w..(y + 1) * w];
            for x in 0..w {
                fa_r[x].re = r1[x] as f64;
                fb_r[x].re = r2[x] as f64;
            }
        }
    }

    fft2d_c64(&mut fa, fh, fw, &row_fwd, &col_fwd);
    fft2d_c64(&mut fb, fh, fw, &row_fwd, &col_fwd);

    // Cross-power spectrum: FB * conj(FA) / |FB * conj(FA)|
    let eps = 1e-15;
    for (va, vb) in fa.iter_mut().zip(fb.iter()) {
        let p = *vb * va.conj();
        let mag = p.norm();
        *va = if mag > eps {
            p / mag
        } else {
            Complex::new(0.0, 0.0)
        };
    }

    fft2d_c64(&mut fa, fh, fw, &row_inv, &col_inv);

    let norm = 1.0 / ((fh * fw) as f64);
    // fftshift into real grid of size (fh, fw)
    let hy = fh / 2;
    let hx = fw / 2;
    let mut shifted = vec![0.0f64; fh * fw];
    let mut max_val = f64::NEG_INFINITY;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    for y in 0..fh {
        let sy = (y + hy) % fh;
        for x in 0..fw {
            let sx = (x + hx) % fw;
            let v = fa[y * fw + x].re * norm;
            shifted[sy * fw + sx] = v;
            if v > max_val {
                max_val = v;
                max_x = sx;
                max_y = sy;
            }
        }
    }

    // 5x5 weighted centroid around (max_x, max_y) matching OpenCV phasecorr.cpp
    let x0 = max_x.saturating_sub(2);
    let x1 = (max_x + 2).min(fw - 1);
    let y0 = max_y.saturating_sub(2);
    let y1 = (max_y + 2).min(fh - 1);

    let mut m00 = 0.0f64;
    let mut m10 = 0.0f64;
    let mut m01 = 0.0f64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let v = shifted[y * fw + x];
            m00 += v;
            m10 += (x as f64) * v;
            m01 += (y as f64) * v;
        }
    }

    if !m00.is_finite() || m00.abs() <= 1e-12 {
        return Ok(((0.0, 0.0), 0.0));
    }

    let cx = m10 / m00;
    let cy = m01 / m00;
    let dx = cx - (fw as f64 * 0.5).floor();
    let dy = cy - (fh as f64 * 0.5).floor();

    // Note: for odd dimensions, OpenCV fftshift puts DC at N/2 = floor(N/2) (wait: (0 + N/2)%N = N/2),
    // which matches (fh/2, fw/2) integer division!
    let dx_exact = cx - hx as f64;
    let dy_exact = cy - hy as f64;
    let _ = (dx, dy);

    Ok(((dx_exact, dy_exact), m00))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_matches_opencv_probe() {
        let hann = create_hanning_window(96, 96).unwrap();
        assert!((hann[[1, 1]] - 0.001_093_186_4).abs() < 1e-6);
        assert!((hann[[1, 48]] - 0.033_058_852).abs() < 1e-6);
    }

    #[test]
    fn phase_correlate_matches_opencv_shift_sign() {
        let mut a = Array2::<f32>::zeros((96, 96));
        let mut b = Array2::<f32>::zeros((96, 96));
        for y in 30..50 {
            for x in 40..60 {
                a[[y, x]] = 1.0;
            }
        }
        for y in 33..53 {
            for x in 45..65 {
                b[[y, x]] = 1.0;
            }
        }
        let hann = create_hanning_window(96, 96).unwrap();
        let ((dx, dy), resp) = phase_correlate(a.view(), b.view(), Some(hann.view())).unwrap();
        assert!((dx - 4.987_379).abs() < 1e-4, "dx={dx}");
        assert!((dy - 2.993_805).abs() < 1e-4, "dy={dy}");
        assert!((resp - 0.984_710).abs() < 1e-4, "resp={resp}");
    }

    #[test]
    fn phase_correlate_nan_returns_zero_shift_without_nan() {
        let mut a = Array2::<f32>::zeros((32, 32));
        a[[5, 5]] = f32::NAN;
        let b = Array2::<f32>::from_elem((32, 32), 1.0);
        let ((dx, dy), resp) = phase_correlate(a.view(), b.view(), None).unwrap();
        assert_eq!((dx, dy), (0.0, 0.0));
        assert_eq!(resp, 0.0);
    }
}
