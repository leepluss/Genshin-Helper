//! Geometric warps: `warp_polar_linear`, `warp_polar_log`, `get_rotation_matrix_2d`,
//! `invert_affine_transform`, and `warp_affine`.

use crate::ops::{BorderMode, CvError, CvResult, Interpolation, map_border_coord};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

/// Computes 2x3 affine rotation matrix matching `cv2.getRotationMatrix2D(center, angle_ccw_deg, scale)`.
/// Positive `angle_ccw_deg` rotates counter-clockwise in image coordinates (+X right, +Y down).
pub fn get_rotation_matrix_2d(center: (f64, f64), angle_ccw_deg: f64, scale: f64) -> [f64; 6] {
    let rad = angle_ccw_deg.to_radians();
    let alpha = scale * rad.cos();
    let beta = scale * rad.sin();
    let (cx, cy) = center;
    [
        alpha,
        beta,
        (1.0 - alpha) * cx - beta * cy,
        -beta,
        alpha,
        beta * cx + (1.0 - alpha) * cy,
    ]
}

/// Inverts a 2x3 affine matrix `[a, b, tx, c, d, ty]` matching `cv2.invertAffineTransform`.
pub fn invert_affine_transform(m: &[f64; 6]) -> Option<[f64; 6]> {
    let det = m[0] * m[4] - m[1] * m[3];
    if det.abs() <= 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let a = m[4] * inv;
    let b = -m[1] * inv;
    let c = -m[3] * inv;
    let d = m[0] * inv;
    let tx = -(a * m[2] + b * m[5]);
    let ty = -(c * m[2] + d * m[5]);
    Some([a, b, tx, c, d, ty])
}

#[inline]
fn sample_u8_2d(
    src: ArrayView2<'_, u8>,
    sx: f64,
    sy: f64,
    interp: Interpolation,
    border: BorderMode,
) -> u8 {
    let (h, w) = src.dim();
    let const_v = match border {
        BorderMode::Constant(v) => v,
        _ => 0.0,
    };
    let fetch = |ix: isize, iy: isize| -> f64 {
        match (map_border_coord(ix, w, border), map_border_coord(iy, h, border)) {
            (Some(px), Some(py)) => src[[py, px]] as f64,
            _ => const_v as f64,
        }
    };
    if matches!(interp, Interpolation::Nearest) {
        let ix = sx.round() as isize;
        let iy = sy.round() as isize;
        return fetch(ix, iy).round().clamp(0.0, 255.0) as u8;
    }
    // Linear (also used when Area/Cubic fallback is requested in warp_affine)
    let x0 = sx.floor() as isize;
    let y0 = sy.floor() as isize;
    let ax = sx - x0 as f64;
    let ay = sy - y0 as f64;
    let v00 = fetch(x0, y0);
    let v10 = fetch(x0 + 1, y0);
    let v01 = fetch(x0, y0 + 1);
    let v11 = fetch(x0 + 1, y0 + 1);
    let top = v00 + ax * (v10 - v00);
    let bot = v01 + ax * (v11 - v01);
    (top + ay * (bot - top)).round().clamp(0.0, 255.0) as u8
}

#[inline]
fn sample_u8_hwc(
    src: ArrayView3<'_, u8>,
    sx: f64,
    sy: f64,
    ch: usize,
    interp: Interpolation,
    border: BorderMode,
) -> u8 {
    let (h, w, _) = src.dim();
    let const_v = match border {
        BorderMode::Constant(v) => v as f64,
        _ => 0.0,
    };
    let fetch = |ix: isize, iy: isize| -> f64 {
        match (map_border_coord(ix, w, border), map_border_coord(iy, h, border)) {
            (Some(px), Some(py)) => src[[py, px, ch]] as f64,
            _ => const_v,
        }
    };
    if matches!(interp, Interpolation::Nearest) {
        return fetch(sx.round() as isize, sy.round() as isize)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    let x0 = sx.floor() as isize;
    let y0 = sy.floor() as isize;
    let ax = sx - x0 as f64;
    let ay = sy - y0 as f64;
    let v00 = fetch(x0, y0);
    let v10 = fetch(x0 + 1, y0);
    let v01 = fetch(x0, y0 + 1);
    let v11 = fetch(x0 + 1, y0 + 1);
    let top = v00 + ax * (v10 - v00);
    let bot = v01 + ax * (v11 - v01);
    (top + ay * (bot - top)).round().clamp(0.0, 255.0) as u8
}

/// Applies a 2x3 affine transformation to a 2D `u8` image.
/// When `inverse_map` is `true` (`WARP_INVERSE_MAP`), `m` maps `dst -> src`; otherwise `m` is inverted.
/// Note: `Interpolation::Area` falls back to `Linear` matching OpenCV `cv2.warpAffine`.
pub fn warp_affine_u8(
    src: ArrayView2<'_, u8>,
    m: &[f64; 6],
    dst_w: usize,
    dst_h: usize,
    interp: Interpolation,
    inverse_map: bool,
    border: BorderMode,
) -> CvResult<Array2<u8>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 || dst_w == 0 || dst_h == 0 {
        return Err(CvError::EmptyInput("warp_affine_u8 empty".into()));
    }
    let inv_m = if inverse_map {
        *m
    } else {
        invert_affine_transform(m).ok_or_else(|| CvError::InvalidArgument("singular affine".into()))?
    };
    let eff_interp = match interp {
        Interpolation::Area => Interpolation::Linear,
        other => other,
    };
    let mut out = Array2::<u8>::zeros((dst_h, dst_w));
    for dy in 0..dst_h {
        let fy = dy as f64;
        for dx in 0..dst_w {
            let fx = dx as f64;
            let sx = inv_m[0] * fx + inv_m[1] * fy + inv_m[2];
            let sy = inv_m[3] * fx + inv_m[4] * fy + inv_m[5];
            out[[dy, dx]] = sample_u8_2d(src, sx, sy, eff_interp, border);
        }
    }
    Ok(out)
}

/// Polar remap (`WARP_POLAR_LINEAR | INTER_LINEAR`) on 2D `u8` restricted to radial slice `rad_range = (r_start, r_end)`
/// out of `dst_w` total radial bins, avoiding wasted work outside the annulus.
pub fn warp_polar_annulus_u8(
    src: ArrayView2<'_, u8>,
    dst_w: usize,
    dst_h: usize,
    rad_range: (usize, usize),
    center: (f64, f64),
    max_radius: f64,
    interp: Interpolation,
) -> CvResult<Array2<u8>> {
    let (r_start, r_end) = rad_range;
    let (h, w) = src.dim();
    if h == 0
        || w == 0
        || dst_w == 0
        || dst_h == 0
        || max_radius <= 0.0
        || r_start >= r_end
        || r_end > dst_w
    {
        return Err(CvError::EmptyInput(
            "warp_polar_annulus_u8 invalid dims".into(),
        ));
    }
    let src_contig;
    let s = if let Some(sl) = src.as_slice() {
        sl
    } else {
        src_contig = src.to_owned();
        src_contig.as_slice().unwrap()
    };
    let out_w = r_end - r_start;
    let (cx, cy) = center;
    let k_angle = (2.0 * std::f64::consts::PI) / dst_h as f64;
    let k_mag = max_radius / dst_w as f64;
    let rhos: Vec<f64> = (r_start..r_end).map(|dx| dx as f64 * k_mag).collect();
    let w_lim = (w - 1) as f64;
    let h_lim = (h - 1) as f64;
    let nearest = matches!(interp, Interpolation::Nearest);
    let mut out = Array2::<u8>::zeros((dst_h, out_w));
    let dst_s = out.as_slice_mut().unwrap();
    for dy in 0..dst_h {
        let phi = dy as f64 * k_angle;
        let (sin_p, cos_p) = phi.sin_cos();
        let row = &mut dst_s[dy * out_w..(dy + 1) * out_w];
        for (ox, &rho) in rhos.iter().enumerate() {
            let sx = cx + rho * cos_p;
            let sy = cy + rho * sin_p;
            if nearest {
                let ix = sx.round() as isize;
                let iy = sy.round() as isize;
                row[ox] = if ix >= 0 && ix < w as isize && iy >= 0 && iy < h as isize {
                    s[iy as usize * w + ix as usize]
                } else {
                    0
                };
            } else if sx >= 0.0 && sx < w_lim && sy >= 0.0 && sy < h_lim {
                let x0 = sx as usize;
                let y0 = sy as usize;
                let ax = (sx - x0 as f64) as f32;
                let ay = (sy - y0 as f64) as f32;
                let i00 = y0 * w + x0;
                let v00 = s[i00] as f32;
                let v10 = s[i00 + 1] as f32;
                let v01 = s[i00 + w] as f32;
                let v11 = s[i00 + w + 1] as f32;
                let top = v00 + ax * (v10 - v00);
                let bot = v01 + ax * (v11 - v01);
                row[ox] = (top + ay * (bot - top) + 0.5) as u8;
            } else {
                row[ox] = sample_u8_2d(src, sx, sy, interp, BorderMode::Constant(0.0));
            }
        }
    }
    Ok(out)
}

/// Polar remap (`WARP_POLAR_LINEAR | INTER_LINEAR`) on 2D `u8` matching `cv2.warpPolar`:
/// `rho = x * max_radius / dst_w`, `phi = y * 2pi / dst_h`, `phi = 0` at `+X`, CW in image coords.
pub fn warp_polar_linear_u8(
    src: ArrayView2<'_, u8>,
    dst_w: usize,
    dst_h: usize,
    center: (f64, f64),
    max_radius: f64,
    interp: Interpolation,
) -> CvResult<Array2<u8>> {
    warp_polar_annulus_u8(src, dst_w, dst_h, (0, dst_w), center, max_radius, interp)
}

/// Polar remap (`WARP_POLAR_LINEAR | INTER_LINEAR`) restricted to radial slice `rad_range = (r_start, r_end)`
/// out of `dst_w` total radial bins, avoiding wasted work outside the annulus.
pub fn warp_polar_annulus_u8_hwc(
    src: ArrayView3<'_, u8>,
    dst_w: usize,
    dst_h: usize,
    rad_range: (usize, usize),
    center: (f64, f64),
    max_radius: f64,
    interp: Interpolation,
) -> CvResult<Array3<u8>> {
    let (r_start, r_end) = rad_range;
    let (h, w, c) = src.dim();
    if h == 0 || w == 0 || c == 0 || dst_w == 0 || dst_h == 0 || max_radius <= 0.0 || r_start >= r_end || r_end > dst_w {
        return Err(CvError::EmptyInput("warp_polar_annulus_u8_hwc invalid dims".into()));
    }
    let src_contig;
    let s = if let Some(sl) = src.as_slice() {
        sl
    } else {
        src_contig = src.to_owned();
        src_contig.as_slice().unwrap()
    };
    let out_w = r_end - r_start;
    let (cx, cy) = center;
    let k_angle = (2.0 * std::f64::consts::PI) / dst_h as f64;
    let k_mag = max_radius / dst_w as f64;
    let rhos: Vec<f64> = (r_start..r_end).map(|dx| dx as f64 * k_mag).collect();
    let w_lim = (w - 1) as f64;
    let h_lim = (h - 1) as f64;
    let row_stride = w * c;
    let nearest = matches!(interp, Interpolation::Nearest);
    let mut out = Array3::<u8>::zeros((dst_h, out_w, c));
    let dst_s = out.as_slice_mut().unwrap();

    if c == 3 && !nearest {
        for dy in 0..dst_h {
            let phi = dy as f64 * k_angle;
            let (sin_p, cos_p) = phi.sin_cos();
            let out_row = &mut dst_s[dy * out_w * 3..(dy + 1) * out_w * 3];
            for (ox, &rho) in rhos.iter().enumerate() {
                let sx = cx + rho * cos_p;
                let sy = cy + rho * sin_p;
                let out_px = &mut out_row[ox * 3..ox * 3 + 3];
                if sx >= 0.0 && sx < w_lim && sy >= 0.0 && sy < h_lim {
                    let x0 = sx as usize;
                    let y0 = sy as usize;
                    let ax = (sx - x0 as f64) as f32;
                    let ay = (sy - y0 as f64) as f32;
                    let i00 = y0 * row_stride + x0 * 3;
                    let i01 = i00 + row_stride;
                    let top0 = s[i00] as f32 + ax * (s[i00 + 3] as f32 - s[i00] as f32);
                    let bot0 = s[i01] as f32 + ax * (s[i01 + 3] as f32 - s[i01] as f32);
                    out_px[0] = (top0 + ay * (bot0 - top0) + 0.5) as u8;

                    let top1 = s[i00 + 1] as f32 + ax * (s[i00 + 4] as f32 - s[i00 + 1] as f32);
                    let bot1 = s[i01 + 1] as f32 + ax * (s[i01 + 4] as f32 - s[i01 + 1] as f32);
                    out_px[1] = (top1 + ay * (bot1 - top1) + 0.5) as u8;

                    let top2 = s[i00 + 2] as f32 + ax * (s[i00 + 5] as f32 - s[i00 + 2] as f32);
                    let bot2 = s[i01 + 2] as f32 + ax * (s[i01 + 5] as f32 - s[i01 + 2] as f32);
                    out_px[2] = (top2 + ay * (bot2 - top2) + 0.5) as u8;
                } else {
                    for (ch, dst_v) in out_px.iter_mut().enumerate() {
                        *dst_v = sample_u8_hwc(src, sx, sy, ch, interp, BorderMode::Constant(0.0));
                    }
                }
            }
        }
        return Ok(out);
    }

    for dy in 0..dst_h {
        let phi = dy as f64 * k_angle;
        let (sin_p, cos_p) = phi.sin_cos();
        let out_row = &mut dst_s[dy * out_w * c..(dy + 1) * out_w * c];
        for (ox, &rho) in rhos.iter().enumerate() {
            let sx = cx + rho * cos_p;
            let sy = cy + rho * sin_p;
            let out_px = &mut out_row[ox * c..(ox + 1) * c];
            if !nearest && sx >= 0.0 && sx < w_lim && sy >= 0.0 && sy < h_lim {
                let x0 = sx as usize;
                let y0 = sy as usize;
                let ax = (sx - x0 as f64) as f32;
                let ay = (sy - y0 as f64) as f32;
                let i00 = y0 * row_stride + x0 * c;
                let i10 = i00 + c;
                let i01 = i00 + row_stride;
                let i11 = i01 + c;
                for ch in 0..c {
                    let v00 = s[i00 + ch] as f32;
                    let v10 = s[i10 + ch] as f32;
                    let v01 = s[i01 + ch] as f32;
                    let v11 = s[i11 + ch] as f32;
                    let top = v00 + ax * (v10 - v00);
                    let bot = v01 + ax * (v11 - v01);
                    out_px[ch] = (top + ay * (bot - top) + 0.5) as u8;
                }
            } else {
                for (ch, dst_v) in out_px.iter_mut().enumerate() {
                    *dst_v = sample_u8_hwc(src, sx, sy, ch, interp, BorderMode::Constant(0.0));
                }
            }
        }
    }
    Ok(out)
}

/// Polar remap (`WARP_POLAR_LINEAR | INTER_LINEAR`) on 3D `(H, W, C)` `u8` matching `cv2.warpPolar`.
pub fn warp_polar_linear_u8_hwc(
    src: ArrayView3<'_, u8>,
    dst_w: usize,
    dst_h: usize,
    center: (f64, f64),
    max_radius: f64,
    interp: Interpolation,
) -> CvResult<Array3<u8>> {
    warp_polar_annulus_u8_hwc(src, dst_w, dst_h, (0, dst_w), center, max_radius, interp)
}

/// Log-polar remap (`WARP_POLAR_LOG`) on 2D `u8`:
/// `rho = exp(x * ln(max_radius) / dst_w) - 1`, `phi = y * 2pi / dst_h`.
pub fn warp_polar_log_u8(
    src: ArrayView2<'_, u8>,
    dst_w: usize,
    dst_h: usize,
    center: (f64, f64),
    max_radius: f64,
    interp: Interpolation,
) -> CvResult<Array2<u8>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 || dst_w == 0 || dst_h == 0 || max_radius <= 1.0 {
        return Err(CvError::EmptyInput("warp_polar_log_u8 invalid dims".into()));
    }
    let (cx, cy) = center;
    let k_angle = (2.0 * std::f64::consts::PI) / dst_h as f64;
    let k_mag = max_radius.ln() / dst_w as f64;
    let mut out = Array2::<u8>::zeros((dst_h, dst_w));
    for dy in 0..dst_h {
        let phi = dy as f64 * k_angle;
        let (sin_p, cos_p) = phi.sin_cos();
        for dx in 0..dst_w {
            let rho = (dx as f64 * k_mag).exp() - 1.0;
            let sx = cx + rho * cos_p;
            let sy = cy + rho * sin_p;
            out[[dy, dx]] = sample_u8_2d(src, sx, sy, interp, BorderMode::Constant(0.0));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_matrix_and_inverse_roundtrip() {
        let m = get_rotation_matrix_2d((50.0, 50.0), 37.0, 1.0);
        let inv = invert_affine_transform(&m).unwrap();
        // M * inv should map (12, 34) back to (12, 34)
        let x1 = m[0] * 12.0 + m[1] * 34.0 + m[2];
        let y1 = m[3] * 12.0 + m[4] * 34.0 + m[5];
        let x2 = inv[0] * x1 + inv[1] * y1 + inv[2];
        let y2 = inv[3] * x1 + inv[4] * y1 + inv[5];
        assert!((x2 - 12.0).abs() < 1e-6);
        assert!((y2 - 34.0).abs() < 1e-6);
    }

    #[test]
    fn warp_polar_east_ray_aligns_at_phi_zero() {
        let mut img = Array2::<u8>::zeros((64, 64));
        for x in 32..60 {
            img[[32, x]] = 200;
        }
        let polar = warp_polar_linear_u8(
            img.view(),
            32,
            64,
            (32.0, 32.0),
            32.0,
            Interpolation::Linear,
        )
        .unwrap();
        // phi = 0 is row 0 (+X East)
        assert!(polar[[0, 16]] > 150);
        assert_eq!(polar[[32, 16]], 0);
    }

    #[test]
    fn warp_polar_log_starts_at_center_minus_one() {
        let mut img = Array2::<u8>::zeros((256, 256));
        for y in 0..256 {
            for x in 0..256 {
                img[[y, x]] = x as u8;
            }
        }
        let logp = warp_polar_log_u8(
            img.view(),
            64,
            180,
            (128.0, 128.0),
            100.0,
            Interpolation::Nearest,
        )
        .unwrap();
        // At phi=0 (row 0), dx=0 has rho = exp(0) - 1 = 0 -> sx = 128
        assert_eq!(logp[[0, 0]], 128);
        // At dx=32, rho = sqrt(100) - 1 = 9 -> sx = 137
        assert_eq!(logp[[0, 32]], 137);
    }
}
