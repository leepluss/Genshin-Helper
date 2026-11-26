//! Image resizing (`Nearest`, `Linear`, `Area` with OpenCV upscale block-replication, `Cubic`)
//! and `copy_make_border` for `u8` (2D / 3D HWC) and `f32` (2D).

use crate::ops::{BorderMode, CvError, CvResult, Interpolation, map_border_coord};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

/// Pads a 2D `u8` array using the specified `BorderMode`.
pub fn copy_make_border_u8(
    src: ArrayView2<'_, u8>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    mode: BorderMode,
) -> CvResult<Array2<u8>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("copy_make_border_u8 empty input".into()));
    }
    let out_h = h + top + bottom;
    let out_w = w + left + right;
    let const_v = match mode {
        BorderMode::Constant(v) => v.round().clamp(0.0, 255.0) as u8,
        _ => 0,
    };
    let mut out = Array2::<u8>::from_elem((out_h, out_w), const_v);
    let y_map: Vec<Option<usize>> = (0..out_h)
        .map(|y| map_border_coord(y as isize - top as isize, h, mode))
        .collect();
    if left == 0 && right == 0
        && let (Some(src_s), Some(out_s)) = (src.as_slice(), out.as_slice_mut())
    {
        for (dy, &sy_opt) in y_map.iter().enumerate() {
            if let Some(sy) = sy_opt {
                out_s[dy * out_w..(dy + 1) * out_w].copy_from_slice(&src_s[sy * w..(sy + 1) * w]);
            }
        }
        return Ok(out);
    }
    let x_map: Vec<Option<usize>> = (0..out_w)
        .map(|x| map_border_coord(x as isize - left as isize, w, mode))
        .collect();
    for (dy, &sy_opt) in y_map.iter().enumerate() {
        if let Some(sy) = sy_opt {
            for (dx, &sx_opt) in x_map.iter().enumerate() {
                if let Some(sx) = sx_opt {
                    out[[dy, dx]] = src[[sy, sx]];
                }
            }
        }
    }
    Ok(out)
}

/// Pads a 3D `(H, W, C)` `u8` array using the specified `BorderMode`.
pub fn copy_make_border_u8_hwc(
    src: ArrayView3<'_, u8>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    mode: BorderMode,
) -> CvResult<Array3<u8>> {
    let (h, w, c) = src.dim();
    if h == 0 || w == 0 || c == 0 {
        return Err(CvError::EmptyInput("copy_make_border_u8_hwc empty input".into()));
    }
    let out_h = h + top + bottom;
    let out_w = w + left + right;
    let const_v = match mode {
        BorderMode::Constant(v) => v.round().clamp(0.0, 255.0) as u8,
        _ => 0,
    };
    let mut out = Array3::<u8>::from_elem((out_h, out_w, c), const_v);
    let y_map: Vec<Option<usize>> = (0..out_h)
        .map(|y| map_border_coord(y as isize - top as isize, h, mode))
        .collect();
    let x_map: Vec<Option<usize>> = (0..out_w)
        .map(|x| map_border_coord(x as isize - left as isize, w, mode))
        .collect();
    for (dy, &sy_opt) in y_map.iter().enumerate() {
        if let Some(sy) = sy_opt {
            for (dx, &sx_opt) in x_map.iter().enumerate() {
                if let Some(sx) = sx_opt {
                    for ch in 0..c {
                        out[[dy, dx, ch]] = src[[sy, sx, ch]];
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Pads a 2D `f32` array using the specified `BorderMode`.
pub fn copy_make_border_f32(
    src: ArrayView2<'_, f32>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    mode: BorderMode,
) -> CvResult<Array2<f32>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("copy_make_border_f32 empty input".into()));
    }
    let out_h = h + top + bottom;
    let out_w = w + left + right;
    let const_v = match mode {
        BorderMode::Constant(v) => v,
        _ => 0.0,
    };
    let mut out = Array2::<f32>::from_elem((out_h, out_w), const_v);
    let y_map: Vec<Option<usize>> = (0..out_h)
        .map(|y| map_border_coord(y as isize - top as isize, h, mode))
        .collect();
    if left == 0 && right == 0
        && let (Some(src_s), Some(out_s)) = (src.as_slice(), out.as_slice_mut())
    {
        for (dy, &sy_opt) in y_map.iter().enumerate() {
            if let Some(sy) = sy_opt {
                out_s[dy * out_w..(dy + 1) * out_w].copy_from_slice(&src_s[sy * w..(sy + 1) * w]);
            }
        }
        return Ok(out);
    }
    let x_map: Vec<Option<usize>> = (0..out_w)
        .map(|x| map_border_coord(x as isize - left as isize, w, mode))
        .collect();
    for (dy, &sy_opt) in y_map.iter().enumerate() {
        if let Some(sy) = sy_opt {
            for (dx, &sx_opt) in x_map.iter().enumerate() {
                if let Some(sx) = sx_opt {
                    out[[dy, dx]] = src[[sy, sx]];
                }
            }
        }
    }
    Ok(out)
}

#[inline]
fn cubic_weights(t: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let w0 = ((A * (t + 1.0) - 5.0 * A) * (t + 1.0) + 8.0 * A) * (t + 1.0) - 4.0 * A;
    let w1 = ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
    let s = 1.0 - t;
    let w2 = ((A + 2.0) * s - (A + 3.0)) * s * s + 1.0;
    let w3 = 1.0 - w0 - w1 - w2;
    [w0, w1, w2, w3]
}

struct AxisTaps {
    // Packed variable-length spans per destination coordinate for Area downscale or fixed 2/4 taps
    spans: Vec<Vec<(usize, f64)>>,
}

fn build_axis_taps(src_len: usize, dst_len: usize, interp: Interpolation) -> AxisTaps {
    let mut spans = Vec::with_capacity(dst_len);
    match interp {
        Interpolation::Nearest => {
            let scale = src_len as f64 / dst_len as f64;
            for dx in 0..dst_len {
                let sx = ((dx as f64) * scale).floor() as usize;
                spans.push(vec![(sx.min(src_len - 1), 1.0)]);
            }
        }
        Interpolation::Linear => {
            let scale = src_len as f64 / dst_len as f64;
            for dx in 0..dst_len {
                let fx = (dx as f64 + 0.5) * scale - 0.5;
                let ix = fx.floor() as isize;
                let alpha = fx - ix as f64;
                let x0 = ix.clamp(0, src_len as isize - 1) as usize;
                let x1 = (ix + 1).clamp(0, src_len as isize - 1) as usize;
                if x0 == x1 {
                    spans.push(vec![(x0, 1.0)]);
                } else {
                    spans.push(vec![(x0, 1.0 - alpha), (x1, alpha)]);
                }
            }
        }
        Interpolation::Area => {
            if dst_len >= src_len {
                // OpenCV INTER_AREA upscale: block replication on integer multiples,
                // weighted boundary interpolation otherwise.
                let inv_scale = src_len as f64 / dst_len as f64;
                let scale = dst_len as f64 / src_len as f64;
                for dx in 0..dst_len {
                    let sx = ((dx as f64) * inv_scale).floor() as usize;
                    let sx_c = sx.min(src_len - 1);
                    let alpha = ((dx + 1) as f64 - (sx + 1) as f64 * scale).clamp(0.0, 1.0);
                    if alpha <= 0.0 || sx_c + 1 >= src_len {
                        spans.push(vec![(sx_c, 1.0)]);
                    } else {
                        spans.push(vec![(sx_c, 1.0 - alpha), (sx_c + 1, alpha)]);
                    }
                }
            } else {
                // Exact fractional box area overlap
                let scale = src_len as f64 / dst_len as f64;
                for dx in 0..dst_len {
                    let fx0 = dx as f64 * scale;
                    let fx1 = (dx + 1) as f64 * scale;
                    let sx0 = fx0.floor() as usize;
                    let sx1 = (fx1.ceil() as usize).min(src_len);
                    let mut row = Vec::with_capacity(sx1.saturating_sub(sx0));
                    let mut w_sum = 0.0f64;
                    for sx in sx0..sx1 {
                        let overlap = (fx1.min((sx + 1) as f64) - fx0.max(sx as f64)).max(0.0);
                        if overlap > 0.0 {
                            row.push((sx, overlap));
                            w_sum += overlap;
                        }
                    }
                    if w_sum > 0.0 {
                        let inv = 1.0 / w_sum;
                        for (_, w) in &mut row {
                            *w *= inv;
                        }
                    } else {
                        row.push((sx0.min(src_len - 1), 1.0));
                    }
                    spans.push(row);
                }
            }
        }
        Interpolation::Cubic => {
            let scale = src_len as f64 / dst_len as f64;
            let last = src_len as isize - 1;
            for dx in 0..dst_len {
                let fx = (dx as f64 + 0.5) * scale - 0.5;
                let ix = fx.floor() as isize;
                let t = fx - ix as f64;
                let w = cubic_weights(t);
                spans.push(vec![
                    ((ix - 1).clamp(0, last) as usize, w[0]),
                    (ix.clamp(0, last) as usize, w[1]),
                    ((ix + 1).clamp(0, last) as usize, w[2]),
                    ((ix + 2).clamp(0, last) as usize, w[3]),
                ]);
            }
        }
    }
    AxisTaps { spans }
}

/// Resizes a 2D `u8` grayscale image to `(dst_w, dst_h)`.
pub fn resize_u8(
    src: ArrayView2<'_, u8>,
    dst_w: usize,
    dst_h: usize,
    interp: Interpolation,
) -> CvResult<Array2<u8>> {
    let (src_h, src_w) = src.dim();
    if src_h == 0 || src_w == 0 || dst_w == 0 || dst_h == 0 {
        return Err(CvError::EmptyInput("resize_u8 zero dimension".into()));
    }
    if src_w == dst_w && src_h == dst_h {
        return Ok(src.to_owned());
    }
    let xtaps = build_axis_taps(src_w, dst_w, interp);
    let ytaps = build_axis_taps(src_h, dst_h, interp);

    if matches!(interp, Interpolation::Nearest) {
        let mut out = Array2::<u8>::zeros((dst_h, dst_w));
        for (dy, yspan) in ytaps.spans.iter().enumerate() {
            let sy = yspan[0].0;
            for (dx, xspan) in xtaps.spans.iter().enumerate() {
                out[[dy, dx]] = src[[sy, xspan[0].0]];
            }
        }
        return Ok(out);
    }

    let mut horiz = vec![0.0f64; src_h * dst_w];
    for sy in 0..src_h {
        let hrow = &mut horiz[sy * dst_w..(sy + 1) * dst_w];
        for (dx, xspan) in xtaps.spans.iter().enumerate() {
            let mut acc = 0.0f64;
            for &(sx, w) in xspan {
                acc += src[[sy, sx]] as f64 * w;
            }
            hrow[dx] = acc;
        }
    }

    let mut out = Array2::<u8>::zeros((dst_h, dst_w));
    for (dy, yspan) in ytaps.spans.iter().enumerate() {
        for dx in 0..dst_w {
            let mut acc = 0.0f64;
            for &(sy, w) in yspan {
                acc += horiz[sy * dst_w + dx] * w;
            }
            out[[dy, dx]] = acc.round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

/// Resizes a 3D `(H, W, C)` `u8` image to `(dst_w, dst_h)`.
pub fn resize_u8_hwc(
    src: ArrayView3<'_, u8>,
    dst_w: usize,
    dst_h: usize,
    interp: Interpolation,
) -> CvResult<Array3<u8>> {
    let (src_h, src_w, c) = src.dim();
    if src_h == 0 || src_w == 0 || c == 0 || dst_w == 0 || dst_h == 0 {
        return Err(CvError::EmptyInput("resize_u8_hwc zero dimension".into()));
    }
    if src_w == dst_w && src_h == dst_h {
        return Ok(src.to_owned());
    }
    if dst_w == src_w * 2 && dst_h == src_h * 2 && matches!(interp, Interpolation::Linear) {
        return Ok(resize_u8_hwc_2x_linear(src));
    }
    let xtaps = build_axis_taps(src_w, dst_w, interp);
    let ytaps = build_axis_taps(src_h, dst_h, interp);

    if matches!(interp, Interpolation::Nearest) {
        let mut out = Array3::<u8>::zeros((dst_h, dst_w, c));
        for (dy, yspan) in ytaps.spans.iter().enumerate() {
            let sy = yspan[0].0;
            for (dx, xspan) in xtaps.spans.iter().enumerate() {
                let sx = xspan[0].0;
                for ch in 0..c {
                    out[[dy, dx, ch]] = src[[sy, sx, ch]];
                }
            }
        }
        return Ok(out);
    }

    let mut horiz = vec![0.0f64; src_h * dst_w * c];
    for sy in 0..src_h {
        for (dx, xspan) in xtaps.spans.iter().enumerate() {
            let base = (sy * dst_w + dx) * c;
            for &(sx, w) in xspan {
                for ch in 0..c {
                    horiz[base + ch] += src[[sy, sx, ch]] as f64 * w;
                }
            }
        }
    }

    let mut out = Array3::<u8>::zeros((dst_h, dst_w, c));
    for (dy, yspan) in ytaps.spans.iter().enumerate() {
        for dx in 0..dst_w {
            let mut acc = [0.0f64; 4];
            if c <= 4 {
                for &(sy, w) in yspan {
                    let base = (sy * dst_w + dx) * c;
                    for ch in 0..c {
                        acc[ch] += horiz[base + ch] * w;
                    }
                }
                for (ch, &v) in acc.iter().take(c).enumerate() {
                    out[[dy, dx, ch]] = v.round().clamp(0.0, 255.0) as u8;
                }
            } else {
                for ch in 0..c {
                    let mut v = 0.0f64;
                    for &(sy, w) in yspan {
                        v += horiz[(sy * dst_w + dx) * c + ch] * w;
                    }
                    out[[dy, dx, ch]] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(out)
}

/// Resizes a 2D `f32` image to `(dst_w, dst_h)`.
pub fn resize_f32(
    src: ArrayView2<'_, f32>,
    dst_w: usize,
    dst_h: usize,
    interp: Interpolation,
) -> CvResult<Array2<f32>> {
    let (src_h, src_w) = src.dim();
    if src_h == 0 || src_w == 0 || dst_w == 0 || dst_h == 0 {
        return Err(CvError::EmptyInput("resize_f32 zero dimension".into()));
    }
    if src_w == dst_w && src_h == dst_h {
        return Ok(src.to_owned());
    }
    if dst_w == src_w * 2 && dst_h == src_h * 2 && matches!(interp, Interpolation::Linear) {
        return Ok(resize_f32_2x_linear(src));
    }
    let xtaps = build_axis_taps(src_w, dst_w, interp);
    let ytaps = build_axis_taps(src_h, dst_h, interp);

    let mut horiz = vec![0.0f64; src_h * dst_w];
    for sy in 0..src_h {
        let hrow = &mut horiz[sy * dst_w..(sy + 1) * dst_w];
        for (dx, xspan) in xtaps.spans.iter().enumerate() {
            let mut acc = 0.0f64;
            for &(sx, w) in xspan {
                acc += src[[sy, sx]] as f64 * w;
            }
            hrow[dx] = acc;
        }
    }

    let mut out = Array2::<f32>::zeros((dst_h, dst_w));
    for (dy, yspan) in ytaps.spans.iter().enumerate() {
        for dx in 0..dst_w {
            let mut acc = 0.0f64;
            for &(sy, w) in yspan {
                acc += horiz[sy * dst_w + dx] * w;
            }
            out[[dy, dx]] = acc as f32;
        }
    }
    Ok(out)
}

/// Computes horizontal 2x linear weights multiplied by 4 (`3*a + b`) for one HWC row.
#[inline]
fn fill_horiz_2x_linear_u16(src: ArrayView3<'_, u8>, sy: usize, dst: &mut [u16]) {
    let (_, src_w, c) = src.dim();
    let row = src.index_axis(ndarray::Axis(0), sy);
    if c == 3 {
        if let Some(s) = row.as_slice() {
            dst[0] = s[0] as u16 * 4;
            dst[1] = s[1] as u16 * 4;
            dst[2] = s[2] as u16 * 4;
            for sx in 0..(src_w - 1) {
                let a = &s[sx * 3..sx * 3 + 3];
                let b = &s[(sx + 1) * 3..(sx + 1) * 3 + 3];
                let o1 = (2 * sx + 1) * 3;
                let o2 = o1 + 3;
                for ch in 0..3 {
                    let va = a[ch] as u16;
                    let vb = b[ch] as u16;
                    dst[o1 + ch] = 3 * va + vb;
                    dst[o2 + ch] = va + 3 * vb;
                }
            }
            let plast = &s[(src_w - 1) * 3..(src_w - 1) * 3 + 3];
            let olast = (2 * src_w - 1) * 3;
            dst[olast] = plast[0] as u16 * 4;
            dst[olast + 1] = plast[1] as u16 * 4;
            dst[olast + 2] = plast[2] as u16 * 4;
            return;
        }
        // Safe strided 3-channel path (e.g. sliced from a 4-channel BGRA image).
        dst[0] = row[[0, 0]] as u16 * 4;
        dst[1] = row[[0, 1]] as u16 * 4;
        dst[2] = row[[0, 2]] as u16 * 4;
        for sx in 0..(src_w - 1) {
            let o1 = (2 * sx + 1) * 3;
            let o2 = o1 + 3;
            for ch in 0..3 {
                let va = row[[sx, ch]] as u16;
                let vb = row[[sx + 1, ch]] as u16;
                dst[o1 + ch] = 3 * va + vb;
                dst[o2 + ch] = va + 3 * vb;
            }
        }
        let olast = (2 * src_w - 1) * 3;
        dst[olast] = row[[src_w - 1, 0]] as u16 * 4;
        dst[olast + 1] = row[[src_w - 1, 1]] as u16 * 4;
        dst[olast + 2] = row[[src_w - 1, 2]] as u16 * 4;
        return;
    }

    for ch in 0..c {
        dst[ch] = row[[0, ch]] as u16 * 4;
    }
    for sx in 0..(src_w - 1) {
        let o1 = (2 * sx + 1) * c;
        let o2 = (2 * sx + 2) * c;
        for ch in 0..c {
            let va = row[[sx, ch]] as u16;
            let vb = row[[sx + 1, ch]] as u16;
            dst[o1 + ch] = 3 * va + vb;
            dst[o2 + ch] = va + 3 * vb;
        }
    }
    let olast = (2 * src_w - 1) * c;
    for ch in 0..c {
        dst[olast + ch] = row[[src_w - 1, ch]] as u16 * 4;
    }
}

/// Bit-exact 2x `Interpolation::Linear` HWC upscaler matching `build_axis_taps(Linear)`.
/// Dyadic 0.75/0.25 weights in f64 have zero rounding error, so `((9*a+3*b+3*c+d+8)>>4)`
/// matches `(f64::round)` for all u8 inputs while eliminating heap tap tables.
fn resize_u8_hwc_2x_linear(src: ArrayView3<'_, u8>) -> Array3<u8> {
    let (src_h, src_w, c) = src.dim();
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let row_elems = dst_w * c;
    let mut out = Array3::<u8>::zeros((dst_h, dst_w, c));
    let out_s = out.as_slice_mut().expect("contiguous output");

    let mut h_prev = vec![0u16; row_elems];
    let mut h_curr = vec![0u16; row_elems];
    fill_horiz_2x_linear_u16(src, 0, &mut h_prev);

    // dy = 0: vertical tap is [(0, 1.0)] == 4 * h0 / 4.
    let dst_r0 = &mut out_s[0..row_elems];
    for (d, &hv) in dst_r0.iter_mut().zip(h_prev.iter()) {
        *d = ((hv as u32 + 2) >> 2) as u8;
    }

    for sy in 0..(src_h - 1) {
        fill_horiz_2x_linear_u16(src, sy + 1, &mut h_curr);
        let dy1 = 2 * sy + 1;
        let dy2 = dy1 + 1;
        let out1 = &mut out_s[dy1 * row_elems..(dy1 + 1) * row_elems];
        for ((d, &a), &b) in out1.iter_mut().zip(h_prev.iter()).zip(h_curr.iter()) {
            *d = ((3 * a as u32 + b as u32 + 8) >> 4) as u8;
        }
        let out2 = &mut out_s[dy2 * row_elems..(dy2 + 1) * row_elems];
        for ((d, &a), &b) in out2.iter_mut().zip(h_prev.iter()).zip(h_curr.iter()) {
            *d = ((a as u32 + 3 * b as u32 + 8) >> 4) as u8;
        }
        std::mem::swap(&mut h_prev, &mut h_curr);
    }

    // dy = dst_h - 1: vertical tap is [(src_h - 1, 1.0)].
    let dst_rlast = &mut out_s[(dst_h - 1) * row_elems..dst_h * row_elems];
    for (d, &hv) in dst_rlast.iter_mut().zip(h_prev.iter()) {
        *d = ((hv as u32 + 2) >> 2) as u8;
    }
    out
}

#[inline]
fn fill_horiz_2x_linear_f64(src: ArrayView2<'_, f32>, sy: usize, dst: &mut [f64]) {
    let (_, src_w) = src.dim();
    let row = src.row(sy);
    if let Some(s) = row.as_slice() {
        dst[0] = s[0] as f64;
        for sx in 0..(src_w - 1) {
            let a = s[sx] as f64;
            let b = s[sx + 1] as f64;
            dst[2 * sx + 1] = a * 0.75 + b * 0.25;
            dst[2 * sx + 2] = a * 0.25 + b * 0.75;
        }
        dst[2 * src_w - 1] = s[src_w - 1] as f64;
    } else {
        dst[0] = row[0] as f64;
        for sx in 0..(src_w - 1) {
            let a = row[sx] as f64;
            let b = row[sx + 1] as f64;
            dst[2 * sx + 1] = a * 0.75 + b * 0.25;
            dst[2 * sx + 2] = a * 0.25 + b * 0.75;
        }
        dst[2 * src_w - 1] = row[src_w - 1] as f64;
    }
}

fn resize_f32_2x_linear(src: ArrayView2<'_, f32>) -> Array2<f32> {
    let (src_h, src_w) = src.dim();
    let dst_w = src_w * 2;
    let dst_h = src_h * 2;
    let mut out = Array2::<f32>::zeros((dst_h, dst_w));
    let out_s = out.as_slice_mut().expect("contiguous output");

    let mut h_prev = vec![0.0f64; dst_w];
    let mut h_curr = vec![0.0f64; dst_w];
    fill_horiz_2x_linear_f64(src, 0, &mut h_prev);

    let r0 = &mut out_s[0..dst_w];
    for (d, &hv) in r0.iter_mut().zip(h_prev.iter()) {
        *d = hv as f32;
    }

    for sy in 0..(src_h - 1) {
        fill_horiz_2x_linear_f64(src, sy + 1, &mut h_curr);
        let dy1 = 2 * sy + 1;
        let dy2 = dy1 + 1;
        let out1 = &mut out_s[dy1 * dst_w..(dy1 + 1) * dst_w];
        for ((d, &a), &b) in out1.iter_mut().zip(h_prev.iter()).zip(h_curr.iter()) {
            *d = (a * 0.75 + b * 0.25) as f32;
        }
        let out2 = &mut out_s[dy2 * dst_w..(dy2 + 1) * dst_w];
        for ((d, &a), &b) in out2.iter_mut().zip(h_prev.iter()).zip(h_curr.iter()) {
            *d = (a * 0.25 + b * 0.75) as f32;
        }
        std::mem::swap(&mut h_prev, &mut h_curr);
    }

    let rlast = &mut out_s[(dst_h - 1) * dst_w..dst_h * dst_w];
    for (d, &hv) in rlast.iter_mut().zip(h_prev.iter()) {
        *d = hv as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    #[test]
    fn inter_area_upscale_matches_opencv_block_and_fractional() {
        let a = array![[10u8, 20, 30], [40, 50, 60]];
        let up2 = resize_u8(a.view(), 6, 4, Interpolation::Area).unwrap();
        assert_eq!(
            up2,
            array![
                [10, 10, 20, 20, 30, 30],
                [10, 10, 20, 20, 30, 30],
                [40, 40, 50, 50, 60, 60],
                [40, 40, 50, 50, 60, 60]
            ]
        );
        let up_frac = resize_u8(a.view(), 7, 5, Interpolation::Area).unwrap();
        assert_eq!(up_frac.row(0).to_vec(), vec![10, 10, 17, 20, 23, 30, 30]);
        assert_eq!(up_frac.row(2).to_vec(), vec![25, 25, 32, 35, 38, 45, 45]);
    }

    #[test]
    fn inter_area_downscale_averages_blocks() {
        let a = array![
            [10u8, 20, 30, 40],
            [10, 20, 30, 40],
            [50, 60, 70, 80],
            [50, 60, 70, 80]
        ];
        let down = resize_u8(a.view(), 2, 2, Interpolation::Area).unwrap();
        assert_eq!(down, array![[15, 35], [55, 75]]);
    }

    #[test]
    fn copy_make_border_wrap_y() {
        let a = array![[1u8, 2], [3, 4], [5, 6]];
        let padded = copy_make_border_u8(a.view(), 2, 2, 0, 0, BorderMode::Wrap).unwrap();
        assert_eq!(padded.row(0).to_vec(), vec![3, 4]);
        assert_eq!(padded.row(1).to_vec(), vec![5, 6]);
        assert_eq!(padded.row(5).to_vec(), vec![1, 2]);
        assert_eq!(padded.row(6).to_vec(), vec![3, 4]);
    }

    #[test]
    fn fast_2x_linear_matches_generic_axis_taps() {
        // Test odd/even/singleton dimensions, multi-channel, and strided BGR view (from BGRA crop).
        let bgra = Array3::from_shape_fn((29, 31, 4), |(y, x, c)| {
            ((y * 37 + x * 53 + c * 97) % 256) as u8
        });
        let bgr_strided = bgra.slice(s![3..28, 2..29, 0..3]);
        let fast_hwc = resize_u8_hwc_2x_linear(bgr_strided);

        // Reference via explicit AxisTaps path (generic dimensions by validating small and odd shapes too).
        let (sh, sw, c) = bgr_strided.dim();
        let xtaps = build_axis_taps(sw, sw * 2, Interpolation::Linear);
        let ytaps = build_axis_taps(sh, sh * 2, Interpolation::Linear);
        let mut horiz = vec![0.0f64; sh * sw * 2 * c];
        for sy in 0..sh {
            for (dx, xspan) in xtaps.spans.iter().enumerate() {
                let base = (sy * sw * 2 + dx) * c;
                for &(sx, w) in xspan {
                    for ch in 0..c {
                        horiz[base + ch] += bgr_strided[[sy, sx, ch]] as f64 * w;
                    }
                }
            }
        }
        let mut ref_hwc = Array3::<u8>::zeros((sh * 2, sw * 2, c));
        for (dy, yspan) in ytaps.spans.iter().enumerate() {
            for dx in 0..sw * 2 {
                let mut acc = [0.0f64; 3];
                for &(sy, w) in yspan {
                    let base = (sy * sw * 2 + dx) * c;
                    for ch in 0..c {
                        acc[ch] += horiz[base + ch] * w;
                    }
                }
                for ch in 0..3 {
                    ref_hwc[[dy, dx, ch]] = acc[ch].round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        assert_eq!(fast_hwc, ref_hwc);

        // Verify singleton dimension edge case (1x1 -> 2x2)
        let one = Array3::from_shape_vec((1, 1, 3), vec![12u8, 200, 255]).unwrap();
        let up_one = resize_u8_hwc(one.view(), 2, 2, Interpolation::Linear).unwrap();
        assert_eq!(up_one[[0, 0, 1]], 200);
        assert_eq!(up_one[[1, 1, 2]], 255);

        // Verify resize_f32 bit-exactness with generic build_axis_taps
        let f_in = Array2::from_shape_fn((17, 19), |(y, x)| (y * 19 + x) as f32 * 0.137 - 5.0);
        let fast_f32 = resize_f32_2x_linear(f_in.view());
        let fx = build_axis_taps(19, 38, Interpolation::Linear);
        let fy = build_axis_taps(17, 34, Interpolation::Linear);
        let mut fh = vec![0.0f64; 17 * 38];
        for sy in 0..17 {
            for (dx, xspan) in fx.spans.iter().enumerate() {
                fh[sy * 38 + dx] = xspan.iter().map(|&(sx, w)| f_in[[sy, sx]] as f64 * w).sum();
            }
        }
        for (dy, yspan) in fy.spans.iter().enumerate() {
            for dx in 0..38 {
                let v: f64 = yspan.iter().map(|&(sy, w)| fh[sy * 38 + dx] * w).sum();
                assert_eq!(fast_f32[[dy, dx]].to_bits(), (v as f32).to_bits());
            }
        }
    }
}
