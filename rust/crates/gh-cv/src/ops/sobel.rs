//! Unnormalized Sobel (3x3 and general odd ksize) and Scharr (3x3, gain 16)
//! derivatives with `BORDER_REFLECT_101`, plus gradient magnitude.

use crate::ops::{CvError, CvResult, border_reflect_101};
use ndarray::{Array2, ArrayView2};

fn sobel_kernels(order: usize, ksize: usize) -> CvResult<Vec<f64>> {
    if order > 2 {
        return Err(CvError::InvalidArgument("derivative order must be <= 2".into()));
    }
    if ksize != 1 && ksize != 3 && ksize != 5 && ksize != 7 {
        return Err(CvError::InvalidArgument(format!(
            "sobel ksize must be 1, 3, 5, or 7, got {ksize}"
        )));
    }
    if ksize == 1 {
        return match order {
            0 => Ok(vec![1.0]),
            1 => Ok(vec![-1.0, 0.0, 1.0]),
            2 => Ok(vec![1.0, -2.0, 1.0]),
            _ => unreachable!(),
        };
    }
    // Pascal triangle / finite-difference convolution matching OpenCV getSobelKernels
    let mut k = vec![0.0f64; ksize + 1];
    k[0] = 1.0;
    for i in 1..=(ksize - order - 1) {
        for j in (1..=i).rev() {
            k[j] += k[j - 1];
        }
    }
    k.truncate(ksize);
    for _ in 0..order {
        let mut next = vec![0.0f64; ksize];
        for j in 0..ksize {
            let prev = if j > 0 { k[j - 1] } else { 0.0 };
            let next_v = if j + 1 < ksize { k[j + 1] } else { 0.0 };
            let _ = next_v;
            // Forward/backward difference: [1, -1] convolved: k'[j] = k[j-1] - k[j] -> sign matches OpenCV [-1, 0, 1]
            let left = if j > 0 { k[j - 1] } else { 0.0 };
            next[j] = k[j] - left;
            let _ = prev;
        }
        // Standard OpenCV Sobel sign: [-1, 0, 1] for dx=1
        // With k[j] - k[j-1]: for [1,1] -> [1, 0, -1], so negate for order==1.
        k = next;
    }
    if order == 1 {
        for v in &mut k {
            *v = -*v;
        }
    }
    Ok(k)
}

fn convolve_sep_reflect101(
    src_f32: ArrayView2<'_, f32>,
    kx: &[f64],
    ky: &[f64],
) -> Array2<f32> {
    let (h, w) = src_f32.dim();
    let rx = (kx.len() / 2) as isize;
    let ry = (ky.len() / 2) as isize;
    let mut tmp = vec![0.0f64; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for (i, &kv) in kx.iter().enumerate() {
                let sx = border_reflect_101(x as isize + i as isize - rx, w);
                acc += src_f32[[y, sx]] as f64 * kv;
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f64;
            for (j, &kv) in ky.iter().enumerate() {
                let sy = border_reflect_101(y as isize + j as isize - ry, h);
                acc += tmp[sy * w + x] * kv;
            }
            out[[y, x]] = acc as f32;
        }
    }
    out
}

/// Fast 3x3 Sobel `dx=1, dy=0` on `u8` (`BORDER_REFLECT_101`, `[-1,0,1] x [1,2,1]`).
pub fn sobel_3x3_dx_u8_f32(src: ArrayView2<'_, u8>) -> CvResult<Array2<f32>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("sobel_3x3_dx_u8_f32 empty".into()));
    }
    let src_owned;
    let s = if let Some(sl) = src.as_slice() {
        sl
    } else {
        src_owned = src.to_owned();
        src_owned.as_slice().unwrap()
    };
    let mut tmp = vec![0.0f32; h * w];
    for y in 0..h {
        let srow = &s[y * w..(y + 1) * w];
        let trow = &mut tmp[y * w..(y + 1) * w];
        if w >= 2 {
            // x=0: xm=1, xp=1 -> 0; x=w-1: xm=w-2, xp=w-2 -> 0
            trow[0] = 0.0;
            for x in 1..(w - 1) {
                trow[x] = srow[x + 1] as f32 - srow[x - 1] as f32;
            }
            trow[w - 1] = 0.0;
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    let out_s = out.as_slice_mut().unwrap();
    for y in 0..h {
        let (ym, yp) = if h == 1 {
            (0, 0)
        } else if y == 0 {
            (1, 1)
        } else if y + 1 == h {
            (h - 2, h - 2)
        } else {
            (y - 1, y + 1)
        };
        let tm = &tmp[ym * w..(ym + 1) * w];
        let tc = &tmp[y * w..(y + 1) * w];
        let tp = &tmp[yp * w..(yp + 1) * w];
        let orow = &mut out_s[y * w..(y + 1) * w];
        for (((dst, &vm), &vc), &vp) in orow.iter_mut().zip(tm.iter()).zip(tc.iter()).zip(tp.iter()) {
            *dst = vm + 2.0 * vc + vp;
        }
    }
    Ok(out)
}

/// Fast 3x3 Sobel `dx=0, dy=1` on `u8` (`BORDER_REFLECT_101`, `[1,2,1] x [-1,0,1]`).
pub fn sobel_3x3_dy_u8_f32(src: ArrayView2<'_, u8>) -> CvResult<Array2<f32>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("sobel_3x3_dy_u8_f32 empty".into()));
    }
    let src_owned;
    let s = if let Some(sl) = src.as_slice() {
        sl
    } else {
        src_owned = src.to_owned();
        src_owned.as_slice().unwrap()
    };
    let mut tmp = vec![0.0f32; h * w];
    for y in 0..h {
        let srow = &s[y * w..(y + 1) * w];
        let trow = &mut tmp[y * w..(y + 1) * w];
        if w == 1 {
            trow[0] = 4.0 * srow[0] as f32;
        } else {
            trow[0] = 2.0 * srow[0] as f32 + 2.0 * srow[1] as f32;
            for x in 1..(w - 1) {
                trow[x] = srow[x - 1] as f32 + 2.0 * srow[x] as f32 + srow[x + 1] as f32;
            }
            trow[w - 1] = 2.0 * srow[w - 2] as f32 + 2.0 * srow[w - 1] as f32;
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    let out_s = out.as_slice_mut().unwrap();
    if h >= 3 {
        // y=0 and y=h-1: ym == yp -> out = 0 (already zero-initialized)
        for y in 1..(h - 1) {
            let tm = &tmp[(y - 1) * w..y * w];
            let tp = &tmp[(y + 1) * w..(y + 2) * w];
            let orow = &mut out_s[y * w..(y + 1) * w];
            for ((dst, &vp), &vm) in orow.iter_mut().zip(tp.iter()).zip(tm.iter()) {
                *dst = vp - vm;
            }
        }
    }
    Ok(out)
}

/// Fast 3x3 Sobel `(dx, dy)` pair on `u8`.
pub fn sobel_3x3_u8_f32(src: ArrayView2<'_, u8>) -> CvResult<(Array2<f32>, Array2<f32>)> {
    Ok((sobel_3x3_dx_u8_f32(src)?, sobel_3x3_dy_u8_f32(src)?))
}

/// General Sobel filter on `f32` (`dx`, `dy` in `0..=2`, `ksize` in `{1, 3, 5, 7}`).
pub fn sobel_f32(
    src: ArrayView2<'_, f32>,
    dx: usize,
    dy: usize,
    ksize: usize,
) -> CvResult<Array2<f32>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("sobel_f32 empty".into()));
    }
    let kx = sobel_kernels(dx, ksize)?;
    let ky = sobel_kernels(dy, ksize)?;
    Ok(convolve_sep_reflect101(src, &kx, &ky))
}

/// 3x3 Scharr `dx=1, dy=0` on `f32` (`[-1,0,1] x [3,10,3]`, gain 16).
pub fn scharr_dx_f32(src: ArrayView2<'_, f32>) -> CvResult<Array2<f32>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("scharr_dx_f32 empty".into()));
    }
    let src_owned;
    let s = if let Some(sl) = src.as_slice() {
        sl
    } else {
        src_owned = src.to_owned();
        src_owned.as_slice().unwrap()
    };
    let mut tmp = vec![0.0f32; h * w];
    if w >= 3 {
        for y in 0..h {
            let srow = &s[y * w..(y + 1) * w];
            let trow = &mut tmp[y * w..(y + 1) * w];
            for ((dst, &vp), &vm) in trow[1..w - 1]
                .iter_mut()
                .zip(srow[2..w].iter())
                .zip(srow[0..w - 2].iter())
            {
                *dst = vp - vm;
            }
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    let out_s = out.as_slice_mut().unwrap();
    for y in 0..h {
        let (ym, yp) = if h == 1 {
            (0, 0)
        } else if y == 0 {
            (1, 1)
        } else if y + 1 == h {
            (h - 2, h - 2)
        } else {
            (y - 1, y + 1)
        };
        let tm = &tmp[ym * w..(ym + 1) * w];
        let tc = &tmp[y * w..(y + 1) * w];
        let tp = &tmp[yp * w..(yp + 1) * w];
        let orow = &mut out_s[y * w..(y + 1) * w];
        for (((dst, &vm), &vc), &vp) in orow.iter_mut().zip(tm.iter()).zip(tc.iter()).zip(tp.iter()) {
            *dst = 3.0 * vm + 10.0 * vc + 3.0 * vp;
        }
    }
    Ok(out)
}

/// 3x3 Scharr `dx=0, dy=1` on `f32` (`[3,10,3] x [-1,0,1]`, gain 16).
pub fn scharr_dy_f32(src: ArrayView2<'_, f32>) -> CvResult<Array2<f32>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("scharr_dy_f32 empty".into()));
    }
    let src_owned;
    let s = if let Some(sl) = src.as_slice() {
        sl
    } else {
        src_owned = src.to_owned();
        src_owned.as_slice().unwrap()
    };
    let mut tmp = vec![0.0f32; h * w];
    for y in 0..h {
        let srow = &s[y * w..(y + 1) * w];
        let trow = &mut tmp[y * w..(y + 1) * w];
        if w == 1 {
            trow[0] = 16.0 * srow[0];
        } else {
            trow[0] = 10.0 * srow[0] + 6.0 * srow[1];
            for (((dst, &vm), &vc), &vp) in trow[1..w - 1]
                .iter_mut()
                .zip(srow[0..w - 2].iter())
                .zip(srow[1..w - 1].iter())
                .zip(srow[2..w].iter())
            {
                *dst = 3.0 * vm + 10.0 * vc + 3.0 * vp;
            }
            trow[w - 1] = 6.0 * srow[w - 2] + 10.0 * srow[w - 1];
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    let out_s = out.as_slice_mut().unwrap();
    if h >= 3 {
        // y=0 and y=h-1: ym == yp -> out = 0 (already zero-initialized)
        for y in 1..(h - 1) {
            let tm = &tmp[(y - 1) * w..y * w];
            let tp = &tmp[(y + 1) * w..(y + 2) * w];
            let orow = &mut out_s[y * w..(y + 1) * w];
            for ((dst, &vp), &vm) in orow.iter_mut().zip(tp.iter()).zip(tm.iter()) {
                *dst = vp - vm;
            }
        }
    }
    Ok(out)
}

/// 3x3 Scharr `dx=0, dy=1` directly on `u8`.
pub fn scharr_dy_u8_f32(src: ArrayView2<'_, u8>) -> CvResult<Array2<f32>> {
    let (h, w) = src.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("scharr_dy_u8_f32 empty".into()));
    }
    let src_owned;
    let s = if let Some(sl) = src.as_slice() {
        sl
    } else {
        src_owned = src.to_owned();
        src_owned.as_slice().unwrap()
    };
    let mut tmp = vec![0.0f32; h * w];
    for y in 0..h {
        let srow = &s[y * w..(y + 1) * w];
        let trow = &mut tmp[y * w..(y + 1) * w];
        if w == 1 {
            trow[0] = 16.0 * srow[0] as f32;
        } else {
            trow[0] = 10.0 * srow[0] as f32 + 6.0 * srow[1] as f32;
            for (((dst, &vm), &vc), &vp) in trow[1..w - 1]
                .iter_mut()
                .zip(srow[0..w - 2].iter())
                .zip(srow[1..w - 1].iter())
                .zip(srow[2..w].iter())
            {
                *dst = 3.0 * vm as f32 + 10.0 * vc as f32 + 3.0 * vp as f32;
            }
            trow[w - 1] = 6.0 * srow[w - 2] as f32 + 10.0 * srow[w - 1] as f32;
        }
    }
    let mut out = Array2::<f32>::zeros((h, w));
    let out_s = out.as_slice_mut().unwrap();
    if h >= 3 {
        for y in 1..(h - 1) {
            let tm = &tmp[(y - 1) * w..y * w];
            let tp = &tmp[(y + 1) * w..(y + 2) * w];
            let orow = &mut out_s[y * w..(y + 1) * w];
            for ((dst, &vp), &vm) in orow.iter_mut().zip(tp.iter()).zip(tm.iter()) {
                *dst = vp - vm;
            }
        }
    }
    Ok(out)
}

/// Computes `sqrt(gx^2 + gy^2)` elementwise matching `cv2.magnitude`.
pub fn magnitude_f32(gx: ArrayView2<'_, f32>, gy: ArrayView2<'_, f32>) -> CvResult<Array2<f32>> {
    if gx.dim() != gy.dim() {
        return Err(CvError::ShapeMismatch("magnitude_f32 shapes differ".into()));
    }
    let mut out = Array2::<f32>::zeros(gx.dim());
    if let (Some(sx), Some(sy), Some(sd)) = (gx.as_slice(), gy.as_slice(), out.as_slice_mut()) {
        for ((dst, &vx), &vy) in sd.iter_mut().zip(sx.iter()).zip(sy.iter()) {
            *dst = (vx * vx + vy * vy).sqrt();
        }
    } else {
        for ((dst, &vx), &vy) in out.iter_mut().zip(gx.iter()).zip(gy.iter()) {
            *dst = (vx * vx + vy * vy).sqrt();
        }
    }
    Ok(out)
}

/// Convenience: 3x3 Sobel gradient magnitude on `u8` grayscale.
pub fn sobel_magnitude_u8_f32(src: ArrayView2<'_, u8>) -> CvResult<Array2<f32>> {
    let (gx, gy) = sobel_3x3_u8_f32(src)?;
    magnitude_f32(gx.view(), gy.view())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sobel_and_scharr_match_opencv_probe() {
        let mut img = Array2::<u8>::zeros((5, 5));
        for y in 0..5 {
            for x in 0..5 {
                img[[y, x]] = (y * 16 + x) as u8;
            }
        }
        let sx = sobel_3x3_dx_u8_f32(img.view()).unwrap();
        assert_eq!(sx.row(1).to_vec(), vec![0.0, 8.0, 8.0, 8.0, 0.0]);

        let sy = scharr_dy_u8_f32(img.view()).unwrap();
        assert_eq!(sy.row(1).to_vec(), vec![512.0, 512.0, 512.0, 512.0, 512.0]);
    }
}
