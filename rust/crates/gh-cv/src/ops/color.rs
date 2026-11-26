//! Color conversions, `FrameView` / `image` / `ndarray` adapters, thresholding, LUT,
//! pixel arithmetic, masking, and 64-bin RGB Bhattacharyya histogram helpers.

use crate::ops::{CvError, CvResult, ThresholdType};
use gh_core::frame::{FrameBuf, FrameView, PixelFormat};
use image::{GrayImage, RgbImage, RgbaImage};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

/// Converts a `FrameView` (Gray8) to a contiguous `Array2<u8>`.
pub fn frame_view_to_gray_array(view: FrameView<'_>) -> CvResult<Array2<u8>> {
    let w = view.width as usize;
    let h = view.height as usize;
    if w == 0 || h == 0 {
        return Err(CvError::EmptyInput("empty FrameView".into()));
    }
    let ch = view.format.channels();
    if view.stride < w * ch {
        return Err(CvError::ShapeMismatch("FrameView stride too small".into()));
    }
    let needed = (h - 1) * view.stride + w * ch;
    if view.data.len() < needed {
        return Err(CvError::ShapeMismatch("FrameView buffer truncated".into()));
    }
    let mut out = Array2::<u8>::zeros((h, w));
    let out_slice = out.as_slice_mut().expect("contiguous");
    match view.format {
        PixelFormat::Gray8 => {
            for y in 0..h {
                let row = &view.data[y * view.stride..y * view.stride + w];
                out_slice[y * w..(y + 1) * w].copy_from_slice(row);
            }
        }
        PixelFormat::Bgr8 | PixelFormat::Bgra8 => {
            for y in 0..h {
                let row = &view.data[y * view.stride..y * view.stride + w * ch];
                let dst = &mut out_slice[y * w..(y + 1) * w];
                for x in 0..w {
                    let px = &row[x * ch..];
                    dst[x] = bgr_px_to_gray(px[0], px[1], px[2]);
                }
            }
        }
        PixelFormat::Rgba8 => {
            for y in 0..h {
                let row = &view.data[y * view.stride..y * view.stride + w * 4];
                let dst = &mut out_slice[y * w..(y + 1) * w];
                for x in 0..w {
                    let px = &row[x * 4..];
                    dst[x] = bgr_px_to_gray(px[2], px[1], px[0]);
                }
            }
        }
    }
    Ok(out)
}

/// Converts a `FrameView` (any format) to a contiguous `Array3<u8>` in BGR or BGRA order.
pub fn frame_view_to_array3(view: FrameView<'_>) -> CvResult<Array3<u8>> {
    let w = view.width as usize;
    let h = view.height as usize;
    if w == 0 || h == 0 {
        return Err(CvError::EmptyInput("empty FrameView".into()));
    }
    let ch = view.format.channels();
    if view.stride < w * ch {
        return Err(CvError::ShapeMismatch("FrameView stride too small".into()));
    }
    let needed = (h - 1) * view.stride + w * ch;
    if view.data.len() < needed {
        return Err(CvError::ShapeMismatch("FrameView buffer truncated".into()));
    }
    let mut out = Array3::<u8>::zeros((h, w, ch));
    let out_slice = out.as_slice_mut().expect("contiguous");
    for y in 0..h {
        let src_row = &view.data[y * view.stride..y * view.stride + w * ch];
        let dst_row = &mut out_slice[y * w * ch..(y + 1) * w * ch];
        dst_row.copy_from_slice(src_row);
    }
    Ok(out)
}

/// Converts a `FrameView` into an owned `Array3<u8>` of shape `(H, W, 3)` in BGR channel order.
pub fn frame_view_to_bgr(view: FrameView<'_>) -> CvResult<Array3<u8>> {
    let w = view.width as usize;
    let h = view.height as usize;
    if w == 0 || h == 0 {
        return Err(CvError::EmptyInput("empty FrameView".into()));
    }
    let ch = view.format.channels();
    if view.stride < w * ch || view.data.len() < (h - 1) * view.stride + w * ch {
        return Err(CvError::ShapeMismatch("invalid FrameView stride/len".into()));
    }
    let mut out = Array3::<u8>::zeros((h, w, 3));
    let dst_all = out.as_slice_mut().expect("contiguous");
    for y in 0..h {
        let src = &view.data[y * view.stride..y * view.stride + w * ch];
        let dst = &mut dst_all[y * w * 3..(y + 1) * w * 3];
        match view.format {
            PixelFormat::Gray8 => {
                for x in 0..w {
                    let v = src[x];
                    dst[x * 3] = v;
                    dst[x * 3 + 1] = v;
                    dst[x * 3 + 2] = v;
                }
            }
            PixelFormat::Bgr8 => dst.copy_from_slice(src),
            PixelFormat::Bgra8 => {
                for x in 0..w {
                    dst[x * 3] = src[x * 4];
                    dst[x * 3 + 1] = src[x * 4 + 1];
                    dst[x * 3 + 2] = src[x * 4 + 2];
                }
            }
            PixelFormat::Rgba8 => {
                for x in 0..w {
                    dst[x * 3] = src[x * 4 + 2];
                    dst[x * 3 + 1] = src[x * 4 + 1];
                    dst[x * 3 + 2] = src[x * 4];
                }
            }
        }
    }
    Ok(out)
}

/// Zero-copy BGR view of a `Bgr8` frame (row stride honoured); `None` for any other format or a
/// malformed view. Hot paths use it instead of [`frame_view_to_bgr`], which copies ~11 MB per
/// 1440p frame.
pub fn frame_view_as_bgr(view: FrameView<'_>) -> Option<ArrayView3<'_, u8>> {
    use ndarray::ShapeBuilder;
    let (w, h) = (view.width as usize, view.height as usize);
    if view.format != PixelFormat::Bgr8 || w == 0 || h == 0 || view.stride < w * 3 {
        return None;
    }
    let data = view.data.get(..(h - 1) * view.stride + w * 3)?;
    ArrayView3::from_shape((h, w, 3).strides((view.stride, 3, 1)), data).ok()
}

/// Converts an `Array2<u8>` into a `FrameBuf` (`Gray8`).
pub fn gray_array_to_frame_buf(arr: ArrayView2<'_, u8>) -> FrameBuf {
    let (h, w) = arr.dim();
    let mut buf = FrameBuf::new_zeroed(w as u32, h as u32, PixelFormat::Gray8);
    if let Some(s) = arr.as_slice() {
        buf.data.copy_from_slice(s);
    } else {
        for (y, row) in arr.rows().into_iter().enumerate() {
            for (x, &v) in row.iter().enumerate() {
                buf.data[y * w + x] = v;
            }
        }
    }
    buf
}

/// Converts an `Array2<u8>` to an `image::GrayImage`.
pub fn array2_to_gray_image(arr: ArrayView2<'_, u8>) -> CvResult<GrayImage> {
    let (h, w) = arr.dim();
    let raw: Vec<u8> = arr.iter().copied().collect();
    GrayImage::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| CvError::ShapeMismatch("GrayImage dimension mismatch".into()))
}

/// Converts an `image::GrayImage` to an `Array2<u8>`.
pub fn gray_image_to_array2(img: &GrayImage) -> Array2<u8> {
    let (w, h) = img.dimensions();
    Array2::from_shape_vec((h as usize, w as usize), img.as_raw().clone())
        .expect("GrayImage raw length matches dimensions")
}

/// Converts an `image::RgbImage` to a BGR `Array3<u8>`.
pub fn rgb_image_to_bgr_array3(img: &RgbImage) -> Array3<u8> {
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    let mut bgr = Vec::with_capacity(raw.len());
    for px in raw.as_chunks::<3>().0 {
        bgr.push(px[2]);
        bgr.push(px[1]);
        bgr.push(px[0]);
    }
    Array3::from_shape_vec((h as usize, w as usize, 3), bgr).expect("RgbImage dims")
}

/// Converts an `image::RgbaImage` to a BGRA `Array3<u8>`.
pub fn rgba_image_to_bgra_array3(img: &RgbaImage) -> Array3<u8> {
    let (w, h) = img.dimensions();
    let raw = img.as_raw();
    let mut bgra = Vec::with_capacity(raw.len());
    for px in raw.as_chunks::<4>().0 {
        bgra.push(px[2]);
        bgra.push(px[1]);
        bgra.push(px[0]);
        bgra.push(px[3]);
    }
    Array3::from_shape_vec((h as usize, w as usize, 4), bgra).expect("RgbaImage dims")
}

/// Exact OpenCV 15-bit BT.601 fixed-point `BGR -> Gray` conversion.
#[inline]
pub fn bgr_px_to_gray(b: u8, g: u8, r: u8) -> u8 {
    (((3735u32 * b as u32) + (19235u32 * g as u32) + (9798u32 * r as u32) + 16_384) >> 15)
        .min(255) as u8
}

/// Converts BGR or BGRA `(H, W, 3|4)` array to `(H, W)` grayscale matching `cv2.cvtColor(COLOR_BGR2GRAY)`.
pub fn bgr_to_gray(src: ArrayView3<'_, u8>) -> CvResult<Array2<u8>> {
    let (h, w, c) = src.dim();
    if c < 3 {
        return Err(CvError::UnsupportedFormat(format!(
            "bgr_to_gray expects >= 3 channels, got {c}"
        )));
    }
    let mut out = Array2::<u8>::zeros((h, w));
    let d = out.as_slice_mut().expect("contiguous");
    if let Some(s) = src.as_slice() {
        for (dst, px) in d.iter_mut().zip(s.chunks_exact(c)) {
            *dst = bgr_px_to_gray(px[0], px[1], px[2]);
        }
    } else {
        for y in 0..h {
            let row = src.index_axis(ndarray::Axis(0), y);
            let dst_row = &mut d[y * w..(y + 1) * w];
            if let Some(rs) = row.as_slice() {
                for (dst, px) in dst_row.iter_mut().zip(rs.chunks_exact(c)) {
                    *dst = bgr_px_to_gray(px[0], px[1], px[2]);
                }
            } else {
                for (x, dst) in dst_row.iter_mut().enumerate() {
                    *dst = bgr_px_to_gray(row[[x, 0]], row[[x, 1]], row[[x, 2]]);
                }
            }
        }
    }
    Ok(out)
}

/// Converts RGB or RGBA `(H, W, 3|4)` array to `(H, W)` grayscale matching `cv2.cvtColor(COLOR_RGB2GRAY)`.
pub fn rgb_to_gray(src: ArrayView3<'_, u8>) -> CvResult<Array2<u8>> {
    let (h, w, c) = src.dim();
    if c < 3 {
        return Err(CvError::UnsupportedFormat(format!(
            "rgb_to_gray expects >= 3 channels, got {c}"
        )));
    }
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            out[[y, x]] = bgr_px_to_gray(src[[y, x, 2]], src[[y, x, 1]], src[[y, x, 0]]);
        }
    }
    Ok(out)
}

/// Strips alpha channel from BGRA `(H, W, 4)` -> BGR `(H, W, 3)`.
pub fn bgra_to_bgr(src: ArrayView3<'_, u8>) -> CvResult<Array3<u8>> {
    let (h, w, c) = src.dim();
    if c < 3 {
        return Err(CvError::UnsupportedFormat(format!(
            "bgra_to_bgr expects >= 3 channels, got {c}"
        )));
    }
    let mut out = Array3::<u8>::zeros((h, w, 3));
    if let (Some(s), Some(d)) = (src.as_slice(), out.as_slice_mut()) {
        for (dst, px) in d.as_chunks_mut::<3>().0.iter_mut().zip(s.chunks_exact(c)) {
            dst[0] = px[0];
            dst[1] = px[1];
            dst[2] = px[2];
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                out[[y, x, 0]] = src[[y, x, 0]];
                out[[y, x, 1]] = src[[y, x, 1]];
                out[[y, x, 2]] = src[[y, x, 2]];
            }
        }
    }
    Ok(out)
}

/// Swaps B and R channels (`BGR <-> RGB`, preserving alpha if 4 channels).
pub fn swap_rb(src: ArrayView3<'_, u8>) -> CvResult<Array3<u8>> {
    let (h, w, c) = src.dim();
    if c != 3 && c != 4 {
        return Err(CvError::UnsupportedFormat(format!(
            "swap_rb expects 3 or 4 channels, got {c}"
        )));
    }
    let mut out = Array3::<u8>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            out[[y, x, 0]] = src[[y, x, 2]];
            out[[y, x, 1]] = src[[y, x, 1]];
            out[[y, x, 2]] = src[[y, x, 0]];
            if c == 4 {
                out[[y, x, 3]] = src[[y, x, 3]];
            }
        }
    }
    Ok(out)
}

const INV_30_F32: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 1usize;
    while i < 256 {
        t[i] = 30.0 / (i as f32);
        i += 1;
    }
    t
};

const RECIP_U32_SHIFT32: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut i = 1usize;
    while i < 256 {
        t[i] = (1u64 << 32).div_ceil(i as u64);
        i += 1;
    }
    t
};

/// Exact OpenCV `COLOR_BGR2HSV` (`H in [0, 179]`, `S, V in [0, 255]`). Accepts 3- or 4-ch BGR/BGRA.
pub fn bgr_to_hsv(src: ArrayView3<'_, u8>) -> CvResult<Array3<u8>> {
    let (h, w, c) = src.dim();
    if c < 3 {
        return Err(CvError::UnsupportedFormat(format!(
            "bgr_to_hsv expects >= 3 channels, got {c}"
        )));
    }
    let mut out = Array3::<u8>::zeros((h, w, 3));
    #[inline(always)]
    fn process(b: u8, g: u8, r: u8) -> [u8; 3] {
        let vmax = b.max(g).max(r);
        let vmin = b.min(g).min(r);
        let diff = vmax - vmin;
        let v = vmax;
        let s = if vmax == 0 {
            0u8
        } else {
            let num = 255u64 * diff as u64 + (vmax as u64 >> 1);
            ((num * RECIP_U32_SHIFT32[vmax as usize]) >> 32).min(255) as u8
        };
        let h_val = if diff == 0 {
            0u8
        } else {
            let inv = INV_30_F32[diff as usize];
            let (base, delta) = if vmax == r {
                (180.0f32, (g as i32 - b as i32) as f32)
            } else if vmax == g {
                (240.0f32, (b as i32 - r as i32) as f32)
            } else {
                (300.0f32, (r as i32 - g as i32) as f32)
            };
            // base + delta * inv is in [150.0, 330.0], strictly positive
            let h_shifted = (base + delta * inv + 0.5) as i32;
            (if h_shifted >= 180 {
                h_shifted - 180
            } else {
                h_shifted
            }) as u8
        };
        [h_val, s, v]
    }

    if let (Some(s), Some(d)) = (src.as_slice(), out.as_slice_mut()) {
        for (dst, px) in d.as_chunks_mut::<3>().0.iter_mut().zip(s.chunks_exact(c)) {
            let hsv = process(px[0], px[1], px[2]);
            dst[0] = hsv[0];
            dst[1] = hsv[1];
            dst[2] = hsv[2];
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                let hsv = process(src[[y, x, 0]], src[[y, x, 1]], src[[y, x, 2]]);
                out[[y, x, 0]] = hsv[0];
                out[[y, x, 1]] = hsv[1];
                out[[y, x, 2]] = hsv[2];
            }
        }
    }
    Ok(out)
}

/// Converts OpenCV-scaled HSV (`H in [0, 179]`, `S, V in [0, 255]`) to BGR `(H, W, 3)`.
pub fn hsv_to_bgr(src: ArrayView3<'_, u8>) -> CvResult<Array3<u8>> {
    let (h, w, c) = src.dim();
    if c != 3 {
        return Err(CvError::UnsupportedFormat("hsv_to_bgr expects 3 channels".into()));
    }
    let mut out = Array3::<u8>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let hue = (src[[y, x, 0]] % 180) as f32 * 2.0;
            let sat = src[[y, x, 1]] as f32 / 255.0;
            let val = src[[y, x, 2]] as f32 / 255.0;
            let c_val = val * sat;
            let sector = hue / 60.0;
            let x_val = c_val * (1.0 - ((sector % 2.0) - 1.0).abs());
            let m = val - c_val;
            let (r1, g1, b1) = if sector < 1.0 {
                (c_val, x_val, 0.0)
            } else if sector < 2.0 {
                (x_val, c_val, 0.0)
            } else if sector < 3.0 {
                (0.0, c_val, x_val)
            } else if sector < 4.0 {
                (0.0, x_val, c_val)
            } else if sector < 5.0 {
                (x_val, 0.0, c_val)
            } else {
                (c_val, 0.0, x_val)
            };
            out[[y, x, 0]] = ((b1 + m) * 255.0).round().clamp(0.0, 255.0) as u8;
            out[[y, x, 1]] = ((g1 + m) * 255.0).round().clamp(0.0, 255.0) as u8;
            out[[y, x, 2]] = ((r1 + m) * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

/// Converts sRGB BGR `(H, W, 3|4)` to OpenCV 8-bit CIE Lab (`L*255/100`, `a+128`, `b+128`).
pub fn bgr_to_lab(src: ArrayView3<'_, u8>) -> CvResult<Array3<u8>> {
    let (h, w, c) = src.dim();
    if c < 3 {
        return Err(CvError::UnsupportedFormat("bgr_to_lab expects >= 3 channels".into()));
    }
    #[inline]
    fn srgb_to_lin(v: u8) -> f32 {
        let s = v as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    }
    #[inline]
    fn f_lab(t: f32) -> f32 {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    }
    let mut out = Array3::<u8>::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            let b = srgb_to_lin(src[[y, x, 0]]);
            let g = srgb_to_lin(src[[y, x, 1]]);
            let r = srgb_to_lin(src[[y, x, 2]]);
            let x_xyz = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456;
            let y_xyz = 0.212671 * r + 0.715160 * g + 0.072169 * b;
            let z_xyz = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754;
            let fx = f_lab(x_xyz);
            let fy = f_lab(y_xyz);
            let fz = f_lab(z_xyz);
            let l_star = if y_xyz > 0.008856 {
                116.0 * fy - 16.0
            } else {
                903.3 * y_xyz
            };
            let a_star = 500.0 * (fx - fy);
            let b_star = 200.0 * (fy - fz);
            out[[y, x, 0]] = (l_star * 2.55).round().clamp(0.0, 255.0) as u8;
            out[[y, x, 1]] = (a_star + 128.0).round().clamp(0.0, 255.0) as u8;
            out[[y, x, 2]] = (b_star + 128.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(out)
}

/// Splits a 3-channel `(H, W, 3)` image into 3 contiguous `(H, W)` planes `[c0, c1, c2]`.
pub fn split_channels(src: ArrayView3<'_, u8>) -> CvResult<Vec<Array2<u8>>> {
    let (h, w, c) = src.dim();
    if c == 0 {
        return Err(CvError::EmptyInput("0 channels".into()));
    }
    let mut planes = vec![Array2::<u8>::zeros((h, w)); c];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..c {
                planes[ch][[y, x]] = src[[y, x, ch]];
            }
        }
    }
    Ok(planes)
}

/// Merges equal-sized `(H, W)` planes into `(H, W, C)`.
pub fn merge_channels(planes: &[ArrayView2<'_, u8>]) -> CvResult<Array3<u8>> {
    if planes.is_empty() {
        return Err(CvError::EmptyInput("no planes to merge".into()));
    }
    let (h, w) = planes[0].dim();
    let c = planes.len();
    for p in planes {
        if p.dim() != (h, w) {
            return Err(CvError::ShapeMismatch("channel dimensions differ".into()));
        }
    }
    let mut out = Array3::<u8>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            for (ch, p) in planes.iter().enumerate() {
                out[[y, x, ch]] = p[[y, x]];
            }
        }
    }
    Ok(out)
}

/// Inclusive 3-channel range check matching `cv2.inRange` (returns `255` inside, `0` outside).
pub fn in_range(src: ArrayView3<'_, u8>, lower: [u8; 3], upper: [u8; 3]) -> CvResult<Array2<u8>> {
    let (h, w, c) = src.dim();
    if c < 3 {
        return Err(CvError::UnsupportedFormat("in_range expects >= 3 channels".into()));
    }
    let mut out = Array2::<u8>::zeros((h, w));
    if let (Some(s), Some(d)) = (src.as_slice(), out.as_slice_mut()) {
        for (dst, px) in d.iter_mut().zip(s.chunks_exact(c)) {
            let ok = px[0] >= lower[0]
                && px[0] <= upper[0]
                && px[1] >= lower[1]
                && px[1] <= upper[1]
                && px[2] >= lower[2]
                && px[2] <= upper[2];
            *dst = if ok { 255 } else { 0 };
        }
    } else {
        for y in 0..h {
            for x in 0..w {
                let p0 = src[[y, x, 0]];
                let p1 = src[[y, x, 1]];
                let p2 = src[[y, x, 2]];
                let ok = p0 >= lower[0]
                    && p0 <= upper[0]
                    && p1 >= lower[1]
                    && p1 <= upper[1]
                    && p2 >= lower[2]
                    && p2 <= upper[2];
                out[[y, x]] = if ok { 255 } else { 0 };
            }
        }
    }
    Ok(out)
}

/// Strict `>` binary threshold matching `cv2.threshold(THRESH_BINARY | THRESH_BINARY_INV)`.
pub fn threshold(
    src: ArrayView2<'_, u8>,
    thresh: u8,
    max_val: u8,
    mode: ThresholdType,
) -> Array2<u8> {
    let (h, w) = src.dim();
    let mut out = Array2::<u8>::zeros((h, w));
    if let (Some(s), Some(d)) = (src.as_slice(), out.as_slice_mut()) {
        match mode {
            ThresholdType::Binary => {
                for (dst, &v) in d.iter_mut().zip(s.iter()) {
                    *dst = if v > thresh { max_val } else { 0 };
                }
            }
            ThresholdType::BinaryInv => {
                for (dst, &v) in d.iter_mut().zip(s.iter()) {
                    *dst = if v > thresh { 0 } else { max_val };
                }
            }
        }
    } else {
        for (dst, &v) in out.iter_mut().zip(src.iter()) {
            *dst = match mode {
                ThresholdType::Binary => {
                    if v > thresh {
                        max_val
                    } else {
                        0
                    }
                }
                ThresholdType::BinaryInv => {
                    if v > thresh {
                        0
                    } else {
                        max_val
                    }
                }
            };
        }
    }
    out
}

/// Bitwise AND of two 2D `u8` arrays with optional mask (`mask != 0`).
pub fn bitwise_and(
    a: ArrayView2<'_, u8>,
    b: ArrayView2<'_, u8>,
    mask: Option<ArrayView2<'_, u8>>,
) -> CvResult<Array2<u8>> {
    if a.dim() != b.dim() {
        return Err(CvError::ShapeMismatch("bitwise_and shapes differ".into()));
    }
    if let Some(m) = mask
        && m.dim() != a.dim()
    {
        return Err(CvError::ShapeMismatch("bitwise_and mask shape differs".into()));
    }
    let mut out = Array2::<u8>::zeros(a.dim());
    if let Some(m) = mask {
        if let (Some(sa), Some(sb), Some(sm), Some(sd)) =
            (a.as_slice(), b.as_slice(), m.as_slice(), out.as_slice_mut())
        {
            for ((dst, &va), (&vb, &vm)) in sd.iter_mut().zip(sa.iter()).zip(sb.iter().zip(sm.iter()))
            {
                *dst = if vm != 0 { va & vb } else { 0 };
            }
        } else {
            for ((dst, &va), (&vb, &vm)) in out.iter_mut().zip(a.iter()).zip(b.iter().zip(m.iter()))
            {
                *dst = if vm != 0 { va & vb } else { 0 };
            }
        }
    } else if let (Some(sa), Some(sb), Some(sd)) = (a.as_slice(), b.as_slice(), out.as_slice_mut()) {
        for ((dst, &va), &vb) in sd.iter_mut().zip(sa.iter()).zip(sb.iter()) {
            *dst = va & vb;
        }
    } else {
        for ((dst, &va), &vb) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = va & vb;
        }
    }
    Ok(out)
}

/// Applies a 256-entry lookup table to a 3D `u8` array matching `cv2.LUT`.
pub fn lut_u8_3d(src: ArrayView3<'_, u8>, lut: &[u8; 256]) -> Array3<u8> {
    let mut out = Array3::<u8>::zeros(src.dim());
    if let (Some(s), Some(d)) = (src.as_slice(), out.as_slice_mut()) {
        for (dst, &v) in d.iter_mut().zip(s.iter()) {
            *dst = lut[v as usize];
        }
    } else {
        for (dst, &v) in out.iter_mut().zip(src.iter()) {
            *dst = lut[v as usize];
        }
    }
    out
}

/// Applies a 256-entry lookup table to a 2D `u8` array.
pub fn lut_u8_2d(src: ArrayView2<'_, u8>, lut: &[u8; 256]) -> Array2<u8> {
    let mut out = Array2::<u8>::zeros(src.dim());
    if let (Some(s), Some(d)) = (src.as_slice(), out.as_slice_mut()) {
        for (dst, &v) in d.iter_mut().zip(s.iter()) {
            *dst = lut[v as usize];
        }
    } else {
        for (dst, &v) in out.iter_mut().zip(src.iter()) {
            *dst = lut[v as usize];
        }
    }
    out
}

/// Elementwise `|a - b|` for 2D `u8` arrays.
pub fn absdiff_u8(a: ArrayView2<'_, u8>, b: ArrayView2<'_, u8>) -> CvResult<Array2<u8>> {
    if a.dim() != b.dim() {
        return Err(CvError::ShapeMismatch("absdiff_u8 shapes differ".into()));
    }
    let mut out = Array2::<u8>::zeros(a.dim());
    if let (Some(sa), Some(sb), Some(sd)) = (a.as_slice(), b.as_slice(), out.as_slice_mut()) {
        for ((dst, &va), &vb) in sd.iter_mut().zip(sa.iter()).zip(sb.iter()) {
            *dst = va.abs_diff(vb);
        }
    } else {
        for ((dst, &va), &vb) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = va.abs_diff(vb);
        }
    }
    Ok(out)
}

/// Elementwise `|a - b|` for 3D `u8` arrays.
pub fn absdiff_u8_3d(a: ArrayView3<'_, u8>, b: ArrayView3<'_, u8>) -> CvResult<Array3<u8>> {
    if a.dim() != b.dim() {
        return Err(CvError::ShapeMismatch("absdiff_u8_3d shapes differ".into()));
    }
    let mut out = Array3::<u8>::zeros(a.dim());
    if let (Some(sa), Some(sb), Some(sd)) = (a.as_slice(), b.as_slice(), out.as_slice_mut()) {
        for ((dst, &va), &vb) in sd.iter_mut().zip(sa.iter()).zip(sb.iter()) {
            *dst = va.abs_diff(vb);
        }
    } else {
        for ((dst, &va), &vb) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = va.abs_diff(vb);
        }
    }
    Ok(out)
}

/// Saturating scalar addition matching `cv2.add(src, val)`.
pub fn add_sat_scalar_u8(src: ArrayView2<'_, u8>, val: u8) -> Array2<u8> {
    let mut out = Array2::<u8>::zeros(src.dim());
    if let (Some(s), Some(d)) = (src.as_slice(), out.as_slice_mut()) {
        for (dst, &v) in d.iter_mut().zip(s.iter()) {
            *dst = v.saturating_add(val);
        }
    } else {
        for (dst, &v) in out.iter_mut().zip(src.iter()) {
            *dst = v.saturating_add(val);
        }
    }
    out
}

/// Saturating array addition matching `cv2.add(a, b)`.
pub fn add_sat_u8(a: ArrayView2<'_, u8>, b: ArrayView2<'_, u8>) -> CvResult<Array2<u8>> {
    if a.dim() != b.dim() {
        return Err(CvError::ShapeMismatch("add_sat_u8 shapes differ".into()));
    }
    let mut out = Array2::<u8>::zeros(a.dim());
    if let (Some(sa), Some(sb), Some(sd)) = (a.as_slice(), b.as_slice(), out.as_slice_mut()) {
        for ((dst, &va), &vb) in sd.iter_mut().zip(sa.iter()).zip(sb.iter()) {
            *dst = va.saturating_add(vb);
        }
    } else {
        for ((dst, &va), &vb) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = va.saturating_add(vb);
        }
    }
    Ok(out)
}

/// Linear blend `round(alpha * a + beta * b + gamma)` clamped to `[0, 255]`.
pub fn add_weighted_u8_3d(
    a: ArrayView3<'_, u8>,
    alpha: f64,
    b: ArrayView3<'_, u8>,
    beta: f64,
    gamma: f64,
) -> CvResult<Array3<u8>> {
    if a.dim() != b.dim() {
        return Err(CvError::ShapeMismatch("add_weighted shapes differ".into()));
    }
    let mut out = Array3::<u8>::zeros(a.dim());
    for ((dst, &va), &vb) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        let v = (alpha * va as f64 + beta * vb as f64 + gamma).round();
        *dst = v.clamp(0.0, 255.0) as u8;
    }
    Ok(out)
}

/// Counts non-zero elements in a 2D `u8` slice/array matching `cv2.countNonZero`.
pub fn count_non_zero(src: ArrayView2<'_, u8>) -> usize {
    src.iter().filter(|&&v| v != 0).count()
}

/// Computes a 1D histogram with `bins` uniform bins in `[0, 256)` normalized to L1 if `normalize`.
pub fn calc_hist_u8(src: ArrayView2<'_, u8>, bins: usize, normalize: bool) -> CvResult<Vec<f32>> {
    if bins == 0 || bins > 256 {
        return Err(CvError::InvalidArgument("bins must be in 1..=256".into()));
    }
    let mut hist = vec![0.0f32; bins];
    for &v in src.iter() {
        let b = (v as usize * bins) / 256;
        hist[b.min(bins - 1)] += 1.0;
    }
    if normalize {
        let sum: f32 = hist.iter().sum();
        if sum > 0.0 {
            for h in &mut hist {
                *h /= sum;
            }
        }
    }
    Ok(hist)
}

/// Computes the 4x4x4 = 64-bin L1-normalized RGB histogram sampled every 8th pixel
/// matching `MapScreenStateMachine._compute_rgb_hist64`.
pub fn rgb_hist64_sampled(frame: ArrayView3<'_, u8>) -> [f32; 64] {
    let (h, w, c) = frame.dim();
    let mut hist = [0.0f32; 64];
    if h == 0 || w == 0 || c == 0 {
        return hist;
    }
    let mut count = 0.0f32;
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            let idx = if c >= 3 {
                let b = frame[[y, x, 0]] as usize >> 6;
                let g = frame[[y, x, 1]] as usize >> 6;
                let r = frame[[y, x, 2]] as usize >> 6;
                (r << 4) | (g << 2) | b
            } else {
                let q = frame[[y, x, 0]] as usize >> 6;
                (q << 4) | (q << 2) | q
            };
            hist[idx] += 1.0;
            count += 1.0;
        }
    }
    if count > 0.0 {
        for v in &mut hist {
            *v /= count;
        }
    }
    hist
}

/// Bhattacharyya distance `sqrt(max(0, 1 - sum(sqrt(h1 * h2))))` in `[0.0, 1.0]`.
pub fn bhattacharyya_distance(h1: &[f32], h2: &[f32]) -> f32 {
    if h1.is_empty() || h1.len() != h2.len() {
        return 0.0;
    }
    let bc: f64 = h1
        .iter()
        .zip(h2.iter())
        .map(|(&a, &b)| (a as f64 * b as f64).max(0.0).sqrt())
        .sum();
    (1.0 - bc).max(0.0).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn bgr_to_gray_and_hsv_match_opencv_values() {
        let bgr = Array3::from_shape_vec(
            (1, 5, 3),
            vec![
                10, 20, 30, //
                255, 0, 0, //
                0, 255, 0, //
                0, 0, 255, //
                255, 255, 255,
            ],
        )
        .unwrap();
        let gray = bgr_to_gray(bgr.view()).unwrap();
        assert_eq!(gray.as_slice().unwrap(), &[22, 29, 150, 76, 255]);

        let hsv = bgr_to_hsv(bgr.view()).unwrap();
        assert_eq!(
            hsv.as_slice().unwrap(),
            &[
                15, 170, 30, //
                120, 255, 255, //
                60, 255, 255, //
                0, 255, 255, //
                0, 0, 255,
            ]
        );
    }

    #[test]
    fn threshold_and_in_range_and_bhatt() {
        let a = array![[240u8, 241], [25, 26]];
        let t_bin = threshold(a.view(), 240, 255, ThresholdType::Binary);
        assert_eq!(t_bin, array![[0, 255], [0, 0]]);
        let t_inv = threshold(a.view(), 25, 255, ThresholdType::BinaryInv);
        assert_eq!(t_inv, array![[0, 0], [255, 0]]);

        let h1 = [0.5f32, 0.5];
        let h2 = [0.5f32, 0.5];
        assert!(bhattacharyya_distance(&h1, &h2) < 1e-6);
        let h3 = [1.0f32, 0.0];
        let h4 = [0.0f32, 1.0];
        assert!((bhattacharyya_distance(&h3, &h4) - 1.0).abs() < 1e-6);
    }
}
