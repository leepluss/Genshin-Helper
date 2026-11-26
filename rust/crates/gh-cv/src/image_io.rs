//! Unicode-path-safe PNG/WebP/JPEG image I/O over `gh_core::atomic_io` (`src/image_io.py`).

use crate::ops::color::{
    array2_to_gray_image, bgr_to_gray, frame_view_to_array3, frame_view_to_gray_array,
    gray_image_to_array2, rgb_image_to_bgr_array3, rgba_image_to_bgra_array3, swap_rb,
};
use crate::ops::{CvError, CvResult};
use gh_core::atomic_io::{
    AtomicTempFile, DEFAULT_REPLACE_RETRIES, DEFAULT_REPLACE_RETRY_DELAY,
};
pub use gh_core::atomic_io::write_json_atomic;
use gh_core::frame::{FrameView, PixelFormat};
use image::{DynamicImage, ImageFormat, RgbImage, RgbaImage};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use std::io::{Cursor, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageReadMode {
    Color,
    Grayscale,
    Unchanged,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecodedImage {
    Gray(Array2<u8>),
    Bgr(Array3<u8>),
    Bgra(Array3<u8>),
}

impl DecodedImage {
    pub fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::Gray(arr) => arr.dim(),
            Self::Bgr(arr) | Self::Bgra(arr) => {
                let (h, w, _) = arr.dim();
                (h, w)
            }
        }
    }

    pub fn into_bgr(self) -> CvResult<Array3<u8>> {
        match self {
            Self::Bgr(arr) => Ok(arr),
            Self::Bgra(arr) => crate::ops::color::bgra_to_bgr(arr.view()),
            Self::Gray(arr) => {
                let (h, w) = arr.dim();
                let mut out = Array3::<u8>::zeros((h, w, 3));
                for y in 0..h {
                    for x in 0..w {
                        let v = arr[[y, x]];
                        out[[y, x, 0]] = v;
                        out[[y, x, 1]] = v;
                        out[[y, x, 2]] = v;
                    }
                }
                Ok(out)
            }
        }
    }

    pub fn into_gray(self) -> CvResult<Array2<u8>> {
        match self {
            Self::Gray(arr) => Ok(arr),
            Self::Bgr(arr) | Self::Bgra(arr) => bgr_to_gray(arr.view()),
        }
    }
}

/// Decodes raw image bytes (`PNG`, `WebP`, `JPEG`) using OpenCV channel semantics (`BGR`/`BGRA`/`Gray`).
pub fn decode_image_bytes(bytes: &[u8], mode: ImageReadMode) -> CvResult<DecodedImage> {
    if bytes.is_empty() {
        return Err(CvError::EmptyInput("empty image bytes".to_owned()));
    }
    let dyn_img = image::load_from_memory(bytes)
        .map_err(|e| CvError::UnsupportedFormat(format!("image decode failed: {e}")))?;
    match mode {
        ImageReadMode::Color => {
            let rgb = dyn_img.to_rgb8();
            Ok(DecodedImage::Bgr(rgb_image_to_bgr_array3(&rgb)))
        }
        ImageReadMode::Grayscale => match dyn_img {
            DynamicImage::ImageLuma8(g) => Ok(DecodedImage::Gray(gray_image_to_array2(&g))),
            DynamicImage::ImageLumaA8(la) => {
                let (w, h) = la.dimensions();
                let raw = la.as_raw();
                let mut gray = Vec::with_capacity((w as usize) * (h as usize));
                for px in raw.as_chunks::<2>().0 {
                    gray.push(px[0]);
                }
                let arr = Array2::from_shape_vec((h as usize, w as usize), gray)
                    .map_err(|e| CvError::ShapeMismatch(e.to_string()))?;
                Ok(DecodedImage::Gray(arr))
            }
            DynamicImage::ImageRgb8(rgb) => {
                let bgr = rgb_image_to_bgr_array3(&rgb);
                Ok(DecodedImage::Gray(bgr_to_gray(bgr.view())?))
            }
            DynamicImage::ImageRgba8(rgba) => {
                let bgra = rgba_image_to_bgra_array3(&rgba);
                Ok(DecodedImage::Gray(bgr_to_gray(bgra.view())?))
            }
            other => {
                let rgb = other.to_rgb8();
                let bgr = rgb_image_to_bgr_array3(&rgb);
                Ok(DecodedImage::Gray(bgr_to_gray(bgr.view())?))
            }
        },
        ImageReadMode::Unchanged => match dyn_img {
            DynamicImage::ImageLuma8(g) => Ok(DecodedImage::Gray(gray_image_to_array2(&g))),
            DynamicImage::ImageRgb8(rgb) => Ok(DecodedImage::Bgr(rgb_image_to_bgr_array3(&rgb))),
            DynamicImage::ImageRgba8(rgba) => Ok(DecodedImage::Bgra(rgba_image_to_bgra_array3(&rgba))),
            other => {
                if other.color().has_alpha() {
                    let rgba = other.to_rgba8();
                    Ok(DecodedImage::Bgra(rgba_image_to_bgra_array3(&rgba)))
                } else if other.color().channel_count() == 1 {
                    let g = other.to_luma8();
                    Ok(DecodedImage::Gray(gray_image_to_array2(&g)))
                } else {
                    let rgb = other.to_rgb8();
                    Ok(DecodedImage::Bgr(rgb_image_to_bgr_array3(&rgb)))
                }
            }
        },
    }
}

/// Reads an image from a Unicode-safe `Path` (`src/image_io.py::read_image`).
pub fn read_image(path: &Path, mode: ImageReadMode) -> CvResult<DecodedImage> {
    let bytes = std::fs::read(path).map_err(|e| CvError::Core(gh_core::CoreError::Io(e)))?;
    decode_image_bytes(&bytes, mode)
}

pub fn read_image_bgr(path: &Path) -> CvResult<Array3<u8>> {
    match read_image(path, ImageReadMode::Color)? {
        DecodedImage::Bgr(arr) => Ok(arr),
        other => other.into_bgr(),
    }
}

pub fn read_image_gray(path: &Path) -> CvResult<Array2<u8>> {
    match read_image(path, ImageReadMode::Grayscale)? {
        DecodedImage::Gray(arr) => Ok(arr),
        other => other.into_gray(),
    }
}

pub fn read_image_unchanged(path: &Path) -> CvResult<DecodedImage> {
    read_image(path, ImageReadMode::Unchanged)
}

pub fn read_image_bgr_opt(path: &Path) -> Option<Array3<u8>> {
    read_image_bgr(path).ok()
}

pub fn read_image_gray_opt(path: &Path) -> Option<Array2<u8>> {
    read_image_gray(path).ok()
}

pub fn read_image_unchanged_opt(path: &Path) -> Option<DecodedImage> {
    read_image_unchanged(path).ok()
}

fn resolve_image_format(path: &Path, ext: Option<&str>) -> CvResult<ImageFormat> {
    let raw_ext = ext
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| path.extension().and_then(|e| e.to_str()))
        .unwrap_or("png");
    let clean = raw_ext.trim_start_matches('.').to_ascii_lowercase();
    match clean.as_str() {
        "png" => Ok(ImageFormat::Png),
        "webp" => Ok(ImageFormat::WebP),
        "jpg" | "jpeg" => Ok(ImageFormat::Jpeg),
        other => Err(CvError::UnsupportedFormat(format!(
            "unsupported image output extension '.{other}'"
        ))),
    }
}

fn write_dynamic_atomic(path: &Path, dyn_img: &DynamicImage, fmt: ImageFormat) -> CvResult<()> {
    let mut encoded = Cursor::new(Vec::new());
    dyn_img
        .write_to(&mut encoded, fmt)
        .map_err(|e| CvError::UnsupportedFormat(format!("image encode failed: {e}")))?;
    let bytes = encoded.into_inner();

    let mut tmp = AtomicTempFile::new(path, "image")?;
    tmp.as_file_mut()?
        .write_all(&bytes)
        .map_err(|e| CvError::Core(gh_core::CoreError::Io(e)))?;
    tmp.commit(DEFAULT_REPLACE_RETRIES, DEFAULT_REPLACE_RETRY_DELAY)?;
    Ok(())
}

/// Atomically writes a BGR `(H, W, 3)`, BGRA `(H, W, 4)`, or 1-channel `(H, W, 1)` image.
pub fn write_image_bgr(path: &Path, img: ArrayView3<'_, u8>, ext: Option<&str>) -> CvResult<()> {
    let (h, w, c) = img.dim();
    if h == 0 || w == 0 || c == 0 {
        return Err(CvError::EmptyInput("write_image_bgr empty image".to_owned()));
    }
    let fmt = resolve_image_format(path, ext)?;
    let dyn_img = match c {
        1 => {
            let view2 = img.index_axis(ndarray::Axis(2), 0);
            DynamicImage::ImageLuma8(array2_to_gray_image(view2)?)
        }
        3 => {
            let rgb = swap_rb(img)?;
            let raw: Vec<u8> = if let Some(s) = rgb.as_slice() {
                s.to_vec()
            } else {
                rgb.iter().copied().collect()
            };
            let rgb_img = RgbImage::from_raw(w as u32, h as u32, raw)
                .ok_or_else(|| CvError::ShapeMismatch("RgbImage buffer mismatch".to_owned()))?;
            DynamicImage::ImageRgb8(rgb_img)
        }
        4 => {
            if matches!(fmt, ImageFormat::Jpeg) {
                let rgb = swap_rb(crate::ops::color::bgra_to_bgr(img)?.view())?;
                let raw: Vec<u8> = rgb.iter().copied().collect();
                let rgb_img = RgbImage::from_raw(w as u32, h as u32, raw)
                    .ok_or_else(|| CvError::ShapeMismatch("RgbImage buffer mismatch".to_owned()))?;
                DynamicImage::ImageRgb8(rgb_img)
            } else {
                let rgba = swap_rb(img)?;
                let raw: Vec<u8> = if let Some(s) = rgba.as_slice() {
                    s.to_vec()
                } else {
                    rgba.iter().copied().collect()
                };
                let rgba_img = RgbaImage::from_raw(w as u32, h as u32, raw).ok_or_else(|| {
                    CvError::ShapeMismatch("RgbaImage buffer mismatch".to_owned())
                })?;
                DynamicImage::ImageRgba8(rgba_img)
            }
        }
        other => {
            return Err(CvError::UnsupportedFormat(format!(
                "write_image_bgr expects 1, 3, or 4 channels, got {other}"
            )));
        }
    };
    write_dynamic_atomic(path, &dyn_img, fmt)
}

/// Atomically writes a 2D grayscale image (`PNG`/`WebP`/`JPEG`).
pub fn write_image_gray(path: &Path, img: ArrayView2<'_, u8>, ext: Option<&str>) -> CvResult<()> {
    let (h, w) = img.dim();
    if h == 0 || w == 0 {
        return Err(CvError::EmptyInput("write_image_gray empty image".to_owned()));
    }
    let fmt = resolve_image_format(path, ext)?;
    let gray_img = array2_to_gray_image(img)?;
    let dyn_img = DynamicImage::ImageLuma8(gray_img);
    write_dynamic_atomic(path, &dyn_img, fmt)
}

/// Atomically writes a `gh_core::frame::FrameView` (`Gray8`, `Bgr8`, `Bgra8`, `Rgba8`).
pub fn write_image_frame(path: &Path, frame: FrameView<'_>, ext: Option<&str>) -> CvResult<()> {
    match frame.format {
        PixelFormat::Gray8 => {
            let gray = frame_view_to_gray_array(frame)?;
            write_image_gray(path, gray.view(), ext)
        }
        PixelFormat::Bgr8 | PixelFormat::Bgra8 => {
            let arr = frame_view_to_array3(frame)?;
            write_image_bgr(path, arr.view(), ext)
        }
        PixelFormat::Rgba8 => {
            let arr = frame_view_to_array3(frame)?;
            let bgra = swap_rb(arr.view())?;
            write_image_bgr(path, bgra.view(), ext)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gh_core::frame::FrameBuf;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEST_SEQ: AtomicUsize = AtomicUsize::new(0);

    fn make_temp_dir(label: &str) -> PathBuf {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "gh_cv_image_io_{label}_{}_{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn unicode_png_and_webp_roundtrip_and_extensionless_write() {
        let base = make_temp_dir("unicode");
        let sub = base.join("tiếng_việt");

        // 1. BGR PNG + WebP lossless bit-exact roundtrip
        let mut bgr = Array3::<u8>::zeros((12, 16, 3));
        for y in 0..12 {
            for x in 0..16 {
                bgr[[y, x, 0]] = (x * 15) as u8;
                bgr[[y, x, 1]] = (y * 20) as u8;
                bgr[[y, x, 2]] = ((x + y) * 9) as u8;
            }
        }
        let png_path = sub.join("ảnh_mẫu.png");
        write_image_bgr(&png_path, bgr.view(), None).expect("write png");
        let read_png = read_image_bgr(&png_path).expect("read png");
        assert_eq!(read_png, bgr);

        let webp_path = sub.join("ảnh_mẫu.webp");
        write_image_bgr(&webp_path, bgr.view(), None).expect("write webp");
        let read_webp = read_image_bgr(&webp_path).expect("read webp");
        assert_eq!(read_webp, bgr);

        // 2. BGRA unchanged roundtrip
        let mut bgra = Array3::<u8>::zeros((8, 10, 4));
        for y in 0..8 {
            for x in 0..10 {
                bgra[[y, x, 0]] = 10;
                bgra[[y, x, 1]] = 120;
                bgra[[y, x, 2]] = 230;
                bgra[[y, x, 3]] = if (x + y) % 2 == 0 { 255 } else { 64 };
            }
        }
        let bgra_path = sub.join("alpha_ảnh.png");
        write_image_bgr(&bgra_path, bgra.view(), None).expect("write bgra");
        let unchanged = read_image_unchanged(&bgra_path).expect("read unchanged");
        assert_eq!(unchanged, DecodedImage::Bgra(bgra));

        // 3. Extensionless path with explicit ext=".png" + sniffed read
        let no_ext = sub.join("không_đuôi");
        write_image_bgr(&no_ext, bgr.view(), Some(".png")).expect("write extensionless");
        assert_eq!(read_image_bgr(&no_ext).expect("read extensionless"), bgr);

        // 4. Grayscale roundtrip & FrameView Rgba8 write
        let gray = Array2::from_shape_fn((6, 8), |(y, x)| ((y * 32 + x * 17) & 0xFF) as u8);
        let gray_path = sub.join("xám.png");
        write_image_gray(&gray_path, gray.view(), None).expect("write gray");
        assert_eq!(read_image_gray(&gray_path).expect("read gray"), gray);

        let mut fbuf = FrameBuf::new_zeroed(4, 3, PixelFormat::Rgba8);
        fbuf.data[0] = 200; // R
        fbuf.data[1] = 100; // G
        fbuf.data[2] = 50; // B
        fbuf.data[3] = 255; // A
        let frame_path = sub.join("khung.png");
        write_image_frame(&frame_path, fbuf.as_view(), None).expect("write frame");
        let got_frame = read_image_bgr(&frame_path).expect("read frame bgr");
        assert_eq!(got_frame[[0, 0, 0]], 50); // B
        assert_eq!(got_frame[[0, 0, 1]], 100); // G
        assert_eq!(got_frame[[0, 0, 2]], 200); // R

        // 5. No leftover .tmp files
        let entries: Vec<String> = std::fs::read_dir(&sub)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(!entries.iter().any(|n| n.ends_with(".tmp")));

        // 6. Missing / empty / corrupt handling
        assert!(read_image_bgr_opt(&sub.join("missing.png")).is_none());
        let empty_path = sub.join("empty.png");
        std::fs::write(&empty_path, b"").expect("write empty");
        assert!(read_image_bgr_opt(&empty_path).is_none());
        let corrupt_path = sub.join("corrupt.png");
        std::fs::write(&corrupt_path, b"not a real png").expect("write corrupt");
        assert!(read_image_bgr_opt(&corrupt_path).is_none());
        assert!(read_image_gray_opt(&corrupt_path).is_none());

        let _ = std::fs::remove_dir_all(&base);
    }
}
