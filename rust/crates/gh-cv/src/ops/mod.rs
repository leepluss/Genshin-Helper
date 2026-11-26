//! Pure-Rust CV primitives: color, resize, Sobel/Scharr, FFT & Padmanabhan (2012) masked 6-FFT NCC,
//! phase correlation, warp polar/affine, contours/Canny/CLAHE, RootSIFT, 4-tree KD-Tree, ORB, RANSAC.

use gh_core::frame::FrameView;
use thiserror::Error;

pub mod color;
pub mod contours;
pub mod filters;
pub mod flann;
pub mod morph;
pub mod ncc;
pub mod orb;
pub mod phase_correlate;
pub mod ransac_similarity;
pub mod resize;
pub mod sift;
pub mod sobel;
pub mod warp;

/// Internal feature extraction/matching boundary allowing Phase 3 Gate 0 tuning
/// without changing `locator.rs`, `global_relocator.rs`, or `minimap_tracker.rs`.
pub trait FeatureMatcher: Send + Sync {
    fn extract_keypoints_and_descriptors(&self, gray: FrameView<'_>) -> Vec<([f32; 2], Vec<f32>)>;
}

#[derive(Debug, Error)]
pub enum CvError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("empty input: {0}")]
    EmptyInput(String),
    #[error("core error: {0}")]
    Core(#[from] gh_core::CoreError),
}

pub type CvResult<T> = Result<T, CvError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Nearest,
    Linear,
    Area,
    Cubic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BorderMode {
    Constant(f32),
    Replicate,
    Reflect101,
    Wrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateMatchMode {
    CcoeffNormed,
    CcorrNormed,
    SqdiffNormed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdType {
    Binary,
    BinaryInv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphShape {
    Rect,
    Ellipse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphOp {
    Erode,
    Dilate,
    Open,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourMode {
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourMethod {
    None,
    Simple,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DMatch {
    pub query_idx: usize,
    pub train_idx: usize,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Point2i {
    pub x: i32,
    pub y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point2f {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// Reflect-101 border index helper matching OpenCV `BORDER_REFLECT_101` (`|dcb|abcd|cba|`).
#[inline]
pub fn border_reflect_101(idx: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (len as isize - 1) * 2;
    let mut i = idx % period;
    if i < 0 {
        i += period;
    }
    if i >= len as isize {
        (period - i) as usize
    } else {
        i as usize
    }
}

/// Border coordinate mapper for general `BorderMode`.
#[inline]
pub fn map_border_coord(idx: isize, len: usize, mode: BorderMode) -> Option<usize> {
    if len == 0 {
        return None;
    }
    if idx >= 0 && (idx as usize) < len {
        return Some(idx as usize);
    }
    match mode {
        BorderMode::Constant(_) => None,
        BorderMode::Replicate => {
            if idx < 0 {
                Some(0)
            } else {
                Some(len - 1)
            }
        }
        BorderMode::Reflect101 => Some(border_reflect_101(idx, len)),
        BorderMode::Wrap => {
            let l = len as isize;
            let mut m = idx % l;
            if m < 0 {
                m += l;
            }
            Some(m as usize)
        }
    }
}
