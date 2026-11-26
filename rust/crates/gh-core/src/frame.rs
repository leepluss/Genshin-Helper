//! Zero-dependency pixel buffer types shared between `gh-win32`, `gh-cv`, and `gh-hud`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    Bgr8,
    Bgra8,
    Gray8,
    Rgba8,
}

impl PixelFormat {
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Bgr8 => 3,
            Self::Bgra8 | Self::Rgba8 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RectI32 {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameBuf {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub format: PixelFormat,
    pub data: Vec<u8>,
}

impl FrameBuf {
    pub fn new_zeroed(width: u32, height: u32, format: PixelFormat) -> Self {
        let stride = (width as usize) * format.channels();
        let data = vec![0u8; stride * (height as usize)];
        Self {
            width,
            height,
            stride,
            format,
            data,
        }
    }

    pub fn as_view(&self) -> FrameView<'_> {
        FrameView {
            width: self.width,
            height: self.height,
            stride: self.stride,
            format: self.format,
            data: &self.data,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameView<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub format: PixelFormat,
    pub data: &'a [u8],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_buf_allocates_exact_stride() {
        let frame = FrameBuf::new_zeroed(16, 8, PixelFormat::Bgra8);
        assert_eq!(frame.stride, 64);
        assert_eq!(frame.data.len(), 512);
        let view = frame.as_view();
        assert_eq!(view.width, 16);
        assert_eq!(view.height, 8);
    }
}
