//! Pure-Rust Computer Vision crate (zero `opencv_world` DLL, zero Win32, zero HTTP).
//!
//! Acyclic dependency: depends ONLY on `gh-core`.

pub mod global_relocator;
pub mod heading360;
pub mod image_io;
pub mod locator;
pub mod map_state;
pub mod minimap_tracker;
pub mod ops;
pub mod tile_cache;
pub mod view_tracker;
pub mod zoom_slider;
