//! Pure-Rust IEEE-754 `f16` scRGB linear HDR -> SDR `Bgra8` tonemapper.
//!
//! Designed for Windows Graphics Capture (WGC) / DXGI `DXGI_FORMAT_R16G16B16A16_FLOAT`
//! scRGB frames where `1.0` linear corresponds to 80 nits and SDR desktop white sits at
//! `sdr_white_nits / 80.0`.
//!
//! Pipeline per channel:
//! 1. Branchless IEEE-754 binary16 -> `f32` decode via a 65,536-entry `OnceLock` LUT
//!    (`NaN -> 0.0`, `±Inf -> ±65504.0`).
//! 2. Normalize linear scRGB by `inv_sdr_scale = 80.0 / sdr_white_nits` so SDR UI elements
//!    (minimap, zoom slider, HUD) preserve 1:1 color fidelity on `[0.0, 0.85]`.
//! 3. Smooth C1-continuous Reinhard shoulder rolloff above `0.85` into `[0.85, 1.0]` so
//!    HDR highlights (`> 1.0`) compress gracefully instead of hard-clipping to flat white.
//! 4. IEC 61966-2-1 sRGB OETF via a 4,096-entry `OnceLock` LUT into `Bgra8` (`A = 255`).
//!
//! Zero `unsafe`, zero external dependencies beyond `gh-core`.

use gh_core::frame::{FrameBuf, PixelFormat};
use std::sync::OnceLock;

/// Windows scRGB reference SDR white level (80 nits = `1.0` linear in scRGB).
pub const DEFAULT_SDR_WHITE_LEVEL_NITS: f32 = 80.0;

/// Minimum clamped Windows SDR white level in nits.
pub const MIN_SDR_WHITE_LEVEL_NITS: f32 = 80.0;

/// Maximum clamped Windows SDR white level in nits.
pub const MAX_SDR_WHITE_LEVEL_NITS: f32 = 1000.0;

/// Maximum finite IEEE-754 binary16 magnitude (`2^15 * (2 - 2^-10) = 65504.0`).
const F16_MAX_FINITE: f32 = 65_504.0;

/// `2^-24` exact in `f32` (`1.0 / 16_777_216.0`), used for IEEE-754 f16 subnormals.
const F16_SUBNORMAL_STEP: f32 = 1.0 / 16_777_216.0;

/// Linear SDR range `[0.0, 0.85]` kept strictly 1:1 for CV template matching fidelity.
const SDR_LINEAR_KNEE: f32 = 0.85;

/// Remaining linear headroom `[0.85, 1.0]` reserved for the Reinhard HDR highlight shoulder.
const SDR_HEADROOM: f32 = 1.0 - SDR_LINEAR_KNEE;

/// Number of entries in the linear -> sRGB `u8` LUT.
const SRGB_LUT_LEN: usize = 4096;
const SRGB_LUT_LAST: usize = SRGB_LUT_LEN - 1;

static F16_TO_F32_LUT: OnceLock<Box<[f32; 65_536]>> = OnceLock::new();
static LINEAR_TO_SRGB_LUT: OnceLock<Box<[u8; SRGB_LUT_LEN]>> = OnceLock::new();

/// Convert `DISPLAYCONFIG_SDR_WHITE_LEVEL.SDRWhiteLevel` (`raw_level * 80 / 1000`)
/// into SDR white nits clamped to `[MIN_SDR_WHITE_LEVEL_NITS, MAX_SDR_WHITE_LEVEL_NITS]`.
///
/// Returns [`DEFAULT_SDR_WHITE_LEVEL_NITS`] (`80.0`) when `raw_level == 0`.
#[inline]
pub fn sdr_white_level_from_windows_raw(raw_level: u32) -> f32 {
    if raw_level == 0 {
        return DEFAULT_SDR_WHITE_LEVEL_NITS;
    }
    let nits = (f64::from(raw_level) * 80.0 / 1000.0) as f32;
    nits.clamp(MIN_SDR_WHITE_LEVEL_NITS, MAX_SDR_WHITE_LEVEL_NITS)
}

/// Sanitize caller-provided `sdr_white_nits` into the valid `[80.0, 1000.0]` range.
#[inline]
fn sanitize_sdr_white_nits(sdr_white_nits: f32) -> f32 {
    if !sdr_white_nits.is_finite() || sdr_white_nits <= 0.0 {
        DEFAULT_SDR_WHITE_LEVEL_NITS
    } else {
        sdr_white_nits.clamp(MIN_SDR_WHITE_LEVEL_NITS, MAX_SDR_WHITE_LEVEL_NITS)
    }
}

/// Exact IEEE-754 binary16 bit decoding used to populate [`F16_TO_F32_LUT`].
///
/// - Zero (`±0.0`) preserves sign bit.
/// - Subnormals (`exp == 0, frac != 0`) decode as `±frac * 2^-24`.
/// - Normals (`1..=30`) map directly to `f32` exponent (`exp + 112`) and mantissa (`frac << 13`).
/// - `±Inf` (`exp == 31, frac == 0`) clamp to `±65504.0` so downstream math never sees `Inf`.
/// - `NaN` (`exp == 31, frac != 0`) maps to `0.0`.
fn decode_f16_exact(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exp = u32::from((bits >> 10) & 0x1F);
    let frac = u32::from(bits & 0x03FF);

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mag = (frac as f32) * F16_SUBNORMAL_STEP;
        return if sign != 0 { -mag } else { mag };
    }

    if exp == 31 {
        if frac == 0 {
            return if sign != 0 {
                -F16_MAX_FINITE
            } else {
                F16_MAX_FINITE
            };
        }
        return 0.0;
    }

    let f32_exp = (exp + 112) << 23;
    let f32_frac = frac << 13;
    f32::from_bits((sign << 31) | f32_exp | f32_frac)
}

fn init_f16_lut() -> Box<[f32; 65_536]> {
    let entries: Vec<f32> = (0u32..=65_535u32)
        .map(|bits| decode_f16_exact(bits as u16))
        .collect();
    entries
        .into_boxed_slice()
        .try_into()
        .expect("f16 LUT always has 65536 elements")
}

#[inline]
fn f16_lut() -> &'static [f32; 65_536] {
    F16_TO_F32_LUT.get_or_init(init_f16_lut)
}

/// Fast IEEE-754 binary16 -> `f32` decode backed by a lazy 65,536-entry `OnceLock` LUT.
///
/// Decodes `±0.0`, subnormals, and normals accurately; clamps `±Inf` to `±65504.0`; maps `NaN` to `0.0`.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    f16_lut()[usize::from(bits)]
}

/// Exact IEC 61966-2-1 linear `[0.0, 1.0]` -> sRGB `u8` OETF.
///
/// Values `<= 0.0` or non-finite clamp to `0`; values `>= 1.0` clamp to `255`.
#[inline]
pub fn linear_to_srgb_u8(v: f32) -> u8 {
    if !v.is_finite() || v <= 0.0 {
        return 0;
    }
    if v >= 1.0 {
        return 255;
    }
    let srgb = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round().clamp(0.0, 255.0) as u8
}

fn init_srgb_lut() -> Box<[u8; SRGB_LUT_LEN]> {
    let inv_last = 1.0 / (SRGB_LUT_LAST as f32);
    let entries: Vec<u8> = (0..SRGB_LUT_LEN)
        .map(|i| linear_to_srgb_u8((i as f32) * inv_last))
        .collect();
    entries
        .into_boxed_slice()
        .try_into()
        .expect("sRGB LUT always has 4096 elements")
}

#[inline]
fn srgb_lut() -> &'static [u8; SRGB_LUT_LEN] {
    LINEAR_TO_SRGB_LUT.get_or_init(init_srgb_lut)
}

/// Highlight-preserving scRGB linear HDR -> SDR `[0.0, 1.0]` tonemapper for a single channel.
///
/// - Scales `linear_scrgb` by `inv_sdr_scale` (`80.0 / sdr_white_nits`).
/// - Clamps negatives and `NaN` to `0.0`, and `+Inf` to `1.0`.
/// - Keeps `[0.0, 0.85]` strictly linear for 1:1 SDR UI / minimap color accuracy.
/// - Applies a C1-continuous Reinhard shoulder `k + h * d / (d + h)` (`k = 0.85`, `h = 0.15`,
///   `d = scaled - k`) above `0.85` into `[0.85, 1.0]` so HDR highlights compress smoothly.
#[inline]
pub fn tonemap_scrgb_channel(linear_scrgb: f32, inv_sdr_scale: f32) -> f32 {
    if !inv_sdr_scale.is_finite() || inv_sdr_scale <= 0.0 {
        return 0.0;
    }
    if linear_scrgb.is_nan() || linear_scrgb <= 0.0 {
        return 0.0;
    }
    if linear_scrgb.is_infinite() {
        return 1.0;
    }
    let scaled = linear_scrgb * inv_sdr_scale;
    if !scaled.is_finite() {
        return 1.0;
    }
    if scaled <= SDR_LINEAR_KNEE {
        scaled
    } else {
        let d = scaled - SDR_LINEAR_KNEE;
        (SDR_LINEAR_KNEE + SDR_HEADROOM * (d / (d + SDR_HEADROOM))).min(1.0)
    }
}

/// Inner-loop specialized tonemap + sRGB LUT lookup for finite LUT-decoded `f16` channels.
#[inline]
fn tonemap_finite_scrgb_to_srgb_u8(
    linear_scrgb: f32,
    inv_sdr_scale: f32,
    lut: &[u8; SRGB_LUT_LEN],
) -> u8 {
    let scaled = linear_scrgb * inv_sdr_scale;
    if scaled <= 0.0 {
        return 0;
    }
    let mapped = if scaled <= SDR_LINEAR_KNEE {
        scaled
    } else {
        let d = scaled - SDR_LINEAR_KNEE;
        SDR_LINEAR_KNEE + SDR_HEADROOM * (d / (d + SDR_HEADROOM))
    };
    let idx = (mapped * (SRGB_LUT_LAST as f32) + 0.5) as usize;
    lut[idx.min(SRGB_LUT_LAST)]
}

/// Convert a row-strided `RGBA16F` scRGB buffer into tightly packed `Bgra8` (`A = 255`).
///
/// Validates `row_pitch_bytes >= width * 8` and ensures `rgba16f_bytes` contains all rows
/// (`(height - 1) * row_pitch_bytes + width * 8` bytes when `height > 0`). Reuses `out_bgra`
/// capacity across frames to avoid per-frame allocations.
pub fn tonemap_rgba16f_to_bgra8(
    rgba16f_bytes: &[u8],
    width: u32,
    height: u32,
    row_pitch_bytes: usize,
    sdr_white_nits: f32,
    out_bgra: &mut Vec<u8>,
) -> Result<(), String> {
    let width_usize = width as usize;
    let height_usize = height as usize;

    let min_row_bytes = width_usize
        .checked_mul(8)
        .ok_or_else(|| format!("RGBA16F row byte width overflow for width={width}"))?;
    if row_pitch_bytes < min_row_bytes {
        return Err(format!(
            "row_pitch_bytes ({row_pitch_bytes}) < width * 8 ({min_row_bytes})"
        ));
    }

    let dst_row_bytes = width_usize
        .checked_mul(4)
        .ok_or_else(|| format!("BGRA8 row byte width overflow for width={width}"))?;
    let dst_total_bytes = dst_row_bytes
        .checked_mul(height_usize)
        .ok_or_else(|| format!("BGRA8 buffer size overflow for {width}x{height}"))?;

    if width_usize == 0 || height_usize == 0 {
        out_bgra.clear();
        return Ok(());
    }

    let required_src_bytes = (height_usize - 1)
        .checked_mul(row_pitch_bytes)
        .and_then(|v| v.checked_add(min_row_bytes))
        .ok_or_else(|| {
            format!("RGBA16F source size overflow for {width}x{height} pitch={row_pitch_bytes}")
        })?;
    if rgba16f_bytes.len() < required_src_bytes {
        return Err(format!(
            "RGBA16F buffer too small: got {} bytes, need at least {} bytes for {}x{} (pitch={})",
            rgba16f_bytes.len(),
            required_src_bytes,
            width,
            height,
            row_pitch_bytes
        ));
    }

    let inv_sdr_scale = DEFAULT_SDR_WHITE_LEVEL_NITS / sanitize_sdr_white_nits(sdr_white_nits);
    let f16 = f16_lut();
    let srgb = srgb_lut();

    out_bgra.resize(dst_total_bytes, 0);

    for y in 0..height_usize {
        let src_offset = y * row_pitch_bytes;
        let src_row = &rgba16f_bytes[src_offset..src_offset + min_row_bytes];
        let dst_offset = y * dst_row_bytes;
        let dst_row = &mut out_bgra[dst_offset..dst_offset + dst_row_bytes];

        for (src_px, dst_px) in src_row
            .as_chunks::<8>()
            .0
            .iter()
            .zip(dst_row.as_chunks_mut::<4>().0.iter_mut())
        {
            let r_bits = u16::from_le_bytes([src_px[0], src_px[1]]);
            let g_bits = u16::from_le_bytes([src_px[2], src_px[3]]);
            let b_bits = u16::from_le_bytes([src_px[4], src_px[5]]);

            let r = tonemap_finite_scrgb_to_srgb_u8(f16[usize::from(r_bits)], inv_sdr_scale, srgb);
            let g = tonemap_finite_scrgb_to_srgb_u8(f16[usize::from(g_bits)], inv_sdr_scale, srgb);
            let b = tonemap_finite_scrgb_to_srgb_u8(f16[usize::from(b_bits)], inv_sdr_scale, srgb);

            *dst_px = [b, g, r, 255];
        }
    }

    Ok(())
}

/// Convenience wrapper around [`tonemap_rgba16f_to_bgra8`] returning a ready-to-use `Bgra8` [`FrameBuf`].
pub fn tonemap_rgba16f_frame(
    rgba16f_bytes: &[u8],
    width: u32,
    height: u32,
    row_pitch_bytes: usize,
    sdr_white_nits: f32,
) -> Result<FrameBuf, String> {
    let mut data = Vec::new();
    tonemap_rgba16f_to_bgra8(
        rgba16f_bytes,
        width,
        height,
        row_pitch_bytes,
        sdr_white_nits,
        &mut data,
    )?;
    let stride = (width as usize) * PixelFormat::Bgra8.channels();
    Ok(FrameBuf {
        width,
        height,
        stride,
        format: PixelFormat::Bgra8,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode an `f32` into IEEE-754 binary16 bits for test fixture generation.
    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let frac = bits & 0x007F_FFFF;

        if exp == 0xFF {
            return if frac == 0 {
                sign | 0x7C00
            } else {
                sign | 0x7E00
            };
        }

        let unbiased = exp - 127;
        if unbiased < -24 {
            return sign;
        }
        if unbiased < -14 {
            let mant = frac | 0x0080_0000;
            let shift = (-14 - unbiased) as u32 + 13;
            let rounded = (mant + (1u32 << (shift - 1))) >> shift;
            return sign | (rounded as u16);
        }
        if unbiased > 15 {
            return sign | 0x7C00;
        }

        let h_exp = ((unbiased + 15) as u16) << 10;
        let h_frac = ((frac + 0x1000) >> 13) as u16;
        if h_frac >= 0x0400 {
            let next = h_exp + 0x0400;
            if next >= 0x7C00 {
                return sign | 0x7C00;
            }
            return sign | next;
        }
        sign | h_exp | h_frac
    }

    fn push_rgba16f_pixel(buf: &mut Vec<u8>, r: f32, g: f32, b: f32, a: f32) {
        buf.extend_from_slice(&f32_to_f16_bits(r).to_le_bytes());
        buf.extend_from_slice(&f32_to_f16_bits(g).to_le_bytes());
        buf.extend_from_slice(&f32_to_f16_bits(b).to_le_bytes());
        buf.extend_from_slice(&f32_to_f16_bits(a).to_le_bytes());
    }

    fn push_raw_rgba16f_bits(buf: &mut Vec<u8>, r: u16, g: u16, b: u16, a: u16) {
        buf.extend_from_slice(&r.to_le_bytes());
        buf.extend_from_slice(&g.to_le_bytes());
        buf.extend_from_slice(&b.to_le_bytes());
        buf.extend_from_slice(&a.to_le_bytes());
    }

    #[test]
    fn f16_to_f32_exact_ieee754_cases() {
        // Signed zeros
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0.0_f32.to_bits());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());

        // Common normals
        assert_eq!(f16_to_f32(0x3C00).to_bits(), 1.0_f32.to_bits());
        assert_eq!(f16_to_f32(0x3800).to_bits(), 0.5_f32.to_bits());
        assert_eq!(f16_to_f32(0xC000).to_bits(), (-2.0_f32).to_bits());
        assert_eq!(f16_to_f32(0x7BFF).to_bits(), 65_504.0_f32.to_bits());
        assert_eq!(f16_to_f32(0xFBFF).to_bits(), (-65_504.0_f32).to_bits());

        // Subnormals and smallest normal
        let min_subnormal = 2.0_f32.powi(-24);
        assert_eq!(f16_to_f32(0x0001).to_bits(), min_subnormal.to_bits());
        assert_eq!(f16_to_f32(0x8001).to_bits(), (-min_subnormal).to_bits());
        assert_eq!(
            f16_to_f32(0x03FF).to_bits(),
            (1023.0_f32 * min_subnormal).to_bits()
        );
        assert_eq!(f16_to_f32(0x0400).to_bits(), 2.0_f32.powi(-14).to_bits());

        // Inf clamps to finite +/-65504.0
        assert_eq!(f16_to_f32(0x7C00).to_bits(), 65_504.0_f32.to_bits());
        assert_eq!(f16_to_f32(0xFC00).to_bits(), (-65_504.0_f32).to_bits());

        // NaNs (quiet and signaling, both signs) map to 0.0
        for nan_bits in [0x7C01, 0x7E00, 0x7FFF, 0xFC01, 0xFE00, 0xFFFF] {
            let decoded = f16_to_f32(nan_bits);
            assert!(!decoded.is_nan(), "bits={nan_bits:#06X} returned NaN");
            assert_eq!(decoded.to_bits(), 0.0_f32.to_bits());
        }

        // Exhaustive finiteness guarantee across all 65,536 bit patterns
        for bits in 0u16..=u16::MAX {
            let v = f16_to_f32(bits);
            assert!(v.is_finite(), "bits={bits:#06X} decoded non-finite: {v}");
            assert!(
                (-F16_MAX_FINITE..=F16_MAX_FINITE).contains(&v),
                "bits={bits:#06X} out of f16 finite range: {v}"
            );
        }
    }

    #[test]
    fn sdr_white_level_from_windows_raw_conversions() {
        assert_eq!(sdr_white_level_from_windows_raw(0).to_bits(), 80.0_f32.to_bits());
        assert_eq!(sdr_white_level_from_windows_raw(1000).to_bits(), 80.0_f32.to_bits());
        assert_eq!(sdr_white_level_from_windows_raw(3000).to_bits(), 240.0_f32.to_bits());
        assert_eq!(sdr_white_level_from_windows_raw(6000).to_bits(), 480.0_f32.to_bits());

        // Clamping at bounds and overflow safety
        assert_eq!(sdr_white_level_from_windows_raw(500).to_bits(), 80.0_f32.to_bits());
        assert_eq!(
            sdr_white_level_from_windows_raw(12_500).to_bits(),
            1000.0_f32.to_bits()
        );
        assert_eq!(
            sdr_white_level_from_windows_raw(50_000).to_bits(),
            1000.0_f32.to_bits()
        );
        assert_eq!(
            sdr_white_level_from_windows_raw(u32::MAX).to_bits(),
            1000.0_f32.to_bits()
        );
    }

    #[test]
    fn linear_to_srgb_exact_and_lut_agreement() {
        assert_eq!(linear_to_srgb_u8(-0.5), 0);
        assert_eq!(linear_to_srgb_u8(f32::NAN), 0);
        assert_eq!(linear_to_srgb_u8(0.0), 0);
        assert_eq!(linear_to_srgb_u8(0.003_130_8), 10);
        assert_eq!(linear_to_srgb_u8(0.18), 118);
        assert_eq!(linear_to_srgb_u8(0.5), 188);
        assert_eq!(linear_to_srgb_u8(1.0), 255);
        assert_eq!(linear_to_srgb_u8(2.5), 255);

        let lut = srgb_lut();
        for step in 0..=10_000 {
            let v = (step as f32) / 10_000.0;
            let exact = linear_to_srgb_u8(v);
            let idx = (v * (SRGB_LUT_LAST as f32) + 0.5) as usize;
            let fast = lut[idx.min(SRGB_LUT_LAST)];
            assert!(
                exact.abs_diff(fast) <= 1,
                "LUT drift > 1 LSB at v={v}: exact={exact}, fast={fast}"
            );
        }
    }

    #[test]
    fn tonemap_channel_c1_knee_and_sdr_linearity() {
        // Strictly linear on [0.0, 0.85]
        for step in 0..=85 {
            let v = (step as f32) / 100.0;
            let mapped = tonemap_scrgb_channel(v, 1.0);
            assert!(
                (mapped - v).abs() < 1e-6,
                "non-linear inside SDR knee at v={v}: mapped={mapped}"
            );
        }

        // Right-hand slope at knee = 1.0 (C1 continuity)
        let eps = 1e-4_f32;
        let slope = (tonemap_scrgb_channel(SDR_LINEAR_KNEE + eps, 1.0)
            - tonemap_scrgb_channel(SDR_LINEAR_KNEE, 1.0))
            / eps;
        assert!(
            (slope - 1.0).abs() < 1e-2,
            "shoulder not C1-continuous at knee: slope={slope}"
        );

        // Edge cases: negatives, NaN, infinities
        assert_eq!(tonemap_scrgb_channel(-1.0, 1.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(tonemap_scrgb_channel(f32::NAN, 1.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(
            tonemap_scrgb_channel(f32::NEG_INFINITY, 1.0).to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            tonemap_scrgb_channel(f32::INFINITY, 1.0).to_bits(),
            1.0_f32.to_bits()
        );
    }

    #[test]
    fn sdr_ui_colors_preserved_across_sdr_white_levels() {
        // Mid-gray and UI colors strictly inside [0.0, 0.80]
        let ui_samples = [0.0_f32, 0.05, 0.18, 0.25, 0.50, 0.75, 0.80];

        let mut src_80 = Vec::with_capacity(ui_samples.len() * 8);
        let mut src_240 = Vec::with_capacity(ui_samples.len() * 8);
        let scale_240 = 240.0_f32 / DEFAULT_SDR_WHITE_LEVEL_NITS;

        for &v in &ui_samples {
            // Asymmetric (R, G, B) = (v, v * 0.5, 0.80 - v) to verify BGRA swizzle too.
            let r = v;
            let g = v * 0.5;
            let b = (0.80 - v).max(0.0);
            push_rgba16f_pixel(&mut src_80, r, g, b, 0.25);
            push_rgba16f_pixel(&mut src_240, r * scale_240, g * scale_240, b * scale_240, 1.0);
        }

        let width = ui_samples.len() as u32;
        let row_pitch = (width as usize) * 8;
        let frame_80 = tonemap_rgba16f_frame(&src_80, width, 1, row_pitch, 80.0).unwrap();
        let frame_240 = tonemap_rgba16f_frame(&src_240, width, 1, row_pitch, 240.0).unwrap();

        assert_eq!(frame_80.format, PixelFormat::Bgra8);
        assert_eq!(frame_80.stride, (width as usize) * 4);
        assert_eq!(frame_80.data.len(), (width as usize) * 4);

        for (i, &v) in ui_samples.iter().enumerate() {
            let r = v;
            let g = v * 0.5;
            let b = (0.80 - v).max(0.0);

            let exp_r = linear_to_srgb_u8(r);
            let exp_g = linear_to_srgb_u8(g);
            let exp_b = linear_to_srgb_u8(b);

            let p80 = &frame_80.data[i * 4..(i + 1) * 4];
            let p240 = &frame_240.data[i * 4..(i + 1) * 4];

            // BGRA channel order + opaque alpha
            assert_eq!(p80[3], 255);
            assert_eq!(p240[3], 255);

            assert!(
                p80[2].abs_diff(exp_r) <= 1 && p240[2].abs_diff(exp_r) <= 1,
                "R mismatch at v={v}: p80={}, p240={}, exp={exp_r}",
                p80[2],
                p240[2]
            );
            assert!(
                p80[1].abs_diff(exp_g) <= 1 && p240[1].abs_diff(exp_g) <= 1,
                "G mismatch at v={v}: p80={}, p240={}, exp={exp_g}",
                p80[1],
                p240[1]
            );
            assert!(
                p80[0].abs_diff(exp_b) <= 1 && p240[0].abs_diff(exp_b) <= 1,
                "B mismatch at v={v}: p80={}, p240={}, exp={exp_b}",
                p80[0],
                p240[0]
            );
            assert!(
                p80[2].abs_diff(p240[2]) <= 1
                    && p80[1].abs_diff(p240[1]) <= 1
                    && p80[0].abs_diff(p240[0]) <= 1,
                "80-nit vs 240-nit drift > 1 LSB at v={v}: p80={p80:?}, p240={p240:?}"
            );
        }
    }

    #[test]
    fn hdr_highlights_strictly_monotonic_and_special_values_safe() {
        // Channel-level strict monotonicity and non-clipping (< 1.0)
        let t_1_0 = tonemap_scrgb_channel(1.0, 1.0);
        let t_1_5 = tonemap_scrgb_channel(1.5, 1.0);
        let t_3_0 = tonemap_scrgb_channel(3.0, 1.0);
        let t_10_0 = tonemap_scrgb_channel(10.0, 1.0);

        assert!(t_1_0 > SDR_LINEAR_KNEE);
        assert!(t_1_0 < t_1_5);
        assert!(t_1_5 < t_3_0);
        assert!(t_3_0 < t_10_0);
        assert!(t_10_0 < 1.0);

        // Full RGBA16F -> BGRA8 frame monotonicity for 1.0, 1.5, 3.0, 10.0
        let mut src = Vec::new();
        for luma in [1.0_f32, 1.5, 3.0, 10.0] {
            push_rgba16f_pixel(&mut src, luma, luma, luma, 1.0);
        }
        let frame = tonemap_rgba16f_frame(&src, 4, 1, 32, 80.0).unwrap();
        let px_r = |idx: usize| frame.data[idx * 4 + 2];
        assert!(
            px_r(0) < px_r(1) && px_r(1) < px_r(2) && px_r(2) < px_r(3),
            "HDR highlights collapsed: {:?}",
            [px_r(0), px_r(1), px_r(2), px_r(3)]
        );
        assert!(px_r(1) < 255 && px_r(2) < 255);

        // NaN, +Inf, -Inf, negative scRGB never panic and clamp cleanly
        let mut special = Vec::new();
        push_raw_rgba16f_bits(&mut special, 0x7E00, 0xFC00, 0xC000, 0x7E00); // NaN, -Inf, -2.0
        push_raw_rgba16f_bits(&mut special, 0x7C00, 0x3C00, 0x0000, 0x0000); // +Inf, 1.0, 0.0
        let special_frame = tonemap_rgba16f_frame(&special, 2, 1, 16, 80.0).unwrap();

        // Pixel 0: R=NaN->0, G=-Inf->0, B=-2.0->0, A=255
        assert_eq!(&special_frame.data[0..4], &[0, 0, 0, 255]);
        // Pixel 1: B=0.0->0, G=1.0->246, R=+Inf->255, A=255
        assert_eq!(special_frame.data[4], 0);
        assert!(special_frame.data[5] >= 240 && special_frame.data[5] < 255);
        assert_eq!(special_frame.data[6], 255);
        assert_eq!(special_frame.data[7], 255);
    }

    #[test]
    fn row_pitch_padding_buffer_reuse_and_error_handling() {
        // 2x2 frame (min_row_bytes = 16) stored with row_pitch_bytes = 32 (16 bytes padding)
        let row_pitch = 32usize;
        let mut src = Vec::with_capacity(row_pitch * 2);

        // Row 0: pure red (0.5, 0, 0) and pure green (0, 0.5, 0)
        push_rgba16f_pixel(&mut src, 0.5, 0.0, 0.0, 0.0);
        push_rgba16f_pixel(&mut src, 0.0, 0.5, 0.0, 0.0);
        src.extend_from_slice(&[0xFF; 16]); // row 0 padding sentinel

        // Row 1: pure blue (0, 0, 0.5) and mid-gray (0.25, 0.25, 0.25)
        push_rgba16f_pixel(&mut src, 0.0, 0.0, 0.5, 0.0);
        push_rgba16f_pixel(&mut src, 0.25, 0.25, 0.25, 0.0);
        src.extend_from_slice(&[0xEE; 16]); // row 1 padding sentinel

        let exp_half = linear_to_srgb_u8(0.5);
        let exp_quarter = linear_to_srgb_u8(0.25);

        let frame = tonemap_rgba16f_frame(&src, 2, 2, row_pitch, 80.0).unwrap();
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.stride, 8);
        assert_eq!(frame.data.len(), 16);

        assert_eq!(&frame.data[0..4], &[0, 0, exp_half, 255]); // (0,0) red -> BGRA
        assert_eq!(&frame.data[4..8], &[0, exp_half, 0, 255]); // (1,0) green -> BGRA
        assert_eq!(&frame.data[8..12], &[exp_half, 0, 0, 255]); // (0,1) blue -> BGRA
        assert_eq!(
            &frame.data[12..16],
            &[exp_quarter, exp_quarter, exp_quarter, 255]
        ); // (1,1) gray

        // Exact minimum source slice without trailing padding after last row must succeed
        let min_slice = &src[..row_pitch + 16];
        let mut reused = vec![0xAA; 64]; // pre-filled with stale data
        tonemap_rgba16f_to_bgra8(min_slice, 2, 2, row_pitch, 80.0, &mut reused).unwrap();
        assert_eq!(reused, frame.data);

        // Error: row_pitch_bytes < width * 8
        let err_pitch = tonemap_rgba16f_frame(&src, 2, 2, 12, 80.0).unwrap_err();
        assert!(err_pitch.contains("row_pitch_bytes"), "unexpected err: {err_pitch}");

        // Error: truncated buffer (missing final byte of row 1)
        let truncated = &src[..row_pitch + 15];
        let err_trunc = tonemap_rgba16f_frame(truncated, 2, 2, row_pitch, 80.0).unwrap_err();
        assert!(
            err_trunc.contains("too small"),
            "unexpected err: {err_trunc}"
        );

        // Error: completely empty buffer for non-empty dimensions
        assert!(tonemap_rgba16f_frame(&[], 2, 2, 16, 80.0).is_err());
    }
}
