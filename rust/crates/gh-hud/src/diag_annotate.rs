//! Pure-Rust CV diagnostic annotator (`src/diagnostics.py::_render_annotated_image`).
//!
//! Renders Midnight Slate diagnostic banner, zoom slider box, orange waypoint candidates,
//! green matched inliers, and red player crosshair/heading arrow onto a `FrameView` with
//! full Vietnamese UTF-8 text via `cosmic-text` + `tiny-skia` (no ASCII transliteration).

use crate::fonts::{FontWeightRole, HudFonts, TextAlign, TextEffect, TextStyle};
use crate::{HudError, HudResult};
use cosmic_text::{Ellipsize, Wrap};
use gh_core::frame::{FrameBuf, FrameView, PixelFormat};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tiny_skia::{
    FillRule, LineCap, LineJoin, Paint, PathBuilder, Pixmap, PremultipliedColorU8, Rect, Stroke,
    Transform,
};

const BANNER_HEIGHT: f32 = 70.0;
const BANNER_FILL: [u8; 4] = [15, 23, 42, 217];
const BANNER_BORDER: [u8; 4] = [56, 189, 248, 255];
const STATUS_OK_COLOR: [u8; 4] = [74, 222, 128, 255];
const STATUS_FAIL_COLOR: [u8; 4] = [248, 113, 113, 255];
const INFO_TEXT_COLOR: [u8; 4] = [226, 232, 240, 255];
const SLIDER_COLOR: [u8; 4] = [255, 230, 0, 255];
const CANDIDATE_COLOR: [u8; 4] = [255, 140, 0, 255];
const INLIER_COLOR: [u8; 4] = [34, 197, 94, 255];
const PLAYER_COLOR: [u8; 4] = [255, 0, 0, 255];
const LABEL_SHADOW: TextEffect = TextEffect::Shadow {
    offset_x: 1,
    offset_y: 1,
    color: [0, 0, 0, 215],
};
const MAX_ANNOTATION_COORD: f64 = 1.0e6;

#[inline]
fn is_valid_annotation_coord(x: f64, y: f64) -> bool {
    x.is_finite()
        && y.is_finite()
        && x.abs() <= MAX_ANNOTATION_COORD
        && y.abs() <= MAX_ANNOTATION_COORD
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SliderAnnotation {
    #[serde(default)]
    pub found: bool,
    #[serde(default)]
    pub success: bool,
    #[serde(default)]
    pub box_coords: Option<[i32; 4]>,
    #[serde(default)]
    pub ratio_t: f64,
    #[serde(default)]
    pub scale: f64,
    #[serde(default)]
    pub score: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CandidateAnnotation {
    pub x: f64,
    pub y: f64,
    #[serde(default)]
    pub r: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InlierAnnotation {
    pub pixel_x: f64,
    pub pixel_y: f64,
    #[serde(default)]
    pub waypoint_id: Option<Value>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagnosticTelemetry {
    #[serde(default)]
    pub has_world_coords: bool,
    #[serde(default)]
    pub map_name: Option<String>,
    #[serde(default)]
    pub lat: f64,
    #[serde(default)]
    pub lng: f64,
    #[serde(default)]
    pub rmse: f64,
    #[serde(default)]
    pub inliers_count: usize,
    #[serde(default)]
    pub is_underground: bool,
    #[serde(default)]
    pub group_id: Option<i64>,
    #[serde(default)]
    pub floor_id: Option<i64>,
    #[serde(default)]
    pub floor_short: Option<String>,
    #[serde(default)]
    pub floor_name: Option<String>,
    #[serde(default)]
    pub floor_idx: Option<i64>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub slider_res: Option<SliderAnnotation>,
    #[serde(default)]
    pub candidates: Vec<CandidateAnnotation>,
    #[serde(default)]
    pub inliers: Vec<InlierAnnotation>,
    #[serde(default)]
    pub pixel_x: Option<f64>,
    #[serde(default)]
    pub pixel_y: Option<f64>,
    #[serde(default)]
    pub heading: Option<f64>,
}

fn frame_view_to_opaque_pixmap(frame: FrameView<'_>) -> HudResult<Pixmap> {
    if frame.width == 0 || frame.height == 0 {
        return Err(HudError::InvalidBuffer(format!(
            "diagnostic frame has zero dimension {}x{}",
            frame.width, frame.height
        )));
    }
    let ch = frame.format.channels();
    let row_bytes = (frame.width as usize)
        .checked_mul(ch)
        .ok_or_else(|| HudError::InvalidBuffer("frame row byte overflow".into()))?;
    if frame.stride < row_bytes {
        return Err(HudError::InvalidBuffer(format!(
            "frame stride {} < minimum row bytes {}",
            frame.stride, row_bytes
        )));
    }
    let needed = frame
        .stride
        .checked_mul(frame.height as usize)
        .ok_or_else(|| HudError::InvalidBuffer("frame buffer size overflow".into()))?;
    if frame.data.len() < needed {
        return Err(HudError::InvalidBuffer(format!(
            "frame data len {} < required {}",
            frame.data.len(),
            needed
        )));
    }

    let mut pixmap = Pixmap::new(frame.width, frame.height).ok_or_else(|| {
        HudError::Render(format!(
            "failed to allocate diagnostic pixmap {}x{}",
            frame.width, frame.height
        ))
    })?;
    let dst = pixmap.pixels_mut();
    let w = frame.width as usize;

    for y in 0..(frame.height as usize) {
        let src_row = &frame.data[y * frame.stride..y * frame.stride + row_bytes];
        let dst_row = &mut dst[y * w..(y + 1) * w];
        match frame.format {
            PixelFormat::Bgr8 => {
                for (px_in, px_out) in src_row.as_chunks::<3>().0.iter().zip(dst_row.iter_mut()) {
                    *px_out =
                        PremultipliedColorU8::from_rgba(px_in[2], px_in[1], px_in[0], 255)
                            .expect("opaque pixel valid");
                }
            }
            PixelFormat::Bgra8 => {
                for (px_in, px_out) in src_row.as_chunks::<4>().0.iter().zip(dst_row.iter_mut()) {
                    *px_out =
                        PremultipliedColorU8::from_rgba(px_in[2], px_in[1], px_in[0], 255)
                            .expect("opaque pixel valid");
                }
            }
            PixelFormat::Rgba8 => {
                for (px_in, px_out) in src_row.as_chunks::<4>().0.iter().zip(dst_row.iter_mut()) {
                    *px_out =
                        PremultipliedColorU8::from_rgba(px_in[0], px_in[1], px_in[2], 255)
                            .expect("opaque pixel valid");
                }
            }
            PixelFormat::Gray8 => {
                for (&g, px_out) in src_row.iter().zip(dst_row.iter_mut()) {
                    *px_out = PremultipliedColorU8::from_rgba(g, g, g, 255)
                        .expect("opaque pixel valid");
                }
            }
        }
    }

    Ok(pixmap)
}

fn format_waypoint_id(val: Option<&Value>) -> String {
    match val {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
    }
}

fn format_info_line(telemetry: &DiagnosticTelemetry) -> String {
    if telemetry.has_world_coords {
        let map_name = telemetry.map_name.as_deref().unwrap_or("Teyvat");
        let ug_str = if telemetry.is_underground {
            let fallback_floor;
            let fshort = if let Some(s) = telemetry.floor_short.as_deref().filter(|s| !s.is_empty())
            {
                s
            } else if let Some(s) = telemetry.floor_name.as_deref().filter(|s| !s.is_empty()) {
                s
            } else {
                fallback_floor = match telemetry.floor_idx {
                    Some(idx) => format!("Floor {idx}"),
                    None => "Floor None".to_owned(),
                };
                &fallback_floor
            };
            if let Some(gid) = telemetry.group_id {
                let fid_str = telemetry
                    .floor_id
                    .map_or_else(|| "None".to_owned(), |id| id.to_string());
                format!(" [G#{gid} Fl#{fid_str} {fshort}]")
            } else {
                let idx_str = telemetry
                    .floor_idx
                    .map_or_else(|| "None".to_owned(), |id| id.to_string());
                format!(" [Floor {idx_str}]")
            }
        } else {
            " [Surface]".to_owned()
        };

        format!(
            "World: ({:.1}, {:.1}){ug_str} | Map: {map_name} | Inliers: {} (RMSE: {:.1})",
            telemetry.lat, telemetry.lng, telemetry.inliers_count, telemetry.rmse
        )
    } else {
        let reason = telemetry.reason.as_deref().unwrap_or("Unknown");
        format!("Reason: {reason}")
    }
}

fn make_paint(rgba: [u8; 4]) -> Paint<'static> {
    let mut paint = Paint {
        anti_alias: true,
        ..Default::default()
    };
    paint.set_color_rgba8(rgba[0], rgba[1], rgba[2], rgba[3]);
    paint
}

fn stroke_circle(pixmap: &mut Pixmap, cx: f32, cy: f32, r: f32, width: f32, color: [u8; 4]) {
    if let Some(circle) = PathBuilder::from_circle(cx, cy, r) {
        let paint = make_paint(color);
        let stroke = Stroke {
            width,
            ..Default::default()
        };
        pixmap.stroke_path(&circle, &paint, &stroke, Transform::identity(), None);
    }
}

/// Renders CV annotations directly onto a copy of `frame` (preserving full Vietnamese UTF-8 text),
/// returning a `tiny_skia::Pixmap` (premultiplied RGBA8).
pub fn render_annotated_pixmap(
    fonts: &mut HudFonts,
    frame: FrameView<'_>,
    telemetry: &DiagnosticTelemetry,
) -> HudResult<Pixmap> {
    let mut pixmap = frame_view_to_opaque_pixmap(frame)?;
    let w = pixmap.width() as f32;

    // 1. Header Banner (85%-opaque Midnight Slate + 2px sky-blue divider at y=70)
    if let Some(banner_rect) = Rect::from_xywh(0.0, 0.0, w, BANNER_HEIGHT) {
        let banner_paint = make_paint(BANNER_FILL);
        pixmap.fill_rect(banner_rect, &banner_paint, Transform::identity(), None);
    }
    let mut div_pb = PathBuilder::new();
    div_pb.move_to(0.0, BANNER_HEIGHT);
    div_pb.line_to(w, BANNER_HEIGHT);
    if let Some(div_path) = div_pb.finish() {
        let div_paint = make_paint(BANNER_BORDER);
        let div_stroke = Stroke {
            width: 2.0,
            ..Default::default()
        };
        pixmap.stroke_path(&div_path, &div_paint, &div_stroke, Transform::identity(), None);
    }

    let (status_color, status_text) = if telemetry.has_world_coords {
        (STATUS_OK_COLOR, "STATUS: SUCCESS")
    } else {
        (STATUS_FAIL_COLOR, "STATUS: FAILED / NO WORLD COORDS")
    };
    let info_line1 = format_info_line(telemetry);

    {
        let mut pm_mut = pixmap.as_mut();
        let status_style = TextStyle::new(20.0, FontWeightRole::Bold, status_color);
        fonts.draw_text(
            &mut pm_mut,
            status_text,
            20.0,
            10.0,
            Some((w - 40.0).max(100.0)),
            TextAlign::Left,
            Wrap::None,
            Ellipsize::None,
            status_style,
            TextEffect::None,
        );

        let info_style = TextStyle::new(15.0, FontWeightRole::Regular, INFO_TEXT_COLOR);
        fonts.draw_text(
            &mut pm_mut,
            &info_line1,
            20.0,
            40.0,
            Some((w - 40.0).max(100.0)),
            TextAlign::Left,
            Wrap::None,
            Ellipsize::None,
            info_style,
            TextEffect::None,
        );
    }

    // 2. Zoom Slider overlay
    if let Some(slider) = telemetry.slider_res.as_ref()
        && (slider.found || slider.success)
        && let Some([x1, y1, x2, y2]) = slider.box_coords
    {
        let rx = x1.min(x2) as f32;
        let ry = y1.min(y2) as f32;
        let rw = (x2 as i64 - x1 as i64).abs().clamp(1, 1_000_000) as f32;
        let rh = (y2 as i64 - y1 as i64).abs().clamp(1, 1_000_000) as f32;
        if is_valid_annotation_coord(rx as f64, ry as f64) {
            let mut pb = PathBuilder::new();
            pb.move_to(rx, ry);
            pb.line_to(rx + rw, ry);
            pb.line_to(rx + rw, ry + rh);
            pb.line_to(rx, ry + rh);
            pb.close();
            if let Some(rect_path) = pb.finish() {
                let paint = make_paint(SLIDER_COLOR);
                let stroke = Stroke {
                    width: 2.0,
                    ..Default::default()
                };
                pixmap.stroke_path(&rect_path, &paint, &stroke, Transform::identity(), None);
            }

            let label = format!(
                "Zoom Slider (t={:.1}%, Scale={:.4})",
                slider.ratio_t * 100.0,
                slider.scale
            );
            let style = TextStyle::new(15.0, FontWeightRole::Semibold, SLIDER_COLOR);
            let mut pm_mut = pixmap.as_mut();
            fonts.draw_text(
                &mut pm_mut,
                &label,
                x1.max(x2) as f32 + 10.0,
                ((y1 as f32 + y2 as f32) * 0.5) - 9.0,
                None,
                TextAlign::Left,
                Wrap::None,
                Ellipsize::None,
                style,
                LABEL_SHADOW,
            );
        }
    }

    // 3. Waypoint Candidates (Orange circles, r=18)
    let cand_style = TextStyle::new(13.0, FontWeightRole::Regular, CANDIDATE_COLOR);
    for c in &telemetry.candidates {
        if !is_valid_annotation_coord(c.x, c.y) {
            continue;
        }
        let cx = c.x as f32;
        let cy = c.y as f32;
        stroke_circle(&mut pixmap, cx, cy, 18.0, 2.0, CANDIDATE_COLOR);
        let mut pm_mut = pixmap.as_mut();
        fonts.draw_text(
            &mut pm_mut,
            "Cand",
            cx - 15.0,
            cy - 36.0,
            None,
            TextAlign::Left,
            Wrap::None,
            Ellipsize::None,
            cand_style,
            LABEL_SHADOW,
        );
    }

    // 4. Matched Inliers (Green circles, r=22 + center dot r=4)
    let inlier_style = TextStyle::new(14.0, FontWeightRole::Bold, INLIER_COLOR);
    let inlier_dot_paint = make_paint(INLIER_COLOR);
    for inp in &telemetry.inliers {
        if !is_valid_annotation_coord(inp.pixel_x, inp.pixel_y) {
            continue;
        }
        let ix = inp.pixel_x as f32;
        let iy = inp.pixel_y as f32;
        stroke_circle(&mut pixmap, ix, iy, 22.0, 3.0, INLIER_COLOR);
        if let Some(dot) = PathBuilder::from_circle(ix, iy, 4.0) {
            pixmap.fill_path(&dot, &inlier_dot_paint, FillRule::Winding, Transform::identity(), None);
        }
        let w_id = format_waypoint_id(inp.waypoint_id.as_ref());
        let tag = format!("WP #{w_id}");
        let mut pm_mut = pixmap.as_mut();
        fonts.draw_text(
            &mut pm_mut,
            &tag,
            ix + 12.0,
            iy - 24.0,
            None,
            TextAlign::Left,
            Wrap::None,
            Ellipsize::None,
            inlier_style,
            LABEL_SHADOW,
        );
    }

    // 5. Player Arrow (Red circle r=24, cross size=32 -> +/-16, heading ray len=45 tipLength=0.35)
    if let (Some(px_val), Some(py_val)) = (telemetry.pixel_x, telemetry.pixel_y)
        && is_valid_annotation_coord(px_val, py_val)
    {
        let px = px_val as f32;
        let py = py_val as f32;
        let heading = telemetry.heading.unwrap_or(0.0);

        stroke_circle(&mut pixmap, px, py, 24.0, 3.0, PLAYER_COLOR);

        let player_paint = make_paint(PLAYER_COLOR);
        let mut cross_pb = PathBuilder::new();
        cross_pb.move_to(px - 16.0, py);
        cross_pb.line_to(px + 16.0, py);
        cross_pb.move_to(px, py - 16.0);
        cross_pb.line_to(px, py + 16.0);
        if let Some(cross_path) = cross_pb.finish() {
            let cross_stroke = Stroke {
                width: 2.0,
                line_cap: LineCap::Round,
                ..Default::default()
            };
            pixmap.stroke_path(
                &cross_path,
                &player_paint,
                &cross_stroke,
                Transform::identity(),
                None,
            );
        }

        // Heading ray (0 deg = Up, clockwise)
        if heading.is_finite() {
            let rad = (heading as f32).to_radians();
            let shaft_len = 45.0f32;
            let tip_x = px + shaft_len * rad.sin();
            let tip_y = py - shaft_len * rad.cos();
            let barb_len = shaft_len * 0.35;
            let back_angle = rad + std::f32::consts::PI;
            let spread = 25.0f32.to_radians();

            let mut arrow_pb = PathBuilder::new();
            arrow_pb.move_to(px, py);
            arrow_pb.line_to(tip_x, tip_y);
            for side in [-1.0f32, 1.0f32] {
                let ang = back_angle + side * spread;
                let bx = tip_x + barb_len * ang.sin();
                let by = tip_y - barb_len * ang.cos();
                arrow_pb.move_to(tip_x, tip_y);
                arrow_pb.line_to(bx, by);
            }
            if let Some(arrow_path) = arrow_pb.finish() {
                let arrow_stroke = Stroke {
                    width: 3.0,
                    line_cap: LineCap::Round,
                    line_join: LineJoin::Round,
                    ..Default::default()
                };
                pixmap.stroke_path(
                    &arrow_path,
                    &player_paint,
                    &arrow_stroke,
                    Transform::identity(),
                    None,
                );
            }
        }

        let label = format!("PLAYER ({heading:.0} deg)");
        let player_style = TextStyle::new(16.0, FontWeightRole::Bold, PLAYER_COLOR);
        let mut pm_mut = pixmap.as_mut();
        fonts.draw_text(
            &mut pm_mut,
            &label,
            px + 28.0,
            py - 4.0,
            None,
            TextAlign::Left,
            Wrap::None,
            Ellipsize::None,
            player_style,
            LABEL_SHADOW,
        );
    }

    Ok(pixmap)
}

/// Convenience wrapper accepting untyped `cv_telemetry` JSON (`Value`) and returning straight `Rgba8` `FrameBuf`.
pub fn render_annotated_frame(
    fonts: &mut HudFonts,
    frame: FrameView<'_>,
    telemetry_json: &Value,
) -> HudResult<FrameBuf> {
    let telemetry: DiagnosticTelemetry = serde_json::from_value(telemetry_json.clone())?;
    let pixmap = render_annotated_pixmap(fonts, frame, &telemetry)?;
    let w = pixmap.width();
    let h = pixmap.height();
    let mut out = FrameBuf::new_zeroed(w, h, PixelFormat::Rgba8);

    for (src, dst) in pixmap
        .pixels()
        .iter()
        .zip(out.data.as_chunks_mut::<4>().0.iter_mut())
    {
        let a = src.alpha();
        if a == 255 {
            dst[0] = src.red();
            dst[1] = src.green();
            dst[2] = src.blue();
            dst[3] = 255;
        } else if a == 0 {
            dst.fill(0);
        } else {
            let af = a as u16;
            dst[0] = (((src.red() as u16) * 255 + af / 2) / af).min(255) as u8;
            dst[1] = (((src.green() as u16) * 255 + af / 2) / af).min(255) as u8;
            dst[2] = (((src.blue() as u16) * 255 + af / 2) / af).min(255) as u8;
            dst[3] = a;
        }
    }
    Ok(out)
}

/// Renders and encodes `04_cv_annotated.png` bytes directly in memory via `Pixmap::encode_png()`.
pub fn render_annotated_png_bytes(
    fonts: &mut HudFonts,
    frame: FrameView<'_>,
    telemetry: &DiagnosticTelemetry,
) -> HudResult<Vec<u8>> {
    let pixmap = render_annotated_pixmap(fonts, frame, telemetry)?;
    pixmap
        .encode_png()
        .map_err(|err| HudError::PngEncode(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn renders_vietnamese_success_and_failure_png_with_expected_pixels() {
        let mut fonts = HudFonts::new();

        // Synthetic 640x360 Bgr8 frame (bright gray terrain so 85% slate banner clearly darkens it)
        let mut frame = FrameBuf::new_zeroed(640, 360, PixelFormat::Bgr8);
        frame.data.fill(200);

        let success_telemetry = DiagnosticTelemetry {
            has_world_coords: true,
            map_name: Some("Bản Đồ Tương Tác Teyvat".to_owned()),
            lat: 1234.5,
            lng: -678.9,
            rmse: 2.3,
            inliers_count: 3,
            is_underground: true,
            group_id: Some(12),
            floor_id: Some(34),
            floor_short: Some("Ngọn Nguồn - Đáy Hồ B1".to_owned()),
            slider_res: Some(SliderAnnotation {
                found: false,
                success: true, // Also exercises older `03_report.json` `success`-only schema!
                box_coords: Some([40, 120, 62, 260]),
                ratio_t: 0.425,
                scale: 1.2345,
                score: 0.91,
            }),
            candidates: vec![CandidateAnnotation {
                x: 180.0,
                y: 160.0,
                r: Some(18.0),
            }],
            inliers: vec![InlierAnnotation {
                pixel_x: 310.0,
                pixel_y: 200.0,
                waypoint_id: Some(json!(50047)),
                name: Some("Điểm dịch chuyển".to_owned()),
            }],
            pixel_x: Some(460.0),
            pixel_y: Some(220.0),
            heading: Some(113.0),
            ..Default::default()
        };

        let png_bytes = render_annotated_png_bytes(&mut fonts, frame.as_view(), &success_telemetry)
            .expect("render success png");
        assert!(png_bytes.starts_with(b"\x89PNG\r\n\x1a\n"));

        let decoded = Pixmap::decode_png(&png_bytes).expect("decode success png");
        assert_eq!(decoded.width(), 640);
        assert_eq!(decoded.height(), 360);

        // Verify Midnight Slate banner darkened background at (10, 10)
        let banner_px = decoded.pixel(10, 10).expect("banner pixel");
        assert!(banner_px.red() < 60 && banner_px.green() < 60 && banner_px.blue() < 80);

        // Verify sky-blue divider line around y=70
        let div_px = decoded.pixel(300, 70).expect("divider pixel");
        assert!(div_px.blue() > 150 && div_px.green() > 120);

        // Verify text pixels rendered inside banner (x in 20..560, y in 10..62)
        let mut bright_text_pixels = 0usize;
        for y in 10..62 {
            for x in 20..560 {
                let px = decoded.pixel(x, y).expect("banner text pixel");
                if px.green() > 160 || px.red() > 160 {
                    bright_text_pixels += 1;
                }
            }
        }
        assert!(
            bright_text_pixels > 150,
            "expected crisp banner glyph pixels, got {bright_text_pixels}"
        );

        // Verify red player center crosshair at (460, 220)
        let player_px = decoded.pixel(460, 220).expect("player center");
        assert!(player_px.red() > 200 && player_px.green() < 50 && player_px.blue() < 50);

        // Verify green inlier center dot at (310, 200)
        let inlier_px = decoded.pixel(310, 200).expect("inlier center");
        assert!(inlier_px.green() > 150 && inlier_px.red() < 80);

        // Failure case with full Vietnamese UTF-8 reason + untyped JSON wrapper
        let fail_json = json!({
            "has_world_coords": false,
            "reason": "Không tìm thấy mũi tên người chơi (Player Arrow) trên màn hình."
        });
        let fail_frame = render_annotated_frame(&mut fonts, frame.as_view(), &fail_json)
            .expect("render failure frame");
        assert_eq!(fail_frame.width, 640);
        assert_eq!(fail_frame.height, 360);
        assert_eq!(fail_frame.format, PixelFormat::Rgba8);

        // Red STATUS: FAILED glyphs in header
        let mut fail_red_pixels = 0usize;
        for y in 10..34 {
            for x in 20..420 {
                let idx = (y * 640 + x) * 4;
                if fail_frame.data[idx] > 180 && fail_frame.data[idx + 1] < 140 {
                    fail_red_pixels += 1;
                }
            }
        }
        assert!(
            fail_red_pixels > 50,
            "expected red failure status glyphs, got {fail_red_pixels}"
        );
    }

    #[test]
    fn supports_bgra8_with_zero_alpha_padding_and_gray8() {
        let mut fonts = HudFonts::new();
        let telemetry = DiagnosticTelemetry::default();

        // GDI BitBlt often leaves byte 3 = 0 in Bgra8; output must still be opaque!
        let mut bgra = FrameBuf::new_zeroed(120, 90, PixelFormat::Bgra8);
        for px in bgra.data.as_chunks_mut::<4>().0 {
            px.copy_from_slice(&[50, 100, 150, 0]);
        }
        let pm_bgra = render_annotated_pixmap(&mut fonts, bgra.as_view(), &telemetry)
            .expect("bgra8 diagnostic render");
        let corner = pm_bgra.pixel(110, 85).expect("corner");
        assert_eq!(corner.alpha(), 255);
        assert_eq!(corner.red(), 150);
        assert_eq!(corner.green(), 100);
        assert_eq!(corner.blue(), 50);

        let mut gray = FrameBuf::new_zeroed(120, 90, PixelFormat::Gray8);
        gray.data.fill(180);
        let pm_gray = render_annotated_pixmap(&mut fonts, gray.as_view(), &telemetry)
            .expect("gray8 diagnostic render");
        let gc = pm_gray.pixel(110, 85).expect("gray corner");
        assert_eq!(gc.alpha(), 255);
        assert_eq!(gc.red(), 180);
    }

    #[test]
    fn outlier_and_non_finite_telemetry_coordinates_do_not_panic() {
        let mut fonts = HudFonts::new();
        let mut frame = FrameBuf::new_zeroed(320, 240, PixelFormat::Bgr8);
        frame.data.fill(160);

        // Rust callers passing f64::INFINITY / f64::NAN / extreme finite values
        let extreme_telemetry = DiagnosticTelemetry {
            has_world_coords: true,
            map_name: Some("Teyvat".to_owned()),
            lat: f64::NAN,
            lng: f64::INFINITY,
            rmse: f64::NEG_INFINITY,
            slider_res: Some(SliderAnnotation {
                found: true,
                success: true,
                box_coords: Some([120, 140, 40, 100]), // inverted box_coords
                ratio_t: 0.5,
                scale: 1.0,
                score: 0.9,
            }),
            candidates: vec![
                CandidateAnnotation {
                    x: 1.0e10,
                    y: 100.0,
                    r: Some(18.0),
                },
                CandidateAnnotation {
                    x: -1.0e10,
                    y: f64::NAN,
                    r: None,
                },
            ],
            inliers: vec![
                InlierAnnotation {
                    pixel_x: 100.0,
                    pixel_y: f64::INFINITY,
                    waypoint_id: Some(json!(1)),
                    name: None,
                },
                InlierAnnotation {
                    pixel_x: f64::NEG_INFINITY,
                    pixel_y: -1.0e10,
                    waypoint_id: Some(json!("2")),
                    name: None,
                },
            ],
            pixel_x: Some(f64::INFINITY),
            pixel_y: Some(100.0),
            heading: Some(f64::NAN),
            ..Default::default()
        };

        let png = render_annotated_png_bytes(&mut fonts, frame.as_view(), &extreme_telemetry)
            .expect("extreme non-finite telemetry must render cleanly");
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));

        // Player finite + NaN heading must also render cleanly
        let mut finite_player = extreme_telemetry.clone();
        finite_player.pixel_x = Some(160.0);
        finite_player.pixel_y = Some(150.0);
        finite_player.heading = Some(f64::NAN);
        let pm_player = render_annotated_pixmap(&mut fonts, frame.as_view(), &finite_player)
            .expect("finite player with NaN heading must render cleanly");
        assert_eq!(pm_player.width(), 320);

        // Valid JSON input with extreme finite outlier coordinates (>= i32::MAX)
        let json_outliers = json!({
            "has_world_coords": false,
            "reason": "Outlier test",
            "slider_res": {
                "found": true,
                "box_coords": [160, 180, 60, 110],
                "ratio_t": 0.25,
                "scale": 1.1
            },
            "candidates": [
                {"x": 1e10, "y": 100.0},
                {"x": -1e10, "y": -1e10}
            ],
            "inliers": [
                {"pixel_x": 100.0, "pixel_y": -1e10, "waypoint_id": 42}
            ],
            "pixel_x": 1e12,
            "pixel_y": 100.0,
            "heading": 45.0
        });
        let out_frame = render_annotated_frame(&mut fonts, frame.as_view(), &json_outliers)
            .expect("JSON outlier coordinates must render cleanly");
        assert_eq!(out_frame.width, 320);
        assert_eq!(out_frame.height, 240);
    }
}
