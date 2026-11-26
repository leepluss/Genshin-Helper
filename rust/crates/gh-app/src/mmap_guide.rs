//! M-Map route guidance stages (pure): from the active route and the registered M-Map view to
//! the [`MMapGuideScene`] the overlay draws.
//!
//! Mirrors the HoYoLAB web flow — whole route → section → teleport — and is driven only by what
//! the map currently SHOWS (the registered view), never by the player position:
//! 1. view on another map → banner naming the route's map;
//! 2. the active section's teleport waypoint on screen and the view zoomed in enough → ring on
//!    the waypoint + "Teleport here" card (section label, underground warning);
//! 3. the section on screen but small → outline + "zoom in here"; zoomed in with the waypoint
//!    off-screen → edge arrow to the waypoint;
//! 4. the section off-screen → edge arrow to it while any of the route is on screen, else an
//!    edge arrow to the whole route.
//!
//! Without a registered view (or with the map drawer open) [`compute_fallback_scene`] gives a
//! text-only banner naming the route, the section and the area / map to go to.

use gh_core::frame::RectI32;
use gh_core::i18n::t_lang;
use gh_core::multimap::MultimapDb;
use gh_core::nav_controller::NavPin;
use gh_core::route_doc::{
    ROUTE_FORMAT, ROUTE_SCHEMA_VERSION, RouteAnchor, RouteDocument, RouteSection, RouteSource,
    RouteStop, RouteStopKind, TELEPORT_LABEL_IDS,
};
use gh_cv::locator::DEFAULT_AREA_NAMES;
use gh_cv::view_tracker::ViewTransform;
use gh_cv::zoom_slider::map_scale_multiplier;
use gh_hud::mmap_overlay::{
    MMapEdgeArrow, MMapGuideScene, MMapGuideStage, MMapOverlayMode, MMapPath, MMapTeleportMarker,
    parse_hex_rgba,
};

/// Distance threshold (in world units / meters) above which a same-map single-pin lock also
/// activates M-Map teleport guidance.
pub const SINGLE_PIN_MMAP_DISTANCE_M: f64 = 180.0;

/// Region where guidance targets count as "on screen" (fractions of the client: x0, y0, x1, y1);
/// keeps clear of the zoom slider, the top bar / right buttons and the bottom-right map switcher.
const SAFE: [f64; 4] = [0.06, 0.08, 0.95, 0.90];
/// Edge arrows sit on this inset rectangle.
const ARROW_RECT: [f64; 4] = [0.09, 0.14, 0.92, 0.86];
/// The section counts as "zoomed in" when its bounding box spans this fraction of the client
/// height, or the view scale (world units per px @1440p, Teyvat-normalised) is at most
/// `TELEPORT_MAX_SCALE_1440` (mid zoom; slider scale runs 1.92 → 0.32).
const TELEPORT_MIN_SECTION_FRAC: f64 = 0.18;
const TELEPORT_MAX_SCALE_1440: f64 = 0.9;
/// Off-screen points are clamped to this many client sizes (keeps path geometry finite).
const CLAMP_SCREENS: f64 = 8.0;
/// A section's area is that of the nearest map waypoint within this many world units.
const AREA_MAX_DIST: f64 = 3000.0;

/// Inputs for one frame. `done[s][i]` = stop `i` of section `s` collected.
pub struct GuideInput<'a> {
    pub doc: &'a RouteDocument,
    pub active: usize,
    pub done: &'a [Vec<bool>],
    pub view: ViewTransform,
    pub game_rect: RectI32,
    pub mode: MMapOverlayMode,
    pub lang: &'a str,
    /// Localised name of the route's map.
    pub route_map_name: &'a str,
    /// Localised name of the map the view shows.
    pub view_map_name: &'a str,
}

fn tr(key: &str, lang: &str, args: &[(&str, &str)]) -> String {
    t_lang(key, Some(lang), args)
}

pub fn section_label(doc: &RouteDocument, idx: usize) -> String {
    let n = doc.sections.len().max(1);
    let name = doc.sections.get(idx).map_or("", |s| s.name.trim());
    if name.is_empty() {
        format!("S{}/{n}", idx + 1)
    } else {
        format!("S{}/{n} · {name}", idx + 1)
    }
}

type Box4 = [f64; 4];

fn grow(b: Option<Box4>, x: f64, y: f64) -> Option<Box4> {
    Some(match b {
        None => [x, y, x, y],
        Some(b) => [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)],
    })
}

fn center(b: Box4) -> (f64, f64) {
    ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)
}

fn inside(r: Box4, x: f64, y: f64) -> bool {
    x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]
}

fn intersects(a: Box4, b: Box4) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

fn frac_rect(f: [f64; 4], w: f64, h: f64) -> Box4 {
    [f[0] * w, f[1] * h, f[2] * w, f[3] * h]
}

fn px(v: f64, extent: f64) -> f32 {
    v.clamp(-CLAMP_SCREENS * extent, (CLAMP_SCREENS + 1.0) * extent).round() as f32
}

/// Edge arrow on [`ARROW_RECT`] pointing from the client centre toward `(x, y)`.
fn edge_arrow(x: f64, y: f64, w: f64, h: f64, label: String) -> MMapEdgeArrow {
    let r = frac_rect(ARROW_RECT, w, h);
    let (cx, cy) = ((r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5);
    let (dx, dy) = (x - cx, y - cy);
    let (hx, hy) = ((r[2] - r[0]) * 0.5, (r[3] - r[1]) * 0.5);
    let k = if dx.abs() < 1e-9 && dy.abs() < 1e-9 {
        0.0
    } else {
        let kx = if dx.abs() > 1e-9 { hx / dx.abs() } else { f64::INFINITY };
        let ky = if dy.abs() > 1e-9 { hy / dy.abs() } else { f64::INFINITY };
        kx.min(ky).min(1.0)
    };
    MMapEdgeArrow {
        x: (cx + dx * k).round() as f32,
        y: (cy + dy * k).round() as f32,
        // Quantised to 1° so a static view does not redraw on sub-pixel noise.
        angle_rad: (dy.atan2(dx).to_degrees().round()).to_radians() as f32,
        label,
    }
}

pub fn compute_guide_scene(inp: &GuideInput<'_>) -> Option<MMapGuideScene> {
    if inp.mode == MMapOverlayMode::Off {
        return None;
    }
    let doc = inp.doc;
    let sec = doc.sections.get(inp.active)?;
    let (w, h) = (f64::from(inp.game_rect.width), f64::from(inp.game_rect.height));
    if w < 320.0 || h < 240.0 || !(inp.view.scale.is_finite() && inp.view.scale > 0.0) {
        return None;
    }
    let lang = inp.lang;
    let accent = parse_hex_rgba(&sec.color, 232);
    let sec_label = section_label(doc, inp.active);
    let mut scene = MMapGuideScene {
        mode: inp.mode,
        game_rect: inp.game_rect,
        stage: MMapGuideStage::Route,
        accent,
        paths: Vec::new(),
        focus_box: None,
        focus_label: String::new(),
        arrow: None,
        teleport: None,
        banner: None,
    };

    if inp.view.map_id != doc.map_id {
        scene.stage = MMapGuideStage::WrongMap;
        scene.banner = Some((
            tr("mmap.wrong_map_title", lang, &[("map", inp.route_map_name)]),
            tr("mmap.wrong_map_sub", lang, &[("current", inp.view_map_name)]),
        ));
        return Some(scene);
    }

    let v = inp.view;
    let to_screen = |lat: f64, lng: f64| v.world_to_screen(lng, lat);

    // Screen-space boxes of the whole route and of the active section: their stops (the
    // teleport only for a section without stops — an auto-suggested waypoint can lie well
    // outside the section and must not drag its centre off-screen).
    let mut route_box: Option<Box4> = None;
    let mut sec_box: Option<Box4> = None;
    for (si, s) in doc.sections.iter().enumerate() {
        let mut pts: Vec<(f64, f64)> = s.stops.iter().map(|st| to_screen(st.lat, st.lng)).collect();
        let anchor_only = pts.is_empty();
        if anchor_only && let Some(a) = &s.anchor_teleport {
            pts.push(to_screen(a.lat, a.lng));
        }
        for &(x, y) in &pts {
            route_box = grow(route_box, x, y);
            if si == inp.active {
                sec_box = grow(sec_box, x, y);
            }
        }
        if anchor_only {
            continue;
        }
        if inp.mode == MMapOverlayMode::Full {
            scene.paths.push(MMapPath {
                points: pts.iter().map(|&(x, y)| [px(x, w), px(y, h)]).collect(),
                edges: s.edges.clone(),
                done: inp.done.get(si).cloned().unwrap_or_default(),
                color: parse_hex_rgba(&s.color, 232),
                emphasized: si == inp.active,
            });
        }
    }
    let (route_box, sec_box) = (route_box?, sec_box?);

    let safe = frac_rect(SAFE, w, h);
    let sec_px = (sec_box[2] - sec_box[0]).max(sec_box[3] - sec_box[1]);
    let scale_1440 = v.scale * h / 1440.0 / map_scale_multiplier(v.map_id).max(1e-6);
    let zoomed_in = sec_px >= TELEPORT_MIN_SECTION_FRAC * h || scale_1440 <= TELEPORT_MAX_SCALE_1440;
    let anchor = sec.anchor_teleport.as_ref().map(|a| (a, to_screen(a.lat, a.lng)));
    let (scx, scy) = center(sec_box);

    if zoomed_in && let Some((a, (ax, ay))) = anchor.filter(|(_, (x, y))| inside(safe, *x, *y)) {
        scene.stage = MMapGuideStage::Teleport;
        let warn = a.is_underground.then(|| {
            let floor = a.floor_name.trim();
            if floor.is_empty() {
                tr("mmap.underground_warn_plain", lang, &[])
            } else {
                tr("mmap.underground_warn", lang, &[("floor", floor)])
            }
        });
        scene.teleport = Some(MMapTeleportMarker {
            x: px(ax, w),
            y: px(ay, h),
            title: tr("mmap.teleport_here", lang, &[]),
            subtitle: sec_label,
            warn,
        });
        return Some(scene);
    }

    if inside(safe, scx, scy) {
        scene.stage = MMapGuideStage::Section;
        if zoomed_in {
            if let Some((_, (ax, ay))) = anchor {
                // Section on screen, its waypoint just off it.
                scene.stage = MMapGuideStage::Teleport;
                scene.arrow = Some(edge_arrow(ax, ay, w, h, tr("mmap.teleport_point", lang, &[("section", &sec_label)])));
            } else {
                scene.focus_box = Some(sec_box.map(|c| c.round() as f32));
                scene.focus_label = sec_label;
            }
        } else {
            scene.focus_box = Some(sec_box.map(|c| c.round() as f32));
            scene.focus_label = tr("mmap.section_zoom_in", lang, &[("section", &sec_label)]);
        }
        return Some(scene);
    }

    if intersects(route_box, safe) {
        scene.stage = MMapGuideStage::Section;
        scene.arrow = Some(edge_arrow(scx, scy, w, h, sec_label));
        return Some(scene);
    }

    scene.stage = MMapGuideStage::Route;
    let title = if doc.title.trim().is_empty() { inp.route_map_name } else { doc.title.trim() };
    let (rcx, rcy) = center(route_box);
    scene.arrow = Some(edge_arrow(rcx, rcy, w, h, tr("mmap.route_label", lang, &[("route", title)])));
    Some(scene)
}

/// Inputs for the text-only fallback banner.
pub struct FallbackInput<'a> {
    pub doc: &'a RouteDocument,
    pub active: usize,
    pub game_rect: RectI32,
    pub mode: MMapOverlayMode,
    pub lang: &'a str,
    /// Localised name of the route's map.
    pub route_map_name: &'a str,
    /// Area of the active section ([`section_area_name`]).
    pub region_name: Option<&'a str>,
    /// The map drawer is open: tell which map to pick.
    pub drawer_open: bool,
}

/// Banner shown while nothing can be pointed at on the M-Map (no registered view, or the map
/// drawer covering the right of the screen).
pub fn compute_fallback_scene(inp: &FallbackInput<'_>) -> Option<MMapGuideScene> {
    if inp.mode == MMapOverlayMode::Off || inp.game_rect.width < 320 || inp.game_rect.height < 240 {
        return None;
    }
    let doc = inp.doc;
    let sec = doc.sections.get(inp.active)?;
    let lang = inp.lang;
    let map = inp.route_map_name.trim();
    let route = if doc.title.trim().is_empty() { map } else { doc.title.trim() };
    let title = tr(
        "mmap.fallback_title",
        lang,
        &[("route", route), ("section", &section_label(doc, inp.active))],
    );
    let sub = match inp.region_name.map(str::trim).filter(|r| !r.is_empty()) {
        _ if inp.drawer_open => tr("mmap.fallback_pick_map", lang, &[("map", map)]),
        Some(region) => tr("mmap.fallback_region", lang, &[("region", region), ("map", map)]),
        None => tr("mmap.fallback_map", lang, &[("map", map)]),
    };
    Some(MMapGuideScene {
        mode: inp.mode,
        game_rect: inp.game_rect,
        stage: MMapGuideStage::Locating,
        accent: parse_hex_rgba(&sec.color, 232),
        paths: Vec::new(),
        focus_box: None,
        focus_label: String::new(),
        arrow: None,
        teleport: None,
        banner: Some((title, sub)),
    })
}

/// Area of the active section: the area of the route map's waypoint nearest to the section's
/// teleport (or first stop), within [`AREA_MAX_DIST`]. Area names come from the HoYoLAB sync
/// (`vi-vn`); English UIs get the built-in English name of the Teyvat nations where known.
pub fn section_area_name(db: &MultimapDb, doc: &RouteDocument, active: usize, lang: &str) -> Option<String> {
    let sec = doc.sections.get(active)?;
    let (lat, lng) = sec
        .anchor_teleport
        .as_ref()
        .map(|a| (a.lat, a.lng))
        .or_else(|| sec.stops.first().map(|s| (s.lat, s.lng)))?;
    let meta = db.maps.get(&doc.map_id)?;
    let dist = |x: f64, y: f64| (x - lng).hypot(y - lat);
    let wp = meta
        .waypoints
        .iter()
        .filter(|w| w.area_id > 0)
        .min_by(|a, b| dist(a.x_pos, a.y_pos).total_cmp(&dist(b.x_pos, b.y_pos)))?;
    if dist(wp.x_pos, wp.y_pos) > AREA_MAX_DIST {
        return None;
    }
    let from_db = meta
        .areas
        .iter()
        .find(|a| a.id == wp.area_id)
        .map(|a| a.name.trim().to_owned())
        .filter(|n| !n.is_empty());
    let builtin = (doc.map_id == 2)
        .then(|| DEFAULT_AREA_NAMES.iter().find(|(id, _)| *id == wp.area_id))
        .flatten()
        .map(|(_, n)| (*n).to_owned());
    if lang.trim().to_ascii_lowercase().starts_with("en") {
        builtin.or(from_db)
    } else {
        from_db.or(builtin)
    }
}

fn find_nearest_single_pin_anchor(db: &MultimapDb, pin: &NavPin) -> Option<RouteAnchor> {
    if !pin.lat.is_finite() || !pin.lng.is_finite() {
        return None;
    }
    let wps = db.waypoints_for_map(pin.map_id);
    if wps.is_empty() {
        return None;
    }
    let has_teleport = wps.iter().any(|wp| {
        wp.x_pos.is_finite()
            && wp.y_pos.is_finite()
            && (wp.is_primary || TELEPORT_LABEL_IDS.contains(&wp.label_id))
    });
    let want_ug = pin.is_underground;
    let want_fid = pin.floor_id.as_deref().and_then(|s| s.parse::<i32>().ok());

    let mut best: Option<(&gh_core::multimap::WaypointEntry, f64)> = None;
    for wp in wps {
        if !wp.x_pos.is_finite() || !wp.y_pos.is_finite() {
            continue;
        }
        if has_teleport && !wp.is_primary && !TELEPORT_LABEL_IDS.contains(&wp.label_id) {
            continue;
        }
        let wp_fid = wp
            .point_group
            .as_ref()
            .and_then(|pg| pg.floor_id)
            .filter(|&f| f > 0);
        let wp_ug = wp_fid.is_some() || wp.z_level != 0 || matches!(wp.icon_sign, 1..=3);
        let dist = (wp.y_pos - pin.lat).hypot(wp.x_pos - pin.lng);
        let penalty = match (want_fid, wp_fid) {
            (Some(wf), Some(pf)) if wf == pf => 0.0,
            _ if want_ug == wp_ug => 40.0,
            _ => 320.0,
        };
        let score = dist + penalty;
        if best.is_none_or(|(_, bscore)| score < bscore) {
            best = Some((wp, score));
        }
    }

    let (wp, _) = best?;
    let wp_fid = wp
        .point_group
        .as_ref()
        .and_then(|pg| pg.floor_id)
        .filter(|&f| f > 0);
    let wp_ug = wp_fid.is_some() || wp.z_level != 0 || matches!(wp.icon_sign, 1..=3);
    let floor_name = wp_fid
        .and_then(|fid| db.resolve_floor_label(pin.map_id, fid))
        .unwrap_or_default();
    Some(RouteAnchor {
        point_id: Some(wp.id.to_string()),
        lat: wp.y_pos,
        lng: wp.x_pos,
        label_id: if wp.label_id > 0 { wp.label_id } else { 3 },
        floor_id: wp_fid.map(|f| f.to_string()),
        floor_name,
        is_underground: wp_ug,
        auto_suggested: true,
    })
}

/// Builds a synthetic 1-section [`RouteDocument`] for a single locked [`NavPin`] so existing
/// [`compute_guide_scene`] and [`compute_fallback_scene`] provide seamless M-Map teleport guidance.
pub fn build_single_pin_route_doc(db: &MultimapDb, pin: &NavPin, map_name: &str) -> RouteDocument {
    let raw_title = pin.display_name();
    let title = if !raw_title.trim().is_empty() {
        raw_title.trim().to_owned()
    } else if !pin.id.trim().is_empty() {
        format!("Pin {}", pin.id.trim())
    } else {
        "Pin".to_owned()
    };
    let resolved_map_name = if !map_name.trim().is_empty() {
        map_name.trim().to_owned()
    } else {
        db.map_name(pin.map_id).unwrap_or("").to_owned()
    };
    let anchor_teleport = find_nearest_single_pin_anchor(db, pin);
    let stop = RouteStop {
        id: pin.id.clone(),
        kind: RouteStopKind::Collectible,
        lat: pin.lat,
        lng: pin.lng,
        label_id: pin.label_id.and_then(|v| i32::try_from(v).ok()).unwrap_or(0),
        floor_id: pin.floor_id.clone(),
        floor_name: pin.floor_name_str().to_owned(),
        is_underground: pin.is_underground,
        name: title.clone(),
        note: pin.description.clone(),
        img_url: pin.image_url.clone(),
        video_url: String::new(),
    };
    RouteDocument {
        format: ROUTE_FORMAT.to_owned(),
        version: ROUTE_SCHEMA_VERSION,
        title: title.clone(),
        description: pin.description.clone(),
        map_id: pin.map_id,
        map_name: resolved_map_name,
        source: RouteSource {
            provider: "single_pin".to_owned(),
            ..Default::default()
        },
        sections: vec![RouteSection {
            index: 0,
            name: title,
            color: "#fbbf24".to_owned(),
            anchor_teleport,
            stops: vec![stop],
            edges: Vec::new(),
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: i32 = 2560;
    const H: i32 = 1440;

    fn stop(lat: f64, lng: f64) -> RouteStop {
        RouteStop {
            lat,
            lng,
            ..Default::default()
        }
    }

    /// Two sections on Teyvat: S1 around world (x 1000..1200, y 500..600), S2 far east around
    /// x 9000..9300, y 800..900, with its teleport at (9100, 950).
    fn doc() -> RouteDocument {
        RouteDocument {
            title: "Băng Tuyết".into(),
            map_id: 2,
            sections: vec![
                RouteSection {
                    index: 0,
                    name: "Đầu tuyến".into(),
                    color: "#38bdf8".into(),
                    anchor_teleport: Some(RouteAnchor { lat: 520.0, lng: 980.0, ..Default::default() }),
                    stops: vec![stop(500.0, 1000.0), stop(600.0, 1200.0)],
                    edges: vec![],
                },
                RouteSection {
                    index: 1,
                    name: "Vực Hải Sao".into(),
                    color: "#fde047".into(),
                    anchor_teleport: Some(RouteAnchor {
                        lat: 950.0,
                        lng: 9100.0,
                        is_underground: true,
                        floor_name: "B1".into(),
                        ..Default::default()
                    }),
                    stops: vec![stop(800.0, 9000.0), stop(900.0, 9300.0)],
                    edges: vec![],
                },
            ],
            ..Default::default()
        }
    }

    /// View whose screen centre shows world (cx, cy) at `scale` world units / px.
    fn view(cx: f64, cy: f64, scale: f64) -> ViewTransform {
        ViewTransform {
            map_id: 2,
            scale,
            tx: cx - scale * f64::from(W) / 2.0,
            ty: cy - scale * f64::from(H) / 2.0,
        }
    }

    fn scene_for(d: &RouteDocument, active: usize, v: ViewTransform) -> MMapGuideScene {
        let done = vec![vec![false; 2]; d.sections.len()];
        compute_guide_scene(&GuideInput {
            doc: d,
            active,
            done: &done,
            view: v,
            game_rect: RectI32 { x: 0, y: 0, width: W, height: H },
            mode: MMapOverlayMode::Full,
            lang: "vi",
            route_map_name: "Teyvat",
            view_map_name: "Enkanomiya",
        })
        .expect("scene")
    }

    #[test]
    fn wrong_map_shows_only_the_banner() {
        let d = doc();
        let mut v = view(1100.0, 550.0, 1.0);
        v.map_id = 7;
        let s = scene_for(&d, 0, v);
        assert_eq!(s.stage, MMapGuideStage::WrongMap);
        let (title, sub) = s.banner.expect("banner");
        assert!(title.contains("Teyvat") && sub.contains("Enkanomiya"), "{title} / {sub}");
        assert!(s.arrow.is_none() && s.teleport.is_none() && s.paths.is_empty());
    }

    #[test]
    fn route_off_screen_points_an_edge_arrow_at_the_route() {
        let d = doc();
        // Looking far west of everything.
        let s = scene_for(&d, 1, view(-20000.0, 700.0, 1.9));
        assert_eq!(s.stage, MMapGuideStage::Route);
        let a = s.arrow.expect("arrow");
        assert!(a.angle_rad.abs() < 0.2, "points east: {}", a.angle_rad);
        assert!(a.x > 0.9 * W as f32, "on the right edge: {}", a.x);
        assert!(a.label.contains("Băng Tuyết"), "{}", a.label);
    }

    #[test]
    fn route_in_view_but_section_off_screen_points_at_the_section() {
        let d = doc();
        // S1 on screen (centre 1100, 550) at a scale where S2 (x ≈ 9150) is off to the east.
        let s = scene_for(&d, 1, view(1100.0, 550.0, 1.9));
        assert_eq!(s.stage, MMapGuideStage::Section);
        let a = s.arrow.expect("arrow");
        assert!(a.label.starts_with("S2/2"), "{}", a.label);
        assert!(a.angle_rad.abs() < 0.2);
        assert_eq!(s.paths.len(), 2);
        assert!(s.paths[1].emphasized && !s.paths[0].emphasized);
    }

    #[test]
    fn small_section_on_screen_is_outlined_with_a_zoom_in_hint() {
        let d = doc();
        // Whole route visible when zoomed far out (scale 8 → S1 spans 25 px).
        let s = scene_for(&d, 0, view(1100.0, 550.0, 8.0));
        assert_eq!(s.stage, MMapGuideStage::Section);
        let fb = s.focus_box.expect("outline");
        assert!(fb[2] - fb[0] < 60.0);
        assert!(s.focus_label.starts_with("S1/2") && s.focus_label.contains("Phóng to"), "{}", s.focus_label);
        assert!(s.teleport.is_none() && s.arrow.is_none());
    }

    #[test]
    fn zoomed_into_the_section_rings_the_teleport_with_the_underground_warning() {
        let d = doc();
        let s = scene_for(&d, 1, view(9150.0, 870.0, 0.6));
        assert_eq!(s.stage, MMapGuideStage::Teleport);
        let tp = s.teleport.expect("teleport");
        // (9100, 950) at scale 0.6 relative to the centre → (1280 - 83, 720 + 133).
        assert!((tp.x - 1197.0).abs() <= 1.0 && (tp.y - 853.0).abs() <= 1.0, "{tp:?}");
        assert_eq!(tp.title, "Dịch chuyển tại đây");
        assert!(tp.subtitle.starts_with("S2/2 · Vực Hải Sao"));
        assert!(tp.warn.expect("warn").contains("B1"));
    }

    #[test]
    fn zoomed_in_with_the_waypoint_just_off_screen_points_at_the_waypoint() {
        let mut d = doc();
        // Move S2's teleport well south of its stops.
        if let Some(a) = d.sections[1].anchor_teleport.as_mut() {
            a.lat = 2000.0;
        }
        let s = scene_for(&d, 1, view(9150.0, 850.0, 0.5));
        assert_eq!(s.stage, MMapGuideStage::Teleport);
        assert!(s.teleport.is_none());
        let a = s.arrow.expect("arrow");
        assert!((a.angle_rad - std::f32::consts::FRAC_PI_2).abs() < 0.2, "points down: {}", a.angle_rad);
        assert!(a.label.contains("S2/2"), "{}", a.label);
    }

    #[test]
    fn unchanged_view_gives_an_equal_scene_and_minimal_mode_drops_paths() {
        let d = doc();
        let v = view(1100.0, 550.0, 1.9);
        let a = scene_for(&d, 1, v);
        let mut v2 = v;
        v2.tx += 0.0004; // sub-pixel tracker noise
        assert_eq!(a, scene_for(&d, 1, v2));
        let done = vec![vec![false; 2]; 2];
        let m = compute_guide_scene(&GuideInput {
            doc: &d,
            active: 1,
            done: &done,
            view: v,
            game_rect: RectI32 { x: 0, y: 0, width: W, height: H },
            mode: MMapOverlayMode::Minimal,
            lang: "en",
            route_map_name: "Teyvat",
            view_map_name: "Teyvat",
        })
        .expect("scene");
        assert!(m.paths.is_empty() && m.arrow.is_some());
    }

    fn fallback(d: &RouteDocument, region: Option<&str>, drawer_open: bool, lang: &str) -> MMapGuideScene {
        compute_fallback_scene(&FallbackInput {
            doc: d,
            active: 1,
            game_rect: RectI32 { x: 0, y: 0, width: W, height: H },
            mode: MMapOverlayMode::Minimal,
            lang,
            route_map_name: "Teyvat",
            region_name: region,
            drawer_open,
        })
        .expect("fallback")
    }

    #[test]
    fn fallback_banner_names_route_section_and_where_to_go() {
        let d = doc();
        let s = fallback(&d, Some("Natlan"), false, "vi");
        assert_eq!(s.stage, MMapGuideStage::Locating);
        assert!(s.arrow.is_none() && s.teleport.is_none() && s.focus_box.is_none() && s.paths.is_empty());
        let (title, sub) = s.banner.expect("banner");
        assert_eq!(title, "Tuyến Băng Tuyết • S2/2 · Vực Hải Sao");
        assert!(sub.contains("Natlan (Teyvat)"), "{sub}");
        let (_, sub) = fallback(&d, None, false, "en").banner.expect("banner");
        assert!(sub.starts_with("Map Teyvat"), "{sub}");
        let (_, sub) = fallback(&d, Some("Natlan"), true, "vi").banner.expect("banner");
        assert_eq!(sub, "Chọn bản đồ: Teyvat");
    }

    #[test]
    fn section_area_comes_from_the_nearest_waypoint_and_is_localised() {
        use gh_core::multimap::{AreaEntry, MapMeta, WaypointEntry};
        let wp = |x: f64, y: f64, area_id: i32| WaypointEntry {
            x_pos: x,
            y_pos: y,
            area_id,
            map_id: 2,
            ..Default::default()
        };
        let mut db = MultimapDb::default();
        db.maps.insert(
            2,
            MapMeta {
                id: 2,
                areas: vec![
                    AreaEntry { id: 11, name: "Natlan (vi)".into() },
                    AreaEntry { id: 77, name: "Vùng Mới".into() },
                ],
                // S2's teleport is at world (9100, 950); S1's at (980, 520).
                waypoints: vec![wp(9000.0, 1000.0, 11), wp(1000.0, 500.0, 77)],
                ..Default::default()
            },
        );
        let d = doc();
        assert_eq!(section_area_name(&db, &d, 1, "vi").as_deref(), Some("Natlan (vi)"));
        assert_eq!(section_area_name(&db, &d, 1, "en").as_deref(), Some("Natlan"));
        assert_eq!(section_area_name(&db, &d, 0, "en").as_deref(), Some("Vùng Mới"), "no built-in name");
        let mut far = d.clone();
        far.sections[0].anchor_teleport = None;
        far.sections[0].stops = vec![stop(-9000.0, -9000.0)];
        assert_eq!(section_area_name(&db, &far, 0, "vi"), None, "beyond AREA_MAX_DIST");
    }

    #[test]
    fn edge_arrow_stays_on_the_inset_rectangle() {
        for (x, y) in [(1e6, 720.0), (-1e6, 720.0), (1280.0, 1e6), (1280.0, -1e6), (1e6, 1e6)] {
            let a = edge_arrow(x, y, f64::from(W), f64::from(H), String::new());
            let r = frac_rect(ARROW_RECT, f64::from(W), f64::from(H));
            assert!(
                f64::from(a.x) >= r[0] - 1.0 && f64::from(a.x) <= r[2] + 1.0
                    && f64::from(a.y) >= r[1] - 1.0 && f64::from(a.y) <= r[3] + 1.0,
                "{a:?}"
            );
        }
    }

    #[test]
    fn single_pin_route_doc_picks_nearest_waypoint_and_feeds_guide_scene() {
        use gh_core::multimap::{FloorEntry, MapMeta, PointGroupEntry, WaypointEntry, WaypointPointGroupRef};

        assert!((SINGLE_PIN_MMAP_DISTANCE_M - 180.0).abs() < 1e-6);

        let mut db = MultimapDb::default();
        db.maps.insert(
            2,
            MapMeta {
                id: 2,
                name: "Teyvat".into(),
                waypoints: vec![
                    WaypointEntry {
                        id: 101,
                        label_id: 3,
                        is_primary: true,
                        x_pos: 1000.0,
                        y_pos: 500.0,
                        area_id: 1,
                        map_id: 2,
                        ..Default::default()
                    },
                    WaypointEntry {
                        id: 102,
                        label_id: 3,
                        is_primary: true,
                        x_pos: 1080.0,
                        y_pos: 540.0,
                        area_id: 1,
                        map_id: 2,
                        point_group: Some(WaypointPointGroupRef {
                            group_id: 10,
                            floor_id: Some(372),
                        }),
                        ..Default::default()
                    },
                    WaypointEntry {
                        id: 103,
                        label_id: 3,
                        is_primary: true,
                        x_pos: 8000.0,
                        y_pos: 2000.0,
                        area_id: 2,
                        map_id: 2,
                        ..Default::default()
                    },
                ],
                point_groups: vec![PointGroupEntry {
                    id: 10,
                    map_id: 2,
                    name: "Cave".into(),
                    floors: vec![FloorEntry {
                        id: 372,
                        floor_name: "Sublevel 1".into(),
                        floor_name_short: "B1".into(),
                        index: 0,
                        overlay: None,
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            },
        );

        // Surface pin selects nearest surface teleport waypoint (101)
        let mut surface_pin = NavPin::new("pin_sf", "Rương Hiếm", 510.0, 1020.0);
        surface_pin.map_id = 2;
        let sf_doc = build_single_pin_route_doc(&db, &surface_pin, "Teyvat");
        assert_eq!(sf_doc.title, "Rương Hiếm");
        assert_eq!(sf_doc.map_id, 2);
        assert_eq!(sf_doc.map_name, "Teyvat");
        assert_eq!(sf_doc.sections.len(), 1);
        assert_eq!(sf_doc.sections[0].color, "#fbbf24");
        assert_eq!(sf_doc.sections[0].stops.len(), 1);
        let sf_anchor = sf_doc.sections[0].anchor_teleport.as_ref().expect("surface anchor");
        assert_eq!(sf_anchor.point_id.as_deref(), Some("101"));
        assert!(!sf_anchor.is_underground);

        // Underground pin prefers matching underground waypoint (102) with resolved floor label B1
        let mut ug_pin = NavPin::new("pin_ug", "Thần Đồng Hang", 520.0, 1025.0);
        ug_pin.map_id = 2;
        ug_pin.is_underground = true;
        ug_pin.floor_id = Some("372".into());
        let ug_doc = build_single_pin_route_doc(&db, &ug_pin, "Teyvat");
        let ug_anchor = ug_doc.sections[0].anchor_teleport.as_ref().expect("ug anchor");
        assert_eq!(ug_anchor.point_id.as_deref(), Some("102"));
        assert!(ug_anchor.is_underground);
        assert_eq!(ug_anchor.floor_name, "B1");

        // Feeds compute_guide_scene cleanly: zoomed in on anchor rings teleport with B1 warning
        let done = vec![vec![false]];
        let scene = compute_guide_scene(&GuideInput {
            doc: &ug_doc,
            active: 0,
            done: &done,
            view: view(1050.0, 530.0, 0.5),
            game_rect: RectI32 { x: 0, y: 0, width: W, height: H },
            mode: MMapOverlayMode::Full,
            lang: "vi",
            route_map_name: "Teyvat",
            view_map_name: "Teyvat",
        })
        .expect("single-pin guide scene");
        assert_eq!(scene.stage, MMapGuideStage::Teleport);
        let tp = scene.teleport.expect("teleport ring");
        assert!(tp.warn.expect("underground warn").contains("B1"));
        assert_eq!(scene.accent, parse_hex_rgba("#fbbf24", 232));

        // Wrong-map fallback & banner
        let mut wrong_v = view(1050.0, 530.0, 0.5);
        wrong_v.map_id = 7;
        let wrong_scene = compute_guide_scene(&GuideInput {
            doc: &ug_doc,
            active: 0,
            done: &done,
            view: wrong_v,
            game_rect: RectI32 { x: 0, y: 0, width: W, height: H },
            mode: MMapOverlayMode::Full,
            lang: "en",
            route_map_name: "Teyvat",
            view_map_name: "Enkanomiya",
        })
        .expect("wrong map scene");
        assert_eq!(wrong_scene.stage, MMapGuideStage::WrongMap);
        let fb = compute_fallback_scene(&FallbackInput {
            doc: &ug_doc,
            active: 0,
            game_rect: RectI32 { x: 0, y: 0, width: W, height: H },
            mode: MMapOverlayMode::Minimal,
            lang: "vi",
            route_map_name: "Teyvat",
            region_name: None,
            drawer_open: false,
        })
        .expect("fallback scene");
        assert_eq!(fb.stage, MMapGuideStage::Locating);
        assert!(fb.banner.expect("banner").0.contains("Thần Đồng Hang"));
    }
}
