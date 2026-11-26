//! Canonical Route Document schema (`.ghroute.json` v1), pluggable adapters
//! (HoYoLAB community/share routes + portable JSON), graph linearization,
//! waypoint/underground enrichment via `MultimapDb`, and local route library I/O.

use crate::atomic_io;
use crate::multimap::{MultimapDb, WaypointEntry};
use crate::{CoreError, CoreResult};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const ROUTE_FORMAT: &str = "genshin-helper-route";
pub const ROUTE_SCHEMA_VERSION: u32 = 1;

pub const TELEPORT_LABEL_IDS: &[i32] = &[2, 3, 154];
const KNOWN_CAVE_WAYPOINT_LABELS: &[i32] = &[150, 167, 318, 426, 437, 499, 542, 650, 651];
const HOYOLAB_COLOR_PALETTE: [&str; 8] = [
    "#fde047", "#ffffff", "#4ade80", "#38bdf8", "#c084fc", "#f472b6", "#fb923c", "#f87171",
];

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RouteSource {
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteStopKind {
    Teleport,
    Waypoint,
    #[default]
    Collectible,
}

impl RouteStopKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Teleport => "teleport",
            Self::Waypoint => "waypoint",
            Self::Collectible => "collectible",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RouteAnchor {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point_id: Option<String>,
    pub lat: f64,
    pub lng: f64,
    pub label_id: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub floor_id: Option<String>,
    pub floor_name: String,
    pub is_underground: bool,
    pub auto_suggested: bool,
}

impl Default for RouteAnchor {
    fn default() -> Self {
        Self {
            point_id: None,
            lat: 0.0,
            lng: 0.0,
            label_id: 3,
            floor_id: None,
            floor_name: String::new(),
            is_underground: false,
            auto_suggested: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RouteStop {
    pub id: String,
    pub kind: RouteStopKind,
    pub lat: f64,
    pub lng: f64,
    pub label_id: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub floor_id: Option<String>,
    pub floor_name: String,
    pub is_underground: bool,
    pub name: String,
    pub note: String,
    pub img_url: String,
    pub video_url: String,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RouteSection {
    pub index: usize,
    pub name: String,
    pub color: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor_teleport: Option<RouteAnchor>,
    pub stops: Vec<RouteStop>,
    pub edges: Vec<(usize, usize)>,
}

impl RouteSection {
    pub fn collectible_count(&self) -> usize {
        self.stops
            .iter()
            .filter(|s| s.kind == RouteStopKind::Collectible)
            .count()
    }

    pub fn focus_world(&self) -> Option<(f64, f64, bool, Option<String>, String)> {
        if let Some(ref anchor) = self.anchor_teleport {
            return Some((
                anchor.lat,
                anchor.lng,
                anchor.is_underground,
                anchor.floor_id.clone(),
                anchor.floor_name.clone(),
            ));
        }
        self.stops.first().map(|s| {
            (
                s.lat,
                s.lng,
                s.is_underground,
                s.floor_id.clone(),
                s.floor_name.clone(),
            )
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RouteDocument {
    pub format: String,
    pub version: u32,
    pub title: String,
    pub description: String,
    pub map_id: i32,
    #[serde(default)]
    pub map_name: String,
    pub source: RouteSource,
    pub sections: Vec<RouteSection>,
}

impl Default for RouteDocument {
    fn default() -> Self {
        Self {
            format: ROUTE_FORMAT.to_owned(),
            version: ROUTE_SCHEMA_VERSION,
            title: String::new(),
            description: String::new(),
            map_id: 2,
            map_name: String::new(),
            source: RouteSource {
                provider: "manual".to_owned(),
                ..Default::default()
            },
            sections: Vec::new(),
        }
    }
}

impl RouteDocument {
    pub fn normalize_and_enrich(&mut self, db: Option<&MultimapDb>) {
        if self.format.trim().is_empty() {
            self.format = ROUTE_FORMAT.to_owned();
        }
        if self.version == 0 {
            self.version = ROUTE_SCHEMA_VERSION;
        }
        if self.map_id <= 0 {
            self.map_id = 2;
        }
        self.map_name = self.map_name.trim().to_owned();
        if self.map_name.is_empty()
            && let Some(name) = db.and_then(|d| d.map_name(self.map_id))
        {
            self.map_name = name.to_owned();
        }
        if self.title.trim().is_empty() {
            self.title = format!("Route (Map {})", self.map_id);
        } else {
            self.title = self.title.trim().to_owned();
        }
        self.description = self.description.trim().to_owned();

        for (i, sec) in self.sections.iter_mut().enumerate() {
            sec.index = i;
            if sec.name.trim().is_empty() {
                sec.name = format!("Section {}", i + 1);
            } else {
                sec.name = sec.name.trim().to_owned();
            }
            if sec.color.trim().is_empty() {
                sec.color = palette_color(i + 1, i);
            }
            for stop in &mut sec.stops {
                enrich_stop_floor(stop, self.map_id, db);
            }
            if sec.anchor_teleport.is_none() {
                sec.anchor_teleport = resolve_section_anchor(&sec.stops, self.map_id, db);
            }
        }
    }

    pub fn total_collectibles(&self) -> usize {
        self.sections.iter().map(RouteSection::collectible_count).sum()
    }

    pub fn suggested_filename(&self) -> String {
        let slug = sanitize_route_filename(&self.title);
        format!("{slug}.ghroute.json")
    }

    pub fn to_json_string_pretty(&self) -> CoreResult<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    pub fn to_value(&self) -> CoreResult<Value> {
        Ok(serde_json::to_value(self)?)
    }
}

/// Pluggable route format adapter so future formats (Kongying, BetterGI, etc.)
/// can plug in cleanly alongside HoYoLAB and `.ghroute.json`.
pub trait RouteAdapter {
    fn provider(&self) -> &'static str;
    fn parse(&self, raw: &Value, db: Option<&MultimapDb>) -> CoreResult<RouteDocument>;
}

pub struct GhRouteJsonAdapter;

impl RouteAdapter for GhRouteJsonAdapter {
    fn provider(&self) -> &'static str {
        "ghroute"
    }

    fn parse(&self, raw: &Value, db: Option<&MultimapDb>) -> CoreResult<RouteDocument> {
        let candidate = raw.get("document").unwrap_or(raw);
        let obj = candidate
            .as_object()
            .ok_or_else(|| CoreError::InvalidData("Route JSON must be an object".to_owned()))?;
        let fmt = obj
            .get("format")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if fmt != ROUTE_FORMAT {
            return Err(CoreError::InvalidData(format!(
                "Unsupported route format '{fmt}'"
            )));
        }
        let mut doc: RouteDocument = serde_json::from_value(candidate.clone())?;
        if doc.version > ROUTE_SCHEMA_VERSION {
            return Err(CoreError::InvalidData(format!(
                "Unsupported route schema version {}",
                doc.version
            )));
        }
        if doc.sections.is_empty() {
            return Err(CoreError::InvalidData(
                "Route document contains no sections".to_owned(),
            ));
        }
        doc.normalize_and_enrich(db);
        Ok(doc)
    }
}

pub struct HoYoLabRouteAdapter;

#[derive(Debug, Clone)]
struct RawPointNode {
    key: String,
    bare_id: String,
    stop: RouteStop,
}

impl RouteAdapter for HoYoLabRouteAdapter {
    fn provider(&self) -> &'static str {
        "hoyolab"
    }

    fn parse(&self, raw: &Value, db: Option<&MultimapDb>) -> CoreResult<RouteDocument> {
        let root_obj = raw
            .as_object()
            .ok_or_else(|| CoreError::InvalidData("HoYoLAB route must be a JSON object".to_owned()))?;
        let payload = root_obj
            .get("payload")
            .or_else(|| root_obj.get("raw"))
            .unwrap_or(raw);
        let path_obj = locate_hoyolab_path_obj(payload)?;
        let raw_sections = path_obj
            .get("sections")
            .and_then(Value::as_array)
            .ok_or_else(|| CoreError::InvalidData("Missing HoYoLAB path.sections".to_owned()))?;
        if raw_sections.is_empty() {
            return Err(CoreError::InvalidData(
                "HoYoLAB route contains no sections".to_owned(),
            ));
        }

        let explicit_map_id = opt_i64(
            root_obj
                .get("map_id")
                .or_else(|| payload.get("map_id"))
                .or_else(|| path_obj.get("map_id")),
        )
        .map(|v| v as i32)
        .filter(|&m| m > 0);
        let inferred_map_id = explicit_map_id.unwrap_or_else(|| infer_map_id_from_sections(raw_sections));

        let path_id = opt_str(
            root_obj
                .get("path_id")
                .or_else(|| path_obj.get("obj_id"))
                .or_else(|| path_obj.get("path_id")),
        );
        let share_url = opt_str(
            root_obj
                .get("share_url")
                .or_else(|| path_obj.get("share_url"))
                .or_else(|| path_obj.get("post_link")),
        );
        let title = opt_str(
            root_obj
                .get("title")
                .or_else(|| root_obj.get("path_name"))
                .or_else(|| path_obj.get("name"))
                .or_else(|| path_obj.get("path_name"))
                .or_else(|| path_obj.get("post_title")),
        )
        .unwrap_or_default();
        let description = opt_str(
            root_obj
                .get("description")
                .or_else(|| root_obj.get("content"))
                .or_else(|| path_obj.get("content")),
        )
        .unwrap_or_default();
        let author = opt_str(root_obj.get("author").or_else(|| path_obj.get("author")));

        let section_names: Vec<String> = root_obj
            .get("section_names")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .map(|v| v.as_str().unwrap_or("").trim().to_owned())
                    .collect()
            })
            .unwrap_or_default();

        let label_names = build_label_name_map(root_obj);
        let mut sections = Vec::with_capacity(raw_sections.len());

        for (sec_idx, sec_val) in raw_sections.iter().enumerate() {
            let Some(sec_obj) = sec_val.as_object() else {
                continue;
            };
            let color_id = opt_i64(sec_obj.get("color_id")).unwrap_or((sec_idx + 1) as i64);
            let explicit_color = opt_str(sec_obj.get("color"));
            let color = explicit_color
                .filter(|c| c.starts_with('#'))
                .unwrap_or_else(|| palette_color(color_id.max(1) as usize, sec_idx));

            let sec_name = section_names
                .get(sec_idx)
                .map(String::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .or_else(|| opt_str(sec_obj.get("name")))
                .unwrap_or_else(|| format!("Route Section {}", sec_idx + 1));

            let nodes = collect_section_nodes(sec_obj, inferred_map_id, &label_names, db);
            if nodes.is_empty() {
                continue;
            }
            let relations = sec_obj
                .get("relation")
                .and_then(Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let (mut stops, edges) = linearize_nodes(&nodes, relations);
            for stop in &mut stops {
                enrich_stop_floor(stop, inferred_map_id, db);
            }
            let anchor_teleport = resolve_section_anchor(&stops, inferred_map_id, db);

            sections.push(RouteSection {
                index: sections.len(),
                name: sec_name,
                color,
                anchor_teleport,
                stops,
                edges,
            });
        }

        if sections.is_empty() {
            return Err(CoreError::InvalidData(
                "No valid stops found in HoYoLAB route".to_owned(),
            ));
        }

        let mut doc = RouteDocument {
            format: ROUTE_FORMAT.to_owned(),
            version: ROUTE_SCHEMA_VERSION,
            title,
            description,
            map_id: inferred_map_id,
            map_name: String::new(),
            source: RouteSource {
                provider: self.provider().to_owned(),
                path_id,
                share_url,
                author,
            },
            sections,
        };
        doc.normalize_and_enrich(db);
        Ok(doc)
    }
}

/// Auto-detects route payload format (`.ghroute.json` v1 or HoYoLAB `get_share_map_spots`).
pub fn parse_route_auto(raw: &Value, db: Option<&MultimapDb>) -> CoreResult<RouteDocument> {
    let gh_candidate = raw
        .get("document")
        .or_else(|| raw.get("raw"))
        .unwrap_or(raw);
    if gh_candidate
        .get("format")
        .and_then(Value::as_str)
        .is_some_and(|f| f.trim() == ROUTE_FORMAT)
    {
        return GhRouteJsonAdapter.parse(gh_candidate, db);
    }
    HoYoLabRouteAdapter.parse(raw, db)
}

/// Extracts `(map_id, share_id)` from HoYoLAB Interactive Map share URLs such as
/// `https://act.hoyolab.com/ys/app/interactive-map/index.html?...#/map/2?share_id=...&share_type=path`.
pub fn parse_hoyolab_route_share_url(raw_url: &str) -> Option<(i32, String)> {
    let trimmed = raw_url.trim();
    if trimmed.is_empty() {
        return None;
    }
    let map_id = extract_hash_map_id(trimmed).unwrap_or(2);
    let share_id = extract_query_param(trimmed, "share_id")?;
    if share_id.is_empty() {
        return None;
    }
    Some((map_id, share_id))
}

pub fn sanitize_route_filename(title: &str) -> String {
    let mut out = String::with_capacity(title.len().min(64));
    let mut prev_sep = false;
    for ch in title.trim().chars() {
        if out.len() >= 48 {
            break;
        }
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_sep = false;
        } else if (ch.is_whitespace() || matches!(ch, '-' | '_' | '.'))
            && !prev_sep
            && !out.is_empty()
        {
            out.push('-');
            prev_sep = true;
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "route".to_owned()
    } else {
        trimmed.to_owned()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedRouteMeta {
    pub filename: String,
    pub title: String,
    pub map_id: i32,
    pub section_count: usize,
    pub collectible_count: usize,
    pub provider: String,
    pub modified_ms: u64,
}

pub fn save_route_to_library(dir: &Path, doc: &RouteDocument) -> CoreResult<PathBuf> {
    fs::create_dir_all(dir)?;
    let fname = doc.suggested_filename();
    let path = dir.join(fname);
    let val = doc.to_value()?;
    atomic_io::write_json_atomic(&path, &val)?;
    Ok(path)
}

pub fn list_saved_routes(dir: &Path) -> CoreResult<Vec<SavedRouteMeta>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !fname.ends_with(".ghroute.json") && !fname.ends_with(".json") {
            continue;
        }
        let Ok(raw_str) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(doc) = serde_json::from_str::<RouteDocument>(&raw_str) else {
            continue;
        };
        if doc.format != ROUTE_FORMAT {
            continue;
        }
        let modified_ms = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        out.push(SavedRouteMeta {
            filename: fname.to_owned(),
            title: doc.title.clone(),
            map_id: doc.map_id,
            section_count: doc.sections.len(),
            collectible_count: doc.total_collectibles(),
            provider: doc.source.provider.clone(),
            modified_ms,
        });
    }
    out.sort_by(|a, b| b.modified_ms.cmp(&a.modified_ms).then_with(|| a.filename.cmp(&b.filename)));
    Ok(out)
}

fn validate_library_filename(filename: &str) -> CoreResult<&str> {
    let trimmed = filename.trim();
    if trimmed.is_empty()
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains("..")
    {
        return Err(CoreError::InvalidArgument(
            "Invalid route filename".to_owned(),
        ));
    }
    Ok(trimmed)
}

pub fn load_saved_route(
    dir: &Path,
    filename: &str,
    db: Option<&MultimapDb>,
) -> CoreResult<RouteDocument> {
    let safe_name = validate_library_filename(filename)?;
    let path = dir.join(safe_name);
    let raw_str = fs::read_to_string(&path)?;
    let val: Value = serde_json::from_str(&raw_str)?;
    GhRouteJsonAdapter.parse(&val, db)
}

pub fn delete_saved_route(dir: &Path, filename: &str) -> CoreResult<bool> {
    let safe_name = validate_library_filename(filename)?;
    let path = dir.join(safe_name);
    if !path.is_file() {
        return Ok(false);
    }
    fs::remove_file(&path)?;
    Ok(true)
}

fn palette_color(color_id: usize, fallback_idx: usize) -> String {
    let idx = if color_id > 0 {
        (color_id - 1) % HOYOLAB_COLOR_PALETTE.len()
    } else {
        fallback_idx % HOYOLAB_COLOR_PALETTE.len()
    };
    HOYOLAB_COLOR_PALETTE[idx].to_owned()
}

fn locate_hoyolab_path_obj(val: &Value) -> CoreResult<&Map<String, Value>> {
    if let Some(p) = val
        .get("data")
        .and_then(|d| d.get("path"))
        .and_then(Value::as_object)
    {
        return Ok(p);
    }
    if let Some(p) = val
        .get("data")
        .and_then(Value::as_object)
        .filter(|d| d.get("sections").is_some_and(Value::is_array))
    {
        return Ok(p);
    }
    if let Some(p) = val.get("path").and_then(Value::as_object) {
        return Ok(p);
    }
    if let Some(obj) = val.as_object()
        && obj.get("sections").is_some_and(Value::is_array)
    {
        return Ok(obj);
    }
    Err(CoreError::InvalidData(
        "Could not locate HoYoLAB path sections object".to_owned(),
    ))
}

fn infer_map_id_from_sections(raw_sections: &[Value]) -> i32 {
    for sec in raw_sections {
        for key in ["system_points", "custom_points"] {
            if let Some(pts) = sec.get(key).and_then(Value::as_array) {
                for pt in pts {
                    if let Some(mid) = opt_i64(pt.get("map_id")).map(|v| v as i32)
                        && mid > 0
                    {
                        return mid;
                    }
                }
            }
        }
    }
    2
}

fn build_label_name_map(root_obj: &Map<String, Value>) -> HashMap<i32, String> {
    let mut map = HashMap::new();
    if let Some(obj) = root_obj.get("label_names").and_then(Value::as_object) {
        for (k, v) in obj {
            if let (Ok(id), Some(name)) = (k.trim().parse::<i32>(), v.as_str()) {
                let trimmed = name.trim();
                if !trimmed.is_empty() {
                    map.insert(id, trimmed.to_owned());
                }
            }
        }
    }
    map
}

fn collect_section_nodes(
    sec_obj: &Map<String, Value>,
    default_map_id: i32,
    label_names: &HashMap<i32, String>,
    db: Option<&MultimapDb>,
) -> Vec<RawPointNode> {
    let mut nodes = Vec::new();
    let mut seen_keys = HashSet::new();

    for (key_name, pt_type) in [("system_points", 1_i32), ("custom_points", 2_i32)] {
        let Some(arr) = sec_obj.get(key_name).and_then(Value::as_array) else {
            continue;
        };
        for raw_pt in arr {
            let Some(obj) = raw_pt.as_object() else {
                continue;
            };
            let Some(node) =
                parse_raw_point_node(obj, pt_type, default_map_id, label_names, db)
            else {
                continue;
            };
            if seen_keys.insert(node.key.clone()) {
                nodes.push(node);
            }
        }
    }
    nodes
}

fn parse_raw_point_node(
    obj: &Map<String, Value>,
    point_type: i32,
    default_map_id: i32,
    label_names: &HashMap<i32, String>,
    db: Option<&MultimapDb>,
) -> Option<RawPointNode> {
    let lat = opt_f64(obj.get("lat").or_else(|| obj.get("y_pos")))?;
    let lng = opt_f64(obj.get("lng").or_else(|| obj.get("x_pos")))?;
    if !lat.is_finite() || !lng.is_finite() {
        return None;
    }
    let bare_id = opt_str(obj.get("id").or_else(|| obj.get("point_id")))
        .unwrap_or_else(|| format!("{lat:.2},{lng:.2}"));
    let label_id = opt_i64(obj.get("label_id").or_else(|| obj.get("labelId")))
        .unwrap_or(0) as i32;
    let map_id = opt_i64(obj.get("map_id"))
        .map(|v| v as i32)
        .filter(|&m| m > 0)
        .unwrap_or(default_map_id);

    let ext_attrs_map = parse_ext_attrs(obj.get("ext_attrs"));
    let mut floor_id = opt_str(
        obj.get("floor_id")
            .or_else(|| obj.get("floorId"))
            .or_else(|| ext_attrs_map.as_ref().and_then(|m| m.get("floor_id"))),
    )
    .filter(|f| f != "0");
    let mut floor_name = opt_str(
        obj.get("floor_name")
            .or_else(|| obj.get("floorName"))
            .or_else(|| ext_attrs_map.as_ref().and_then(|m| m.get("floor_name"))),
    )
    .unwrap_or_default();

    let z_level = opt_i64(obj.get("z_level")).unwrap_or(0);
    let icon_sign = opt_i64(obj.get("icon_sign")).unwrap_or(0);
    let explicit_ug = obj.get("is_underground").and_then(Value::as_bool);

    let mut db_wp: Option<&WaypointEntry> = None;
    if let Some(m) = db.and_then(|d| d.map(map_id)) {
        db_wp = bare_id
            .parse::<i64>()
            .ok()
            .and_then(|id_num| m.waypoints.iter().find(|w| w.id == id_num));
        if floor_id.is_none()
            && let Some(wp) = db_wp
            && let Some(ref pg) = wp.point_group
            && let Some(fid) = pg.floor_id
            && fid > 0
        {
            floor_id = Some(fid.to_string());
        }
    }
    if floor_name.is_empty()
        && let (Some(fid_s), Some(d)) = (floor_id.as_deref(), db)
        && let Ok(fid_num) = fid_s.parse::<i32>()
        && let Some(lbl) = d.resolve_floor_label(map_id, fid_num)
    {
        floor_name = lbl;
    }

    let is_underground = explicit_ug.unwrap_or_else(|| {
        floor_id.is_some() || z_level != 0 || matches!(icon_sign, 1..=3)
    });

    let raw_name = opt_str(obj.get("name").or_else(|| obj.get("title")));
    let label_title = label_names.get(&label_id).cloned();
    let name = raw_name
        .or_else(|| {
            if let Some(m) = db.and_then(|d| d.map(map_id)) {
                m.landmark_labels
                    .iter()
                    .find(|l| l.id == label_id)
                    .map(|l| l.name.clone())
            } else {
                None
            }
        })
        .or(label_title.clone())
        .unwrap_or_else(|| {
            if point_type == 2 {
                format!("Custom ({bare_id})")
            } else {
                format!("Pin {bare_id}")
            }
        });

    let kind = classify_stop_kind(point_type, label_id, &name, label_title.as_deref(), db, map_id, db_wp);

    let note = opt_str(obj.get("content").or_else(|| obj.get("description"))).unwrap_or_default();
    let img_url = opt_str(obj.get("img").or_else(|| obj.get("image_url"))).unwrap_or_default();
    let video_url = opt_str(obj.get("video")).unwrap_or_default();

    Some(RawPointNode {
        key: format!("{point_type}:{bare_id}"),
        bare_id: bare_id.clone(),
        stop: RouteStop {
            id: bare_id,
            kind,
            lat,
            lng,
            label_id,
            floor_id,
            floor_name,
            is_underground,
            name,
            note,
            img_url,
            video_url,
        },
    })
}

fn classify_stop_kind(
    point_type: i32,
    label_id: i32,
    name: &str,
    label_name: Option<&str>,
    db: Option<&MultimapDb>,
    map_id: i32,
    db_wp: Option<&WaypointEntry>,
) -> RouteStopKind {
    if TELEPORT_LABEL_IDS.contains(&label_id) || db_wp.is_some_and(|w| w.is_primary) {
        return RouteStopKind::Teleport;
    }
    let name_low = name.to_ascii_lowercase();
    let lbl_low = label_name.unwrap_or("").to_ascii_lowercase();
    let combined = format!("{name_low} {lbl_low}");
    if combined.contains("teleport")
        || combined.contains("dịch chuyển")
        || combined.contains("statue of the seven")
        || combined.contains("thất thiên thần tượng")
    {
        return RouteStopKind::Teleport;
    }

    if db_wp.is_some()
        || KNOWN_CAVE_WAYPOINT_LABELS.contains(&label_id)
        || db.and_then(|d| d.map(map_id)).is_some_and(|m| {
            m.landmark_labels.iter().any(|l| l.id == label_id)
        })
        || (point_type == 2 && label_id == 0)
        || combined.contains("cave entrance")
        || combined.contains("lối vào hang")
        || combined.contains("cửa hang")
        || combined.contains("underground entrance")
        || combined.contains("waverider")
    {
        return RouteStopKind::Waypoint;
    }

    RouteStopKind::Collectible
}

fn linearize_nodes(
    nodes: &[RawPointNode],
    relations: &[Value],
) -> (Vec<RouteStop>, Vec<(usize, usize)>) {
    let mut by_key: HashMap<String, usize> = HashMap::with_capacity(nodes.len());
    let mut by_bare: HashMap<String, Vec<usize>> = HashMap::with_capacity(nodes.len());
    for (idx, n) in nodes.iter().enumerate() {
        by_key.insert(n.key.clone(), idx);
        by_bare.entry(n.bare_id.clone()).or_default().push(idx);
    }

    let lookup_idx = |id_val: Option<&Value>, type_val: Option<&Value>| -> Option<usize> {
        let id_str = opt_str(id_val)?;
        if let Some(t) = opt_i64(type_val) {
            let k = format!("{t}:{id_str}");
            if let Some(&idx) = by_key.get(&k) {
                return Some(idx);
            }
        }
        by_bare.get(&id_str).and_then(|v| v.first().copied())
    };

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut in_deg: Vec<usize> = vec![0; nodes.len()];
    let mut parsed_edges: Vec<(usize, usize)> = Vec::new();

    for rel in relations {
        let Some(obj) = rel.as_object() else {
            continue;
        };
        let Some(from_orig) = lookup_idx(obj.get("start_id"), obj.get("start_type")) else {
            continue;
        };
        let Some(to_orig) = lookup_idx(obj.get("end_id"), obj.get("end_type")) else {
            continue;
        };
        if from_orig == to_orig {
            continue;
        }
        if !adj[from_orig].contains(&to_orig) {
            adj[from_orig].push(to_orig);
            in_deg[to_orig] += 1;
        }
        parsed_edges.push((from_orig, to_orig));
    }

    // Order outgoing branches nearest-first from parent coordinates for intuitive pathing.
    for (from_idx, nbrs) in adj.iter_mut().enumerate() {
        if nbrs.len() > 1 {
            let p = &nodes[from_idx].stop;
            nbrs.sort_by(|&a, &b| {
                let da = (nodes[a].stop.lat - p.lat).hypot(nodes[a].stop.lng - p.lng);
                let db = (nodes[b].stop.lat - p.lat).hypot(nodes[b].stop.lng - p.lng);
                da.total_cmp(&db).then_with(|| a.cmp(&b))
            });
        }
    }

    let mut starts: Vec<usize> = (0..nodes.len()).filter(|&i| in_deg[i] == 0).collect();
    starts.sort_by_key(|&i| {
        let prio = match nodes[i].stop.kind {
            RouteStopKind::Teleport => 0,
            RouteStopKind::Waypoint => 1,
            RouteStopKind::Collectible => 2,
        };
        (prio, i)
    });

    let mut visited = vec![false; nodes.len()];
    let mut order_orig = Vec::with_capacity(nodes.len());

    fn dfs_walk(
        cur: usize,
        adj: &[Vec<usize>],
        visited: &mut [bool],
        order: &mut Vec<usize>,
    ) {
        if visited[cur] {
            return;
        }
        visited[cur] = true;
        order.push(cur);
        for &nxt in &adj[cur] {
            if !visited[nxt] {
                dfs_walk(nxt, adj, visited, order);
            }
        }
    }

    for s in starts {
        dfs_walk(s, &adj, &mut visited, &mut order_orig);
    }

    // Append any nodes trapped in cycles or without explicit start edges in nearest-neighbor order.
    while order_orig.len() < nodes.len() {
        let last_pos = order_orig
            .last()
            .map(|&idx| (nodes[idx].stop.lat, nodes[idx].stop.lng));
        let mut best_next: Option<usize> = None;
        let mut best_score = f64::INFINITY;
        for (idx, n) in nodes.iter().enumerate() {
            if visited[idx] {
                continue;
            }
            let score = match last_pos {
                Some((plat, plng)) => (n.stop.lat - plat).hypot(n.stop.lng - plng),
                None => idx as f64,
            };
            if score < best_score {
                best_score = score;
                best_next = Some(idx);
            }
        }
        let Some(nxt) = best_next else {
            break;
        };
        dfs_walk(nxt, &adj, &mut visited, &mut order_orig);
    }

    let mut orig_to_new = vec![0usize; nodes.len()];
    let mut stops = Vec::with_capacity(nodes.len());
    for (new_idx, orig_idx) in order_orig.into_iter().enumerate() {
        orig_to_new[orig_idx] = new_idx;
        stops.push(nodes[orig_idx].stop.clone());
    }

    let mut seen_edges = HashSet::new();
    let mut edges = Vec::new();
    for (u, v) in parsed_edges {
        let nu = orig_to_new[u];
        let nv = orig_to_new[v];
        if nu != nv && seen_edges.insert((nu, nv)) {
            edges.push((nu, nv));
        }
    }
    if edges.is_empty() && stops.len() >= 2 {
        for i in 0..stops.len() - 1 {
            edges.push((i, i + 1));
        }
    }

    (stops, edges)
}

fn enrich_stop_floor(stop: &mut RouteStop, map_id: i32, db: Option<&MultimapDb>) {
    stop.name = stop.name.trim().to_owned();
    stop.note = stop.note.trim().to_owned();
    stop.img_url = stop.img_url.trim().to_owned();
    stop.video_url = stop.video_url.trim().to_owned();
    stop.floor_id = stop
        .floor_id
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty() && *s != "0")
        .map(str::to_owned);

    if stop.is_underground && stop.floor_id.is_none()
        && let Some(d) = db
        && let Some((fid, lbl)) = d.infer_unique_underground_floor(map_id, stop.lat, stop.lng, 24.0)
    {
        stop.floor_id = Some(fid.to_string());
        if stop.floor_name.is_empty() {
            stop.floor_name = lbl;
        }
    }
    if stop.floor_name.is_empty()
        && let (Some(fid_s), Some(d)) = (stop.floor_id.as_deref(), db)
        && let Ok(fid_num) = fid_s.parse::<i32>()
        && let Some(lbl) = d.resolve_floor_label(map_id, fid_num)
    {
        stop.floor_name = lbl;
    }
}

fn resolve_section_anchor(
    stops: &[RouteStop],
    map_id: i32,
    db: Option<&MultimapDb>,
) -> Option<RouteAnchor> {
    if let Some(tp) = stops.iter().find(|s| s.kind == RouteStopKind::Teleport) {
        return Some(RouteAnchor {
            point_id: Some(tp.id.clone()),
            lat: tp.lat,
            lng: tp.lng,
            label_id: if tp.label_id > 0 { tp.label_id } else { 3 },
            floor_id: tp.floor_id.clone(),
            floor_name: tp.floor_name.clone(),
            is_underground: tp.is_underground,
            auto_suggested: false,
        });
    }
    let ref_stop = stops.first()?;
    let d = db?;
    let wps = d.waypoints_for_map(map_id);
    let want_ug = ref_stop.is_underground;
    let want_fid = ref_stop.floor_id.as_deref().and_then(|s| s.parse::<i32>().ok());

    let mut best: Option<(&WaypointEntry, f64)> = None;
    for wp in wps {
        if !wp.is_primary && !TELEPORT_LABEL_IDS.contains(&wp.label_id) {
            continue;
        }
        let wp_fid = wp
            .point_group
            .as_ref()
            .and_then(|pg| pg.floor_id)
            .filter(|&f| f > 0);
        let wp_ug = wp_fid.is_some();
        let dist = (wp.y_pos - ref_stop.lat).hypot(wp.x_pos - ref_stop.lng);
        // Strongly prefer same floor/layer so underground sections suggest underground waypoints first.
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
    let floor_name = wp_fid
        .and_then(|fid| d.resolve_floor_label(map_id, fid))
        .unwrap_or_default();
    Some(RouteAnchor {
        point_id: Some(wp.id.to_string()),
        lat: wp.y_pos,
        lng: wp.x_pos,
        label_id: wp.label_id,
        floor_id: wp_fid.map(|f| f.to_string()),
        floor_name,
        is_underground: wp_fid.is_some(),
        auto_suggested: true,
    })
}

fn parse_ext_attrs(val: Option<&Value>) -> Option<Map<String, Value>> {
    match val {
        Some(Value::Object(m)) => Some(m.clone()),
        Some(Value::String(s)) => {
            let trimmed = s.trim();
            if trimmed.starts_with('{') {
                serde_json::from_str::<Map<String, Value>>(trimmed).ok()
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_hash_map_id(url: &str) -> Option<i32> {
    let idx = url.find("/map/")?;
    let rest = &url[idx + 5..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<i32>().ok().filter(|&m| m > 0)
}

fn extract_query_param(url: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=");
    let pos = url.find(&needle)?;
    let rest = &url[pos + needle.len()..];
    let val: String = rest
        .chars()
        .take_while(|&c| c != '&' && c != '#')
        .collect();
    let trimmed = val.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn opt_str(val: Option<&Value>) -> Option<String> {
    let v = val?;
    match v {
        Value::Null => None,
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_owned())
            }
        }
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn opt_i64(val: Option<&Value>) -> Option<i64> {
    let v = val?;
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn opt_f64(val: Option<&Value>) -> Option<f64> {
    let v = val?;
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimap::{FloorEntry, FloorOverlayMeta, MapMeta, PointGroupEntry, WaypointPointGroupRef};
    use serde_json::json;
    use std::collections::BTreeMap;

    fn mock_db() -> MultimapDb {
        let mut maps = BTreeMap::new();
        maps.insert(
            2,
            MapMeta {
                id: 2,
                name: "Teyvat".to_owned(),
                priority: 1,
                waypoints: vec![
                    WaypointEntry {
                        id: 36674,
                        label_id: 2,
                        is_primary: true,
                        x_pos: -4735.75,
                        y_pos: 4322.5,
                        area_id: 4,
                        map_id: 2,
                        z_level: 0,
                        icon_sign: 0,
                        point_group: None,
                    },
                    WaypointEntry {
                        id: 99001,
                        label_id: 3,
                        is_primary: true,
                        x_pos: 100.0,
                        y_pos: 200.0,
                        area_id: 1,
                        map_id: 2,
                        z_level: 0,
                        icon_sign: 0,
                        point_group: None,
                    },
                    WaypointEntry {
                        id: 99002,
                        label_id: 3,
                        is_primary: true,
                        x_pos: 500.0,
                        y_pos: 600.0,
                        area_id: 1,
                        map_id: 2,
                        z_level: -1,
                        icon_sign: 1,
                        point_group: Some(WaypointPointGroupRef {
                            group_id: 10,
                            floor_id: Some(372),
                        }),
                    },
                ],
                point_groups: vec![PointGroupEntry {
                    id: 10,
                    map_id: 2,
                    name: "Desert Cave".to_owned(),
                    floors: vec![FloorEntry {
                        id: 372,
                        floor_name: "Underground B1".to_owned(),
                        floor_name_short: "B1".to_owned(),
                        index: 0,
                        overlay: Some(FloorOverlayMeta {
                            url: "https://webstatic.hoyoverse.com/cave.png".to_owned(),
                            l_x: 450.0,
                            l_y: 550.0,
                            r_x: 560.0,
                            r_y: 660.0,
                            reflect_y: Some(false),
                            ..Default::default()
                        }),
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        MultimapDb {
            version: "4.0.0".to_owned(),
            maps,
        }
    }

    #[test]
    fn parses_hoyolab_route_classifies_nodes_linearizes_and_roundtrips() {
        let db = mock_db();
        let sample = json!({
            "title": "Scarab Harvest Routes",
            "description": "Scarab * 4",
            "map_id": 2,
            "data": {
                "path": {
                    "obj_id": "64b9f7ece2d64a14a41a57aa",
                    "sections": [{
                        "color_id": 2,
                        "relation": [
                            { "start_id": 36674, "start_type": 1, "end_id": 40001, "end_type": 1 },
                            { "start_id": 40001, "start_type": 1, "end_id": 37227, "end_type": 1 }
                        ],
                        "system_points": [
                            {
                                "id": 37227,
                                "map_id": 2,
                                "label_id": 430,
                                "x_pos": 510.0,
                                "y_pos": 610.0,
                                "content": "Under sand dune",
                                "img": "https://webstatic.hoyoverse.com/a.png",
                                "z_level": -1,
                                "icon_sign": 1
                            },
                            {
                                "id": 40001,
                                "map_id": 2,
                                "label_id": 318,
                                "x_pos": 490.0,
                                "y_pos": 595.0,
                                "z_level": 0
                            },
                            {
                                "id": 36674,
                                "map_id": 2,
                                "label_id": 2,
                                "x_pos": -4735.75,
                                "y_pos": 4322.5
                            }
                        ]
                    }]
                }
            }
        });

        let doc = parse_route_auto(&sample, Some(&db)).expect("parse HoYoLAB route");
        assert_eq!(doc.format, ROUTE_FORMAT);
        assert_eq!(doc.version, ROUTE_SCHEMA_VERSION);
        assert_eq!(doc.title, "Scarab Harvest Routes");
        assert_eq!(doc.sections.len(), 1);

        let sec = &doc.sections[0];
        assert_eq!(sec.color, "#ffffff");
        // Linearized start->end even though raw system_points were reversed!
        assert_eq!(
            sec.stops.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
            vec!["36674", "40001", "37227"]
        );
        assert_eq!(sec.stops[0].kind, RouteStopKind::Teleport);
        assert_eq!(sec.stops[1].kind, RouteStopKind::Waypoint);
        assert_eq!(sec.stops[2].kind, RouteStopKind::Collectible);
        assert!(sec.stops[2].is_underground);
        assert_eq!(sec.stops[2].floor_id.as_deref(), Some("372"));
        assert_eq!(sec.stops[2].floor_name, "B1");
        assert_eq!(sec.edges, vec![(0, 1), (1, 2)]);

        let anchor = sec.anchor_teleport.as_ref().expect("teleport anchor");
        assert_eq!(anchor.point_id.as_deref(), Some("36674"));
        assert!(!anchor.auto_suggested);

        // Lossless .ghroute.json round-trip
        let exported = doc.to_value().unwrap();
        let reimported = parse_route_auto(&exported, Some(&db)).unwrap();
        assert_eq!(reimported, doc);
    }

    #[test]
    fn suggests_matching_underground_teleport_when_section_has_no_teleport() {
        let db = mock_db();
        let raw = json!({
            "title": "Underground Farm",
            "map_id": 2,
            "data": {
                "path": {
                    "sections": [{
                        "color_id": 1,
                        "relation": [{ "start_id": 50001, "start_type": 1, "end_id": 50002, "end_type": 1 }],
                        "system_points": [
                            { "id": 50001, "map_id": 2, "label_id": 430, "x_pos": 505.0, "y_pos": 605.0, "z_level": -1, "icon_sign": 1 },
                            { "id": 50002, "map_id": 2, "label_id": 430, "x_pos": 520.0, "y_pos": 615.0, "z_level": -1, "icon_sign": 1 }
                        ]
                    }]
                }
            }
        });
        let doc = parse_route_auto(&raw, Some(&db)).unwrap();
        let anchor = doc.sections[0].anchor_teleport.as_ref().expect("auto suggested");
        assert!(anchor.auto_suggested);
        assert_eq!(anchor.point_id.as_deref(), Some("99002"));
        assert_eq!(anchor.floor_id.as_deref(), Some("372"));
        assert!(anchor.is_underground);
    }

    #[test]
    fn linearizes_cyclic_branching_relations_and_parses_share_url() {
        let raw = json!({
            "title": "Branch & Cycle",
            "map_id": 9,
            "sections": [{
                "color_id": 4,
                "relation": [
                    { "start_id": 1, "end_id": 2 },
                    { "start_id": 2, "end_id": 3 },
                    { "start_id": 3, "end_id": 1 },
                    { "start_id": 2, "end_id": 4 }
                ],
                "system_points": [
                    { "id": 1, "label_id": 17, "x_pos": 0.0, "y_pos": 0.0 },
                    { "id": 2, "label_id": 17, "x_pos": 10.0, "y_pos": 0.0 },
                    { "id": 3, "label_id": 17, "x_pos": 30.0, "y_pos": 0.0 },
                    { "id": 4, "label_id": 17, "x_pos": 15.0, "y_pos": 0.0 }
                ]
            }]
        });
        let doc = parse_route_auto(&raw, None).unwrap();
        assert_eq!(doc.map_id, 9);
        assert_eq!(doc.sections[0].stops.len(), 4);

        let parsed = parse_hoyolab_route_share_url(
            "https://act.hoyolab.com/ys/app/interactive-map/index.html?lang=en-us#/map/34?share_id=1768-abc-123&index=15&share_type=path",
        );
        assert_eq!(parsed, Some((34, "1768-abc-123".to_owned())));
    }

    #[test]
    fn route_library_save_list_load_delete_and_path_traversal_guard() {
        let dir = std::env::temp_dir().join(format!("gh_core_routes_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let doc = RouteDocument {
            title: "Inazuma Crystal Route!".to_owned(),
            map_id: 2,
            sections: vec![RouteSection {
                index: 0,
                name: "S1".to_owned(),
                color: "#38bdf8".to_owned(),
                anchor_teleport: None,
                stops: vec![RouteStop {
                    id: "1".to_owned(),
                    kind: RouteStopKind::Collectible,
                    lat: 10.0,
                    lng: 20.0,
                    ..Default::default()
                }],
                edges: Vec::new(),
            }],
            ..Default::default()
        };

        let saved_path = save_route_to_library(&dir, &doc).unwrap();
        assert!(saved_path.is_file());
        let list = list_saved_routes(&dir).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].title, "Inazuma Crystal Route!");

        let loaded = load_saved_route(&dir, &list[0].filename, None).unwrap();
        assert_eq!(loaded.title, "Inazuma Crystal Route!");

        assert!(load_saved_route(&dir, "../escape.json", None).is_err());
        assert!(delete_saved_route(&dir, &list[0].filename).unwrap());
        assert!(list_saved_routes(&dir).unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn preserves_outer_envelope_metadata_and_personal_data_sections_shape() {
        let db = mock_db();
        let enveloped = json!({
            "action": "load",
            "title": "Winter Icelea",
            "description": "Winter Icelea * 69",
            "share_url": "https://act.hoyolab.com/ys/app/interactive-map/index.html#/map/2?share_id=abc",
            "path_id": "696714fe",
            "map_id": 2,
            "raw": {
                "retcode": 0,
                "data": {
                    "path": {
                        "obj_id": "696714fe",
                        "post_title": "",
                        "sections": [{
                            "color_id": 1,
                            "system_points": [
                                { "id": 36674, "label_id": 2, "x_pos": -4735.75, "y_pos": 4322.5 },
                                { "id": 77001, "label_id": 770, "x_pos": -4700.0, "y_pos": 4300.0 }
                            ],
                            "relation": [{ "start_id": 36674, "end_id": 77001 }]
                        }]
                    }
                }
            }
        });
        let doc = parse_route_auto(&enveloped, Some(&db)).expect("parse enveloped route");
        assert_eq!(doc.title, "Winter Icelea");
        assert_eq!(doc.description, "Winter Icelea * 69");
        assert_eq!(doc.source.path_id.as_deref(), Some("696714fe"));
        assert!(doc.source.share_url.as_deref().unwrap_or("").contains("share_id=abc"));
        assert_eq!(doc.map_id, 2);

        let personal = json!({
            "action": "load",
            "title": "Personal Route",
            "raw": {
                "retcode": 0,
                "data": {
                    "sections": [{
                        "color_id": 3,
                        "system_points": [
                            { "id": 88001, "map_id": 2, "label_id": 17, "x_pos": 10.0, "y_pos": 20.0 }
                        ]
                    }]
                }
            }
        });
        let pdoc = parse_route_auto(&personal, Some(&db)).expect("parse data.sections shape");
        assert_eq!(pdoc.title, "Personal Route");
        assert_eq!(pdoc.sections.len(), 1);
    }
}
