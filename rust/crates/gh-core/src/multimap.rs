//! Multi-map metadata model & loader (`multimap_v4.json`, split from `src/hd_assets.py`).
//! Pure JSON parsing shared by `gh-core::tsp_route`, `gh-net`, and `gh-cv`.

use crate::assets;
use crate::{CoreError, CoreResult};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const DEFAULT_MULTIMAP_VERSION: &str = "4.0.0";
pub const DEFAULT_TILE_BASE: &str = "https://act-webstatic.hoyoverse.com/map_manage/map";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MultimapDb {
    pub version: String,
    #[serde(
        deserialize_with = "deserialize_maps_compat",
        serialize_with = "serialize_maps_sorted"
    )]
    pub maps: BTreeMap<i32, MapMeta>,
}

impl Default for MultimapDb {
    fn default() -> Self {
        Self {
            version: DEFAULT_MULTIMAP_VERSION.to_owned(),
            maps: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MapMeta {
    pub id: i32,
    pub name: String,
    #[serde(default = "default_priority")]
    pub priority: i32,
    pub detail_v2: Option<DetailV2Meta>,
    pub areas: Vec<AreaEntry>,
    pub landmark_labels: Vec<LandmarkLabelEntry>,
    pub waypoints: Vec<WaypointEntry>,
    pub point_groups: Vec<PointGroupEntry>,
}

impl Default for MapMeta {
    fn default() -> Self {
        Self {
            id: 0,
            name: String::new(),
            priority: default_priority(),
            detail_v2: None,
            areas: Vec::new(),
            landmark_labels: Vec::new(),
            waypoints: Vec::new(),
            point_groups: Vec::new(),
        }
    }
}

fn default_priority() -> i32 {
    99
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DetailV2Meta {
    pub map_version: String,
    pub total_size: [f64; 2],
    pub padding: [f64; 2],
    pub origin: [f64; 2],
    pub original_map_size: Option<[f64; 2]>,
    pub min_zoom: i32,
    pub max_zoom: i32,
}

impl DetailV2Meta {
    pub fn max_tile_xy(&self, zoom: i32) -> (i32, i32) {
        let s = 2f64.powi(zoom);
        let max_tx = ((self.total_size[0] * s - 1.0) / 256.0).floor().max(0.0) as i32;
        let max_ty = ((self.total_size[1] * s - 1.0) / 256.0).floor().max(0.0) as i32;
        (max_tx, max_ty)
    }

    pub fn tile_url(&self, tile_base: &str, map_id: i32, zoom: i32, tx: i32, ty: i32) -> String {
        let trimmed = tile_base.trim().trim_end_matches('/');
        let base = if trimmed.is_empty() {
            DEFAULT_TILE_BASE
        } else {
            trimmed
        };
        let z_str = if zoom < 0 {
            format!("N{}", -zoom)
        } else {
            format!("P{zoom}")
        };
        format!("{base}/{map_id}/{}/{tx}_{ty}_{z_str}.webp", self.map_version)
    }

    pub fn tile_range_for_world_rect(
        &self,
        wx_min: f64,
        wx_max: f64,
        wy_min: f64,
        wy_max: f64,
        zoom: i32,
    ) -> (i32, i32, i32, i32) {
        let ox = self.origin[0];
        let oy = self.origin[1];
        let s = 2f64.powi(zoom);
        let tx0 = (((wx_min + ox) * s / 256.0).floor() as i32).max(0);
        let tx1 = (((wx_max + ox) * s / 256.0).floor() as i32).max(tx0);
        let ty0 = (((wy_min + oy) * s / 256.0).floor() as i32).max(0);
        let ty1 = (((wy_max + oy) * s / 256.0).floor() as i32).max(ty0);
        (tx0, tx1, ty0, ty1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AreaEntry {
    pub id: i32,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LandmarkLabelEntry {
    pub id: i32,
    pub name: String,
    pub is_primary: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WaypointPointGroupRef {
    pub group_id: i32,
    pub floor_id: Option<i32>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct WaypointEntry {
    pub id: i64,
    pub label_id: i32,
    pub is_primary: bool,
    pub x_pos: f64,
    pub y_pos: f64,
    pub area_id: i32,
    pub map_id: i32,
    pub z_level: i32,
    pub icon_sign: i32,
    pub point_group: Option<WaypointPointGroupRef>,
}

pub type WaypointRecord = WaypointEntry;
pub type PointGroupRecord = PointGroupEntry;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PointGroupEntry {
    pub id: i32,
    pub map_id: i32,
    pub name: String,
    #[serde(default = "default_ground_floor_name")]
    pub g_floor_name: String,
    pub default_floor_index: i32,
    pub floors: Vec<FloorEntry>,
}

impl Default for PointGroupEntry {
    fn default() -> Self {
        Self {
            id: 0,
            map_id: 0,
            name: String::new(),
            g_floor_name: default_ground_floor_name(),
            default_floor_index: 0,
            floors: Vec::new(),
        }
    }
}

fn default_ground_floor_name() -> String {
    "Ground".to_owned()
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FloorEntry {
    pub id: i32,
    pub floor_name: String,
    #[serde(alias = "floor_short")]
    pub floor_name_short: String,
    pub overlay: Option<FloorOverlayMeta>,
    pub index: i32,
}

impl FloorEntry {
    /// Short uppercase label used by `locator` / HUD (`B1`, `F2`, ...).
    pub fn short_label(&self) -> String {
        self.short_label_at(self.index.max(0) as usize)
    }

    /// Short uppercase label using `0`-based floor position `f_idx` as fallback (`B{f_idx + 1}`),
    /// matching `src/locator.py:2475-2478`.
    pub fn short_label_at(&self, f_idx: usize) -> String {
        let short = self.floor_name_short.trim();
        if !short.is_empty() {
            return short.to_ascii_uppercase();
        }
        let full = self.floor_name.trim();
        if !full.is_empty() {
            return full.to_owned();
        }
        format!("B{}", f_idx + 1)
    }

    /// Display label used by `hd_assets` (`floor_name_short` || `floor_name` || `F{id}`).
    pub fn overlay_label(&self) -> String {
        let short = self.floor_name_short.trim();
        if !short.is_empty() {
            return short.to_owned();
        }
        let full = self.floor_name.trim();
        if !full.is_empty() {
            return full.to_owned();
        }
        format!("F{}", self.id)
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct FloorOverlayMeta {
    pub url: String,
    pub l_x: f64,
    pub l_y: f64,
    pub r_x: f64,
    pub r_y: f64,
    #[serde(rename = "minLng")]
    pub min_lng: Option<f64>,
    #[serde(rename = "maxLng")]
    pub max_lng: Option<f64>,
    #[serde(rename = "minLat")]
    pub min_lat: Option<f64>,
    #[serde(rename = "maxLat")]
    pub max_lat: Option<f64>,
    pub width: Option<f64>,
    pub height: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflect_y: Option<bool>,
}

impl FloorOverlayMeta {
    /// Returns `(min_lng, max_lng, min_lat, max_lat)` preferring explicit finite bounds
    /// and falling back to `(min(l_x, r_x), max(l_x, r_x), min(l_y, r_y), max(l_y, r_y))`.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let (fallback_lx, fallback_rx) = (self.l_x.min(self.r_x), self.l_x.max(self.r_x));
        let (fallback_ly, fallback_ry) = (self.l_y.min(self.r_y), self.l_y.max(self.r_y));

        match (self.min_lng, self.max_lng, self.min_lat, self.max_lat) {
            (Some(a), Some(b), Some(c), Some(d))
                if a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite() =>
            {
                (a.min(b), a.max(b), c.min(d), c.max(d))
            }
            _ => (fallback_lx, fallback_rx, fallback_ly, fallback_ry),
        }
    }
}

/// Flattened, normalized underground floor record used by `gh-net::hd_assets`,
/// `gh-cv::locator`, `gh-cv::tile_cache`, and `gh-cv::minimap_tracker`.
#[derive(Debug, Clone, PartialEq)]
pub struct FloorOverlayRecord {
    pub map_id: i32,
    pub group_id: i32,
    pub floor_id: i32,
    pub floor_index: usize,
    pub floor_short: String,
    pub floor_name: String,
    pub overlay: FloorOverlayMeta,
    pub lx: f64,
    pub rx: f64,
    pub ly: f64,
    pub ry: f64,
}

fn deserialize_maps_compat<'de, D>(deserializer: D) -> Result<BTreeMap<i32, MapMeta>, D::Error>
where
    D: Deserializer<'de>,
{
    let val = Value::deserialize(deserializer)?;
    let mut out = BTreeMap::new();
    match val {
        Value::Null => {}
        Value::Object(obj) => {
            for (k, v) in obj {
                let Ok(mid) = k.trim().parse::<i32>() else {
                    continue;
                };
                let mut meta: MapMeta = serde_json::from_value(v).map_err(|e| {
                    serde::de::Error::custom(format!("invalid map entry '{k}': {e}"))
                })?;
                if meta.id == 0 {
                    meta.id = mid;
                }
                out.insert(mid, meta);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                let meta: MapMeta = serde_json::from_value(item).map_err(|e| {
                    serde::de::Error::custom(format!("invalid map array entry: {e}"))
                })?;
                if meta.id != 0 {
                    out.insert(meta.id, meta);
                }
            }
        }
        _ => {
            return Err(serde::de::Error::custom(
                "'maps' must be a JSON object or array",
            ));
        }
    }
    Ok(out)
}

fn serialize_maps_sorted<S>(maps: &BTreeMap<i32, MapMeta>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut map_ser = serializer.serialize_map(Some(maps.len()))?;
    for (k, v) in maps {
        map_ser.serialize_entry(&k.to_string(), v)?;
    }
    map_ser.end()
}

pub fn resolve_multimap_json_path(explicit: Option<&Path>, data_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Some(dir) = data_dir {
        let candidate = dir.join("multimap_v4.json");
        if candidate.is_file() {
            return candidate;
        }
    }
    assets::resolve_data_asset("multimap_v4.json")
}

pub fn load_multimap_db(path: &Path) -> CoreResult<MultimapDb> {
    let raw = std::fs::read_to_string(path).map_err(CoreError::Io)?;
    MultimapDb::from_json_str(&raw)
}

impl MultimapDb {
    pub fn from_json_str(raw: &str) -> CoreResult<Self> {
        let mut db: Self = serde_json::from_str(raw)?;
        for (&mid, meta) in &mut db.maps {
            if meta.id == 0 {
                meta.id = mid;
            }
        }
        Ok(db)
    }

    pub fn to_json_string_pretty(&self) -> CoreResult<String> {
        serde_json::to_string_pretty(self).map_err(CoreError::Json)
    }

    pub fn map(&self, map_id: i32) -> Option<&MapMeta> {
        self.maps.get(&map_id)
    }

    pub fn map_name(&self, map_id: i32) -> Option<&str> {
        self.maps
            .get(&map_id)
            .map(|m| m.name.trim())
            .filter(|name| !name.is_empty())
    }

    pub fn local_map_versions(&self) -> BTreeMap<i32, String> {
        let mut out = BTreeMap::new();
        for (&mid, meta) in &self.maps {
            if let Some(d2) = &meta.detail_v2 {
                let ver = d2.map_version.trim();
                if !ver.is_empty() {
                    out.insert(mid, ver.to_owned());
                }
            }
        }
        out
    }

    pub fn waypoints_for_map(&self, map_id: i32) -> &[WaypointEntry] {
        self.maps
            .get(&map_id)
            .map(|m| m.waypoints.as_slice())
            .unwrap_or(&[])
    }

    pub fn primary_waypoints_for_map(&self, map_id: i32) -> Vec<&WaypointEntry> {
        self.waypoints_for_map(map_id)
            .iter()
            .filter(|wp| wp.is_primary)
            .collect()
    }

    pub fn area_names_map(&self) -> BTreeMap<i32, String> {
        let mut out = BTreeMap::new();
        // Insert non-Teyvat submaps first so Map 2 (Teyvat) wins on overlapping area_id
        for (&mid, meta) in &self.maps {
            if mid == 2 {
                continue;
            }
            for area in &meta.areas {
                if !area.name.trim().is_empty() {
                    out.insert(area.id, area.name.clone());
                }
            }
        }
        if let Some(teyvat) = self.maps.get(&2) {
            for area in &teyvat.areas {
                if !area.name.trim().is_empty() {
                    out.insert(area.id, area.name.clone());
                }
            }
        }
        out
    }

    pub fn floor_overlays_for_map(&self, map_id: i32) -> Vec<FloorOverlayRecord> {
        let Some(meta) = self.maps.get(&map_id) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for grp in &meta.point_groups {
            for (idx, fl) in grp.floors.iter().enumerate() {
                let Some(ov) = &fl.overlay else {
                    continue;
                };
                if ov.url.trim().is_empty() {
                    continue;
                }
                let lx = ov.l_x.min(ov.r_x);
                let rx = ov.l_x.max(ov.r_x);
                let ly = ov.l_y.min(ov.r_y);
                let ry = ov.l_y.max(ov.r_y);
                let floor_index = if fl.index <= 0 && idx > 0 {
                    idx
                } else {
                    fl.index.max(0) as usize
                };
                out.push(FloorOverlayRecord {
                    map_id,
                    group_id: grp.id,
                    floor_id: fl.id,
                    floor_index,
                    floor_short: fl.overlay_label(),
                    floor_name: fl.floor_name.clone(),
                    overlay: ov.clone(),
                    lx,
                    rx,
                    ly,
                    ry,
                });
            }
        }
        out
    }

    pub fn find_overlapping_floors(
        &self,
        map_id: i32,
        wx: f64,
        wy: f64,
        pad: f64,
    ) -> Vec<FloorOverlayRecord> {
        if !wx.is_finite() || !wy.is_finite() || !pad.is_finite() {
            return Vec::new();
        }
        self.floor_overlays_for_map(map_id)
            .into_iter()
            .filter(|rec| {
                (rec.lx - pad) <= wx
                    && wx <= (rec.rx + pad)
                    && (rec.ly - pad) <= wy
                    && wy <= (rec.ry + pad)
            })
            .collect()
    }

    pub fn get_floor_overlay(
        &self,
        map_id: i32,
        floor_id: i32,
        group_id: Option<i32>,
    ) -> Option<FloorOverlayRecord> {
        let records = self.floor_overlays_for_map(map_id);
        if let Some(gid) = group_id
            && let Some(found) = records
                .iter()
                .find(|r| r.group_id == gid && r.floor_id == floor_id)
        {
            return Some(found.clone());
        }
        records.into_iter().find(|r| r.floor_id == floor_id)
    }

    pub fn get_floor_overlay_str(&self, map_id: i32, floor_id: &str) -> Option<FloorOverlayRecord> {
        let fid = floor_id.trim().parse::<i32>().ok()?;
        self.get_floor_overlay(map_id, fid, None)
    }

    pub fn resolve_floor_label(&self, map_id: i32, floor_id: i32) -> Option<String> {
        if floor_id <= 0 {
            return None;
        }
        let meta = self.maps.get(&map_id)?;
        for grp in &meta.point_groups {
            for (f_idx, fl) in grp.floors.iter().enumerate() {
                if fl.id == floor_id {
                    return Some(fl.short_label_at(f_idx));
                }
            }
        }
        None
    }

    /// Resolve the parent `group_id` owning `floor_id` on `map_id`.
    pub fn resolve_floor_group_id(&self, map_id: i32, floor_id: i32) -> Option<i32> {
        if floor_id <= 0 {
            return None;
        }
        let meta = self.maps.get(&map_id)?;
        for grp in &meta.point_groups {
            if grp.floors.iter().any(|fl| fl.id == floor_id) {
                return Some(grp.id);
            }
        }
        None
    }

    /// Infer `(area_id, area_name)` from the nearest waypoint with `area_id > 0` on `map_id`.
    /// Returns `None` if coordinates are non-finite or the map has no area-tagged waypoints.
    pub fn nearest_area_for_pos(
        &self,
        map_id: i32,
        world_x: f64,
        world_y: f64,
    ) -> Option<(i32, Option<String>)> {
        if !world_x.is_finite() || !world_y.is_finite() {
            return None;
        }
        let meta = self.maps.get(&map_id)?;
        let wp = meta
            .waypoints
            .iter()
            .filter(|w| w.area_id > 0)
            .min_by(|a, b| {
                let da = (a.x_pos - world_x).powi(2) + (a.y_pos - world_y).powi(2);
                let db = (b.x_pos - world_x).powi(2) + (b.y_pos - world_y).powi(2);
                da.total_cmp(&db)
            })?;
        let name = meta
            .areas
            .iter()
            .find(|a| a.id == wp.area_id)
            .map(|a| a.name.trim().to_owned())
            .filter(|n| !n.is_empty());
        Some((wp.area_id, name))
    }

    /// Mirrors `src/locator.py:2500-2552`: returns `Some((floor_id, label))` ONLY when
    /// coordinates are finite and **exactly 1** distinct non-zero `floor_id` overlay covers
    /// `(lng, lat)` within `margin.max(0.0)`.
    pub fn infer_unique_underground_floor(
        &self,
        map_id: i32,
        lat: f64,
        lng: f64,
        margin: f64,
    ) -> Option<(i32, String)> {
        if !lat.is_finite() || !lng.is_finite() {
            return None;
        }
        let meta = self.maps.get(&map_id)?;
        let pad = if margin.is_finite() {
            margin.max(0.0)
        } else {
            0.0
        };

        let mut matches: BTreeMap<i32, String> = BTreeMap::new();
        for grp in &meta.point_groups {
            for (f_idx, fl) in grp.floors.iter().enumerate() {
                if fl.id == 0 {
                    continue;
                }
                let Some(ov) = &fl.overlay else {
                    continue;
                };
                if ov.url.trim().is_empty() {
                    continue;
                }
                let (min_lng, max_lng, min_lat, max_lat) = ov.bounds();
                if (min_lng - pad) <= lng
                    && lng <= (max_lng + pad)
                    && (min_lat - pad) <= lat
                    && lat <= (max_lat + pad)
                {
                    matches
                        .entry(fl.id)
                        .or_insert_with(|| fl.short_label_at(f_idx));
                }
            }
        }

        if matches.len() == 1 {
            matches.into_iter().next()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_dummy_and_array_maps_compat() {
        let dummy = r#"{"version": "test-v1", "maps": {"2": {"name": "Teyvat"}}}"#;
        let db = MultimapDb::from_json_str(dummy).expect("minimal dummy");
        assert_eq!(db.version, "test-v1");
        let m2 = db.map(2).expect("map 2");
        assert_eq!(m2.id, 2);
        assert_eq!(m2.name, "Teyvat");
        assert!(m2.detail_v2.is_none());

        let arr_json = r#"{"version": "4.0.0", "maps": [{"id": 7, "name": "Enkanomiya"}]}"#;
        let db_arr = MultimapDb::from_json_str(arr_json).expect("array maps");
        assert_eq!(db_arr.map_name(7), Some("Enkanomiya"));
    }

    #[test]
    fn roundtrips_sample_multimap_v4_and_missing_optional_fields() {
        let sample = r#"{
            "version": "4.0.0",
            "maps": {
                "2": {
                    "id": 2,
                    "name": "Teyvat",
                    "priority": 1,
                    "detail_v2": {
                        "map_version": "abc123",
                        "total_size": [36864, 32768],
                        "padding": [7625, 6881],
                        "origin": [24207, 16411],
                        "min_zoom": -4,
                        "max_zoom": 0
                    },
                    "areas": [{"id": 10, "name": "Mondstadt"}],
                    "waypoints": [
                        {"id": 9000000001, "label_id": 2, "is_primary": true, "x_pos": 100.5, "y_pos": -200.25, "area_id": 10, "map_id": 2, "point_group": null},
                        {"id": 9000000002, "label_id": 3, "is_primary": false, "x_pos": 50.0, "y_pos": 60.0, "area_id": 10, "map_id": 2, "point_group": {"group_id": 235, "floor_id": null}}
                    ],
                    "point_groups": [
                        {
                            "id": 235,
                            "map_id": 2,
                            "name": "Cave",
                            "floors": [
                                {"id": 301, "floor_name": "Upper", "floor_short": "b1", "index": 0, "overlay": null}
                            ]
                        }
                    ]
                }
            }
        }"#;

        let db = MultimapDb::from_json_str(sample).expect("parse sample");
        assert_eq!(db.waypoints_for_map(2).len(), 2);
        assert_eq!(db.primary_waypoints_for_map(2).len(), 1);
        assert_eq!(db.waypoints_for_map(2)[0].id, 9_000_000_001);
        assert_eq!(db.resolve_floor_label(2, 301).as_deref(), Some("B1"));

        let d2 = db.map(2).unwrap().detail_v2.as_ref().unwrap();
        assert_eq!(d2.max_tile_xy(0), (143, 127));
        assert_eq!(
            d2.tile_url("", 2, -2, 3, 4),
            "https://act-webstatic.hoyoverse.com/map_manage/map/2/abc123/3_4_N2.webp"
        );
        assert_eq!(
            d2.tile_url("https://cdn.example.com/", 2, 0, 1, 2),
            "https://cdn.example.com/2/abc123/1_2_P0.webp"
        );

        let versions = db.local_map_versions();
        assert_eq!(versions.get(&2).map(String::as_str), Some("abc123"));

        let pretty = db.to_json_string_pretty().expect("pretty");
        let reparsed = MultimapDb::from_json_str(&pretty).expect("reparse");
        assert_eq!(db, reparsed);
    }

    #[test]
    fn floor_queries_and_unique_inference() {
        let sample = r#"{
            "version": "4.0.0",
            "maps": {
                "2": {
                    "id": 2,
                    "name": "Teyvat",
                    "point_groups": [
                        {
                            "id": 100,
                            "map_id": 2,
                            "name": "Ruins",
                            "floors": [
                                {
                                    "id": 501,
                                    "floor_name": "Level 1",
                                    "floor_name_short": "b1",
                                    "index": 0,
                                    "overlay": {"url": "https://x/1.png", "l_x": 10.0, "l_y": 10.0, "r_x": 50.0, "r_y": 50.0}
                                },
                                {
                                    "id": 502,
                                    "floor_name": "Level 2",
                                    "floor_name_short": "b2",
                                    "index": 1,
                                    "overlay": {"url": "https://x/2.png", "l_x": 40.0, "l_y": 40.0, "r_x": 80.0, "r_y": 80.0}
                                }
                            ]
                        }
                    ]
                }
            }
        }"#;

        let db = MultimapDb::from_json_str(sample).expect("parse floors");
        assert_eq!(db.find_overlapping_floors(2, 20.0, 20.0, 0.0).len(), 1);
        assert_eq!(db.find_overlapping_floors(2, 45.0, 45.0, 0.0).len(), 2);

        assert!(db.get_floor_overlay(2, 501, Some(100)).is_some());
        assert!(db.get_floor_overlay_str(2, "502").is_some());

        // Unique match -> Some
        assert_eq!(
            db.infer_unique_underground_floor(2, 20.0, 20.0, 0.0),
            Some((501, "B1".to_owned()))
        );
        // Zero matches -> None
        assert_eq!(db.infer_unique_underground_floor(2, 200.0, 200.0, 0.0), None);
        // Ambiguous (2 floors) -> None
        assert_eq!(db.infer_unique_underground_floor(2, 45.0, 45.0, 0.0), None);
        // Non-finite -> None
        assert_eq!(db.infer_unique_underground_floor(2, f64::NAN, 20.0, 0.0), None);
    }

    #[test]
    fn resolve_floor_label_rejects_zero_or_negative_id_and_uses_position_fallback() {
        let sample = r#"{
            "version": "4.0.0",
            "maps": {
                "2": {
                    "id": 2,
                    "name": "Teyvat",
                    "point_groups": [
                        {
                            "id": 14,
                            "map_id": 2,
                            "name": "Placeholder Group",
                            "floors": [
                                {"id": 0, "floor_name": "", "floor_name_short": "", "index": 0, "overlay": null},
                                {"id": 302, "floor_name": "", "floor_name_short": "", "overlay": null}
                            ]
                        }
                    ]
                }
            }
        }"#;
        let db = MultimapDb::from_json_str(sample).expect("parse synthetic");
        assert_eq!(db.resolve_floor_label(2, 0), None);
        assert_eq!(db.resolve_floor_label(2, -1), None);
        assert_eq!(db.resolve_floor_label(2, 302).as_deref(), Some("B2"));
        assert_eq!(db.resolve_floor_group_id(2, 0), None);
        assert_eq!(db.resolve_floor_group_id(2, -1), None);
        assert_eq!(db.resolve_floor_group_id(2, 302), Some(14));
        assert_eq!(db.resolve_floor_group_id(2, 999), None);
    }

    #[test]
    fn nearest_area_for_pos_skips_untagged_and_non_finite() {
        let sample = r#"{
            "version": "4.0.0",
            "maps": {
                "2": {
                    "id": 2,
                    "name": "Teyvat",
                    "areas": [
                        {"id": 1, "name": "Mondstadt"},
                        {"id": 11, "name": "Natlan"}
                    ],
                    "waypoints": [
                        {"id": 1, "label_id": 2, "is_primary": true, "x_pos": 0.0, "y_pos": 0.0, "area_id": 0},
                        {"id": 2, "label_id": 2, "is_primary": true, "x_pos": 100.0, "y_pos": 100.0, "area_id": 1},
                        {"id": 3, "label_id": 2, "is_primary": true, "x_pos": -4000.0, "y_pos": 2500.0, "area_id": 11},
                        {"id": 4, "label_id": 2, "is_primary": true, "x_pos": 9000.0, "y_pos": 9000.0, "area_id": 99}
                    ]
                }
            }
        }"#;
        let db = MultimapDb::from_json_str(sample).expect("parse areas");
        // Closest waypoint (0, 0) has area_id=0 and must be skipped in favor of Mondstadt (100, 100).
        assert_eq!(
            db.nearest_area_for_pos(2, 1.0, 1.0),
            Some((1, Some("Mondstadt".to_owned())))
        );
        assert_eq!(
            db.nearest_area_for_pos(2, -3950.0, 2480.0),
            Some((11, Some("Natlan".to_owned())))
        );
        // Waypoint with area_id not listed in `areas` still returns its id with None name.
        assert_eq!(db.nearest_area_for_pos(2, 8990.0, 9010.0), Some((99, None)));
        assert_eq!(db.nearest_area_for_pos(2, f64::NAN, 0.0), None);
        assert_eq!(db.nearest_area_for_pos(2, 0.0, f64::INFINITY), None);
        assert_eq!(db.nearest_area_for_pos(7, 0.0, 0.0), None);
    }

    #[test]
    fn loads_real_multimap_v4_json_when_present() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("data")
            .join("multimap_v4.json");
        if !path.is_file() {
            return;
        }
        let db = load_multimap_db(&path).expect("load real multimap_v4.json");
        for mid in [2, 7, 9, 34, 36, 37, 40] {
            assert!(db.map(mid).is_some(), "missing map {mid}");
        }
        assert!(db.waypoints_for_map(2).len() >= 800);
        assert!(db.map(2).unwrap().point_groups.len() >= 200);
        assert_eq!(db.resolve_floor_label(2, 0), None);
        assert_eq!(db.resolve_floor_label(2, -1), None);
    }
}
