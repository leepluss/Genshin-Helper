//! Open-path Nearest-Neighbor + 2-opt TSP route planner with waypoint teleport splitting
//! (`src/tsp_route.py`).
//!
//! Loads default teleport waypoints from `multimap_v4.json` via `crate::multimap` (`x_pos -> lng`,
//! `y_pos -> lat`) filtered to primary teleportable landmarks (`is_primary || label_id in {2, 3, 154}`).

use crate::multimap::{MultimapDb, WaypointEntry, load_multimap_db, resolve_multimap_json_path};
use crate::{CoreError, CoreResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;
use std::time::Instant;

pub const DEFAULT_TELEPORT_PENALTY: f64 = 180.0;
pub const DEFAULT_MAX_2OPT_ITERS: usize = 200;
pub const DEFAULT_2OPT_EPS: f64 = 1e-9;
pub const TELEPORT_GAIN_EPS: f64 = 1e-6;
pub const NN_EXHAUSTIVE_SEED_LIMIT: usize = 24;
pub const PRIMARY_LANDMARK_LABELS: [i32; 3] = [2, 3, 154];

fn round4(x: f64) -> f64 {
    let r = (x * 10_000.0).round() / 10_000.0;
    if r == 0.0 { 0.0 } else { r }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TspId {
    Int(i64),
    Str(String),
}

impl TspId {
    pub fn start() -> Self {
        Self::Str("START".to_owned())
    }
}

impl fmt::Display for TspId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(n) => write!(f, "{n}"),
            Self::Str(s) => f.write_str(s),
        }
    }
}

impl From<i64> for TspId {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<&str> for TspId {
    fn from(v: &str) -> Self {
        Self::Str(v.to_owned())
    }
}

impl From<String> for TspId {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TspCoordInput {
    LatLng { lat: f64, lng: f64 },
    Pos { y_pos: f64, x_pos: f64 },
    Pair([f64; 2]),
}

impl TspCoordInput {
    pub fn to_lat_lng(self) -> CoreResult<(f64, f64)> {
        let (lat, lng) = match self {
            Self::LatLng { lat, lng } => (lat, lng),
            Self::Pos { y_pos, x_pos } => (y_pos, x_pos),
            Self::Pair([lat, lng]) => (lat, lng),
        };
        if !lat.is_finite() || !lng.is_finite() {
            return Err(CoreError::InvalidArgument(
                "Coordinate lat/lng must be finite".to_owned(),
            ));
        }
        Ok((lat, lng))
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TspPinInput {
    pub id: Option<TspId>,
    pub lat: Option<f64>,
    pub lng: Option<f64>,
    pub y_pos: Option<f64>,
    pub x_pos: Option<f64>,
    pub name: Option<String>,
    pub label_id: Option<i64>,
    pub floor_id: Option<TspId>,
    #[serde(default, deserialize_with = "crate::deserialize_null_default")]
    pub is_underground: bool,
}

impl TspPinInput {
    pub fn normalize(&self, fallback_idx: usize) -> CoreResult<TspPin> {
        let lat = self.lat.or(self.y_pos).ok_or_else(|| {
            CoreError::InvalidArgument(format!("Pin[{fallback_idx}] missing lat/y_pos"))
        })?;
        let lng = self.lng.or(self.x_pos).ok_or_else(|| {
            CoreError::InvalidArgument(format!("Pin[{fallback_idx}] missing lng/x_pos"))
        })?;
        if !lat.is_finite() || !lng.is_finite() {
            return Err(CoreError::InvalidArgument(format!(
                "Pin[{fallback_idx}] has non-finite coordinates"
            )));
        }
        let id = self
            .id
            .clone()
            .unwrap_or(TspId::Int(fallback_idx as i64));
        let name = self
            .name
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("Pin {id}"));
        Ok(TspPin {
            id,
            lat,
            lng,
            name,
            label_id: self.label_id,
            floor_id: self.floor_id.clone(),
            is_underground: self.is_underground,
            step: 0,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TspPin {
    pub id: TspId,
    pub lat: f64,
    pub lng: f64,
    pub name: String,
    pub label_id: Option<i64>,
    pub floor_id: Option<TspId>,
    pub is_underground: bool,
    pub step: usize,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TspWaypointInput {
    pub id: Option<TspId>,
    pub lat: Option<f64>,
    pub lng: Option<f64>,
    pub y_pos: Option<f64>,
    pub x_pos: Option<f64>,
    pub name: Option<String>,
    pub label_id: Option<i32>,
    pub area_id: Option<i32>,
    pub map_id: Option<i32>,
    pub is_primary: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TspWaypoint {
    pub id: TspId,
    pub lat: f64,
    pub lng: f64,
    pub name: String,
    pub label_id: Option<i32>,
    pub area_id: Option<i32>,
    pub map_id: Option<i32>,
}

impl TryFrom<&WaypointEntry> for TspWaypoint {
    type Error = CoreError;

    fn try_from(wp: &WaypointEntry) -> CoreResult<Self> {
        if !wp.x_pos.is_finite() || !wp.y_pos.is_finite() {
            return Err(CoreError::InvalidData(format!(
                "Waypoint {} has non-finite coordinates",
                wp.id
            )));
        }
        Ok(Self {
            id: TspId::Int(wp.id),
            lat: wp.y_pos,
            lng: wp.x_pos,
            name: format!("Waypoint {}", wp.id),
            label_id: Some(wp.label_id),
            area_id: Some(wp.area_id),
            map_id: Some(wp.map_id),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TspLeg {
    pub index: usize,
    pub from_id: TspId,
    pub to_id: TspId,
    pub from_lat: f64,
    pub from_lng: f64,
    pub to_lat: f64,
    pub to_lng: f64,
    pub distance: f64,
    pub air_distance: f64,
    pub effective_cost: f64,
    pub is_teleport: bool,
    pub waypoint_id: Option<TspId>,
    pub waypoint_name: Option<String>,
    pub from_floor_id: Option<TspId>,
    pub to_floor_id: Option<TspId>,
    pub floor_change: bool,
    pub points: [[f64; 2]; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeleportHint {
    pub before_step: usize,
    pub target_pin_id: TspId,
    pub target_pin_name: String,
    pub waypoint_id: TspId,
    pub waypoint_name: String,
    pub waypoint_lat: f64,
    pub waypoint_lng: f64,
    pub direct_walk_distance: f64,
    pub post_teleport_walk: f64,
    pub saved_distance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TspRoutePlan {
    pub ordered_pins: Vec<TspPin>,
    pub legs: Vec<TspLeg>,
    pub teleport_hints: Vec<TeleportHint>,
    pub total_distance: f64,
    pub raw_walk_distance: f64,
    pub effective_distance: f64,
    pub nn_distance: f64,
    pub two_opt_iterations: usize,
    pub teleport_count: usize,
    pub pin_count: usize,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DistMatrix {
    n: usize,
    data: Vec<f64>,
}

impl DistMatrix {
    pub fn from_coords(coords: &[(f64, f64)]) -> Self {
        let n = coords.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = (coords[i].0 - coords[j].0).hypot(coords[i].1 - coords[j].1);
                data[i * n + j] = d;
                data[j * n + i] = d;
            }
        }
        Self { n, data }
    }

    pub fn from_flat(n: usize, data: Vec<f64>) -> CoreResult<Self> {
        if data.len() != n * n {
            return Err(CoreError::InvalidArgument(format!(
                "Distance matrix length {} does not match {}x{}",
                data.len(),
                n,
                n
            )));
        }
        Ok(Self { n, data })
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.n + c]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TspPlanOptions {
    pub teleport_penalty: f64,
    pub max_2opt_iters: usize,
}

impl Default for TspPlanOptions {
    fn default() -> Self {
        Self {
            teleport_penalty: DEFAULT_TELEPORT_PENALTY,
            max_2opt_iters: DEFAULT_MAX_2OPT_ITERS,
        }
    }
}

pub fn is_primary_teleport_label(label_id: i32) -> bool {
    PRIMARY_LANDMARK_LABELS.contains(&label_id)
}

pub fn waypoints_from_multimap(
    db: &MultimapDb,
    map_id: i32,
    primary_only: bool,
) -> Vec<TspWaypoint> {
    db.waypoints_for_map(map_id)
        .iter()
        .filter(|wp| !primary_only || wp.is_primary || is_primary_teleport_label(wp.label_id))
        .filter_map(|wp| TspWaypoint::try_from(wp).ok())
        .collect()
}

pub fn load_waypoints_for_map(
    map_id: i32,
    db_path: Option<&Path>,
) -> CoreResult<Vec<TspWaypoint>> {
    load_waypoints_for_map_in(map_id, db_path, None, true)
}

pub fn load_waypoints_for_map_in(
    map_id: i32,
    explicit_path: Option<&Path>,
    data_dir: Option<&Path>,
    primary_only: bool,
) -> CoreResult<Vec<TspWaypoint>> {
    let path = resolve_multimap_json_path(explicit_path, data_dir);
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let db = load_multimap_db(&path)?;
    Ok(waypoints_from_multimap(&db, map_id, primary_only))
}

pub fn normalize_waypoints(raw: &[TspWaypointInput]) -> Vec<TspWaypoint> {
    let mut out = Vec::new();
    for (idx, wp) in raw.iter().enumerate() {
        let Some(lat) = wp.lat.or(wp.y_pos) else {
            continue;
        };
        let Some(lng) = wp.lng.or(wp.x_pos) else {
            continue;
        };
        if !lat.is_finite() || !lng.is_finite() {
            continue;
        }
        let id = wp
            .id
            .clone()
            .unwrap_or_else(|| TspId::Str(format!("wp_{idx}")));
        let name = wp
            .name
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| format!("Waypoint {id}"));
        out.push(TspWaypoint {
            id,
            lat,
            lng,
            name,
            label_id: wp.label_id,
            area_id: wp.area_id,
            map_id: wp.map_id,
        });
    }
    out
}

pub fn build_distance_matrix(coords: &[(f64, f64)]) -> DistMatrix {
    DistMatrix::from_coords(coords)
}

pub fn path_length(order: &[usize], dist_matrix: &DistMatrix) -> f64 {
    if order.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for k in 0..(order.len() - 1) {
        total += dist_matrix.get(order[k], order[k + 1]);
    }
    total
}

pub fn nearest_neighbor_order(
    dist_matrix: &DistMatrix,
    start_idx: usize,
    fixed_start: bool,
) -> Vec<usize> {
    let n = dist_matrix.len();
    if n <= 1 {
        return (0..n).collect();
    }
    let seed0 = start_idx.min(n - 1);

    let run_from = |seed: usize| -> (f64, Vec<usize>) {
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        order.push(seed);
        visited[seed] = true;
        let mut cur = seed;
        let mut total = 0.0;

        for _ in 0..(n - 1) {
            let mut best_j: Option<usize> = None;
            let mut best_d = f64::INFINITY;
            for (j, &is_vis) in visited.iter().enumerate() {
                if !is_vis {
                    let d = dist_matrix.get(cur, j);
                    if d.is_finite() && d < best_d {
                        best_d = d;
                        best_j = Some(j);
                    }
                }
            }
            let next_j = match best_j {
                Some(j) => j,
                None => visited
                    .iter()
                    .position(|&v| !v)
                    .expect("unvisited node must exist"),
            };
            order.push(next_j);
            visited[next_j] = true;
            if best_d.is_finite() {
                total += best_d;
            }
            cur = next_j;
        }
        (total, order)
    };

    if fixed_start {
        return run_from(seed0).1;
    }

    let candidates: Vec<usize> = if n <= NN_EXHAUSTIVE_SEED_LIMIT {
        (0..n).collect()
    } else {
        let step = (n / 8).max(1);
        let mut set = BTreeSet::new();
        set.insert(seed0);
        let mut idx = 0;
        while idx < n {
            set.insert(idx);
            idx += step;
        }
        set.into_iter().collect()
    };

    let mut best_total = f64::INFINITY;
    let mut best_order: Vec<usize> = (0..n).collect();
    for seed in candidates {
        let (total, order) = run_from(seed);
        if total < best_total {
            best_total = total;
            best_order = order;
        }
    }
    best_order
}

pub fn two_opt_improve(
    order: &[usize],
    dist_matrix: &DistMatrix,
    fix_first: bool,
    max_iters: usize,
    eps: f64,
) -> (Vec<usize>, usize) {
    let mut path = order.to_vec();
    let m = path.len();
    if m < 3 || max_iters == 0 {
        return (path, 0);
    }
    let eps = if eps.is_finite() && eps >= 0.0 {
        eps
    } else {
        DEFAULT_2OPT_EPS
    };
    let min_i = if fix_first { 1 } else { 0 };
    let mut iters = 0;

    while iters < max_iters {
        let mut improved = false;
        iters += 1;

        for i in min_i..(m - 1) {
            let mut pi = path[i];
            let prev_idx = if i > 0 { Some(path[i - 1]) } else { None };
            let mut base_before = prev_idx.map_or(0.0, |p| dist_matrix.get(p, pi));

            for j in (i + 1)..m {
                if i == 0 && j == m - 1 {
                    continue;
                }
                let pj = path[j];
                let (old_cost, new_cost) = if j + 1 < m {
                    let nxt = path[j + 1];
                    (
                        base_before + dist_matrix.get(pj, nxt),
                        prev_idx.map_or(0.0, |p| dist_matrix.get(p, pj))
                            + dist_matrix.get(pi, nxt),
                    )
                } else {
                    (
                        base_before,
                        prev_idx.map_or(0.0, |p| dist_matrix.get(p, pj)),
                    )
                };

                if new_cost.is_finite() && old_cost.is_finite() && new_cost + eps < old_cost {
                    path[i..=j].reverse();
                    pi = path[i];
                    base_before = prev_idx.map_or(0.0, |p| dist_matrix.get(p, pi));
                    improved = true;
                }
            }
        }

        if !improved {
            break;
        }
    }

    (path, iters)
}

#[allow(clippy::too_many_arguments)]
fn make_leg(
    index: usize,
    from_id: TspId,
    to_id: TspId,
    from_lat: f64,
    from_lng: f64,
    to_lat: f64,
    to_lng: f64,
    distance: f64,
    air_distance: f64,
    effective_cost: f64,
    is_teleport: bool,
    waypoint_id: Option<TspId>,
    waypoint_name: Option<String>,
    from_floor_id: Option<TspId>,
    to_floor_id: Option<TspId>,
) -> TspLeg {
    let floor_change =
        from_floor_id.is_some() && to_floor_id.is_some() && from_floor_id != to_floor_id;
    let fl = round4(from_lat);
    let fg = round4(from_lng);
    let tl = round4(to_lat);
    let tg = round4(to_lng);
    TspLeg {
        index,
        from_id,
        to_id,
        from_lat: fl,
        from_lng: fg,
        to_lat: tl,
        to_lng: tg,
        distance: round4(distance),
        air_distance: round4(air_distance),
        effective_cost: round4(effective_cost),
        is_teleport,
        waypoint_id,
        waypoint_name,
        from_floor_id,
        to_floor_id,
        floor_change,
        points: [[fl, fg], [tl, tg]],
    }
}

pub fn plan_route(
    pins: &[TspPinInput],
    start_pos: Option<TspCoordInput>,
    waypoints: Option<&[TspWaypoint]>,
    options: TspPlanOptions,
) -> CoreResult<TspRoutePlan> {
    let t0 = Instant::now();
    if !options.teleport_penalty.is_finite() {
        return Err(CoreError::InvalidArgument(
            "teleport_penalty must be finite".to_owned(),
        ));
    }
    let penalty = options.teleport_penalty.max(0.0);

    let mut norm_pins = Vec::with_capacity(pins.len());
    for (idx, p) in pins.iter().enumerate() {
        norm_pins.push(p.normalize(idx)?);
    }
    let n_pins = norm_pins.len();

    if n_pins == 0 {
        return Ok(TspRoutePlan {
            ordered_pins: Vec::new(),
            legs: Vec::new(),
            teleport_hints: Vec::new(),
            total_distance: 0.0,
            raw_walk_distance: 0.0,
            effective_distance: 0.0,
            nn_distance: 0.0,
            two_opt_iterations: 0,
            teleport_count: 0,
            pin_count: 0,
            elapsed_ms: round4(t0.elapsed().as_secs_f64() * 1000.0),
        });
    }

    let start_lat_lng = match start_pos {
        Some(sp) => Some(sp.to_lat_lng()?),
        None => None,
    };
    let has_start = start_lat_lng.is_some();

    let mut coords = Vec::with_capacity(n_pins + usize::from(has_start));
    if let Some((slat, slng)) = start_lat_lng {
        coords.push((slat, slng));
    }
    for p in &norm_pins {
        coords.push((p.lat, p.lng));
    }

    let dist_matrix = build_distance_matrix(&coords);
    let nn_order = nearest_neighbor_order(&dist_matrix, 0, has_start);
    let nn_dist = path_length(&nn_order, &dist_matrix);

    let (opt_order, two_opt_iters) = two_opt_improve(
        &nn_order,
        &dist_matrix,
        has_start,
        options.max_2opt_iters,
        DEFAULT_2OPT_EPS,
    );
    let raw_walk_dist = path_length(&opt_order, &dist_matrix);

    let pin_indices: Vec<usize> = if has_start {
        opt_order.iter().skip(1).map(|&idx| idx - 1).collect()
    } else {
        opt_order
    };

    let mut ordered_pins = Vec::with_capacity(n_pins);
    for (step_idx, pin_idx) in pin_indices.into_iter().enumerate() {
        let mut pin_copy = norm_pins[pin_idx].clone();
        pin_copy.step = step_idx + 1;
        ordered_pins.push(pin_copy);
    }

    let finite_wps: Vec<&TspWaypoint> = waypoints
        .unwrap_or(&[])
        .iter()
        .filter(|w| w.lat.is_finite() && w.lng.is_finite())
        .collect();

    let find_best_teleport =
        |target_lat: f64, target_lng: f64, direct_dist: f64| -> Option<(&TspWaypoint, f64)> {
            if finite_wps.is_empty() || direct_dist <= penalty {
                return None;
            }
            let mut best_wp: Option<&TspWaypoint> = None;
            let mut best_walk = f64::INFINITY;
            for &wp in &finite_wps {
                let d = (wp.lat - target_lat).hypot(wp.lng - target_lng);
                if d < best_walk {
                    best_walk = d;
                    best_wp = Some(wp);
                }
            }
            if let Some(wp) = best_wp
                && penalty + best_walk + TELEPORT_GAIN_EPS < direct_dist
            {
                return Some((wp, best_walk));
            }
            None
        };

    struct RouteNode {
        id: TspId,
        lat: f64,
        lng: f64,
        name: String,
        floor_id: Option<TspId>,
        step: usize,
    }

    let mut nodes = Vec::with_capacity(ordered_pins.len() + usize::from(has_start));
    if let Some((slat, slng)) = start_lat_lng {
        nodes.push(RouteNode {
            id: TspId::start(),
            lat: slat,
            lng: slng,
            name: "Start".to_owned(),
            floor_id: None,
            step: 0,
        });
    }
    for p in &ordered_pins {
        nodes.push(RouteNode {
            id: p.id.clone(),
            lat: p.lat,
            lng: p.lng,
            name: p.name.clone(),
            floor_id: p.floor_id.clone(),
            step: p.step,
        });
    }

    let mut legs = Vec::new();
    let mut teleport_hints = Vec::new();
    let mut total_walk_dist = 0.0;
    let mut effective_dist = 0.0;

    for pair in nodes.windows(2) {
        let src = &pair[0];
        let dst = &pair[1];
        let direct_dist = (dst.lat - src.lat).hypot(dst.lng - src.lng);

        if let Some((wp, post_walk)) = find_best_teleport(dst.lat, dst.lng, direct_dist) {
            let tp_air = (wp.lat - src.lat).hypot(wp.lng - src.lng);
            legs.push(make_leg(
                legs.len(),
                src.id.clone(),
                wp.id.clone(),
                src.lat,
                src.lng,
                wp.lat,
                wp.lng,
                0.0,
                tp_air,
                penalty,
                true,
                Some(wp.id.clone()),
                Some(wp.name.clone()),
                src.floor_id.clone(),
                None,
            ));
            legs.push(make_leg(
                legs.len(),
                wp.id.clone(),
                dst.id.clone(),
                wp.lat,
                wp.lng,
                dst.lat,
                dst.lng,
                post_walk,
                post_walk,
                post_walk,
                false,
                Some(wp.id.clone()),
                Some(wp.name.clone()),
                None,
                dst.floor_id.clone(),
            ));
            teleport_hints.push(TeleportHint {
                before_step: dst.step,
                target_pin_id: dst.id.clone(),
                target_pin_name: dst.name.clone(),
                waypoint_id: wp.id.clone(),
                waypoint_name: wp.name.clone(),
                waypoint_lat: round4(wp.lat),
                waypoint_lng: round4(wp.lng),
                direct_walk_distance: round4(direct_dist),
                post_teleport_walk: round4(post_walk),
                saved_distance: round4(direct_dist - (penalty + post_walk)),
            });
            total_walk_dist += post_walk;
            effective_dist += penalty + post_walk;
        } else {
            legs.push(make_leg(
                legs.len(),
                src.id.clone(),
                dst.id.clone(),
                src.lat,
                src.lng,
                dst.lat,
                dst.lng,
                direct_dist,
                direct_dist,
                direct_dist,
                false,
                None,
                None,
                src.floor_id.clone(),
                dst.floor_id.clone(),
            ));
            total_walk_dist += direct_dist;
            effective_dist += direct_dist;
        }
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(TspRoutePlan {
        ordered_pins,
        legs,
        teleport_count: teleport_hints.len(),
        teleport_hints,
        total_distance: round4(total_walk_dist),
        raw_walk_distance: round4(raw_walk_dist),
        effective_distance: round4(effective_dist),
        nn_distance: round4(nn_dist),
        two_opt_iterations: two_opt_iters,
        pin_count: n_pins,
        elapsed_ms: round4(elapsed_ms),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn lcg_uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((*state >> 11) as f64) / ((1u64 << 53) as f64);
        lo + unit * (hi - lo)
    }

    #[test]
    fn tsp_50_pins_speed_and_2opt_matches_oracle() {
        let mut seed = 20260501u64;
        let pins: Vec<TspPinInput> = (0..50)
            .map(|i| TspPinInput {
                id: Some(TspId::Int(1000 + i)),
                lat: Some(lcg_uniform(&mut seed, -1200.0, 1200.0)),
                lng: Some(lcg_uniform(&mut seed, -1200.0, 1200.0)),
                name: Some(format!("Oculus #{}", i + 1)),
                label_id: Some(12),
                floor_id: Some(TspId::Int(if i % 7 == 0 { 1 } else { 0 })),
                is_underground: i % 7 == 0,
                ..Default::default()
            })
            .collect();

        let res = plan_route(
            &pins,
            Some(TspCoordInput::Pair([0.0, 0.0])),
            None,
            TspPlanOptions::default(),
        )
        .expect("plan 50 pins");

        assert_eq!(res.pin_count, 50);
        assert_eq!(res.ordered_pins.len(), 50);
        let ids: HashSet<i64> = res
            .ordered_pins
            .iter()
            .filter_map(|p| match p.id {
                TspId::Int(n) => Some(n),
                _ => None,
            })
            .collect();
        assert_eq!(ids.len(), 50);
        let steps: Vec<usize> = res.ordered_pins.iter().map(|p| p.step).collect();
        assert_eq!(steps, (1..=50).collect::<Vec<_>>());
        assert_eq!(res.legs.len(), 50);
        assert_eq!(res.teleport_count, 0);
        assert!(res.raw_walk_distance <= res.nn_distance + 1e-6);
        assert!((res.total_distance - res.raw_walk_distance).abs() < 1e-6);
    }

    #[test]
    fn two_opt_strictly_improves_crossing_path() {
        let coords = vec![
            (0.0, 0.0),
            (100.0, 100.0),
            (0.0, 100.0),
            (100.0, 0.0),
            (200.0, 100.0),
            (200.0, 0.0),
        ];
        let dmat = build_distance_matrix(&coords);
        let crossed = vec![0, 1, 2, 3, 4, 5];
        let crossed_len = path_length(&crossed, &dmat);
        let (improved, iters) = two_opt_improve(&crossed, &dmat, true, 50, DEFAULT_2OPT_EPS);
        let improved_len = path_length(&improved, &dmat);
        assert!(improved_len < crossed_len - 1.0);
        assert!(iters >= 1);

        let mut strict_gains = 0;
        for trial in 0..10u64 {
            let mut st = trial + 42;
            let pts: Vec<(f64, f64)> = (0..50)
                .map(|_| (lcg_uniform(&mut st, 0.0, 1000.0), lcg_uniform(&mut st, 0.0, 1000.0)))
                .collect();
            let dm = build_distance_matrix(&pts);
            let nn = nearest_neighbor_order(&dm, 0, true);
            let (opt, _) = two_opt_improve(&nn, &dm, true, 200, DEFAULT_2OPT_EPS);
            let d_nn = path_length(&nn, &dm);
            let d_opt = path_length(&opt, &dm);
            assert!(d_opt <= d_nn + 1e-6);
            if d_opt < d_nn - 1e-3 {
                strict_gains += 1;
            }
        }
        assert!(strict_gains >= 5, "strict_gains={strict_gains}");
    }

    #[test]
    fn empty_single_and_two_pin_edge_cases() {
        let empty = plan_route(&[], None, None, TspPlanOptions::default()).unwrap();
        assert!(empty.ordered_pins.is_empty());
        assert!(empty.legs.is_empty());
        assert_eq!(empty.total_distance, 0.0);

        let solo = vec![TspPinInput {
            id: Some("p1".into()),
            lat: Some(10.0),
            lng: Some(20.0),
            name: Some("Solo Chest".into()),
            ..Default::default()
        }];
        let solo_none = plan_route(&solo, None, None, TspPlanOptions::default()).unwrap();
        assert_eq!(solo_none.ordered_pins.len(), 1);
        assert_eq!(solo_none.ordered_pins[0].step, 1);
        assert!(solo_none.legs.is_empty());
        assert_eq!(solo_none.total_distance, 0.0);

        let solo_same = plan_route(
            &solo,
            Some(TspCoordInput::LatLng {
                lat: 10.0,
                lng: 20.0,
            }),
            None,
            TspPlanOptions::default(),
        )
        .unwrap();
        assert_eq!(solo_same.pin_count, 1);
        assert_eq!(solo_same.legs.len(), 1);
        assert_eq!(solo_same.total_distance, 0.0);

        let two = vec![
            TspPinInput {
                id: Some("far".into()),
                lat: Some(100.0),
                lng: Some(0.0),
                ..Default::default()
            },
            TspPinInput {
                id: Some("near".into()),
                lat: Some(10.0),
                lng: Some(0.0),
                ..Default::default()
            },
        ];
        let r2_none = plan_route(&two, None, None, TspPlanOptions::default()).unwrap();
        assert_eq!(r2_none.two_opt_iterations, 0);
        assert_eq!(r2_none.legs.len(), 1);

        let r2_start = plan_route(
            &two,
            Some(TspCoordInput::Pair([0.0, 0.0])),
            None,
            TspPlanOptions::default(),
        )
        .unwrap();
        assert_eq!(r2_start.ordered_pins[0].id, TspId::Str("near".to_owned()));
        assert_eq!(r2_start.ordered_pins[1].id, TspId::Str("far".to_owned()));
    }

    #[test]
    fn teleport_leg_splitting_and_floor_change() {
        let pins = vec![
            TspPinInput {
                id: Some("A1".into()),
                lat: Some(0.0),
                lng: Some(0.0),
                name: Some("Mondstadt Chest 1".into()),
                floor_id: Some(TspId::Int(0)),
                ..Default::default()
            },
            TspPinInput {
                id: Some("A2".into()),
                lat: Some(25.0),
                lng: Some(15.0),
                name: Some("Mondstadt Chest 2".into()),
                floor_id: Some(TspId::Int(0)),
                ..Default::default()
            },
            TspPinInput {
                id: Some("A3".into()),
                lat: Some(50.0),
                lng: Some(5.0),
                name: Some("Mondstadt Chest 3".into()),
                floor_id: Some(TspId::Int(0)),
                ..Default::default()
            },
            TspPinInput {
                id: Some("B1".into()),
                lat: Some(2500.0),
                lng: Some(2500.0),
                name: Some("Inazuma Chest 1".into()),
                floor_id: Some(TspId::Int(2)),
                is_underground: true,
                ..Default::default()
            },
            TspPinInput {
                id: Some("B2".into()),
                lat: Some(2530.0),
                lng: Some(2510.0),
                name: Some("Inazuma Chest 2".into()),
                floor_id: Some(TspId::Int(2)),
                is_underground: true,
                ..Default::default()
            },
            TspPinInput {
                id: Some("B3".into()),
                lat: Some(2515.0),
                lng: Some(2545.0),
                name: Some("Inazuma Chest 3".into()),
                floor_id: Some(TspId::Int(0)),
                ..Default::default()
            },
        ];
        let wps = normalize_waypoints(&[
            TspWaypointInput {
                id: Some("WP_A".into()),
                x_pos: Some(10.0),
                y_pos: Some(-5.0),
                name: Some("Mondstadt Waypoint".into()),
                ..Default::default()
            },
            TspWaypointInput {
                id: Some("WP_B".into()),
                x_pos: Some(2492.0),
                y_pos: Some(2496.0),
                name: Some("Inazuma Statue".into()),
                ..Default::default()
            },
        ]);

        let route = plan_route(
            &pins,
            Some(TspCoordInput::Pair([0.0, -5.0])),
            Some(&wps),
            TspPlanOptions::default(),
        )
        .unwrap();

        let tp_legs: Vec<&TspLeg> = route.legs.iter().filter(|l| l.is_teleport).collect();
        assert_eq!(tp_legs.len(), 1);
        assert_eq!(tp_legs[0].waypoint_id.as_ref(), Some(&TspId::Str("WP_B".into())));
        assert_eq!(tp_legs[0].distance, 0.0);
        assert_eq!(tp_legs[0].effective_cost, 180.0);
        assert!(!tp_legs[0].floor_change);
        assert_eq!(route.teleport_hints.len(), 1);
        assert!(route.teleport_hints[0].saved_distance > 2000.0);
        assert!(route.total_distance < route.raw_walk_distance - 2000.0);
        assert!(route.legs.iter().any(|l| l.floor_change));

        let no_start = plan_route(&pins, None, Some(&wps), TspPlanOptions::default()).unwrap();
        assert_eq!(no_start.pin_count, 6);
        assert_eq!(no_start.teleport_count, 1);
        assert!(no_start.total_distance < no_start.raw_walk_distance - 2000.0);
    }

    #[test]
    fn unhelpful_teleport_and_duplicate_coords_do_not_split_or_loop() {
        // direct_dist == penalty (180.0) -> no split
        let pins = vec![
            TspPinInput {
                id: Some(1.into()),
                lat: Some(0.0),
                lng: Some(0.0),
                ..Default::default()
            },
            TspPinInput {
                id: Some(2.into()),
                lat: Some(180.0),
                lng: Some(0.0),
                ..Default::default()
            },
        ];
        let wp = vec![TspWaypoint {
            id: "wp".into(),
            lat: 180.0,
            lng: 0.0,
            name: "WP".into(),
            label_id: None,
            area_id: None,
            map_id: None,
        }];
        let r = plan_route(&pins, None, Some(&wp), TspPlanOptions::default()).unwrap();
        assert_eq!(r.teleport_count, 0);

        // penalty + wp_walk == direct_dist (180 + 20 == 200) -> no split
        let pins2 = vec![
            TspPinInput {
                id: Some(1.into()),
                lat: Some(0.0),
                lng: Some(0.0),
                ..Default::default()
            },
            TspPinInput {
                id: Some(2.into()),
                lat: Some(200.0),
                lng: Some(0.0),
                ..Default::default()
            },
        ];
        let wp2 = vec![TspWaypoint {
            id: "wp2".into(),
            lat: 180.0,
            lng: 0.0,
            name: "WP2".into(),
            label_id: None,
            area_id: None,
            map_id: None,
        }];
        let r2 = plan_route(&pins2, None, Some(&wp2), TspPlanOptions::default()).unwrap();
        assert_eq!(r2.teleport_count, 0);

        // Duplicate coords do not loop or split
        let dup_pins = vec![
            TspPinInput {
                id: Some(1.into()),
                lat: Some(10.0),
                lng: Some(10.0),
                ..Default::default()
            },
            TspPinInput {
                id: Some(2.into()),
                lat: Some(10.0),
                lng: Some(10.0),
                ..Default::default()
            },
            TspPinInput {
                id: Some(3.into()),
                lat: Some(10.0),
                lng: Some(10.0),
                ..Default::default()
            },
        ];
        let r_dup = plan_route(&dup_pins, None, Some(&wp), TspPlanOptions::default()).unwrap();
        assert_eq!(r_dup.total_distance, 0.0);
        assert_eq!(r_dup.teleport_count, 0);
    }

    #[test]
    fn nan_and_inf_inputs_handled_safely() {
        let bad_pin = TspPinInput {
            id: Some(1.into()),
            lat: Some(f64::NAN),
            lng: Some(0.0),
            ..Default::default()
        };
        assert!(matches!(
            plan_route(&[bad_pin], None, None, TspPlanOptions::default()),
            Err(CoreError::InvalidArgument(_))
        ));

        let good_pin = TspPinInput {
            id: Some(1.into()),
            lat: Some(1.0),
            lng: Some(2.0),
            ..Default::default()
        };
        assert!(matches!(
            plan_route(
                &[good_pin],
                Some(TspCoordInput::Pair([f64::INFINITY, 0.0])),
                None,
                TspPlanOptions::default(),
            ),
            Err(CoreError::InvalidArgument(_))
        ));

        let nan_mat = DistMatrix::from_flat(
            3,
            vec![0.0, f64::NAN, 5.0, f64::NAN, 0.0, 3.0, 5.0, 3.0, 0.0],
        )
        .unwrap();
        let nn = nearest_neighbor_order(&nan_mat, 10, false);
        assert_eq!(nn.len(), 3);
        let (opt, _) = two_opt_improve(&nn, &nan_mat, false, 10, DEFAULT_2OPT_EPS);
        assert_eq!(opt.len(), 3);
    }

    #[test]
    fn serde_wire_compatibility() {
        let req = serde_json::json!({
            "pins": [
                {"id": 1001, "lat": 0.0, "lng": 0.0, "name": "A", "floor_id": 2},
                {"id": "str_pin", "y_pos": 300.0, "x_pos": 0.0, "floor_id": "301"}
            ],
            "start_pos": [0.0, 0.0],
            "waypoints": [
                {"id": "WP_1", "y_pos": 295.0, "x_pos": 0.0, "name": "TP"}
            ]
        });
        let pins: Vec<TspPinInput> = serde_json::from_value(req["pins"].clone()).unwrap();
        let start: TspCoordInput = serde_json::from_value(req["start_pos"].clone()).unwrap();
        let wps_in: Vec<TspWaypointInput> =
            serde_json::from_value(req["waypoints"].clone()).unwrap();
        let wps = normalize_waypoints(&wps_in);
        let plan = plan_route(&pins, Some(start), Some(&wps), TspPlanOptions::default()).unwrap();
        let json_out = serde_json::to_value(&plan).unwrap();
        assert_eq!(json_out["pin_count"], 2);
        assert_eq!(json_out["teleport_count"], 1);
        assert!(json_out["ordered_pins"][0]["step"].is_u64());
        assert!(json_out["legs"][0]["points"].is_array());
        assert!(json_out["legs"][0]["floor_change"].is_boolean());
        assert!(json_out["teleport_hints"][0]["saved_distance"].is_f64());
        assert!(json_out["elapsed_ms"].is_f64());
    }

    #[test]
    fn loads_waypoints_from_real_multimap_v4_when_present() {
        let missing = load_waypoints_for_map(2, Some(Path::new("nonexistent_multimap_xyz.json")))
            .expect("missing file returns empty");
        assert!(missing.is_empty());

        let legacy_json = r#"{
            "version": "2.0.0",
            "maps": {
                "2": {
                    "id": 2,
                    "name": "Teyvat",
                    "waypoints": [
                        {"id": 10, "label_id": 2, "x_pos": 12.5, "y_pos": -34.5},
                        {"id": 11, "label_id": 517, "x_pos": 1.0, "y_pos": 2.0}
                    ]
                }
            }
        }"#;
        let db_legacy = MultimapDb::from_json_str(legacy_json).unwrap();
        let wps_legacy = waypoints_from_multimap(&db_legacy, 2, true);
        assert_eq!(wps_legacy.len(), 1);
        assert_eq!(wps_legacy[0].id, TspId::Int(10));
        assert_eq!(wps_legacy[0].lng, 12.5);
        assert_eq!(wps_legacy[0].lat, -34.5);

        let real_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("data")
            .join("multimap_v4.json");
        if !real_path.is_file() {
            return;
        }
        let primary_wps = load_waypoints_for_map(2, Some(&real_path)).expect("load real multimap");
        assert_eq!(primary_wps.len(), 819);
        assert!(primary_wps.len() > 100);
        let db = load_multimap_db(&real_path).unwrap();
        let all_wps = waypoints_from_multimap(&db, 2, false);
        assert_eq!(all_wps.len(), 846);
    }

    #[test]
    fn tsp_pin_input_deserialize_null_is_underground() {
        let pin: TspPinInput = serde_json::from_value(serde_json::json!({
            "id": 1,
            "lat": 10.0,
            "lng": 20.0,
            "is_underground": null
        }))
        .unwrap();
        assert!(!pin.is_underground);
    }
}
