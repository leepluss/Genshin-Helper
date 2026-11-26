//! Measurement gate for the v5.5 M-Map route-guidance redesign: arrow-free registration of the
//! displayed M-Map view (`MapLocator::locate_view`). Reports accuracy against `locate_player`
//! ground truth on real `scratch/` fixtures; no UI may consume `locate_view` before the gate:
//!   registered >= 95 % · max error <= 12 px @1440p · confident-but-wrong <= 0.5 % · <= 100 ms.
//!
//! Run (prints the report; JSON copy under CARGO_TARGET_TMPDIR):
//! `cargo nextest run --manifest-path rust/Cargo.toml -p gh-cv --profile cv-heavy --cargo-profile cv-bench --run-ignored all -E "test(e3_)" --no-capture`

use gh_core::CoreResult;
use gh_core::frame::FrameView;
use gh_core::traits::OcrProvider;
use gh_cv::image_io::read_image_bgr;
use gh_cv::locator::{LocatorPaths, MapLocator, ViewHint, ViewOptions, ViewRegistration};
use ndarray::Array3;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const OK_PX_1440: f64 = 12.0;
const COARSE_PX_1440: f64 = 40.0;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("..")
}

fn collect_fixtures(root: &Path) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root.join("scratch").join("diagnostics")) {
        for ent in entries.flatten() {
            let name = ent.file_name().to_string_lossy().into_owned();
            let game = ent.path().join("01_game.png");
            if name.starts_with("diag_") && game.is_file() {
                out.push((name, game));
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(root.join("scratch").join("Test-images")) {
        for ent in entries.flatten() {
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("png")) {
                out.push((ent.file_name().to_string_lossy().into_owned(), p));
            }
        }
    }
    out.sort();
    out
}

struct FixtureOcrProvider {
    current: Mutex<String>,
}

impl OcrProvider for FixtureOcrProvider {
    fn recognize_text(&self, _frame: FrameView<'_>) -> CoreResult<String> {
        Ok(self.current.lock().map(|g| g.clone()).unwrap_or_default())
    }
}

/// Region-label OCR text per fixture (copied from `e1_oracle_parity.rs`), used only for the
/// `locate_player` ground truth and the optional `use_region_hint` variant.
fn ocr_fixture_text(key: &str) -> &'static str {
    match key {
        "diag_20260920_214102" => "Nüi Thånh Vién",
        "diag_20260920_214145" => "Nüi Thånh Vién cb",
        "diag_20260920_214359" => "I Dei Sånh Fontame",
        "diag_20260920_214528" => "Khu Beryl",
        "diag_20260920_214615" => "Khu Morte",
        "diag_20260920_214801" => "Lüng Tequemecan 0101",
        "diag_20260920_214910" => "Dåo Lempo",
        "diag_20260920_220106" => "Dei snh Fontaine",
        "diag_20260920_220225" => "Bech Du'0'ng Tuyét",
        "diag_20260920_220317" => "Lüng Löng Vü Lü'a",
        "diag_20260920_220416" => "Cao Nguyen Träng",
        "diag_20260920_220447" => "Cao Nguyen Träng",
        "diag_20260920_220624" => "Dåo Narukami",
        "diag_20260920_220745" => "CÖc - Thu'Q'ng C6c",
        "diag_20260920_221245" => "Lüng Tequemecan",
        "diag_20260920_222052" => "Lüng Tequemecan",
        "diag_20261120_204445" => "Ong Chåo Ven HÖa",
        "diag_20261120_204655" => "I Dai ånh Fontaine",
        "diag_20261120_205011" => "Novokitezhgrad",
        "diag_20261120_213109" => "I Da ånh Fontaine",
        "diag_20261120_231651" => "Novokitezhgrad",
        "diag_20261120_232016" => "Dinh Bäng Giå",
        "diag_20261120_232232" => "Lülhköng Vü Lü'a",
        "diag_20261120_232623" => "Riyng Lokapala",
        "diag_20261121_011215" => "Riyng Lokapala",
        "diag_20261121_011322" => "I Dai Sånh Fontaine",
        "diag_20261121_011520" => "Dich Chuyén",
        "diag_20261121_204617" => "Dinh Såi Cånh",
        "diag_20261123_003509" => "Tién ;öng Chåo Yen HÖa",
        "Screenshot 2026-09-19 232904.png" => "Dåo Lempe",
        "Screenshot 2026-09-19 234037.png" => "Ong Chåo Ven HÖa",
        "Screenshot 2026-09-19 234051.png" => "Löng Chåo Ven HÖa",
        "Screenshot 2026-09-20 004819.png" => "Chåo Van Höa",
        "Screenshot 2026-09-20 024639.png" => "Ngoqi Ö Vissudha",
        "Screenshot 2026-09-20 102526.png" => "Mac Hadramaveth",
        "Screenshot 2026-09-20 102641.png" => "Dåo Watatsumi",
        "Screenshot 2026-09-20 102719.png" => "Cao Nguyen Träng",
        "Screenshot 2026-09-20 102751.png" => "Cao Nguyén Träng",
        "Screenshot 2026-09-20 102847.png" => "Bién KY Nguyen Cü",
        "Screenshot 2026-09-20 102938.png" => "Bech Du'0'ng Tuyét",
        "Screenshot 2026-09-20 103023.png" => "Bech Du'0'ng Tuyét",
        "Screenshot 2026-09-20 103056.png" => "Dåo Lempo",
        "Screenshot 2026-09-20 103143.png" => "Dåo Hiisi",
        "Screenshot 2026-09-20 103226.png" => "Nüi Thånh Vién cb",
        "Screenshot 2026-09-20 103325.png" => "au - Khu MÖ Ngam",
        "Screenshot 2026-09-20 103356.png" => "au - Khu MÖ Ngam",
        "Screenshot 2026-09-20 103817.png" => "I Dai Sånh Fon me",
        "Screenshot 2026-09-20 103843.png" => "I Dai Sånh Fontaine",
        "Screenshot 2026-09-20 103958.png" => "I Dai Sånh Fontaine",
        "Screenshot 2026-09-20 104045.png" => "Khu Morte",
        "Screenshot 2026-09-20 104129.png" => "dya Söng Quynh co'",
        "Screenshot 2026-09-20 104218.png" => "Döi Phong Tiéu",
        _ => "",
    }
}

#[derive(Clone, Copy)]
struct Truth {
    map_id: i32,
    scale: f64,
    tx: f64,
    ty: f64,
    arrow: (f64, f64),
}

/// Max reprojection error (px, normalised to 1440p) over a 3x3 grid in the central 80 % of
/// the frame: where the estimate puts each screen point vs where ground truth puts it.
fn max_err_px_1440(est: &ViewRegistration, gt: &Truth, w: usize, h: usize) -> f64 {
    let mut worst = 0.0f64;
    for fy in [0.1, 0.5, 0.9] {
        for fx in [0.1, 0.5, 0.9] {
            let (x, y) = (fx * w as f64, fy * h as f64);
            let (wx, wy) = est.screen_to_world(x, y);
            let gx = (wx - gt.tx) / gt.scale;
            let gy = (wy - gt.ty) / gt.scale;
            worst = worst.max((gx - x).hypot(gy - y));
        }
    }
    worst * 1440.0 / h as f64
}

fn mean_color_ring(img: &Array3<u8>, cx: f64, cy: f64, r0: f64, r1: f64) -> [u8; 3] {
    let (h, w, _) = img.dim();
    let mut acc = [0u64; 3];
    let mut n = 0u64;
    let (x0, x1) = ((cx - r1).max(0.0) as usize, ((cx + r1) as usize).min(w.saturating_sub(1)));
    let (y0, y1) = ((cy - r1).max(0.0) as usize, ((cy + r1) as usize).min(h.saturating_sub(1)));
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = (x as f64 - cx).hypot(y as f64 - cy);
            if d >= r0 && d <= r1 {
                for (ch, a) in acc.iter_mut().enumerate() {
                    *a += u64::from(img[[y, x, ch]]);
                }
                n += 1;
            }
        }
    }
    if n == 0 {
        return [0, 0, 0];
    }
    [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8]
}

/// Removes the player arrow (fills a disk with the surrounding ring colour).
fn mask_arrow(img: &Array3<u8>, cx: f64, cy: f64) -> Array3<u8> {
    let (h, w, _) = img.dim();
    let r = 34.0 * h as f64 / 1440.0;
    let col = mean_color_ring(img, cx, cy, r, r * 1.5);
    let mut out = img.clone();
    for y in 0..h {
        for x in 0..w {
            if (x as f64 - cx).hypot(y as f64 - cy) <= r {
                for ch in 0..3 {
                    out[[y, x, ch]] = col[ch];
                }
            }
        }
    }
    out
}

/// Simulates "dragged away from the player": paints the half of the map that contains the
/// player with a flat ocean-like colour (keeps the left zoom-slider strip and top UI band).
fn mask_player_half(img: &Array3<u8>, arrow_x: f64) -> Array3<u8> {
    let (h, w, _) = img.dim();
    let strip = (0.07 * w as f64) as usize;
    let top = (0.09 * h as f64) as usize;
    let (x0, x1) = if arrow_x < w as f64 / 2.0 { (strip, w / 2) } else { (w / 2, w) };
    let mut out = img.clone();
    for y in top..h {
        for x in x0..x1 {
            out[[y, x, 0]] = 70;
            out[[y, x, 1]] = 48;
            out[[y, x, 2]] = 24;
        }
    }
    out
}

#[derive(Default)]
struct Tally {
    total: usize,
    ok: usize,
    coarse: usize,
    wrong: usize,
    miss: usize,
    ms: Vec<f64>,
}

impl Tally {
    fn add(&mut self, outcome: &str, ms: f64) {
        self.total += 1;
        self.ms.push(ms);
        match outcome {
            "ok" => self.ok += 1,
            "coarse" => self.coarse += 1,
            "wrong" => self.wrong += 1,
            _ => self.miss += 1,
        }
    }

    fn summary(&self) -> Value {
        let mut ms = self.ms.clone();
        ms.sort_by(f64::total_cmp);
        let pct = |n: usize| if self.total == 0 { 0.0 } else { 100.0 * n as f64 / self.total as f64 };
        let p = |q: f64| ms.get(((ms.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(0.0);
        json!({
            "total": self.total,
            "ok_pct": pct(self.ok),
            "ok_or_coarse_pct": pct(self.ok + self.coarse),
            "wrong_pct": pct(self.wrong),
            "miss_pct": pct(self.miss),
            "counts": {"ok": self.ok, "coarse": self.coarse, "wrong": self.wrong, "miss": self.miss},
            "ms_p50": p(0.5),
            "ms_p95": p(0.95),
            "ms_max": ms.last().copied().unwrap_or(0.0),
        })
    }
}

fn classify(est: Option<&ViewRegistration>, gt: &Truth, w: usize, h: usize) -> (&'static str, Option<f64>) {
    match est {
        None => ("miss", None),
        Some(e) if e.map_id != gt.map_id => ("wrong", None),
        Some(e) => {
            let err = max_err_px_1440(e, gt, w, h);
            if err <= OK_PX_1440 {
                ("ok", Some(err))
            } else if err <= COARSE_PX_1440 {
                ("coarse", Some(err))
            } else {
                ("wrong", Some(err))
            }
        }
    }
}

fn reg_json(r: Option<&ViewRegistration>) -> Value {
    r.map_or(Value::Null, |r| {
        json!({
            "map_id": r.map_id, "scale": r.scale, "inliers": r.inliers_count, "rmse": r.rmse,
            "terrain_score": r.terrain_score, "terrain_only": r.terrain_only,
            "candidates": r.candidates_count, "zoom_t": r.zoom_ratio_t,
        })
    })
}

#[test]
#[ignore]
fn e3_view_registration_measurement_report() {
    let root = repo_root();
    let ocr = Arc::new(FixtureOcrProvider { current: Mutex::new(String::new()) });
    let loc = MapLocator::with_paths(
        LocatorPaths {
            db_path: Some(root.join("data").join("multimap_v4.json")),
            terrain_npz_path: Some(root.join("data").join("terrain_n2_v4.npz")),
            cave_masks_npz_path: Some(root.join("data").join("cave_masks_v4.npz")),
            ui_anchors_npz_path: Some(root.join("data").join("ui_anchors_v4.npz")),
            legacy_underground_db_path: Some(root.join("data").join("underground_db.json")),
        },
        Some(ocr.clone()),
    );
    let fixtures = collect_fixtures(&root);
    assert!(fixtures.len() >= 50, "expected scratch/ diagnostics + Test-images fixtures");

    // Frames where the M-Map is NOT open: any registration here is a false positive.
    let not_map: HashSet<&str> = [
        "diag_20261121_205434",
        "diag_20261122_032842",
        "diag_20261122_115106",
        "diag_20261122_121803",
        "diag_20261122_161651",
    ]
    .into_iter()
    .collect();

    let variants = [
        "full",
        "arrow_masked",
        "region_hint",
        "map_only_hint",
        "player_half_masked",
        "half_masked_map_hint",
        // Live-tracking settings (no terrain-only search), without / with translation votes.
        "map_only_hint_live",
        "map_only_hint_votes",
        "half_masked_map_hint_live",
        "half_masked_map_hint_votes",
    ];
    let mut tallies: Vec<Tally> = variants.iter().map(|_| Tally::default()).collect();
    // Same runs after the `is_confident()` gate (non-confident registrations become "miss").
    let mut gated: Vec<Tally> = variants.iter().map(|_| Tally::default()).collect();
    let mut rows = Vec::new();
    let mut negatives = Vec::new();
    let mut unlabeled = Vec::new();

    for (name, path) in &fixtures {
        let img = read_image_bgr(path).expect("read fixture");
        let (h, w, _) = img.dim();
        ocr.set_text(ocr_fixture_text(name));

        if not_map.contains(name.as_str()) {
            let t0 = Instant::now();
            let r = loc.locate_view(img.view(), None, ViewHint::None);
            let live_votes = ViewOptions { allow_terrain_only: false, translation_votes: true };
            let rv = loc.locate_view_with(img.view(), None, ViewHint::RegionOcrMapOnly, live_votes);
            negatives.push(json!({
                "name": name, "registered": r.is_some(),
                "confident": r.as_ref().is_some_and(ViewRegistration::is_confident),
                "confident_live_votes": rv.as_ref().is_some_and(ViewRegistration::is_confident),
                "ms": t0.elapsed().as_secs_f64() * 1000.0,
                "reg": reg_json(r.as_ref()),
            }));
            continue;
        }

        loc.reset_tracking_state();
        let gt = loc.locate_player(img.view(), false, None).and_then(|res| {
            let (Some(sc), Some(wx), Some(wy), Some(mid)) = (res.scale, res.world_x, res.world_y, res.map_id)
            else {
                return None;
            };
            res.has_world_coords.then_some(Truth {
                map_id: mid,
                scale: sc,
                tx: wx - sc * res.pixel_x,
                ty: wy - sc * res.pixel_y,
                arrow: (res.pixel_x, res.pixel_y),
            })
        });
        let Some(gt) = gt else {
            // No ground truth (e.g. player arrow already off-screen): record what we get.
            let r = loc.locate_view(img.view(), None, ViewHint::None);
            unlabeled.push(json!({"name": name, "reg": reg_json(r.as_ref())}));
            continue;
        };

        let arrow_masked = mask_arrow(&img, gt.arrow.0, gt.arrow.1);
        let half_masked = mask_player_half(&arrow_masked, gt.arrow.0);
        let mut row = json!({"name": name, "w": w, "h": h, "gt_map": gt.map_id, "gt_scale": gt.scale});
        for (vi, variant) in variants.iter().enumerate() {
            let (frame, hint) = match *variant {
                "full" => (&img, ViewHint::None),
                "arrow_masked" => (&arrow_masked, ViewHint::None),
                "region_hint" => (&arrow_masked, ViewHint::RegionOcr),
                "map_only_hint" => (&arrow_masked, ViewHint::RegionOcrMapOnly),
                "player_half_masked" => (&half_masked, ViewHint::None),
                "map_only_hint_live" | "map_only_hint_votes" => (&arrow_masked, ViewHint::RegionOcrMapOnly),
                _ => (&half_masked, ViewHint::RegionOcrMapOnly),
            };
            let opts = if variant.ends_with("_votes") {
                Some(ViewOptions { allow_terrain_only: false, translation_votes: true })
            } else if variant.ends_with("_live") {
                Some(ViewOptions { allow_terrain_only: false, translation_votes: false })
            } else {
                None
            };
            let t0 = Instant::now();
            let r = match opts {
                Some(o) => loc.locate_view_with(frame.view(), None, hint, o),
                None => loc.locate_view(frame.view(), None, hint),
            };
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let (outcome, err) = classify(r.as_ref(), &gt, w, h);
            tallies[vi].add(outcome, ms);
            let confident = r.as_ref().is_some_and(ViewRegistration::is_confident);
            gated[vi].add(if confident { outcome } else { "miss" }, ms);
            row[*variant] = json!({"outcome": outcome, "err_px_1440": err, "ms": ms, "reg": reg_json(r.as_ref())});
        }
        rows.push(row);
    }

    let summary: Value = variants
        .iter()
        .zip(&tallies)
        .map(|(v, t)| ((*v).to_owned(), t.summary()))
        .collect::<serde_json::Map<String, Value>>()
        .into();
    let summary_gated: Value = variants
        .iter()
        .zip(&gated)
        .map(|(v, t)| ((*v).to_owned(), t.summary()))
        .collect::<serde_json::Map<String, Value>>()
        .into();
    let false_pos = negatives.iter().filter(|n| n["registered"] == true).count();
    let false_pos_gated = negatives.iter().filter(|n| n["confident"] == true).count();
    let false_pos_live_votes = negatives.iter().filter(|n| n["confident_live_votes"] == true).count();
    let report = json!({
        "gate": {"ok_px_1440": OK_PX_1440, "coarse_px_1440": COARSE_PX_1440},
        "summary": summary,
        "summary_confidence_gated": summary_gated,
        "negatives_not_map_open": {
            "total": negatives.len(),
            "false_positives": false_pos,
            "false_positives_gated": false_pos_gated,
            "false_positives_live_votes_gated": false_pos_live_votes,
            "items": negatives,
        },
        "unlabeled": unlabeled,
        "rows": rows,
    });

    println!("\n===== e3 view registration report =====");
    for row in report["rows"].as_array().into_iter().flatten() {
        let cell = |v: &str| {
            let c = &row[v];
            format!(
                "{:<6}{:>6}",
                c["outcome"].as_str().unwrap_or("?"),
                c["err_px_1440"].as_f64().map_or("-".to_owned(), |e| format!("{e:.1}"))
            )
        };
        println!(
            "{:<34} full {} | hint {} | map-hint {} | half {} | half+map {}",
            row["name"].as_str().unwrap_or(""),
            cell("full"),
            cell("region_hint"),
            cell("map_only_hint"),
            cell("player_half_masked"),
            cell("half_masked_map_hint")
        );
    }
    let line = |label: &str, key: &str| {
        println!("--- {label} ---");
        for v in variants {
            let t = &report[key][v];
            println!(
                "{v:<27} ok {:>5.1}% | ok+coarse {:>5.1}% | WRONG {:>4.1}% | miss {:>5.1}% | ms p50 {:>6.1} p95 {:>6.1} max {:>6.1}",
                t["ok_pct"].as_f64().unwrap_or(0.0),
                t["ok_or_coarse_pct"].as_f64().unwrap_or(0.0),
                t["wrong_pct"].as_f64().unwrap_or(0.0),
                t["miss_pct"].as_f64().unwrap_or(0.0),
                t["ms_p50"].as_f64().unwrap_or(0.0),
                t["ms_p95"].as_f64().unwrap_or(0.0),
                t["ms_max"].as_f64().unwrap_or(0.0),
            );
        }
    };
    line("summary (raw)", "summary");
    line("summary (is_confident gate)", "summary_confidence_gated");
    println!(
        "--- not-map-open negatives: {false_pos} raw / {false_pos_gated} gated / {false_pos_live_votes} gated live+votes false positive(s) of {} ---",
        report["negatives_not_map_open"]["total"]
    );
    println!("--- unlabeled (no locate_player ground truth) ---");
    for u in &report["unlabeled"].as_array().cloned().unwrap_or_default() {
        println!("{} -> {}", u["name"], u["reg"]);
    }

    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("e3_view_registration_report.json");
    std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).expect("write report");
    println!("JSON report: {}", out.display());
    assert!(!rows.is_empty(), "no fixture produced locate_player ground truth");
}

impl FixtureOcrProvider {
    fn set_text(&self, text: &str) {
        if let Ok(mut g) = self.current.lock() {
            *g = text.to_owned();
        }
    }
}
