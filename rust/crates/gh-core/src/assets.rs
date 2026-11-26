//! Dataset manifest verification, staged promotion & rollback (`src/asset_setup.py`).
//! Orchestrates `DatasetSyncRunner` without depending on `gh-net`.

use crate::atomic_io;
use crate::config::HoyolabSyncConfig;
use crate::paths::{self, AppPaths};
use crate::traits::DatasetSyncRunner;
use crate::{CoreError, CoreResult};
use md5::{Digest, Md5};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use zip::ZipArchive;

pub const REQUIRED_ASSETS: [&str; 3] = [
    "multimap_v4.json",
    "terrain_n2_v4.npz",
    "cave_masks_v4.npz",
];
pub const OPTIONAL_ASSETS: [&str; 1] = ["ui_anchors_v4.npz"];
pub const MANIFEST_FILENAME: &str = "manifest.json";
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

static STAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataHealth {
    Ready,
    Missing,
    Corrupt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataStatus {
    pub ready: bool,
    pub data_dir: PathBuf,
    pub missing: Vec<String>,
    pub corrupted: Vec<String>,
    pub mismatched: Vec<String>,
    pub manifest_present: bool,
    pub source: String,
}

impl DataStatus {
    pub fn health(&self) -> DataHealth {
        if self.ready {
            DataHealth::Ready
        } else if !self.corrupted.is_empty() || !self.mismatched.is_empty() {
            DataHealth::Corrupt
        } else {
            DataHealth::Missing
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub md5: String,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetManifest {
    pub schema_version: u32,
    pub generated_at: String,
    pub source: String,
    pub files: BTreeMap<String, ManifestEntry>,
}

/// Convenience adapter allowing closures/functions to act as a `DatasetSyncRunner`.
pub struct ClosureSyncRunner<F>(pub F)
where
    F: Fn(&Path, &HoyolabSyncConfig) -> CoreResult<()> + Send + Sync;

impl<F> DatasetSyncRunner for ClosureSyncRunner<F>
where
    F: Fn(&Path, &HoyolabSyncConfig) -> CoreResult<()> + Send + Sync,
{
    fn check_online_map_versions(
        &self,
        _sync_cfg: &HoyolabSyncConfig,
        _local_versions: &BTreeMap<i32, String>,
    ) -> CoreResult<crate::traits::MapVersionCheckResult> {
        Ok(Default::default())
    }

    fn hydrate_into(&self, staging_dir: &Path, sync_cfg: &HoyolabSyncConfig) -> CoreResult<()> {
        (self.0)(staging_dir, sync_cfg)
    }
}

pub fn get_user_data_cache_dir() -> PathBuf {
    AppPaths::resolve().data_dir
}

pub fn get_default_bundle_data_dir() -> PathBuf {
    let app_paths = paths::resolve_paths();
    if let Some(dev_data) = app_paths.dev_data_dir {
        return dev_data;
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(exe_dir) = exe.parent()
    {
        let installed_bundle = exe_dir.join("data");
        if installed_bundle.is_dir() {
            return installed_bundle;
        }
    }
    app_paths.data_dir
}

pub fn resolve_data_asset_in(filename: &str, bundle_dir: &Path, cache_dir: &Path) -> PathBuf {
    let bundle_candidate = bundle_dir.join(filename);
    if bundle_candidate.is_file() {
        return bundle_candidate;
    }
    let cache_candidate = cache_dir.join(filename);
    if cache_candidate.is_file() {
        return cache_candidate;
    }
    bundle_candidate
}

pub fn resolve_data_asset(filename: &str) -> PathBuf {
    resolve_data_asset_in(
        filename,
        &get_default_bundle_data_dir(),
        &get_user_data_cache_dir(),
    )
}

pub fn resolve_data_dir() -> PathBuf {
    let bundle_dir = get_default_bundle_data_dir();
    if verify_data_dir(&bundle_dir, &REQUIRED_ASSETS, false, "bundle").ready {
        return bundle_dir;
    }
    let cache_dir = get_user_data_cache_dir();
    if verify_data_dir(&cache_dir, &REQUIRED_ASSETS, false, "cache").ready {
        return cache_dir;
    }
    bundle_dir
}

pub fn compute_md5(path: &Path) -> CoreResult<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut hasher = Md5::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(format!("{digest:x}"))
}

pub fn validate_asset_file(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() || meta.len() == 0 {
        return false;
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    match ext.as_str() {
        "json" => {
            let Ok(bytes) = std::fs::read(path) else {
                return false;
            };
            let Ok(val) = serde_json::from_slice::<Value>(&bytes) else {
                return false;
            };
            match val {
                Value::Object(map) => !map.is_empty(),
                Value::Array(arr) => !arr.is_empty(),
                _ => false,
            }
        }
        "npz" => {
            let Ok(file) = File::open(path) else {
                return false;
            };
            let Ok(mut archive) = ZipArchive::new(BufReader::new(file)) else {
                return false;
            };
            if archive.is_empty() {
                return false;
            }
            let mut npy_count = 0usize;
            for idx in 0..archive.len() {
                let Ok(mut entry) = archive.by_index(idx) else {
                    return false;
                };
                if entry.is_dir() {
                    continue;
                }
                let Ok(name) = entry.name() else {
                    return false;
                };
                if !name.ends_with(".npy") {
                    return false;
                }
                let mut magic = [0u8; 6];
                if entry.read_exact(&mut magic).is_err() || &magic != b"\x93NUMPY" {
                    return false;
                }
                npy_count += 1;
            }
            npy_count > 0
        }
        _ => true,
    }
}

pub fn read_manifest(data_dir: &Path) -> Option<AssetManifest> {
    let path = data_dir.join(MANIFEST_FILENAME);
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<AssetManifest>(&raw).ok()
}

pub fn build_manifest(
    data_dir: &Path,
    required_assets: &[&str],
    optional_assets: &[&str],
    source: &str,
) -> CoreResult<AssetManifest> {
    let mut files = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for &name in required_assets.iter().chain(optional_assets.iter()) {
        if !seen.insert(name) {
            continue;
        }
        let path = data_dir.join(name);
        if path.is_file() {
            let size = std::fs::metadata(&path)?.len();
            let md5 = compute_md5(&path)?;
            files.insert(name.to_owned(), ManifestEntry { md5, size });
        }
    }

    Ok(AssetManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        generated_at: utc_now_iso8601(),
        source: source.to_owned(),
        files,
    })
}

pub fn write_manifest(data_dir: &Path, manifest: &AssetManifest) -> CoreResult<PathBuf> {
    std::fs::create_dir_all(data_dir)?;
    let target = data_dir.join(MANIFEST_FILENAME);
    let mut json = serde_json::to_string_pretty(manifest)?;
    json.push('\n');
    atomic_io::write_bytes_atomic(&target, json.as_bytes())?;
    Ok(target)
}

pub fn verify_data_dir(
    data_dir: &Path,
    required_assets: &[&str],
    check_manifest: bool,
    source: &str,
) -> DataStatus {
    let manifest = if check_manifest {
        read_manifest(data_dir)
    } else {
        None
    };
    let manifest_present = manifest.is_some();

    let mut missing = Vec::new();
    let mut corrupted = Vec::new();
    let mut mismatched = Vec::new();

    for &name in required_assets {
        let asset_path = data_dir.join(name);
        if !asset_path.is_file() {
            missing.push(name.to_owned());
            continue;
        }
        if !validate_asset_file(&asset_path) {
            corrupted.push(name.to_owned());
            continue;
        }
        if let Some(m) = &manifest
            && let Some(entry) = m.files.get(name)
        {
            let expected = entry.md5.trim().to_ascii_lowercase();
            if !expected.is_empty() {
                match compute_md5(&asset_path) {
                    Ok(actual) if actual.eq_ignore_ascii_case(&expected) => {}
                    _ => mismatched.push(name.to_owned()),
                }
            }
        }
    }

    let ready = missing.is_empty() && corrupted.is_empty() && mismatched.is_empty();
    DataStatus {
        ready,
        data_dir: data_dir.to_path_buf(),
        missing,
        corrupted,
        mismatched,
        manifest_present,
        source: source.to_owned(),
    }
}

pub fn promote_staged_assets(pairs: &[(PathBuf, PathBuf)]) -> CoreResult<()> {
    struct PromotedRecord {
        dest: PathBuf,
        previous: PathBuf,
        had_previous: bool,
    }

    let mut promoted: Vec<PromotedRecord> = Vec::with_capacity(pairs.len());

    for (staged, dest) in pairs {
        if let Some(parent) = dest.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        let file_name = dest
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| CoreError::InvalidArgument("destination missing filename".to_owned()))?;
        let previous = dest.with_file_name(format!(".{file_name}.previous"));
        if previous.exists() {
            remove_path(&previous)?;
        }

        let had_previous = dest.exists();
        if had_previous {
            atomic_io::replace_with_retry(
                dest,
                &previous,
                atomic_io::DEFAULT_REPLACE_RETRIES,
                atomic_io::DEFAULT_REPLACE_RETRY_DELAY,
            )?;
        }

        if let Err(err) = atomic_io::replace_with_retry(
            staged,
            dest,
            atomic_io::DEFAULT_REPLACE_RETRIES,
            atomic_io::DEFAULT_REPLACE_RETRY_DELAY,
        ) {
            if had_previous && previous.exists() && !dest.exists() {
                let _ = atomic_io::replace_with_retry(
                    &previous,
                    dest,
                    atomic_io::DEFAULT_REPLACE_RETRIES,
                    atomic_io::DEFAULT_REPLACE_RETRY_DELAY,
                );
            }
            for rec in promoted.into_iter().rev() {
                let _ = remove_path(&rec.dest);
                if rec.had_previous && rec.previous.exists() {
                    let _ = atomic_io::replace_with_retry(
                        &rec.previous,
                        &rec.dest,
                        atomic_io::DEFAULT_REPLACE_RETRIES,
                        atomic_io::DEFAULT_REPLACE_RETRY_DELAY,
                    );
                }
            }
            return Err(err);
        }

        promoted.push(PromotedRecord {
            dest: dest.clone(),
            previous,
            had_previous,
        });
    }

    for rec in promoted {
        if rec.previous.exists() {
            let _ = remove_path(&rec.previous);
        }
    }

    Ok(())
}

pub struct EnsureDataOptions<'a> {
    pub cache_dir: Option<&'a Path>,
    pub fallback_bundle_dir: Option<&'a Path>,
    pub required_assets: &'a [&'a str],
    pub auto_sync: bool,
    pub sync_cfg: &'a HoyolabSyncConfig,
}

impl<'a> EnsureDataOptions<'a> {
    pub fn with_defaults(sync_cfg: &'a HoyolabSyncConfig) -> Self {
        Self {
            cache_dir: None,
            fallback_bundle_dir: None,
            required_assets: &REQUIRED_ASSETS,
            auto_sync: true,
            sync_cfg,
        }
    }
}

struct StageDirGuard(PathBuf);

impl StageDirGuard {
    fn create_under(parent: &Path) -> CoreResult<Self> {
        std::fs::create_dir_all(parent)?;
        let pid = std::process::id();
        for _ in 0..256 {
            let seq = STAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let candidate = parent.join(format!(".genshin_helper_stage_{pid}_{seq}"));
            if !candidate.exists() {
                std::fs::create_dir_all(&candidate)?;
                return Ok(Self(candidate));
            }
        }
        Err(CoreError::Io(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "failed to allocate unique staging directory",
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for StageDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn ensure_data_ready(
    opts: &EnsureDataOptions<'_>,
    sync_runner: Option<&dyn DatasetSyncRunner>,
) -> CoreResult<PathBuf> {
    let target_cache = opts
        .cache_dir
        .map(PathBuf::from)
        .unwrap_or_else(get_user_data_cache_dir);

    let cache_status = verify_data_dir(&target_cache, opts.required_assets, true, "cache");
    if cache_status.ready {
        return Ok(target_cache);
    }

    let bundle_dir = opts
        .fallback_bundle_dir
        .map(PathBuf::from)
        .unwrap_or_else(get_default_bundle_data_dir);
    let bundle_status = verify_data_dir(&bundle_dir, opts.required_assets, false, "bundle");
    if bundle_status.ready {
        return Ok(bundle_dir);
    }

    if !opts.auto_sync {
        return Err(CoreError::NotFound(format!(
            "required map assets unavailable in {} (missing={:?}, corrupted={:?}, mismatched={:?})",
            target_cache.display(),
            cache_status.missing,
            cache_status.corrupted,
            cache_status.mismatched
        )));
    }

    let runner = sync_runner.ok_or_else(|| {
        CoreError::InvalidArgument("auto_sync=true requires a DatasetSyncRunner".to_owned())
    })?;

    let cache_parent = target_cache
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(&target_cache);
    let stage = StageDirGuard::create_under(cache_parent)?;

    // Seed valid assets from bundle_dir into staging directory
    for &name in opts
        .required_assets
        .iter()
        .chain(OPTIONAL_ASSETS.iter())
    {
        let src = bundle_dir.join(name);
        if validate_asset_file(&src) {
            let dst = stage.path().join(name);
            let _ = std::fs::copy(&src, &dst);
        }
    }

    runner.hydrate_into(stage.path(), opts.sync_cfg)?;

    let staged_status = verify_data_dir(stage.path(), opts.required_assets, false, "staged");
    if !staged_status.ready {
        return Err(CoreError::InvalidData(format!(
            "staged sync incomplete (missing={:?}, corrupted={:?})",
            staged_status.missing, staged_status.corrupted
        )));
    }

    let manifest = build_manifest(
        stage.path(),
        opts.required_assets,
        &OPTIONAL_ASSETS,
        "sync_hoyolab_data",
    )?;
    let staged_manifest = write_manifest(stage.path(), &manifest)?;

    std::fs::create_dir_all(&target_cache)?;
    let mut pairs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut seen = BTreeSet::new();
    for &name in opts.required_assets {
        if seen.insert(name) {
            pairs.push((stage.path().join(name), target_cache.join(name)));
        }
    }
    for &name in &OPTIONAL_ASSETS {
        if seen.insert(name) {
            let candidate = stage.path().join(name);
            if candidate.is_file() {
                pairs.push((candidate, target_cache.join(name)));
            }
        }
    }
    pairs.push((staged_manifest, target_cache.join(MANIFEST_FILENAME)));

    promote_staged_assets(&pairs)?;
    drop(stage);

    let final_status = verify_data_dir(&target_cache, opts.required_assets, true, "cache");
    if !final_status.ready {
        return Err(CoreError::InvalidData(format!(
            "promoted cache failed verification (missing={:?}, corrupted={:?}, mismatched={:?})",
            final_status.missing, final_status.corrupted, final_status.mismatched
        )));
    }

    Ok(target_cache)
}

fn remove_path(path: &Path) -> CoreResult<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)?;
    } else if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn utc_now_iso8601() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3600;
    let min = (rem % 3600) / 60;
    let sec = rem % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + i64::from(m <= 2);
    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npz::{NpyArray, NpzArchive};
    use std::io::Write;
    use std::sync::atomic::AtomicUsize;

    static TEST_SEQ: AtomicUsize = AtomicUsize::new(0);

    fn make_temp_dir(label: &str) -> PathBuf {
        let seq = TEST_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "gh_core_assets_{label}_{}_{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_sample_npz(path: &Path, key: &str, vals: Vec<i32>) {
        let mut archive = NpzArchive::new();
        let len = vals.len();
        archive.insert(key, NpyArray::from_i32(vec![len], vals).expect("arr"));
        archive.write_file_atomic(path).expect("write npz");
    }

    #[test]
    fn validate_asset_file_checks_json_and_npz() {
        let base = make_temp_dir("validate");

        let empty_json = base.join("empty.json");
        std::fs::write(&empty_json, "").expect("write");
        assert!(!validate_asset_file(&empty_json));

        let empty_obj = base.join("empty_obj.json");
        std::fs::write(&empty_obj, "{}").expect("write");
        assert!(!validate_asset_file(&empty_obj));

        let bad_json = base.join("bad.json");
        std::fs::write(&bad_json, "{bad").expect("write");
        assert!(!validate_asset_file(&bad_json));

        let good_json = base.join("good.json");
        std::fs::write(&good_json, r#"{"version":"4.0.0","maps":{}}"#).expect("write");
        assert!(validate_asset_file(&good_json));

        let bad_npz = base.join("not_zip.npz");
        std::fs::write(&bad_npz, b"plain bytes").expect("write");
        assert!(!validate_asset_file(&bad_npz));

        // Zip with non-.npy entry
        let wrong_ext_npz = base.join("wrong_ext.npz");
        {
            let file = File::create(&wrong_ext_npz).expect("create");
            let mut zw = zip::ZipWriter::new(file);
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("readme.txt", opts).expect("start");
            zw.write_all(b"hello").expect("write");
            zw.finish().expect("finish");
        }
        assert!(!validate_asset_file(&wrong_ext_npz));

        // Zip with .npy entry missing \x93NUMPY magic
        let bad_magic_npz = base.join("bad_magic.npz");
        {
            let file = File::create(&bad_magic_npz).expect("create");
            let mut zw = zip::ZipWriter::new(file);
            let opts = zip::write::SimpleFileOptions::default();
            zw.start_file("arr.npy", opts).expect("start");
            zw.write_all(b"NOT_NUMPY_HEADER").expect("write");
            zw.finish().expect("finish");
        }
        assert!(!validate_asset_file(&bad_magic_npz));

        let valid_npz = base.join("valid.npz");
        write_sample_npz(&valid_npz, "payload", vec![1, 2, 3]);
        assert!(validate_asset_file(&valid_npz));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn promote_staged_assets_success_and_rollback() {
        let base = make_temp_dir("promote");
        let stage = base.join("stage");
        let dest = base.join("dest");
        std::fs::create_dir_all(&stage).expect("stage");
        std::fs::create_dir_all(&dest).expect("dest");

        // (a) Successful promotion
        std::fs::write(dest.join("a.txt"), "old-a").expect("old-a");
        std::fs::write(stage.join("a.txt"), "new-a").expect("new-a");
        std::fs::write(stage.join("b.txt"), "new-b").expect("new-b");

        promote_staged_assets(&[
            (stage.join("a.txt"), dest.join("a.txt")),
            (stage.join("b.txt"), dest.join("b.txt")),
        ])
        .expect("promote success");

        assert_eq!(std::fs::read_to_string(dest.join("a.txt")).unwrap(), "new-a");
        assert_eq!(std::fs::read_to_string(dest.join("b.txt")).unwrap(), "new-b");
        assert!(!dest.join(".a.txt.previous").exists());
        assert!(!dest.join(".b.txt.previous").exists());

        // (b) Failure on 3rd missing staged file triggers full rollback
        std::fs::write(dest.join("first.txt"), "orig-first").expect("orig-first");
        std::fs::write(dest.join("third.txt"), "orig-third").expect("orig-third");
        std::fs::write(stage.join("first.txt"), "bad-first").expect("bad-first");
        std::fs::write(stage.join("second.txt"), "bad-second").expect("bad-second");
        // stage/third.txt intentionally missing

        let err = promote_staged_assets(&[
            (stage.join("first.txt"), dest.join("first.txt")),
            (stage.join("second.txt"), dest.join("second.txt")),
            (stage.join("third.txt"), dest.join("third.txt")),
        ])
        .expect_err("must rollback on 3rd missing staged file");
        assert!(matches!(err, CoreError::Io(_)));

        assert_eq!(
            std::fs::read_to_string(dest.join("first.txt")).unwrap(),
            "orig-first"
        );
        assert!(!dest.join("second.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dest.join("third.txt")).unwrap(),
            "orig-third"
        );
        assert!(!dest.join(".first.txt.previous").exists());
        assert!(!dest.join(".third.txt.previous").exists());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn ensure_data_ready_bundle_hydration_cache_hit_and_tamper() {
        let base = make_temp_dir("ensure");
        let cache_dir = base.join("cache_data");
        let bundle_dir = base.join("bundle_data");
        let sync_cfg = HoyolabSyncConfig::default();

        // (a) Bundle fallback with auto_sync = false
        std::fs::create_dir_all(&bundle_dir).expect("bundle");
        std::fs::write(
            bundle_dir.join("multimap_v4.json"),
            r#"{"version":"v1","maps":{"2":{"name":"Teyvat"}}}"#,
        )
        .expect("mm");
        write_sample_npz(&bundle_dir.join("terrain_n2_v4.npz"), "arr", vec![1, 2]);
        write_sample_npz(&bundle_dir.join("cave_masks_v4.npz"), "arr", vec![3, 4]);

        let opts_bundle = EnsureDataOptions {
            cache_dir: Some(&cache_dir),
            fallback_bundle_dir: Some(&bundle_dir),
            required_assets: &REQUIRED_ASSETS,
            auto_sync: false,
            sync_cfg: &sync_cfg,
        };
        let resolved = ensure_data_ready(&opts_bundle, None).expect("bundle fallback");
        assert_eq!(resolved, bundle_dir);

        // (b) Empty bundle + auto_sync = true hydrates into cache_dir with 4 manifest entries
        let empty_bundle = base.join("empty_bundle");
        std::fs::create_dir_all(&empty_bundle).expect("empty bundle");
        let hydrate_calls = AtomicUsize::new(0);
        let runner = ClosureSyncRunner(|stage_dir, _cfg| {
            hydrate_calls.fetch_add(1, Ordering::SeqCst);
            std::fs::write(
                stage_dir.join("multimap_v4.json"),
                r#"{"version":"v2","maps":{"2":{"name":"Teyvat"}}}"#,
            )?;
            write_sample_npz(&stage_dir.join("terrain_n2_v4.npz"), "arr", vec![10, 20]);
            write_sample_npz(&stage_dir.join("cave_masks_v4.npz"), "arr", vec![30, 40]);
            write_sample_npz(&stage_dir.join("ui_anchors_v4.npz"), "arr", vec![50, 60]);
            Ok(())
        });

        let opts_sync = EnsureDataOptions {
            cache_dir: Some(&cache_dir),
            fallback_bundle_dir: Some(&empty_bundle),
            required_assets: &REQUIRED_ASSETS,
            auto_sync: true,
            sync_cfg: &sync_cfg,
        };
        let hydrated_dir = ensure_data_ready(&opts_sync, Some(&runner)).expect("hydrate");
        assert_eq!(hydrated_dir, cache_dir);
        assert_eq!(hydrate_calls.load(Ordering::SeqCst), 1);

        let manifest = read_manifest(&cache_dir).expect("manifest");
        assert_eq!(manifest.files.len(), 4);
        assert!(manifest.files.contains_key("ui_anchors_v4.npz"));

        // Cache hit on second call does NOT re-invoke runner
        let second = ensure_data_ready(&opts_sync, Some(&runner)).expect("cache hit");
        assert_eq!(second, cache_dir);
        assert_eq!(hydrate_calls.load(Ordering::SeqCst), 1);

        // (d) Tampering with multimap_v4.json is detected via mismatched
        std::fs::write(
            cache_dir.join("multimap_v4.json"),
            r#"{"version":"tampered","maps":{"2":{"name":"Teyvat"}}}"#,
        )
        .expect("tamper");
        let tampered_status = verify_data_dir(&cache_dir, &REQUIRED_ASSETS, true, "cache");
        assert!(!tampered_status.ready);
        assert_eq!(tampered_status.health(), DataHealth::Corrupt);
        assert_eq!(tampered_status.mismatched, vec!["multimap_v4.json"]);

        // (c) Clean install without optional ui_anchors_v4.npz or underground_db.json succeeds with 3 manifest entries
        let cache_no_opt = base.join("cache_no_opt");
        let runner_no_opt = ClosureSyncRunner(|stage_dir, _cfg| {
            std::fs::write(
                stage_dir.join("multimap_v4.json"),
                r#"{"version":"v3","maps":{"2":{"name":"Teyvat"}}}"#,
            )?;
            write_sample_npz(&stage_dir.join("terrain_n2_v4.npz"), "arr", vec![1]);
            write_sample_npz(&stage_dir.join("cave_masks_v4.npz"), "arr", vec![2]);
            Ok(())
        });
        let opts_no_opt = EnsureDataOptions {
            cache_dir: Some(&cache_no_opt),
            fallback_bundle_dir: Some(&empty_bundle),
            required_assets: &REQUIRED_ASSETS,
            auto_sync: true,
            sync_cfg: &sync_cfg,
        };
        ensure_data_ready(&opts_no_opt, Some(&runner_no_opt)).expect("no optional assets ok");
        let m3 = read_manifest(&cache_no_opt).expect("m3");
        assert_eq!(m3.files.len(), 3);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_data_asset_in_prefers_bundle_then_cache() {
        let base = make_temp_dir("resolve_in");
        let bundle = base.join("bundle");
        let cache = base.join("cache");
        std::fs::create_dir_all(&bundle).expect("bundle");
        std::fs::create_dir_all(&cache).expect("cache");

        std::fs::write(bundle.join("multimap_v4.json"), "{}").expect("b mm");
        std::fs::write(cache.join("multimap_v4.json"), "{}").expect("c mm");
        std::fs::write(cache.join("terrain_n2_v4.npz"), "z").expect("c t");

        assert_eq!(
            resolve_data_asset_in("multimap_v4.json", &bundle, &cache),
            bundle.join("multimap_v4.json")
        );
        assert_eq!(
            resolve_data_asset_in("terrain_n2_v4.npz", &bundle, &cache),
            cache.join("terrain_n2_v4.npz")
        );
        assert_eq!(
            resolve_data_asset_in("missing.npz", &bundle, &cache),
            bundle.join("missing.npz")
        );

        let _ = std::fs::remove_dir_all(&base);
    }
}
