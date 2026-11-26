# Migration Map (`rust/MAP.md`)

> Single source of truth for wave progress, unit ownership, and exit gates across `/teamwork` sessions.
> Full architecture, HTTP/WS/Native-Messaging contracts, invariants, and crate pins live in [docs/rust-migration-plan.md](../docs/rust-migration-plan.md).

## Global Invariants (Quick Reminder for Every Worker)
- **Untouched**: `extension/**` (100% immutable), `scratch/**` (preserve all offline fixtures). Legacy Python runtime is archived at tag `v5.4.0-python-final` (`940543f`) and retired post-cutover (`Test-Path src == False`).
- **External build cache (`rust/target/` clean)**: Repo root gitignored `/.cargo/config.toml` sets an external `target-dir`. Parallel worktree workers MUST set a per-worker `$env:CARGO_TARGET_DIR` (e.g. `C:\CargoTarget\Genshin-Helper-<worker>`) to avoid `.cargo-lock` contention.
- **Cargo manifest ownership**: Only the Coordinator edits root `rust/Cargo.toml` `[workspace.dependencies]` and this `rust/MAP.md`.
- **Game safety**: Read-only only; ZERO `SendInput` against the game window; GDI `SRCCOPY` only (never `CAPTUREBLT`); `WDA_EXCLUDEFROMCAPTURE` refreshed every 2s & toggled via `0x8015`; `WORLD_ANCHOR_Y_OFFSET = -8.5`.
- **Gate before reporting done**:
  - `cargo nextest run --manifest-path rust/Cargo.toml --workspace`
  - `cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings`
  - `.\.venv\Scripts\python.exe scripts/audit_release.py --dir rust --strict-ip`

---

## Wave 1 — Foundation & Workspace Scaffolding (`done`)

| Unit | Target | Status | Gate |
|---|---|---|---|
| Architecture & contract spec | `docs/rust-migration-plan.md` | `done` | `audit_release.py --dir docs --strict-ip` PASS |
| Workspace + release hygiene + nextest profiles | `rust/Cargo.toml`, `rust/.cargo/config.toml`, `rust/.config/nextest.toml`, `/.cargo/config.toml` | `done` | `cargo nextest` + `cargo clippy -- -D warnings` PASS; 0 bytes in `rust/target` |
| Shared constants, `AppConfig` + `HoyolabSyncConfig`, `FrameBuf`, `protocol` DTOs, acyclic `traits` | `gh-core::{lib,config,paths,frame,ui_layout,nav_controller,protocol,traits}` | `done` | 7/7 `gh-core` tests PASS |
| Crate skeletons | `gh-win32`, `gh-net`, `gh-cv`, `gh-hud`, `gh-host`, `gh-app` | `done` | Acyclic DAG verified via `cargo metadata` |

---

## Wave 2A — `gh-core` Full Implementation (`done`)

| Python Source | Rust Target | Status | Owner | Oracle / Gate |
|---|---|---|---|---|
| `src/paths.py` | `gh-core::paths` | `done` | `M1` | `verify_v45_release.py` parity; `gh-core` unit tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |
| `src/image_io.py` (atomic JSON/bytes) | `gh-core::atomic_io` | `done` | `M1` | `verify_perf_and_settings_v5.py` parity; `gh-core` retry/atomic tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |
| NumPy `.npy`/`.npz` I/O (`u8/i32/i64/f32`) | `gh-core::npz` | `done` | `M1` | Cross-read `data/*.npz` / `scratch/`; `gh-core` `.npy`/`.npz` (`u8/i32/i64/f32`) cross-read tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |
| `src/i18n.py` + `assets/locales/*.json` | `gh-core::i18n` | `done` | `M1` | `verify_i18n_and_ext_v5.py` parity; `gh-core` VI/EN 99-key tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |
| `src/game_settings.py` (CanvasScaler + HDR) | `gh-core::ui_layout` | `done` | `M1` | `verify_perf_and_settings_v5.py`, `verify_minimap_tracker_v5.py` parity; `gh-core` CanvasScaler + HDR tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |
| `src/hd_assets.py` (MultimapDb model) | `gh-core::multimap` | `done` | `M1` | `verify_v53_production.py` parity; `gh-core` `MultimapDb` tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |
| `src/asset_setup.py` | `gh-core::assets` | `done` | `M1` | `verify_public_readiness_v5.py` parity; `gh-core` MD5/staging/rollback tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |
| `src/nav_controller.py` | `gh-core::nav_controller` | `done` | `M1` | `verify_nav_hud_v5.py`, `verify_nav_ux_v54.py` parity; `gh-core` nav/hold/undo/floor tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |
| `src/tsp_route.py` | `gh-core::tsp_route` | `done` | `M1` | `verify_tsp_and_lan_v5.py` parity; `gh-core` NN + 2-opt (`multimap_v4.json`) tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |
| `src/map_bridge.py` + `category_aliases.json` | `gh-core::map_bridge` | `done` | `M1` | `verify_map_bridge_v5.py` parity; `gh-core` map_bridge tests (`73/73 nextest, clippy -D warnings, audit_release PASS`) |

---

## Wave 2B — Parallel Platform, Net, Host & HUD (`done`)

### Track B: `gh-win32`
| Python Source | Rust Target | Status | Owner | Oracle / Gate |
|---|---|---|---|---|
| `src/capture.py` | `gh-win32::capture` | `done` | `M2` | `verify_capture_no_flicker.py`, `verify_hotkey_conflicts_and_capture.py`; `49/49 gh-win32 nextest, clippy -D warnings, audit_release PASS` |
| `src/hotkey.py` | `gh-win32::hotkey` | `done` | `M2` | `verify_single_monitor.py`, `verify_hotkey_conflicts_and_capture.py`; `49/49 gh-win32 nextest, clippy -D warnings, audit_release PASS` |
| `src/hotkey_conflicts.py` | `gh-win32::hotkey_conflicts` | `done` | `M2` | `verify_hotkey_conflicts_and_capture.py` (temp path + `write_registry=false`); `49/49 gh-win32 nextest, clippy -D warnings, audit_release PASS` |
| `src/game_settings.py` (Win32) | `gh-win32::game_settings` | `done` | `M2` | `verify_perf_and_settings_v5.py`, `verify_nav_hud_v5.py`; `49/49 gh-win32 nextest, clippy -D warnings, audit_release PASS` |
| `src/win_ocr.py` | `gh-win32::win_ocr` | `done` | `M2` | `scratch/ocr_query_*.png` (`OcrProvider`); `49/49 gh-win32 nextest, clippy -D warnings, audit_release PASS` |
| `src/tray.py` | `gh-win32::tray` | `done` | `M2` | `verify_v45_release.py`; `49/49 gh-win32 nextest, clippy -D warnings, audit_release PASS` |
| `src/single_monitor.py` (audio/switch) | `gh-win32::{audio,switcher}` | `done` | `M2` | `verify_single_monitor.py`; `49/49 gh-win32 nextest, clippy -D warnings, audit_release PASS` |
| `src/i18n.py` (Win32 locale) + mutex/timer | `gh-win32::{locale,runtime}` | `done` | `M2` | `gh-win32` unit tests; `49/49 gh-win32 nextest, clippy -D warnings, audit_release PASS` |

### Track C: `gh-net` + `gh-host`
| Python Source | Rust Target | Status | Owner | Oracle / Gate |
|---|---|---|---|---|
| `src/server.py` | `gh-net::sync_server` | `done` | `M3 / M_E2E` | 27 routes, `/poll?since=<seq>`, `/shutdown` 403, 32 MiB body, takeover; `28/28 gh-net + 9/9 gh-host nextest, 79/79 blackbox, clippy -D warnings, audit_release PASS` |
| `src/lan_companion.py` | `gh-net::lan_companion` | `done` | `M3 / M_E2E` | `verify_tsp_and_lan_v5.py` (QR SVG, PIN 429, WS 15 Hz); `28/28 gh-net + 9/9 gh-host nextest, 79/79 blackbox, clippy -D warnings, audit_release PASS` |
| `src/hoyolab_sync.py` | `gh-net::hoyolab_sync` | `done` | `M3` | `verify_public_readiness_v5.py` + `check_online_map_versions`; `28/28 gh-net + 9/9 gh-host nextest, 79/79 blackbox, clippy -D warnings, audit_release PASS` |
| `src/hd_assets.py` (async providers) | `gh-net::hd_assets` | `done` | `M3` | `verify_v53_production.py`; `28/28 gh-net + 9/9 gh-host nextest, 79/79 blackbox, clippy -D warnings, audit_release PASS` |
| `src/pin_guide.py` | `gh-net::pin_guide` | `done` | `M3` | `verify_nav_ux_v54.py`, `verify_hotkey_conflicts_and_capture.py`; `28/28 gh-net + 9/9 gh-host nextest, 79/79 blackbox, clippy -D warnings, audit_release PASS` |
| `src/native_host.py` | `gh-host` (`GenshinHelperHost.exe`) | `done` | `M3 / M_E2E` | Black-box 4-byte LE stdio test + `verify_v45_release.py`; `28/28 gh-net + 9/9 gh-host nextest, 79/79 blackbox, clippy -D warnings, audit_release PASS` |

### Track D: `gh-hud`
| Python Source | Rust Target | Status | Owner | Oracle / Gate |
|---|---|---|---|---|
| `src/nav_hud.py` (layered host + fonts) | `gh-hud::{layered,fonts}` | `done` | `M4` | `WM_APP_CAPTURE_AFFINITY` (`0x8015`), 2s WDA refresh; `35/35 gh-hud nextest (<0.4ms HUD), clippy -D warnings, audit_release PASS` |
| `src/nav_hud.py` (renderer + Guide Card) | `gh-hud::nav_hud` | `done` | `M4` | `verify_nav_hud_v5.py`, `verify_nav_ux_v54.py` (headless pixel assertions); `35/35 gh-hud nextest (<0.4ms HUD), clippy -D warnings, audit_release PASS` |
| `src/single_monitor.py` (toast) | `gh-hud::toast` | `done` | `M4` | `verify_single_monitor.py`; `35/35 gh-hud nextest (<0.4ms HUD), clippy -D warnings, audit_release PASS` |
| `src/diagnostics.py` (image annotation) | `gh-hud::diag_annotate` | `done` | `M4` | Vietnamese UTF-8 PNG render test; `35/35 gh-hud nextest (<0.4ms HUD), clippy -D warnings, audit_release PASS` |

---

## Wave 3 — Pure-Rust Computer Vision (`gh-cv`, `done`)

### Gate 0 (Must Pass Before Track E1/E2)
| Scope | Rust Target | Status | Owner | Gate |
|---|---|---|---|---|
| Pure-Rust CV primitives (Sobel/Scharr, FFT & Padmanabhan masked 6-FFT NCC, phaseCorrelate, warpPolar/warpAffine, contours/Canny/CLAHE, DoG RootSIFT, 4-tree KD-Tree, ORB, RANSAC) | `gh-cv::ops` | `done` | `M5` | `--profile cv-heavy --cargo-profile cv-bench` benchmark on `scratch/minimap_rec/` & `scratch/yt_bench/` vs Python oracle (`28/28 unit + 34/34 cv-bench, clippy -D warnings, audit_release PASS`) |

### Track E1 & E2 (2 Parallel Workers after Gate 0)
| Python Source | Rust Target | Status | Owner | Oracle / Gate |
|---|---|---|---|---|
| `src/image_io.py` (PNG/WebP) | `gh-cv::image_io` | `done` | `M6` | Unicode path PNG/WebP tests; `63/63 fast + 77/77 cv-bench nextest, clippy -D warnings, audit_release PASS` |
| `src/zoom_slider.py` | `gh-cv::zoom_slider` | `done` | `M6` | `scratch/0zoom.png`, `scratch/100zoom.png`; `63/63 fast + 77/77 cv-bench nextest, clippy -D warnings, audit_release PASS` |
| `src/map_state.py` | `gh-cv::map_state` | `done` | `M6` | `verify_map_state_v5.py` (`0.88`, `0.975`/`0.65`, `0.82`); `63/63 fast + 77/77 cv-bench nextest, clippy -D warnings, audit_release PASS` |
| `src/heading360.py` + `src/tracker.py` | `gh-cv::heading360` | `done` | `M6` | `verify_heading360_v5.py` (<2 deg, <3 ms); `63/63 fast + 77/77 cv-bench nextest, clippy -D warnings, audit_release PASS` |
| `src/locator.py` | `gh-cv::locator` | `done` | `M6` | `verify_v4_suite.py`, `verify_alignment_v5.py` (`-8.5` Y offset); `63/63 fast + 77/77 cv-bench nextest, clippy -D warnings, audit_release PASS` |
| `src/tile_cache.py` | `gh-cv::tile_cache` | `done` | `M6` | `verify_minimap_tracker_v5.py`, `verify_v53_production.py`; `63/63 fast + 77/77 cv-bench nextest, clippy -D warnings, audit_release PASS` |
| `src/global_relocator.py` | `gh-cv::global_relocator` | `done` | `M6` | `verify_v53_production.py` (`INDEX_VERSION = 3`); `63/63 fast + 77/77 cv-bench nextest, clippy -D warnings, audit_release PASS` |
| `src/minimap_tracker.py` | `gh-cv::minimap_tracker` | `done` | `M6` | `verify_minimap_tracker_v5.py`, `verify_v53_production.py`, `verify_nav_ux_v54.py`; `63/63 fast + 77/77 cv-bench nextest, clippy -D warnings, audit_release PASS` |

---

## Wave 4 — App Wiring, Packaging, Cutover & Docs Sync (Batch 3)

| Scope | Target | Status | Owner | Gate |
|---|---|---|---|---|
| `main.py` + `src/diagnostics.py` + `src/minimap_recorder.py` | `gh-app` (`GenshinHelper.exe`) | `done` | `M7a / M7b` | `257/257 workspace nextest, 0 clippy warnings, 79/79 test_blackbox_contracts.py --backend rust (w/ GH_TEST_HOST_COLD_LAUNCH=1), verify_i18n_and_ext_v5.py PASS` |
| Release scripts & Inno Setup | `rust/.cargo/config.toml`, `installer/{build_release.ps1,GenshinHelper.iss}`, `scripts/{audit_release.py,collect_licenses.py,register_dev_native_host.ps1}` | `done` | `M7b` | `audit_release.py --strict-ip --artifact dist/GenshinHelper & --dir rust PASS (0 external DLLs) + GenshinHelper-Setup-v5.4.0.exe + Test-Path rust/target == False` |
| Cutover & Python retirement (Plan 7.1) | Tag `v5.4.0-python-final` (`940543f`), relocate 4 static assets to `assets/`, move dev datasets to `data/`, retire Python runtime & 16 `verify_*.py` scripts | `done` | `M8` | `git rev-parse v5.4.0-python-final` -> `940543f`; `src/`, `main.py`, `requirements.txt`, `GenshinHelper.spec` removed; `extension/` & `scratch/` untouched |
| Documentation & Index sync (Plan 7.2) | `README*.md`, `PRODUCT.md`, `SECURITY.md`, `CONTRIBUTING.md`, `.github/*`, `THIRD_PARTY_NOTICES.md`, `AGENTS.md`, `CLAUDE.md`, `.github/workflows/*`, `npx gitnexus analyze` | `done` | `M8` | Post-cutover release rebuild, `clippy`, `nextest`, `cv-bench`, 79/79 `test_blackbox_contracts.py --backend rust`, `verify_i18n_and_ext_v5.py`, `audit_release.py --strict-ip`, `gitnexus status` up-to-date |
