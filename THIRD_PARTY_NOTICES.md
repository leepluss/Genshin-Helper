# Third-Party Notices & Attributions

Genshin-Helper is licensed under the [MIT License](LICENSE) for original source code authored in this repository. It incorporates ported MIT-licensed algorithms, relies on open-source runtime dependencies, and interacts locally with third-party map services whose content remains outside this project's license.

---

## 1. Ported Open-Source Modules (MIT License)

The following open-source repositories provided algorithms or modules originally ported into Genshin-Helper under the MIT License (rewritten in pure Rust in v5.4.0 while preserving attribution to the upstream commits and original source paths listed below).

### GengGode/cvAutoTrack (`@336a4a6`, MIT License)
- Repository: <https://github.com/GengGode/cvAutoTrack>
- License: MIT License (Copyright (c) GengGode / cvAutoTrack Contributors)
- Ported modules in Genshin-Helper:
  - `rust/crates/gh-cv/src/heading360.rs` — Polar-coordinate 360-degree camera FOV cone extraction (alpha + RGB fallback) and 3-vertex contour avatar direction extraction (ported from `source/algorithms/algorithms.rotation.h` and `source/algorithms/algorithms.direction.h`).
  - `rust/crates/gh-core/src/ui_layout.rs` & `rust/crates/gh-win32/src/game_settings.rs` — Unity `CanvasScaler` multi-aspect HUD layout normalization (`16:9`, `16:10`, `21:9`, `4:3`, keyboard & gamepad safe-area) and O(1) Paimon anchor fast-path checks (ported from `source/genshin/genshin.handle.cpp` and `source/genshin/check/paimon/genshin.check.paimon.cpp`).
- Scope note: No patented SURF algorithms and no binary game-icon assets from `cvAutoTrack` are included in Genshin-Helper.

### MaxWay12/Genshin-Navigator (`@99e1e13`, MIT License)
- Repository: <https://github.com/MaxWay12/Genshin-Navigator>
- License: MIT License (Copyright (c) MaxWay12 / Genshin-Navigator Contributors)
- Ported modules in Genshin-Helper:
  - `rust/crates/gh-cv/src/image_io.rs` — Unicode-path-safe image I/O helpers.
  - `rust/crates/gh-core/src/assets.rs` — Local asset staging, integrity validation, and atomic asset promotion with full rollback (`src/genshin_navigator/asset_setup.py`).
  - `rust/crates/gh-cv/src/minimap_tracker.rs` — 4-state minimap tracking FSM (`LOST` / `ACQUIRING` / `TRACKING` / `RELOCATING`), screen gate, and backoff scheduling (`src/genshin_navigator/tracker.py`, `screen_gate.py`, `performance.py`).
  - `rust/crates/gh-core/src/nav_controller.rs` — Sticky target navigation state, undo stack, and hold-to-collect confirmation guard (`src/genshin_navigator/navigation.py`).
  - `scripts/audit_release.py` — Tracked-file and release-artifact privacy/secret/IP leak auditor (`scripts/audit_release.py`).
  - `scripts/collect_licenses.py` — Cargo & third-party dependency license collector (`scripts/collect_licenses.py`).

---

## 2. Runtime & Build Dependencies

Run `.\.venv\Scripts\python.exe scripts/collect_licenses.py --cargo-manifest rust/Cargo.toml` to inspect the full transitive Cargo crate BOM and licenses (packaged builds include the generated `THIRD_PARTY_LICENSES.md` and `licenses/` directory), or pass `--copy-dir <dir>` to export full license texts.

| Component | Purpose | License Family | Project |
| --- | --- | --- | --- |
| Rust (`windows` / `windows-rs`) | Typed Win32 GDI capture, hooks, OCR & tray bindings | MIT / Apache-2.0 | <https://github.com/microsoft/windows-rs> |
| `axum` + `tower-http` + `tokio` | Localhost Sync Server, WebSocket & LAN Companion HTTP/WS | MIT | <https://github.com/tokio-rs/axum> |
| `reqwest` (`rustls`) | HoYoLAB asset hydration & version checks (pure-Rust TLS) | MIT / Apache-2.0 | <https://github.com/seanmonstar/reqwest> |
| `tiny-skia` + `cosmic-text` | Layered HUD vector rendering & bilingual `%WINDIR%\Fonts` text | BSD-3-Clause / MIT / Apache-2.0 | <https://github.com/RazrFalcon/tiny-skia> |
| `image` + `ndarray` + `rustfft` | Pure-Rust image decoding, arrays & FFT/CV ops | MIT / Apache-2.0 | <https://github.com/image-rs/image> |
| `zip` + `md-5` + `serde` / `serde_json` | `.npz` archive I/O, checksum validation & protocol JSON | MIT / Apache-2.0 | <https://serde.rs/> |
| `clap` + `tracing` + `subtle` + `rand` + `thiserror` | CLI parsing, logging, constant-time auth & RNG | MIT / Apache-2.0 / BSD-3-Clause | <https://crates.io/> |
| `winresource` (build-only) | Windows icon & version resource embedding | MIT | <https://github.com/mxre/winres> |
| Inno Setup (build-only) | Windows installer packaging | Inno Setup License | <https://jrsoftware.org/isinfo.php> |

> This table summarizes top-level dependencies; zero GPL/AGPL runtime crates are linked (dual-licensed crates such as `self_cell` are used under `Apache-2.0`). Where a crate ships its own `LICENSE` or `NOTICE` text, that license text controls (bundled in `dist/GenshinHelper/THIRD_PARTY_LICENSES.md`).

---

## 3. HoYoverse / HoYoLAB Third-Party IP Disclaimer

**Genshin Impact**, **HoYoLAB**, **HoYoverse**, **miHoYo**, and **Cognosphere** are trademarks, service marks, or copyrights of **COGNOSPHERE PTE. LTD.** and/or **miHoYo Co., Ltd.**

- Genshin-Helper is an independent, unofficial community tool and is **not affiliated with, endorsed by, sponsored by, or approved by** HoYoverse / COGNOSPHERE PTE. LTD.
- HoYoLAB interactive map tiles, terrain images, underground floor overlays, point-of-interest (POI) coordinates, icons, screenshots, and community descriptions are **not licensed under this project's MIT License**.
- Map datasets hydrated locally by `GenshinHelper.exe` (on first launch or via `--sync-data`) into the user's local cache (`%LOCALAPPDATA%\GenshinHelper\data`) are downloaded directly from public HoYoLAB endpoints by the end user for personal reference only and are excluded from this repository's MIT License.
- Original application branding such as `assets/icons/app_compass_origin.png` is original artwork created for Genshin-Helper with zero HoYoverse assets.
