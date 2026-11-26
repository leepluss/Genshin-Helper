# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| `5.x` (current) | Yes |
| `4.5.x` | Security fixes only |
| `< 4.5` | No |

---

## Security & Anti-Cheat Architecture Guarantees

Genshin-Helper is designed as a **strictly passive, local-only companion tool**:

1. **No Game Memory Access or Injection:** Genshin-Helper never calls `ReadProcessMemory` / `WriteProcessMemory`, never injects DLLs, and never hooks game engine internals.
2. **No Synthetic Game Input:** Genshin-Helper never calls `SendInput`, `mouse_event`, or `keybd_event` against the Genshin Impact window. Global hotkeys use read-only Win32 hooks (`SetWindowsHookExW` / `WH_KEYBOARD_LL` / `WH_MOUSE_LL`) via typed `windows` (`windows-rs`) bindings in `gh-win32`.
3. **Localhost-Only Transport & Hardened Opt-In LAN Mode:** The desktop sync server (`gh-net::sync_server`) binds to `127.0.0.1` (default port `8765`) and restricts `/shutdown` to loopback `POST` (`403` otherwise). Optional read-only LAN Companion mode (`0.0.0.0:8766`) enforces PIN rate limiting (`429`), constant-time token comparison (`subtle`), and strict `no-store`/CSP headers. No telemetry or analytics are transmitted to external servers.
4. **Zero Credential Collection:** Genshin-Helper never asks for, stores, or transmits HoYoLAB passwords or authentication cookies (`ltoken_v2`, `cookie_token`). Cross-map pin export runs passively inside the user's existing browser session and outputs local files only.
5. **Pinned Native Messaging Origin:** The Chromium MV3 Extension uses a pinned Extension ID (`nfkijdoblmphmdljfogooajhbldkeojh`) enforced in `installer/native_host_manifest.json` so only this extension can invoke `GenshinHelperHost.exe`.
6. **Privacy-Safe Diagnostics & Release Auditing:**
   - Diagnostic bundles (`Alt+F12`) automatically blur the in-game UID region in captured screenshots.
   - Release binaries (`GenshinHelper.exe` & `GenshinHelperHost.exe`) are compiled with static CRT (`+crt-static`), stripped symbols (`strip = "symbols"`), `/PDBALTPATH:%_PDB%`, and `--remap-path-prefix` so local developer paths never leak into released executables.
   - `scripts/audit_release.py --strict-ip` scans git-tracked files, `rust/`, and compiled `dist/GenshinHelper` binaries to block accidental leaks of cookies, private keys, 9-10 digit UIDs, personal filesystem paths, HoYoLAB `.npz` assets, or `scratch/` dumps.

---

## Reporting a Vulnerability

If you discover a security vulnerability (such as a localhost authentication bypass, Native Messaging command injection, path traversal, or accidental credential/UID leak):

1. **Do not open a public GitHub issue** with exploit details or sensitive traces.
2. Report privately via **GitHub Security Advisories** ("Report a vulnerability" under the repository's **Security** tab), or contact the repository maintainers directly.
3. Include:
   - Affected component and version/commit SHA.
   - Clear reproduction steps or minimal proof of concept.
   - Potential impact assessment.
4. **Never include** personal HoYoLAB cookies (`ltoken_v2`, `cookie_token`), private keys, or unredacted Genshin UIDs in reports.

We aim to acknowledge security reports within **72 hours** and publish a coordinated fix once verified.
