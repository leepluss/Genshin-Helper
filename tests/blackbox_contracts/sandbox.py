"""Isolated process sandbox and pure-stdlib synthetic asset generator."""
from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
import uuid
import zipfile

from .config import TargetConfig, get_config, repo_root

FORBIDDEN_DEFAULT_PORTS = frozenset({8765, 8766})
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

_ACTIVE_HANDLES: List["ServerHandle"] = []
_ACTIVE_SANDBOXES: List["BackendSandbox"] = []


def _atexit_cleanup() -> None:
    for handle in list(_ACTIVE_HANDLES):
        try:
            handle.stop()
        except Exception:
            pass
    for sb in list(_ACTIVE_SANDBOXES):
        try:
            sb.cleanup()
        except Exception:
            pass


atexit.register(_atexit_cleanup)


def allocate_free_port(host: str = "127.0.0.1") -> int:
    """Allocates an ephemeral unused TCP port, never returning 8765 or 8766."""
    for _ in range(32):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            port = int(sock.getsockname()[1])
        if port not in FORBIDDEN_DEFAULT_PORTS:
            return port
    raise RuntimeError("Unable to allocate an ephemeral non-default port")


def _encode_npy(shape: tuple[int, ...], descr: str, raw_bytes: bytes) -> bytes:
    """Encodes a C-contiguous NumPy .npy v1.0 buffer using stdlib only."""
    shape_repr = f"({shape[0]},)" if len(shape) == 1 else repr(shape)
    header_dict = (
        f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_repr}, }}"
    ).encode("ascii")
    # Total prefix (10 bytes) + header must be divisible by 64 for .npy v1.0.
    pad_len = (64 - ((10 + len(header_dict) + 1) % 64)) % 64
    header = header_dict + (b" " * pad_len) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + raw_bytes


def _write_npz(dest: Path, arrays: Dict[str, tuple[tuple[int, ...], str, bytes]]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, (shape, descr, raw_bytes) in arrays.items():
            zf.writestr(f"{name}.npy", _encode_npy(shape, descr, raw_bytes))


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_multimap_payload() -> Dict[str, Any]:
    return {
        "version": "4.0.0",
        "maps": {
            "2": {
                "id": 2,
                "name": "Teyvat Interactive Map",
                "priority": 1,
                "detail_v2": {
                    "total_size": [36864, 32768],
                    "padding": [7625, 6881],
                    "origin": [24207, 16411],
                    "map_version": "sandbox_v4",
                    "min_zoom": -4,
                    "max_zoom": 0,
                    "original_map_size": [21613, 19006],
                },
                "areas": [
                    {"id": 1, "name": "Mondstadt"},
                    {"id": 2, "name": "Liyue"},
                ],
                "waypoints": [
                    {
                        "id": 900001,
                        "label_id": 2,
                        "area_id": 1,
                        "x_pos": -100.0,
                        "y_pos": 100.0,
                        "name": "Sandbox Statue",
                    },
                    {
                        "id": 900002,
                        "label_id": 3,
                        "area_id": 1,
                        "x_pos": 0.0,
                        "y_pos": 0.0,
                        "name": "Sandbox Waypoint Origin",
                    },
                    {
                        "id": 900003,
                        "label_id": 3,
                        "area_id": 2,
                        "x_pos": 500.0,
                        "y_pos": 500.0,
                        "name": "Sandbox Waypoint East",
                    },
                ],
                "point_groups": [],
            }
        },
    }


def generate_synthetic_data_dir(data_dir: Path) -> None:
    """Generates valid synthetic map assets + MD5 manifest strictly inside temp dir."""
    data_dir.mkdir(parents=True, exist_ok=True)

    multimap_path = data_dir / "multimap_v4.json"
    multimap_path.write_text(
        json.dumps(_synthetic_multimap_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 64x64 grayscale pattern + 5x int64 metadata [tx_start, ty_start, z, orig_x, orig_y].
    gray_rows = bytearray()
    for y in range(64):
        for x in range(64):
            gray_rows.append(((x * 7) ^ (y * 11)) & 0xFF)
    meta_bytes = struct.pack("<5q", 14, 10, -2, 24207, 16411)
    terrain_path = data_dir / "terrain_n2_v4.npz"
    _write_npz(
        terrain_path,
        {
            "map_2_gray": ((64, 64), "|u1", bytes(gray_rows)),
            "map_2_meta": ((5,), "<i8", meta_bytes),
        },
    )

    # Minimal non-empty cave mask archive.
    cave_path = data_dir / "cave_masks_v4.npz"
    _write_npz(
        cave_path,
        {
            "mask_2_1_1": ((16, 16), "|u1", bytes([255] * 256)),
        },
    )

    files_meta: Dict[str, Any] = {}
    for fname in ("multimap_v4.json", "terrain_n2_v4.npz", "cave_masks_v4.npz"):
        fpath = data_dir / fname
        files_meta[fname] = {
            "md5": _md5_file(fpath),
            "size": fpath.stat().st_size,
        }

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "sandbox_synthetic",
        "files": files_meta,
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def default_sandbox_config(
    ws_port: int,
    lan_port: int,
    *,
    lan_enabled: bool = False,
    language: str = "en",
    audio_enabled: bool = False,
    audio_volume: int = 0,
) -> Dict[str, Any]:
    return {
        "hotkey": "alt+f11",
        "hotkey_sync": "alt+f11",
        "hotkey_debug": "alt+f12",
        "hotkey_switch": "alt+q",
        "hotkey_nav_mark": "alt+k",
        "hotkey_nav_undo": "alt+shift+k",
        "hotkey_nav_guide": "alt+h",
        "hotkey_nav_next": "alt+n",
        "hotkey_nav_layout": "alt+shift+h",
        "hotkey_nav_toggle": "alt+x",
        "hotkey_suppress_conflicts": True,
        "audio_enabled": audio_enabled,
        "audio_volume": audio_volume,
        "chest_search_radius": 30.0,
        "ws_port": int(ws_port),
        "auto_sync_m": True,
        "single_monitor_mode": False,
        "language": language,
        "minimap_tracking": True,
        "nav_hud_enabled": True,
        "nav_hud_capture_visible": False,
        "nav_hud_position": "under_minimap",
        "nav_hud_custom_pos": None,
        "nav_hud_guide_zoom_mode": "cycle",
        "lan_companion_enabled": bool(lan_enabled),
        "lan_companion_port": int(lan_port),
        "hoyolab_sync": {
            "auto_check_on_startup": False,
            "auto_update_mode": "off",
        },
        "instructions": "Isolated black-box test configuration",
    }


@dataclass
class ServerHandle:
    process: subprocess.Popen[bytes]
    ws_port: int
    lan_port: int
    sandbox: "BackendSandbox"
    config_path: Path
    diagnostics_dir: Path
    log_path: Path
    mutex_name: str
    _log_fh: Any = field(default=None, repr=False)
    _stopped: bool = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.ws_port}"

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def wait_until_ready(self, timeout_s: float = 25.0) -> None:
        deadline = time.monotonic() + timeout_s
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            rc = self.process.poll()
            if rc is not None:
                tail = self.read_log_tail()
                raise RuntimeError(
                    f"Server exited prematurely with code {rc} before /status became ready.\n"
                    f"Log tail ({self.log_path}):\n{tail}"
                )
            conn: Optional[http.client.HTTPConnection] = None
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.ws_port, timeout=1.0)
                conn.request("GET", "/status")
                resp = conn.getresponse()
                resp.read()
                if resp.status == 200:
                    return
            except Exception as exc:
                last_err = exc
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
            time.sleep(0.1)
        tail = self.read_log_tail()
        raise TimeoutError(
            f"Timed out waiting for server on 127.0.0.1:{self.ws_port}/status "
            f"(last error: {last_err}).\nLog tail ({self.log_path}):\n{tail}"
        )

    def read_log_tail(self, max_chars: int = 3000) -> str:
        if not self.log_path.is_file():
            return "<no log>"
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
            return text[-max_chars:]
        except Exception as exc:
            return f"<failed to read log: {exc}>"

    def stop(self) -> int:
        if self._stopped and self.process.poll() is not None:
            return int(self.process.returncode or 0)
        self._stopped = True

        if self.process.poll() is None:
            conn: Optional[http.client.HTTPConnection] = None
            try:
                conn = http.client.HTTPConnection("127.0.0.1", self.ws_port, timeout=1.5)
                conn.request(
                    "POST",
                    "/shutdown",
                    body=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                resp = conn.getresponse()
                resp.read()
            except Exception:
                pass
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
            try:
                self.process.wait(timeout=3.5)
            except subprocess.TimeoutExpired:
                pass

        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except Exception:
                pass

        if self.process.poll() is None and sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                    creationflags=CREATE_NO_WINDOW,
                    check=False,
                )
                self.process.wait(timeout=2.0)
            except Exception:
                pass
        elif self.process.poll() is None:
            try:
                self.process.kill()
                self.process.wait(timeout=2.0)
            except Exception:
                pass

        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

        return int(self.process.returncode if self.process.returncode is not None else -1)


class BackendSandbox:
    """Isolated temp workspace for running Python or Rust Genshin-Helper processes."""

    def __init__(self, config: Optional[TargetConfig] = None, *, prefix: str = "gh_e2e_") -> None:
        self.config = config or get_config()
        self.root = Path(tempfile.mkdtemp(prefix=prefix))
        self.appdata = self.root / "appdata"
        self.localappdata = self.root / "localappdata"
        self.data_dir = self.localappdata / "GenshinHelper" / "data"
        self.config_dir = self.appdata / "GenshinHelper"
        self.config_path = self.config_dir / "config.json"
        self.diagnostics_dir = self.config_dir / "diagnostics"
        self.bundle_dir = self.root / "bundle"
        self.run_dir = self.root / "run"
        self.logs_dir = self.root / "logs"

        for d in (
            self.appdata,
            self.localappdata,
            self.data_dir,
            self.config_dir,
            self.bundle_dir,
            self.run_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        self._handles: List[ServerHandle] = []
        self._cleaned = False
        _ACTIVE_SANDBOXES.append(self)

        self._seed_assets()

    def _seed_assets(self) -> None:
        generate_synthetic_data_dir(self.data_dir)

        bundle_data = self.bundle_dir / "src" / "data"
        bundle_locales = self.bundle_dir / "src" / "locales"
        bundle_data.mkdir(parents=True, exist_ok=True)
        bundle_locales.mkdir(parents=True, exist_ok=True)

        # Mirror synthetic data into bundle_dir/src/data as well.
        for fname in ("multimap_v4.json", "terrain_n2_v4.npz", "cave_masks_v4.npz", "manifest.json"):
            shutil.copy2(self.data_dir / fname, bundle_data / fname)

        root = repo_root()
        # Copy static non-IP config/locale files from src/ (or assets/ post-cutover).
        alias_src = next(
            (
                p
                for p in (
                    root / "src" / "data" / "category_aliases.json",
                    root / "assets" / "data" / "category_aliases.json",
                )
                if p.is_file()
            ),
            None,
        )
        if alias_src is not None:
            shutil.copy2(alias_src, bundle_data / "category_aliases.json")

        for lang in ("en", "vi"):
            loc_src = next(
                (
                    p
                    for p in (
                        root / "src" / "locales" / f"{lang}.json",
                        root / "assets" / "locales" / f"{lang}.json",
                    )
                    if p.is_file()
                ),
                None,
            )
            if loc_src is not None:
                shutil.copy2(loc_src, bundle_locales / f"{lang}.json")

        thumb_src = next(
            (
                p
                for p in (
                    root / "src" / "zoom_thumb_template.png",
                    root / "assets" / "zoom_thumb_template.png",
                )
                if p.is_file()
            ),
            None,
        )
        if thumb_src is not None:
            shutil.copy2(thumb_src, self.bundle_dir / "src" / "zoom_thumb_template.png")

    def build_child_env(
        self,
        *,
        mutex_name: Optional[str] = None,
        extra: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        env = os.environ.copy()
        env.pop("GENSHIN_HELPER_DEV_ROOT", None)
        env["APPDATA"] = str(self.appdata)
        env["LOCALAPPDATA"] = str(self.localappdata)
        env["GENSHIN_HELPER_DATA_DIR"] = str(self.data_dir)
        env["PYTHONUTF8"] = "1"
        env.setdefault("GH_NO_ELEVATE", "1")
        env.setdefault("GH_HUD_OFFSCREEN", "1")
        # Never touch the user's desktop from a sandboxed instance: no focus changes from
        # /quick_switch or the switch hotkey, and no global keyboard/mouse hooks.
        env["GH_DESKTOP_SANDBOX"] = "1"
        env.setdefault("GH_LAN_BIND_HOST", "127.0.0.1")
        if mutex_name:
            env["GH_SINGLE_INSTANCE_MUTEX"] = mutex_name
        if extra:
            env.update(extra)
        return env

    def write_config(self, payload: Dict[str, Any]) -> Path:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        return self.config_path

    def read_config(self) -> Dict[str, Any]:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def start_server(
        self,
        *,
        ws_port: Optional[int] = None,
        lan_port: Optional[int] = None,
        standby: bool = True,
        lan_enabled: bool = False,
        language: str = "en",
        mutex_name: Optional[str] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> ServerHandle:
        explicit_ws = ws_port is not None
        attempts = 1 if explicit_ws else 3
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            chosen_ws = int(ws_port) if explicit_ws else allocate_free_port()
            chosen_lan = int(lan_port) if lan_port is not None else allocate_free_port()
            while chosen_lan == chosen_ws:
                chosen_lan = allocate_free_port()

            cfg_data = default_sandbox_config(
                chosen_ws,
                chosen_lan,
                lan_enabled=lan_enabled,
                language=language,
            )
            if extra_config:
                cfg_data.update(extra_config)
            self.write_config(cfg_data)

            chosen_mutex = mutex_name or f"Local\\GH_E2E_Mutex_{uuid.uuid4().hex}"
            env = self.build_child_env(mutex_name=chosen_mutex)
            log_path = self.logs_dir / f"server_{len(self._handles)}_{attempt}.log"
            log_fh = log_path.open("wb")

            if self.config.server_exe is None or not self.config.server_exe.is_file():
                log_fh.close()
                raise FileNotFoundError(
                    f"Rust server executable not found: {self.config.server_exe}"
                )
            cmd = [
                str(self.config.server_exe),
                "--no-tray",
                "--minimized",
                "--port",
                str(chosen_ws),
                "--lan-port",
                str(chosen_lan),
                "--config",
                str(self.config_path),
            ]
            if standby:
                cmd.append("--standby")

            flags = (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else 0
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.run_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            handle = ServerHandle(
                process=proc,
                ws_port=chosen_ws,
                lan_port=chosen_lan,
                sandbox=self,
                config_path=self.config_path,
                diagnostics_dir=self.diagnostics_dir,
                log_path=log_path,
                mutex_name=chosen_mutex,
                _log_fh=log_fh,
            )
            self._handles.append(handle)
            _ACTIVE_HANDLES.append(handle)

            try:
                handle.wait_until_ready(timeout_s=self.config.startup_timeout_s)
                return handle
            except Exception as exc:
                last_error = exc
                handle.stop()

        assert last_error is not None
        raise last_error

    def host_command(self) -> List[str]:
        if self.config.host_exe is None or not self.config.host_exe.is_file():
            raise FileNotFoundError(
                f"Rust host executable not found: {self.config.host_exe}"
            )
        return [str(self.config.host_exe)]

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        for h in list(self._handles):
            try:
                h.stop()
            except Exception:
                pass
        if not self.config.keep_sandbox:
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "BackendSandbox":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.cleanup()
