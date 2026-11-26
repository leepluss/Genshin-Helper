"""Configuration and target discovery for the Black-Box Contract Test Suite."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _discover_rust_exe(name: str) -> Optional[Path]:
    root = repo_root()
    candidates: list[Path] = [
        root / "dist" / "GenshinHelper" / name,
    ]
    cargo_target = os.environ.get("CARGO_TARGET_DIR", "").strip()
    if cargo_target:
        tdir = Path(cargo_target)
        candidates.extend([tdir / "release" / name, tdir / "debug" / name])
    candidates.extend([
        root / "rust" / "target" / "release" / name,
        root / "rust" / "target" / "debug" / name,
    ])
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    manifest = root / "rust" / "Cargo.toml"
    if manifest.is_file():
        try:
            out = subprocess.check_output(
                ["cargo", "metadata", "--manifest-path", str(manifest), "--no-deps", "--format-version", "1"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
            tdir_str = json.loads(out).get("target_directory")
            if tdir_str:
                tdir = Path(tdir_str)
                for profile in ("release", "debug"):
                    cand = tdir / profile / name
                    if cand.is_file():
                        return cand.resolve()
        except Exception:
            pass
    return None


@dataclass
class TargetConfig:
    server_exe: Optional[Path] = None
    host_exe: Optional[Path] = None
    backend: str = "rust"  # "auto" | "rust"
    keep_sandbox: bool = False
    allow_cold_host_launch: bool = False
    startup_timeout_s: float = 25.0
    http_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.server_exe is None:
            self.server_exe = _discover_rust_exe("GenshinHelper.exe")
        if self.host_exe is None:
            self.host_exe = _discover_rust_exe("GenshinHelperHost.exe")

    @property
    def resolved_backend(self) -> str:
        return "rust"

    @property
    def is_rust(self) -> bool:
        return self.resolved_backend == "rust"


def _env_path(name: str) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def load_config_from_env() -> TargetConfig:
    backend = os.environ.get("GH_BLACKBOX_BACKEND", "rust").strip().lower() or "rust"
    if backend not in ("auto", "rust"):
        backend = "rust"
    return TargetConfig(
        server_exe=_env_path("GH_SERVER_EXE"),
        host_exe=_env_path("GH_HOST_EXE"),
        backend=backend,
        keep_sandbox=_env_flag("GH_KEEP_SANDBOX"),
        allow_cold_host_launch=_env_flag("GH_TEST_HOST_COLD_LAUNCH"),
    )


_ACTIVE_CONFIG: TargetConfig = load_config_from_env()


def get_config() -> TargetConfig:
    return _ACTIVE_CONFIG


def configure(
    *,
    server_exe: Optional[str | Path] = None,
    host_exe: Optional[str | Path] = None,
    backend: Optional[str] = None,
    keep_sandbox: Optional[bool] = None,
    allow_cold_host_launch: Optional[bool] = None,
) -> TargetConfig:
    cfg = get_config()
    if server_exe is not None:
        cfg.server_exe = Path(server_exe).expanduser().resolve()
    if host_exe is not None:
        cfg.host_exe = Path(host_exe).expanduser().resolve()
    if backend is not None:
        norm = backend.strip().lower()
        if norm not in ("auto", "rust"):
            raise ValueError(f"Unsupported backend: {backend!r}")
        cfg.backend = norm
    if keep_sandbox is not None:
        cfg.keep_sandbox = bool(keep_sandbox)
    if allow_cold_host_launch is not None:
        cfg.allow_cold_host_launch = bool(allow_cold_host_launch)
    return cfg


def python_executable() -> str:
    return sys.executable
