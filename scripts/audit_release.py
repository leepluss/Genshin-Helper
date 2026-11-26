# Ported from MaxWay12/Genshin-Navigator@99e1e13:scripts/audit_release.py (MIT License)
"""
Release privacy, secret-leak, personal-path, and HoYoverse IP hygiene auditor.

Scans git-tracked files (and optional --artifact / --dir directories) for:
- Hardcoded personal filesystem paths (C:\\Users\\, E:\\Dev Soft, %USERPROFILE%)
- Leaked HoYoLAB cookies/tokens (ltoken_v2, cookie_token, auth values) and private keys
- Leaked 9-10 digit Genshin UID numbers
- Tracked scratch/ or personal runtime state leaks
- Copyrighted HoYoverse IP assets (paimon_master.png, data/*.npz, assets/data/*.npz, src/data/*.npz) —
  warning by default, hard failure when --strict-ip is enabled for pre-public GitHub push gating.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

FORBIDDEN_ARTIFACT_PARTS = {
    "scratch",
    "diagnostics",
    "backups",
    "ocr_cache",
    "failures",
    "scenarios",
    ".git",
    ".venv",
}

SENSITIVE_TEXT_SUFFIXES = {
    ".json",
    ".txt",
    ".log",
    ".ini",
    ".cfg",
    ".yaml",
    ".yml",
    ".env",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".ps1",
    ".bat",
    ".iss",
    ".spec",
    ".html",
    ".css",
    ".db",
    ".sqlite",
    ".rs",
    ".toml",
}

BINARY_SKIP_AUTH_SUFFIXES = {
    ".exe",
    ".dll",
    ".pyd",
    ".so",
    ".dylib",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".ico",
    ".npz",
    ".npy",
    ".zip",
}

# Patterns intentionally use character classes / concatenation so this file never self-matches.
HARDCODED_USERS_PATH = re.compile(rb"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"']+", re.I)
HARDCODED_DEV_SOFT_PATH = re.compile(rb"E:[\\/]Dev\s+Soft(?:[\\/]|$|\s|\"|')", re.I)
UID_WITH_VALUE = re.compile(rb"\bUID\s*[:=]?\s*\d{9,10}\b", re.I)
COOKIE_SECRET_VALUE = re.compile(
    rb"(?:ltoken_v2|cookie_token_v2|cookie_token|ltoken|ltuid_v2|ltuid|account_id_v2|account_id|authorization)\s*[:=]\s*['\"]?[^\s,;}\"']{8,}",
    re.I,
)
PRIVATE_KEY_HEADER = re.compile(
    rb"-----" + rb"BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----"
)

COPYRIGHTED_IP_FILENAMES = {
    "paimon_master.png",
}


def tracked_files(root: Path) -> list[Path]:
    """Return all git-tracked files under root."""
    output = subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", "ls-files", "-z"],
        cwd=root,
    )
    return [root / item.decode("utf-8") for item in output.split(b"\0") if item]


def is_copyrighted_ip_asset(path: Path, root: Path | None = None) -> str | None:
    """Return a reason string if path matches a known copyrighted HoYoverse asset."""
    name_lower = path.name.lower()
    if name_lower in COPYRIGHTED_IP_FILENAMES:
        return "copyrighted HoYoverse character artwork (paimon_master.png)"

    rel = path
    if root is not None:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
    parts_lower = [part.lower() for part in rel.parts]
    if name_lower.endswith(".npz") and (
        (len(parts_lower) >= 2 and parts_lower[-2] == "data")
        or (
            len(parts_lower) >= 3
            and parts_lower[-3] in {"src", "assets"}
            and parts_lower[-2] == "data"
        )
    ):
        return "HoYoLAB map data binary (*.npz under data/ — keep in local cache, exclude from public git)"
    return None


def inspect_file(
    path: Path,
    *,
    root: Path | None = None,
    artifact: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Inspect a single file and return (failures, warnings).
    """
    failures: list[str] = []
    warnings: list[str] = []

    rel = path
    if root is not None:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
    relative_parts = {part.lower() for part in rel.parts}

    if "scratch" in relative_parts:
        failures.append("scratch/ directory leak")
    if artifact and (relative_parts & FORBIDDEN_ARTIFACT_PARTS):
        failures.append("forbidden personal-state or build-cache path in artifact")

    ip_reason = is_copyrighted_ip_asset(path, root=root)
    if ip_reason is not None:
        warnings.append(ip_reason)

    try:
        data = path.read_bytes()
    except OSError as error:
        failures.append(f"unreadable: {error}")
        return failures, warnings

    suffix = path.suffix.lower()
    checks: list[tuple[str, re.Pattern[bytes]]] = [
        ("hardcoded developer workspace path", HARDCODED_DEV_SOFT_PATH),
        ("private key material", PRIVATE_KEY_HEADER),
        ("UID value", UID_WITH_VALUE),
    ]
    if not (artifact and suffix in BINARY_SKIP_AUTH_SUFFIXES):
        checks.insert(0, ("hardcoded Windows user path", HARDCODED_USERS_PATH))

    profile = os.environ.get("USERPROFILE", "").strip()
    if profile and len(profile) >= 6:
        profile_re = re.compile(re.escape(profile.encode("utf-8")), re.I)
        checks.insert(0, ("local USERPROFILE path", profile_re))

    if suffix not in BINARY_SKIP_AUTH_SUFFIXES and (
        not artifact or suffix in SENSITIVE_TEXT_SUFFIXES
    ):
        checks.append(("secret cookie/token value", COOKIE_SECRET_VALUE))

    seen_labels: set[str] = set()
    for label, pattern in checks:
        if label not in seen_labels and pattern.search(data):
            failures.append(label)
            seen_labels.add(label)

    return failures, warnings


def audit_targets(
    targets: Sequence[tuple[Path, bool]],
    *,
    root: Path | None = None,
    strict_ip: bool = False,
) -> tuple[list[str], list[str]]:
    """Run inspection across all (path, is_artifact) targets."""
    all_failures: list[str] = []
    all_warnings: list[str] = []

    for path, is_artifact in targets:
        file_failures, file_warnings = inspect_file(path, root=root, artifact=is_artifact)
        if root is not None:
            try:
                shown = path.relative_to(root).as_posix()
            except ValueError:
                shown = str(path)
        else:
            shown = str(path)

        for issue in file_failures:
            all_failures.append(f"{shown}: {issue}")
        for warn in file_warnings:
            entry = f"{shown}: {warn}"
            if strict_ip:
                all_failures.append(f"[strict-ip] {entry}")
            else:
                all_warnings.append(entry)

    return all_failures, all_warnings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit tracked and release files for private state, secrets, and IP hygiene."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Optional release artifact directory to audit in addition to git-tracked files.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Audit only files inside this directory (skips git ls-files; useful for isolated checks).",
    )
    parser.add_argument(
        "--strict-ip",
        action="store_true",
        help="Treat copyrighted HoYoverse IP warnings (paimon_master.png, data/*.npz) as hard failures.",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    targets: list[tuple[Path, bool]] = []
    audit_root: Path = root

    if args.dir is not None:
        audit_root = args.dir.resolve()
        targets.extend(
            (path, False) for path in sorted(audit_root.rglob("*")) if path.is_file()
        )
    else:
        targets.extend((path, False) for path in tracked_files(root) if path.is_file())

    if args.artifact is not None:
        artifact_dir = args.artifact.resolve()
        targets.extend(
            (path, True) for path in sorted(artifact_dir.rglob("*")) if path.is_file()
        )

    failures, warnings = audit_targets(
        targets, root=audit_root, strict_ip=args.strict_ip
    )

    if warnings:
        print(f"Release IP hygiene warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  [WARN] {warning}")

    if failures:
        print(f"Release privacy/security audit FAILED ({len(failures)} issue(s)):")
        for failure in failures:
            print(f"  [FAIL] {failure}")
        return 2

    print(
        f"Release privacy/security audit passed "
        f"({len(targets)} files checked, {len(warnings)} IP warning(s))."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
