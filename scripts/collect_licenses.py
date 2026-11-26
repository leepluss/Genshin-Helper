# Ported from MaxWay12/Genshin-Navigator@99e1e13:scripts/collect_licenses.py (MIT License)
"""
Collect dependency license metadata for Rust (rust/Cargo.toml via `cargo metadata`,
default) or legacy Python requirements manifests, and optionally export full
third-party license texts for portable/installer builds.
"""
from __future__ import annotations

import argparse
from collections import deque
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

LICENSE_WORDS = ("license", "copying", "notice", "authors")
CARGO_LICENSE_PREFIXES = ("license", "licence", "copying", "notice", "unlicense")
REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
SPDX_TOKEN_RE = re.compile(r"\(|\)|/|[^\s()/]+")
SPDX_RESTRICTED_RE = re.compile(r"^(?:A?GPL|LGPL|SSPL)(?:$|-)", re.I)


def parse_requirements(requirements_path: Path) -> list[str]:
    """Parse top-level distribution names from a requirements.txt file."""
    if not requirements_path.is_file():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    names: list[str] = []
    seen: set[str] = set()
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = REQ_NAME_RE.match(line)
        if not match:
            continue
        pkg = match.group(1)
        norm = pkg.lower().replace("_", "-")
        if norm not in seen:
            seen.add(norm)
            names.append(pkg)
    return names


def _resolve_license_label(metadata: importlib.metadata.PackageMetadata) -> str:
    """Resolve a concise license identifier from PEP 639 / Core Metadata fields."""
    lic_expr = (metadata.get("License-Expression") or "").strip()
    if lic_expr:
        return lic_expr

    raw_lic = (metadata.get("License") or "").strip()
    if raw_lic and raw_lic.upper() != "UNKNOWN":
        first_line = next((ln.strip() for ln in raw_lic.splitlines() if ln.strip()), "")
        if first_line and len(first_line) <= 100:
            return first_line

    classifiers = [
        c for c in (metadata.get_all("Classifier") or []) if c.startswith("License ::")
    ]
    if classifiers:
        labels = [c.split("::")[-1].strip() for c in classifiers]
        return " / ".join(dict.fromkeys(labels))

    if raw_lic and raw_lic.upper() != "UNKNOWN":
        return "Custom (see package LICENSE file)"
    return "Unknown"


def _resolve_homepage(metadata: importlib.metadata.PackageMetadata) -> str:
    home = (metadata.get("Home-Page") or "").strip()
    if home and home.upper() != "UNKNOWN":
        return home
    for entry in metadata.get_all("Project-URL") or []:
        if "," in entry:
            label, url = [part.strip() for part in entry.split(",", 1)]
            if label.lower() in {"homepage", "home", "repository", "source", "source code"}:
                return url
    for entry in metadata.get_all("Project-URL") or []:
        if "," in entry:
            return entry.split(",", 1)[1].strip()
    return "-"


def _find_license_files(distribution: importlib.metadata.Distribution) -> list[Path]:
    found: list[Path] = []
    for item in distribution.files or ():
        filename = Path(str(item)).name.lower()
        if not any(word in filename for word in LICENSE_WORDS):
            continue
        source = Path(distribution.locate_file(item))
        if source.is_file():
            found.append(source)
    return found


def collect_metadata_summary(
    packages: Sequence[str],
    *,
    include_python: bool = True,
) -> list[dict[str, Any]]:
    """Inspect importlib.metadata for each requested package."""
    entries: list[dict[str, Any]] = []
    if include_python:
        entries.append(
            {
                "package": "Python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "license": "PSF-2.0",
                "installed": True,
                "homepage": "https://www.python.org/",
                "license_files": 1 if (Path(sys.base_prefix) / "LICENSE.txt").is_file() else 0,
            }
        )

    for pkg in packages:
        try:
            dist = importlib.metadata.distribution(pkg)
        except importlib.metadata.PackageNotFoundError:
            entries.append(
                {
                    "package": pkg,
                    "version": "-",
                    "license": "Not installed",
                    "installed": False,
                    "homepage": "-",
                    "license_files": 0,
                }
            )
            continue

        md = dist.metadata
        lic_files = _find_license_files(dist)
        entries.append(
            {
                "package": md.get("Name", pkg),
                "version": dist.version,
                "license": _resolve_license_label(md),
                "installed": True,
                "homepage": _resolve_homepage(md),
                "license_files": len(lic_files),
            }
        )
    return entries


def format_markdown_summary(
    entries: Sequence[dict[str, Any]],
    *,
    title: str = "Third-Party Python Dependency Licenses",
    installed_label: str = "installed",
) -> str:
    """Format package license metadata as a clean Markdown table."""
    lines = [
        f"# {title}",
        "",
        "| Package | Version | License | Status | Homepage |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in entries:
        status = installed_label if item["installed"] else "not installed"
        lines.append(
            f"| {item['package']} | {item['version']} | {item['license']} | "
            f"{status} | {item['homepage']} |"
        )
    return "\n".join(lines) + "\n"


def is_spdx_expression_permissive(expr: str) -> bool:
    """
    Evaluate an SPDX license expression (supporting `OR`, `AND`, `/`, `WITH`, and
    parentheses) and return True iff at least one compliant license branch satisfies
    all required clauses without unconditional strong copyleft or unknown licenses.
    """
    raw = (expr or "").strip()
    if not raw:
        return False

    tokens = SPDX_TOKEN_RE.findall(raw)
    if not tokens:
        return False
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def consume() -> str:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def parse_or() -> bool:
        val = parse_and()
        while (tok := peek()) is not None and (tok.upper() == "OR" or tok == "/"):
            consume()
            rhs = parse_and()
            val = val or rhs
        return val

    def parse_and() -> bool:
        val = parse_primary()
        while (tok := peek()) is not None and tok.upper() == "AND":
            consume()
            rhs = parse_primary()
            val = val and rhs
        return val

    def parse_primary() -> bool:
        tok = peek()
        if tok is None:
            raise ValueError("unexpected end of SPDX expression")
        if tok == "(":
            consume()
            val = parse_or()
            if peek() != ")":
                raise ValueError("unclosed parenthesis in SPDX expression")
            consume()
            return val
        if tok in {")", "/"} or tok.upper() in {"AND", "OR", "WITH"}:
            raise ValueError(f"unexpected operator {tok!r} in SPDX expression")
        ident = consume()
        if (nxt := peek()) is not None and nxt.upper() == "WITH":
            consume()
            if peek() is None or peek() in {"(", ")", "/"}:
                raise ValueError("missing exception after WITH")
            consume()
        upper = ident.upper()
        if upper in {"UNKNOWN", "NONE", "CUSTOM"}:
            return False
        return not bool(SPDX_RESTRICTED_RE.match(ident))

    try:
        result = parse_or()
    except ValueError:
        return False
    return result and pos == len(tokens)


def _find_cargo_license_files(pkg_dir: Path, license_file: str | None) -> list[Path]:
    found: dict[Path, Path] = {}
    if pkg_dir.is_dir():
        for child in sorted(pkg_dir.iterdir()):
            if child.is_file() and child.name.lower().startswith(CARGO_LICENSE_PREFIXES):
                found[child.resolve()] = child
    if license_file:
        candidate = (pkg_dir / license_file).resolve()
        try:
            candidate.relative_to(pkg_dir.resolve())
        except ValueError:
            candidate = Path()
        if candidate.is_file():
            found.setdefault(candidate, candidate)
    return list(found.values())


def collect_cargo_metadata(
    manifest_path: Path,
    packages: Sequence[str],
    *,
    target: str = "x86_64-pc-windows-msvc",
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[Path]]]:
    """
    Collect transitive normal + build dependencies of selected workspace crates via
    `cargo metadata`, excluding workspace members themselves and never exposing local
    `manifest_path` values in the returned metadata entries.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cargo manifest not found: {manifest_path}")

    raw = subprocess.check_output(
        [
            "cargo",
            "metadata",
            "--manifest-path",
            str(manifest_path),
            "--format-version",
            "1",
            "--filter-platform",
            target,
        ],
        text=True,
        encoding="utf-8",
    )
    meta = json.loads(raw)

    pkg_by_id: dict[str, dict[str, Any]] = {p["id"]: p for p in meta.get("packages", [])}
    ws_ids: set[str] = set(meta.get("workspace_members") or [])
    nodes_by_id: dict[str, dict[str, Any]] = {
        n["id"]: n for n in (meta.get("resolve") or {}).get("nodes", [])
    }

    requested = list(packages)
    seeds: list[str] = []
    for req_name in requested:
        matches = [
            pid
            for pid, pkg in pkg_by_id.items()
            if pkg.get("name") == req_name and pid in ws_ids
        ]
        if not matches:
            matches = [
                pid for pid, pkg in pkg_by_id.items() if pkg.get("name") == req_name
            ]
        if not matches:
            raise ValueError(f"Cargo package {req_name!r} not found in {manifest_path}")
        seeds.extend(matches)

    visited: set[str] = set(seeds)
    queue: deque[str] = deque(seeds)
    third_party_ids: set[str] = set()

    while queue:
        cur = queue.popleft()
        if cur not in ws_ids:
            third_party_ids.add(cur)
        node = nodes_by_id.get(cur) or {}
        for dep in node.get("deps", []):
            dep_kinds = dep.get("dep_kinds") or []
            if dep_kinds and not any(
                dk.get("kind") in (None, "build") for dk in dep_kinds
            ):
                continue
            dep_id = dep.get("pkg")
            if dep_id and dep_id not in visited:
                visited.add(dep_id)
                queue.append(dep_id)

    entries: list[dict[str, Any]] = []
    license_sources: dict[tuple[str, str], list[Path]] = {}
    for pid in third_party_ids:
        pkg = pkg_by_id[pid]
        name = str(pkg["name"])
        version = str(pkg["version"])
        lic = (pkg.get("license") or "").strip()
        if not lic:
            lic = "Custom (see package LICENSE file)" if pkg.get("license_file") else "Unknown"
        homepage = (pkg.get("repository") or pkg.get("homepage") or "-").strip() or "-"
        pkg_dir = Path(pkg["manifest_path"]).parent
        lic_files = _find_cargo_license_files(pkg_dir, pkg.get("license_file"))
        license_sources[(name, version)] = lic_files
        entries.append(
            {
                "package": name,
                "version": version,
                "license": lic,
                "installed": True,
                "homepage": homepage,
                "license_files": len(lic_files),
            }
        )

    entries.sort(key=lambda item: (item["package"].lower(), item["version"]))
    return entries, license_sources


def copy_cargo_license_texts(
    license_sources: dict[tuple[str, str], list[Path]],
    output_dir: Path,
) -> dict[str, list[str]]:
    """Copy crate license texts into output_dir/<name>-<version>/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, list[str]] = {}
    for (name, version), files in sorted(license_sources.items()):
        if not files:
            continue
        destination = output_dir / f"{name}-{version}"
        destination.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for src in files:
            if src.is_file():
                target = destination / src.name
                shutil.copy2(src, target)
                names.append(target.name)
        if names:
            copied[f"{name}=={version}"] = sorted(set(names))
    return copied


def copy_license_texts(packages: Sequence[str], output_dir: Path) -> dict[str, list[str]]:
    """
    Copy distributable license texts from installed package dist-info folders
    (and Python's LICENSE.txt) into output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, list[str]] = {}

    for pkg in packages:
        try:
            dist = importlib.metadata.distribution(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue

        destination = output_dir / f"{dist.metadata['Name']}-{dist.version}"
        found_names: list[str] = []
        for item in dist.files or ():
            filename = Path(str(item)).name.lower()
            if not any(word in filename for word in LICENSE_WORDS):
                continue
            source = Path(dist.locate_file(item))
            if not source.is_file():
                continue
            safe_name = "__".join(
                part for part in Path(str(item)).parts if part not in {".", ".."}
            )
            target = destination / safe_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            found_names.append(target.name)

        license_text = dist.metadata.get("License")
        if not found_names and license_text and len(license_text.strip()) > 80:
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / "LICENSE-from-package-metadata.txt"
            target.write_text(license_text.strip() + "\n", encoding="utf-8")
            found_names.append(target.name)

        copied[f"{dist.metadata['Name']}=={dist.version}"] = sorted(set(found_names))

    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        py_dest = output_dir / f"Python-{sys.version_info.major}.{sys.version_info.minor}"
        py_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(python_license, py_dest / "LICENSE.txt")
        copied[f"Python {sys.version_info.major}.{sys.version_info.minor}"] = ["LICENSE.txt"]

    return copied


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Collect dependency license metadata from Cargo.toml (default: rust/Cargo.toml)."
    )
    parser.add_argument(
        "--source",
        choices=("auto", "cargo", "python"),
        default="auto",
        help="Metadata source: 'auto' selects Cargo (default unless --requirements exists).",
    )
    parser.add_argument(
        "--cargo-manifest",
        type=Path,
        default=None,
        help="Path to Cargo.toml (implies --source cargo when --source is auto).",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="Workspace crate(s) whose transitive normal+build deps to collect (default: gh-app, gh-host).",
    )
    parser.add_argument(
        "--target",
        default="x86_64-pc-windows-msvc",
        help="Target triple passed to `cargo metadata --filter-platform` (default: x86_64-pc-windows-msvc).",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=root / "requirements.txt",
        help="Path to requirements.txt (default: <repo>/requirements.txt).",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Additional Python package name(s) to include alongside requirements.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path for the summary (writes to stdout if omitted).",
    )
    parser.add_argument(
        "--copy-dir",
        type=Path,
        default=None,
        help="Optional directory to copy full third-party license text files into.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output summary as JSON instead of Markdown.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any package is missing (Python) or has Unknown/copyleft license (Cargo).",
    )
    args = parser.parse_args(argv)

    source = args.source
    if source == "auto":
        if args.cargo_manifest is not None or not args.requirements.is_file():
            source = "cargo"
        else:
            source = "python"

    if source == "cargo":
        manifest = args.cargo_manifest or (root / "rust" / "Cargo.toml")
        cargo_packages = args.package or ["gh-app", "gh-host"]
        entries, license_sources = collect_cargo_metadata(
            manifest,
            cargo_packages,
            target=args.target,
        )
        rendered = (
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n"
            if args.as_json
            else format_markdown_summary(
                entries,
                title="Third-Party Rust Dependency Licenses",
                installed_label="bundled",
            )
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(f"Wrote license summary ({len(entries)} entries) to {args.output}")
        else:
            sys.stdout.write(rendered)

        if args.copy_dir is not None:
            copied = copy_cargo_license_texts(license_sources, args.copy_dir)
            print(f"Copied license texts for {len(copied)} crates into {args.copy_dir}")

        if args.strict:
            non_compliant = [
                f"{item['package']}=={item['version']} ({item['license']})"
                for item in entries
                if not is_spdx_expression_permissive(item["license"])
            ]
            if non_compliant:
                print(
                    f"Non-permissive or unknown Cargo licenses: {', '.join(non_compliant)}",
                    file=sys.stderr,
                )
                return 1
        return 0

    packages = parse_requirements(args.requirements)
    seen = {p.lower().replace("_", "-") for p in packages}
    for extra in args.extra:
        norm = extra.lower().replace("_", "-")
        if norm not in seen:
            seen.add(norm)
            packages.append(extra)

    entries = collect_metadata_summary(packages, include_python=True)
    rendered = (
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n"
        if args.as_json
        else format_markdown_summary(entries)
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote license summary ({len(entries)} entries) to {args.output}")
    else:
        sys.stdout.write(rendered)

    if args.copy_dir is not None:
        copied = copy_license_texts(packages, args.copy_dir)
        print(f"Copied license texts for {len(copied)} distributions into {args.copy_dir}")

    if args.strict and any(not item["installed"] for item in entries):
        missing = [item["package"] for item in entries if not item["installed"]]
        print(f"Missing installed packages: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
