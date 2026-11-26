"""Permanent Black-Box Contract Test Suite entrypoint for Genshin-Helper.

Supports both standard unittest discovery:
    python -m unittest discover -s tests -p "test_blackbox_contracts.py" -v
and direct CLI execution with target binary overrides:
    python tests/test_blackbox_contracts.py --server-exe ... --host-exe ... -v
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.blackbox_contracts.config import configure


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    pkg_dir = _THIS_DIR / "blackbox_contracts"
    return loader.discover(start_dir=str(pkg_dir), pattern="test_*.py", top_level_dir=str(_REPO_ROOT))


def _parse_cli_and_configure(argv: list[str]) -> list[str]:
    parser = argparse.ArgumentParser(
        description="Run Genshin-Helper Black-Box Contract Test Suite (stdlib-only).",
        add_help=False,
    )
    parser.add_argument("--server-exe", default=None, help="Path to GenshinHelper.exe")
    parser.add_argument("--host-exe", default=None, help="Path to GenshinHelperHost.exe")
    parser.add_argument(
        "--backend",
        choices=("auto", "rust"),
        default=None,
        help="Target backend override (default: rust)",
    )
    parser.add_argument(
        "--keep-sandbox",
        action="store_true",
        default=None,
        help="Preserve temporary sandbox directories after exit",
    )
    parser.add_argument(
        "--cold-host-launch",
        action="store_true",
        default=None,
        help="Enable Native Host cold start_server spawn test",
    )
    args, remaining = parser.parse_known_args(argv)
    configure(
        server_exe=args.server_exe,
        host_exe=args.host_exe,
        backend=args.backend,
        keep_sandbox=True if args.keep_sandbox else None,
        allow_cold_host_launch=True if args.cold_host_launch else None,
    )
    return remaining


if __name__ == "__main__":
    remaining_args = _parse_cli_and_configure(sys.argv[1:])
    unittest.main(module="tests.test_blackbox_contracts", argv=[sys.argv[0], *remaining_args])
