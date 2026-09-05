#!/usr/bin/env python3
"""Thin executable entry point for the importable experiment harness."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.orchestration.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
