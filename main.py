#!/usr/bin/env python3
"""
Entrypoint for the quant-derivatives CLI.
Supports both running from the repository root and as an installed package.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from quant_derivatives.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
