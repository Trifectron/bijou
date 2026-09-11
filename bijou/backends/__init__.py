"""Vendored model implementations. The only package importing third_party.

Importing this package puts the vendored nanoDiff on sys.path, so any module
below it can import nanodiff regardless of which one is loaded first.
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "nanoDiff"

if VENDOR.is_dir() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
