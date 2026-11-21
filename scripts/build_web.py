"""Bundle the retirement_mc package for the browser demo.

GitHub Pages serves only the docs/ folder, so the demo can't import the package
from the repository directly. This script zips the package into
docs/retirement_mc.zip; the page downloads it and unpacks it inside Pyodide.

Run it after changing anything in retirement_mc/:

    python scripts/build_web.py

The test suite checks that the zip matches the source, so a stale demo fails CI
instead of quietly showing old results.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "retirement_mc"
OUTPUT = ROOT / "docs" / "retirement_mc.zip"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)  # same bytes on every rebuild, so git only sees real changes


def package_files() -> list[Path]:
    """Every file the package needs at run time: the code and its data."""
    files = [p for p in PACKAGE.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".csv", ".md"}]
    return sorted(files)


def build(output: Path = OUTPUT) -> Path:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in package_files():
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return output


if __name__ == "__main__":
    out = build()
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB, {len(package_files())} files)")
    sys.exit(0)
