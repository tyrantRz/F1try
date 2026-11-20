"""Refresh the Kaggle-equivalent historical F1 tables from TracingInsights/RaceData."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "racedata"
URL = "https://github.com/TracingInsights/RaceData/releases/latest/download/data.zip"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    req = Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=120) as resp:
        payload = resp.read()
    zip_path = OUT_DIR / "data.zip"
    zip_path.write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(OUT_DIR)
    print(f"Extracted to {OUT_DIR}")
    for path in sorted(OUT_DIR.glob("*.csv")):
        print(f"  {path.name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
