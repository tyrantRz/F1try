"""
Download Race-session lap/sector tables via tif1 (TracingInsights CDN).

Used for Segment Analysis in the F1 Circuit Dashboard.
Saves one CSV per year under data/raw/tif1_sectors/.
Resume-safe: skips years that already have a non-empty CSV unless --force.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import tif1

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "tif1_sectors"

KEEP_COLS = [
    "Year",
    "EventName",
    "RoundNumber",
    "CircuitName",
    "Location",
    "Country",
    "Driver",
    "DriverNumber",
    "Team",
    "LapNumber",
    "LapTimeSeconds",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Stint",
    "Position",
    "TrackStatus",
    "IsPersonalBest",
    "Deleted",
    "IsAccurate",
    "AirTemp",
    "TrackTemp",
    "Rainfall",
]


def _to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return series


def download_year(year: int) -> pd.DataFrame:
    schedule = tif1.get_event_schedule(year)
    frames: list[pd.DataFrame] = []

    for _, event in schedule.iterrows():
        event_name = event["EventName"]
        if "Testing" in str(event_name):
            continue
        try:
            session = tif1.get_session(year, event_name, "Race")
            laps = session.laps.copy()
        except Exception as exc:  # noqa: BLE001 - resume across missing events
            print(f"  skip {year} {event_name}: {exc}")
            continue

        if laps is None or len(laps) == 0:
            print(f"  empty {year} {event_name}")
            continue

        df = pd.DataFrame(laps)
        df["Year"] = year
        df["EventName"] = event_name
        df["RoundNumber"] = event.get("RoundNumber")
        df["CircuitName"] = event.get("CircuitName")
        df["Location"] = event.get("Location")
        df["Country"] = event.get("Country")

        for col in ("Sector1Time", "Sector2Time", "Sector3Time"):
            if col in df.columns:
                df[col] = _to_seconds(df[col])

        cols = [c for c in KEEP_COLS if c in df.columns]
        frames.append(df[cols])
        print(f"  ok   {year} {event_name}: {len(df)} laps")
        time.sleep(0.2)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLS)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=2018)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(args.start, args.end + 1):
        out = OUT_DIR / f"race_sector_laps_{year}.csv"
        if out.exists() and out.stat().st_size > 0 and not args.force:
            print(f"exists {out.name}, skip (use --force to redo)")
            continue
        print(f"Downloading {year} ...")
        df = download_year(year)
        df.to_csv(out, index=False)
        print(f"wrote {out.name}: {len(df)} rows")


if __name__ == "__main__":
    main()
