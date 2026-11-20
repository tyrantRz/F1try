"""Precompute compact tables used by the dashboard (Cloud-friendly)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_grid_results, load_pit_stops, load_sector_laps, load_status
from src.maps import fetch_circuit_outline, match_circuit_key
from src.metrics import compute_grid_metrics

OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Priority circuits for the mock + common calendar tracks
PRIORITY_LABELS = [
    "Mexico City (Hermanos Rodríguez)",
    "Monaco (Monte-Carlo)",
    "Monza",
    "Spa-Francorchamps",
    "Silverstone",
    "Suzuka",
    "Montreal",
    "São Paulo",
    "Bahrain (Sakhir)",
    "Budapest",
    "Montmeló",
    "Abu Dhabi",
    "Baku",
    "Austin",
    "Marina Bay",
    "Melbourne",
    "Zandvoort",
    "Imola",
    "Shanghai",
    "Miami",
    "Jeddah",
    "Las Vegas",
    "Spielberg",
    "Al Daayen",
    "Sochi",
    "Le Castellet",
    "Portimão",
    "Hockenheim",
    "Istanbul",
    "Nürburg",
    "Mugello",
]


def build_tables(grid: pd.DataFrame, pits: pd.DataFrame, sectors: pd.DataFrame, status: pd.DataFrame) -> None:
    lean_results = grid[
        [
            "raceId",
            "year",
            "circuit_label",
            "location",
            "circuit_official",
            "grid",
            "position",
            "driverId",
            "code",
            "constructor_name",
            "statusId",
        ]
    ].copy()
    lean_results.to_parquet(OUT / "results_lean.parquet", index=False)

    for y0, y1, tag in [(2000, 2026, "2000_2026"), (2018, 2025, "2018_2025")]:
        m = compute_grid_metrics(grid, y0, y1, max_grid=20)
        m.to_parquet(OUT / f"grid_metrics_{tag}.parquet", index=False)

    if not sectors.empty:
        sec = sectors[
            [
                "Year",
                "EventName",
                "Location",
                "Country",
                "Driver",
                "Team",
                "LapNumber",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Compound",
                "Deleted",
                "event_key",
            ]
        ].copy()
        sec.to_parquet(OUT / "sectors_lean.parquet", index=False)

    race_map = grid[["raceId", "year", "circuit_label"]].drop_duplicates()
    pit_lean = pits.merge(race_map, on="raceId", how="inner")
    pit_lean.to_parquet(OUT / "pits_lean.parquet", index=False)
    status.to_parquet(OUT / "status.parquet", index=False)


def build_track_outlines(grid: pd.DataFrame, year: int = 2024, limit: int = 40) -> None:
    """Snapshot MultiViewer outlines into JSON so Cloud needs no live map API."""
    counts = grid.groupby("circuit_label")["raceId"].nunique().sort_values(ascending=False)
    labels: list[str] = []
    for lab in PRIORITY_LABELS:
        if lab in counts.index and lab not in labels:
            labels.append(lab)
    for lab in counts.index.tolist():
        if lab not in labels:
            labels.append(lab)
        if len(labels) >= limit:
            break

    meta = (
        grid.groupby("circuit_label")
        .agg(location=("location", "first"), circuit_official=("circuit_official", "first"))
        .to_dict(orient="index")
    )
    outlines: dict[str, dict] = {}
    for lab in labels:
        info = meta.get(lab, {})
        location = str(info.get("location", lab))
        official = str(info.get("circuit_official", location))
        key = match_circuit_key(location, official)
        if key is None:
            print(f"  skip {lab}: no MultiViewer match")
            continue
        payload = fetch_circuit_outline(key, year=year)
        if not payload:
            print(f"  skip {lab}: outline fetch failed")
            continue
        # Keep only JSON-serializable fields needed by the map figure
        outlines[lab] = {
            "circuitName": payload.get("circuitName") or lab,
            "x": payload.get("x"),
            "y": payload.get("y"),
            "corners": payload.get("corners") or [],
            "_resolved_year": payload.get("_resolved_year"),
            "mv_key": key,
            "location": location,
        }
        print(f"  ok   {lab} (year={outlines[lab]['_resolved_year']})")

    path = OUT / "track_outlines.json"
    path.write_text(json.dumps(outlines), encoding="utf-8")
    print(f"wrote {path.name}: {len(outlines)} circuits, {path.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["tables", "maps"], help="build just one part")
    parser.add_argument("--map-year", type=int, default=2024)
    args = parser.parse_args()

    grid = load_grid_results()
    pits = load_pit_stops()
    sectors = load_sector_laps()
    status = load_status()

    if args.only in (None, "tables"):
        build_tables(grid, pits, sectors, status)
        sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in OUT.glob("*.parquet")}
        print("wrote tables", sizes)

    if args.only in (None, "maps"):
        print("Building track outlines from MultiViewer…")
        build_track_outlines(grid, year=args.map_year)


if __name__ == "__main__":
    main()
