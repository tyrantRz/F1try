"""Load and lightly clean the dashboard source tables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RACEDATA = ROOT / "data" / "raw" / "racedata"
SECTORS = ROOT / "data" / "raw" / "tif1_sectors"

LOCATION_ALIASES = {
    "Mexico City": "Mexico City (Hermanos Rodríguez)",
    "Monte-Carlo": "Monaco (Monte-Carlo)",
    "Spa": "Spa-Francorchamps",
    "Monza": "Monza",
    "Silverstone": "Silverstone",
    "Suzuka": "Suzuka",
    "Sakhir": "Bahrain (Sakhir)",
}


def _friendly_location(location: str, official_name: str) -> str:
    if location in LOCATION_ALIASES:
        return LOCATION_ALIASES[location]
    if pd.isna(location) or str(location).strip() == "":
        return official_name
    return str(location)


@lru_cache(maxsize=1)
def load_grid_results() -> pd.DataFrame:
    results = pd.read_csv(RACEDATA / "results.csv")
    races = pd.read_csv(RACEDATA / "races.csv")
    circuits = pd.read_csv(RACEDATA / "circuits.csv")
    drivers = pd.read_csv(RACEDATA / "drivers.csv")
    constructors = pd.read_csv(RACEDATA / "constructors.csv")

    results["grid"] = pd.to_numeric(results["grid"], errors="coerce")
    results["position"] = pd.to_numeric(results["position"], errors="coerce")

    df = (
        results.merge(
            races[["raceId", "year", "round", "circuitId", "name"]],
            on="raceId",
            how="left",
        )
        .merge(
            circuits[["circuitId", "name", "location", "country"]],
            on="circuitId",
            how="left",
            suffixes=("_race", "_circuit"),
        )
        .merge(
            drivers[["driverId", "code", "forename", "surname"]],
            on="driverId",
            how="left",
        )
        .merge(
            constructors[["constructorId", "name"]],
            on="constructorId",
            how="left",
        )
    )
    df = df.rename(
        columns={
            "name_race": "race_name",
            "name_circuit": "circuit_official",
            "name": "constructor_name",
        }
    )
    df["circuit_label"] = [
        _friendly_location(loc, official)
        for loc, official in zip(df["location"], df["circuit_official"])
    ]
    df["driver_name"] = (
        df["forename"].fillna("").astype(str) + " " + df["surname"].fillna("").astype(str)
    ).str.strip()
    df = df[df["grid"].notna() & (df["grid"] >= 1)].copy()
    df["grid"] = df["grid"].astype(int)
    return df


@lru_cache(maxsize=1)
def load_pit_stops() -> pd.DataFrame:
    return pd.read_csv(RACEDATA / "pit_stops.csv")


@lru_cache(maxsize=1)
def load_status() -> pd.DataFrame:
    return pd.read_csv(RACEDATA / "status.csv")


@lru_cache(maxsize=1)
def load_sector_laps() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(SECTORS.glob("race_sector_laps_*.csv"))]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ("Sector1Time", "Sector2Time", "Sector3Time", "LapTimeSeconds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["event_key"] = df["EventName"].fillna(df["Location"]).astype(str)
    df["location_key"] = df["Location"].fillna(df["EventName"]).astype(str)
    return df


def circuit_options(df: pd.DataFrame | None = None) -> list[str]:
    data = df if df is not None else load_grid_results()
    # Prefer circuits with more history first in the selector? Keep alpha for scanability.
    counts = data.groupby("circuit_label")["raceId"].nunique().sort_values(ascending=False)
    return counts.index.tolist()


def sector_events_for_circuit(sector_df: pd.DataFrame, circuit_label: str, location: str) -> list[str]:
    """Map a RaceData circuit to tif1 event names via location / name overlap."""
    needles = [
        circuit_label.split("(")[0].strip().lower(),
        str(location).lower(),
    ]
    keys = []
    for event in sector_df["event_key"].dropna().unique():
        el = str(event).lower()
        if any(n and n in el for n in needles):
            keys.append(event)
    # Mexico special-case
    if "mexico" in circuit_label.lower() or "mexico" in str(location).lower():
        keys.extend([e for e in sector_df["event_key"].unique() if "mexico" in str(e).lower()])
    if "monaco" in circuit_label.lower() or "monte" in str(location).lower():
        keys.extend([e for e in sector_df["event_key"].unique() if "monaco" in str(e).lower()])
    return sorted(set(keys))
