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


def _fold(text: str) -> str:
    """Lowercase + strip accents for fuzzy circuit matching."""
    raw = str(text or "").lower().strip()
    repl = {
        "á": "a",
        "à": "a",
        "ä": "a",
        "â": "a",
        "ã": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "ô": "o",
        "ö": "o",
        "õ": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ñ": "n",
        "ç": "c",
        "ß": "ss",
        "-": " ",
        "_": " ",
        "'": "",
    }
    for a, b in repl.items():
        raw = raw.replace(a, b)
    return " ".join(raw.split())


# RaceData circuit_label / location → tif1 Location values (folded)
_LOCATION_ALIASES: dict[str, set[str]] = {
    "silverstone": {"silverstone"},
    "spa": {"spa francorchamps"},
    "spa francorchamps": {"spa francorchamps"},
    "monza": {"monza"},
    "spielberg": {"spielberg"},
    "budapest": {"budapest"},
    "montmelo": {"barcelona"},
    "barcelona": {"barcelona"},
    "baku": {"baku"},
    "austin": {"austin"},
    "montreal": {"montreal"},
    "marina bay": {"marina bay", "singapore"},
    "singapore": {"marina bay", "singapore"},
    "melbourne": {"melbourne"},
    "suzuka": {"suzuka"},
    "zandvoort": {"zandvoort"},
    "imola": {"imola"},
    "shanghai": {"shanghai"},
    "jeddah": {"jeddah"},
    "sochi": {"sochi"},
    "le castellet": {"le castellet"},
    "al daayen": {"lusail"},
    "lusail": {"lusail"},
    "portimao": {"portimao"},
    "hockenheim": {"hockenheim"},
    "istanbul": {"istanbul"},
    "nurburg": {"nurburgring"},
    "nurburgring": {"nurburgring"},
    "mugello": {"mugello"},
    "sakhir": {"sakhir"},
    "bahrain": {"sakhir"},
    "mexico city": {"mexico city"},
    "monaco": {"monaco", "monte carlo"},
    "monte carlo": {"monaco", "monte carlo"},
    "sao paulo": {"sao paulo"},
    "miami": {"miami", "miami gardens"},
    "las vegas": {"las vegas"},
    "abu dhabi": {"yas island", "yas marina"},
    "yas island": {"yas island", "yas marina"},
    "yas marina": {"yas island", "yas marina"},
}

# Extra event_key substrings when location alone is ambiguous / renamed
_EVENT_ALIASES: dict[str, set[str]] = {
    "silverstone": {"british", "70th anniversary"},
    "spa francorchamps": {"belgian"},
    "spa": {"belgian"},
    "monza": {"italian"},
    "spielberg": {"austrian", "styrian"},
    "budapest": {"hungarian"},
    "montmelo": {"spanish"},
    "barcelona": {"spanish"},
    "baku": {"azerbaijan"},
    "austin": {"united states"},
    "montreal": {"canadian"},
    "marina bay": {"singapore"},
    "singapore": {"singapore"},
    "melbourne": {"australian"},
    "suzuka": {"japanese"},
    "zandvoort": {"dutch"},
    "imola": {"emilia romagna"},
    "shanghai": {"chinese"},
    "jeddah": {"saudi"},
    "sochi": {"russian"},
    "le castellet": {"french"},
    "al daayen": {"qatar"},
    "lusail": {"qatar"},
    "portimao": {"portuguese"},
    "hockenheim": {"german"},
    "istanbul": {"turkish"},
    "nurburg": {"eifel", "german"},
    "nurburgring": {"eifel", "german"},
    "mugello": {"tuscan"},
    "sakhir": {"bahrain", "sakhir"},
    "mexico city": {"mexican", "mexico city"},
    "monaco": {"monaco"},
    "monte carlo": {"monaco"},
    "sao paulo": {"brazilian", "sao paulo"},
    "miami": {"miami"},
    "las vegas": {"las vegas"},
    "abu dhabi": {"abu dhabi"},
}


def sector_events_for_circuit(sector_df: pd.DataFrame, circuit_label: str, location: str) -> list[str]:
    """Map a RaceData circuit to tif1 event_key values via Location + aliases.

    Prefer Location match (British GP is event_key='British Grand Prix' but
    Location='Silverstone'). Event aliases are secondary and location-gated so
    'spa' never matches 'Spanish Grand Prix'.
    """
    if sector_df.empty:
        return []

    seeds = {
        _fold(circuit_label.split("(")[0]),
        _fold(location),
        _fold(circuit_label),
    }
    seeds.discard("")

    target_locations: set[str] = set()
    event_needles: set[str] = set()
    for seed in seeds:
        target_locations |= _LOCATION_ALIASES.get(seed, {seed})
        event_needles |= _EVENT_ALIASES.get(seed, set())
        for k, locs in _LOCATION_ALIASES.items():
            if seed == k or seed in locs:
                target_locations |= locs
                event_needles |= _EVENT_ALIASES.get(k, set())

    cols = [c for c in ("event_key", "Location", "EventName") if c in sector_df.columns]
    pairs = sector_df[cols].drop_duplicates()
    keys: list[str] = []
    for row in pairs.itertuples(index=False):
        event = str(getattr(row, "event_key", "") or "")
        if not event:
            continue
        loc = _fold(getattr(row, "Location", "") or "")
        ename = _fold(getattr(row, "EventName", "") or event)

        loc_hit = False
        for t in target_locations:
            if not t:
                continue
            if loc == t or (len(t) >= 4 and (t in loc or loc in t)):
                loc_hit = True
                break

        if loc_hit:
            keys.append(event)
            continue

        event_hit = any(n and n in ename for n in event_needles)
        if event_hit and any(seed and len(seed) >= 4 and seed in loc for seed in seeds):
            keys.append(event)

    return sorted(set(keys))
