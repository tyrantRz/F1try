"""Metric helpers aligned with the F1 Circuit Explorer prototype."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_LABELS = {
    "p1_pct": "P1 % (first-place finish rate)",
    "podium_pct": "Podium % (P1–P3)",
    "avg_finish": "Average finish position",
}


def compute_grid_metrics(
    df: pd.DataFrame,
    year_min: int,
    year_max: int,
    circuit_label: str | None = None,
    max_grid: int = 20,
) -> pd.DataFrame:
    subset = df[(df["year"] >= year_min) & (df["year"] <= year_max) & (df["grid"] <= max_grid)]
    if circuit_label:
        subset = subset[subset["circuit_label"] == circuit_label]
    grouped = subset.groupby(["circuit_label", "grid"], as_index=False).agg(
        n_starts=("raceId", "count"),
        n_classified=("position", lambda s: int(s.notna().sum())),
        n_p1=("position", lambda s: int((s == 1).sum())),
        n_podium=("position", lambda s: int(s.isin([1, 2, 3]).sum())),
        avg_finish=("position", "mean"),
    )
    grouped["p1_pct"] = grouped["n_p1"] / grouped["n_starts"] * 100
    grouped["podium_pct"] = grouped["n_podium"] / grouped["n_starts"] * 100
    return grouped


def circuit_summary(df: pd.DataFrame, year_min: int, year_max: int, circuit_label: str) -> dict:
    subset = df[
        (df["circuit_label"] == circuit_label)
        & (df["year"] >= year_min)
        & (df["year"] <= year_max)
    ].copy()
    races = subset["raceId"].nunique()
    pole = subset[subset["grid"] == 1]
    pole_starts = len(pole)
    pole_wins = int((pole["position"] == 1).sum())
    winners = subset[subset["position"] == 1]
    winners_top3 = int((winners["grid"] <= 3).sum()) if len(winners) else 0

    # Grid-finish Spearman rho per race, then average.
    rhos = []
    for _, race in subset.groupby("raceId"):
        classified = race.dropna(subset=["position"])
        if len(classified) < 5:
            continue
        rhos.append(classified["grid"].corr(classified["position"], method="spearman"))
    rho = float(np.nanmean(rhos)) if rhos else np.nan

    dnf_mask = subset["position"].isna()
    dnf_per_race = float(dnf_mask.sum() / races) if races else np.nan

    return {
        "circuit_label": circuit_label,
        "n_races": int(races),
        "pole_starts": int(pole_starts),
        "pole_wins": int(pole_wins),
        "pole_win_pct": (pole_wins / pole_starts * 100) if pole_starts else np.nan,
        "winners": int(len(winners)),
        "winners_top3": winners_top3,
        "grid_finish_rho": rho,
        "dnf_per_race": dnf_per_race,
    }


def all_circuit_summaries(df: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    rows = []
    for circuit in sorted(df["circuit_label"].dropna().unique()):
        rows.append(circuit_summary(df, year_min, year_max, circuit))
    return pd.DataFrame(rows)


def sector_times(
    df: pd.DataFrame,
    event_keys: list[str],
    segment: int,
    driver: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    col = {1: "Sector1Time", 2: "Sector2Time", 3: "Sector3Time"}[segment]
    event = df[df["event_key"].isin(event_keys)].copy()
    # Prefer clean-ish laps: drop deleted / inaccurate when available.
    if "Deleted" in event.columns:
        event = event[event["Deleted"].fillna(False) == False]  # noqa: E712
    field = pd.to_numeric(event[col], errors="coerce")
    field = field[(field > 5) & (field < 120)].dropna()
    if driver:
        drv = pd.to_numeric(event.loc[event["Driver"] == driver, col], errors="coerce")
        drv = drv[(drv > 5) & (drv < 120)].dropna()
    else:
        drv = pd.Series(dtype=float)
    return field, drv


def fastest_by_driver(
    df: pd.DataFrame,
    event_keys: list[str],
    segment: int,
    top_n: int = 10,
) -> pd.DataFrame:
    col = {1: "Sector1Time", 2: "Sector2Time", 3: "Sector3Time"}[segment]
    event = df[df["event_key"].isin(event_keys)].copy()
    event[col] = pd.to_numeric(event[col], errors="coerce")
    event = event[(event[col] > 5) & (event[col] < 120)]
    return (
        event.groupby("Driver", as_index=False)[col]
        .min()
        .rename(columns={col: "best_sector_s"})
        .sort_values("best_sector_s")
        .head(top_n)
    )


def pit_stop_profile(
    results: pd.DataFrame,
    pit_stops: pd.DataFrame,
    year_min: int,
    year_max: int,
    circuit_label: str,
) -> pd.DataFrame:
    race_ids = results[
        (results["circuit_label"] == circuit_label)
        & (results["year"] >= year_min)
        & (results["year"] <= year_max)
    ]["raceId"].unique()
    pits = pit_stops[pit_stops["raceId"].isin(race_ids)].copy()
    if pits.empty:
        return pits
    pits["lap"] = pd.to_numeric(pits["lap"], errors="coerce")
    return pits.dropna(subset=["lap"])


def retirement_causes(
    results: pd.DataFrame,
    status: pd.DataFrame,
    year_min: int,
    year_max: int,
    circuit_label: str,
) -> pd.DataFrame:
    subset = results[
        (results["circuit_label"] == circuit_label)
        & (results["year"] >= year_min)
        & (results["year"] <= year_max)
        & (results["position"].isna())
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=["bucket", "count"])
    merged = subset.merge(status, on="statusId", how="left")
    accident_kw = ("accident", "collision", "spun", "crash")
    mechanical_kw = (
        "engine",
        "gear",
        "transmission",
        "brake",
        "hydraulic",
        "electrical",
        "fuel",
        "wheel",
        "suspension",
        "clutch",
        "overheating",
        "exhaust",
        "oil",
        "turbo",
        "power",
        "ers",
        "battery",
    )

    def bucket(status_text: str) -> str:
        s = str(status_text).lower()
        if any(k in s for k in accident_kw):
            return "Accident / collision"
        if any(k in s for k in mechanical_kw):
            return "Mechanical"
        return "Other / not recorded"

    merged["bucket"] = merged["status"].map(bucket)
    return (
        merged.groupby("bucket", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
