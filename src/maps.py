"""Circuit outline helpers.

Prefer pre-built outlines in data/processed/track_outlines.json (Cloud-friendly).
Fall back to the live MultiViewer API when a circuit is missing locally.
"""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SEGMENT_COLORS = {1: "#2563eb", 2: "#ea580c", 3: "#16a34a"}
ROOT = Path(__file__).resolve().parents[1]
OUTLINES_PATH = ROOT / "data" / "processed" / "track_outlines.json"


@lru_cache(maxsize=1)
def load_local_outlines() -> dict[str, dict]:
    if not OUTLINES_PATH.exists():
        return {}
    with OUTLINES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def local_outline(circuit_label: str) -> dict | None:
    outlines = load_local_outlines()
    hit = outlines.get(circuit_label)
    if hit and hit.get("x") and hit.get("y"):
        return hit
    # fuzzy: match by location substring
    needle = _norm(circuit_label.split("(")[0])
    for key, payload in outlines.items():
        if needle and needle in _norm(key):
            return payload
    return None


@lru_cache(maxsize=1)
def list_multiviewer_circuits() -> dict:
    req = urllib.request.Request(
        "https://api.multiviewer.app/api/v1/circuits",
        headers={"User-Agent": "F1CircuitDashboard/0.1"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


@lru_cache(maxsize=64)
def fetch_circuit_outline(circuit_key: int, year: int = 2024) -> dict | None:
    # MultiViewer keeps year-specific layouts; fall back across nearby years.
    for y in range(year, year - 8, -1):
        url = f"https://api.multiviewer.app/api/v1/circuits/{circuit_key}/{y}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "F1CircuitDashboard/0.1"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
            if isinstance(payload, dict) and payload.get("x") and payload.get("y"):
                payload["_resolved_year"] = y
                return payload
        except Exception:  # noqa: BLE001
            continue
    return None


def resolve_outline(circuit_label: str, location: str, official_name: str, year: int = 2024) -> dict | None:
    """Local JSON first (deploy path), then live MultiViewer."""
    local = local_outline(circuit_label)
    if local:
        return local
    key = match_circuit_key(location, official_name)
    if key is None:
        return None
    return fetch_circuit_outline(key, year=year)


def _norm(text: str) -> str:
    return (
        str(text or "")
        .lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )


# Explicit bridges between RaceData labels and MultiViewer circuit names.
MANUAL_MATCHES = {
    "monte-carlo": "Monte Carlo",
    "monaco": "Monte Carlo",
    "mexico city": "Mexico City",
    "spa": "Spa-Francorchamps",
    "spa francorchamps": "Spa-Francorchamps",
    "sao paulo": "Interlagos",
    "são paulo": "Interlagos",
    "montreal": "Montreal",
    "yas marina island": "Yas Marina Circuit",
    "yas island": "Yas Marina Circuit",
}


def match_circuit_key(location: str, official_name: str) -> int | None:
    circuits = list_multiviewer_circuits()
    name_to_key = {str(meta.get("name", "")): int(key) for key, meta in circuits.items()}

    for candidate in (location, official_name):
        manual = MANUAL_MATCHES.get(_norm(candidate))
        if manual and manual in name_to_key:
            return name_to_key[manual]

    loc = _norm(location)
    official = _norm(official_name)
    best = None
    best_score = 0
    for key, meta in circuits.items():
        name = _norm(meta.get("name", ""))
        if not name:
            continue
        if name == loc or name == official or name in loc or loc in name or name in official:
            return int(key)
        tokens = set(name.split())
        score = len(tokens & set(official.split()))
        if score > best_score and score >= 2:
            best_score = score
            best = int(key)
    return best


def _segment_index_by_distance(x: list[float], y: list[float]) -> np.ndarray:
    pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    deltas = np.diff(pts, axis=0)
    dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(deltas, axis=1))])
    if dist[-1] <= 0:
        return np.ones(len(x), dtype=int)
    frac = dist / dist[-1]
    seg = np.ones(len(x), dtype=int)
    seg[frac >= 1 / 3] = 2
    seg[frac >= 2 / 3] = 3
    return seg


def circuit_map_figure(
    outline: dict,
    selected_segment: int,
) -> go.Figure:
    x = list(outline["x"])
    y = list(outline["y"])
    segs = _segment_index_by_distance(x, y)
    fig = go.Figure()
    # faint full outline under segments (not clickable)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
            line=dict(color="rgba(137,135,129,0.25)", width=14),
        )
    )
    for segment in (1, 2, 3):
        idx = np.where(segs == segment)[0]
        if len(idx) == 0:
            continue
        # include first point of next segment so the stroke has no gap
        nxt = np.where(segs == (segment % 3) + 1)[0]
        if segment < 3 and len(nxt):
            idx = np.concatenate([idx, nxt[:1]])
        xs = [x[i] for i in idx]
        ys = [y[i] for i in idx]
        on = selected_segment == segment
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=f"Segment {segment}",
                line=dict(color=SEGMENT_COLORS[segment], width=9 if on else 4),
                marker=dict(size=12, color="rgba(0,0,0,0.01)"),  # invisible hit targets
                opacity=1.0 if on else 0.45,
                customdata=np.full(len(xs), segment),
                hovertemplate=f"<b>Segment {segment}</b><br>click to select<extra></extra>",
            )
        )

    corners = outline.get("corners") or []
    if corners:
        fig.add_trace(
            go.Scatter(
                x=[c["trackPosition"]["x"] for c in corners],
                y=[c["trackPosition"]["y"] for c in corners],
                mode="text",
                text=[str(c.get("number", "")) for c in corners],
                textposition="top center",
                name="Corners",
                hoverinfo="skip",
                showlegend=False,
                textfont=dict(size=9, color="#898781"),
            )
        )

    fig.update_layout(
        title=(
            f"{outline.get('circuitName', 'Circuit')} · {outline.get('_resolved_year', '')} layout"
            f" · selected Segment {selected_segment}"
        ),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, fixedrange=True),
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, itemclick=False, itemdoubleclick=False),
        plot_bgcolor="#f7f7f5",
        paper_bgcolor="rgba(0,0,0,0)",
        clickmode="event+select",
        dragmode=False,
    )
    fig.update_yaxes(autorange="reversed", fixedrange=True)
    fig.update_xaxes(fixedrange=True)
    return fig
