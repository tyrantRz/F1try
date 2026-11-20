"""
F1 Circuit Explorer — Streamlit prototype.

Layout and chart encodings follow:
C:\\Users\\DELL\\Downloads\\F1_Combined_Dashboard_Mock.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import sector_events_for_circuit  # noqa: E402
from src.maps import circuit_map_figure, resolve_outline  # noqa: E402

PROCESSED = ROOT / "data" / "processed"

# Colors aligned with the HTML mock CSS variables
SEQ_LO = "#cde2fb"
SEQ_HI = "#0d366b"
S1 = "#2a78d6"
S1_SOFT = "#9ec5f4"
S2 = "#eb6834"
SOFT, MEDIUM, HARD, INTER = "#e34948", "#eda100", "#898781", "#008300"
TYRE_COLORS = {"SOFT": SOFT, "MEDIUM": MEDIUM, "HARD": HARD, "INTERMEDIATE": INTER, "WET": "#1d4ed8"}
SMALL_CIRCUITS = [
    "Monaco (Monte-Carlo)",
    "Monza",
    "Spa-Francorchamps",
    "Suzuka",
]

st.set_page_config(page_title="F1 Circuit Explorer", page_icon="🏎️", layout="wide")

# No pan/zoom — keep click selection only (same idea as Streamlit-proto)
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": False,
}


def lock_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(dragmode=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def show_chart(fig: go.Figure, **kwargs):
    """Render a Plotly chart with drag/zoom disabled."""
    return st.plotly_chart(
        lock_figure(fig),
        use_container_width=True,
        config=PLOTLY_CONFIG,
        **kwargs,
    )


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    path = PROCESSED / "results_lean.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # fallback to CSV loader path
    from src.data_loader import load_grid_results

    return load_grid_results()


@st.cache_data(show_spinner=False)
def load_sectors() -> pd.DataFrame:
    path = PROCESSED / "sectors_lean.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from src.data_loader import load_sector_laps

    return load_sector_laps()


@st.cache_data(show_spinner=False)
def load_pits() -> pd.DataFrame:
    path = PROCESSED / "pits_lean.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_status() -> pd.DataFrame:
    path = PROCESSED / "status.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(ROOT / "data" / "raw" / "racedata" / "status.csv")


def metric_value(row: pd.Series, metric: str) -> float:
    if metric == "finish":
        return row["avg_finish"]
    if metric == "podium":
        return row["podium_pct"]
    return row["p1_pct"]


def fmt_metric(v: float, metric: str) -> str:
    if pd.isna(v):
        return "—"
    if metric == "finish":
        return f"P{v:.1f}"
    return f"{v:.0f}%"


def grid_metrics(df: pd.DataFrame, y0: int, y1: int, max_grid: int = 10) -> pd.DataFrame:
    subset = df[(df["year"] >= y0) & (df["year"] <= y1) & (df["grid"] <= max_grid)]
    g = subset.groupby(["circuit_label", "grid"], as_index=False).agg(
        n_starts=("raceId", "count"),
        n_p1=("position", lambda s: int((s == 1).sum())),
        n_podium=("position", lambda s: int(s.isin([1, 2, 3]).sum())),
        avg_finish=("position", "mean"),
    )
    g["p1_pct"] = g["n_p1"] / g["n_starts"] * 100
    g["podium_pct"] = g["n_podium"] / g["n_starts"] * 100
    return g


def accident_count(df: pd.DataFrame, status: pd.DataFrame, circuit: str, y0: int, y1: int) -> int:
    sub = df[(df["circuit_label"] == circuit) & (df["year"] >= y0) & (df["year"] <= y1)]
    merged = sub.merge(status, on="statusId", how="left")
    text = merged["status"].fillna("").str.lower()
    mask = text.str.contains("accident|collision|spun", regex=True)
    return int(mask.sum())


def _norm_team(name: str) -> str:
    t = str(name or "").lower().strip()
    for junk in ("f1 team", "racing", "team"):
        t = t.replace(junk, " ")
    return " ".join(t.split())


# Map RaceData constructor_name ↔ tif1 Team labels
_TEAM_CANON = {
    "red bull": "red bull",
    "alpine": "alpine",
    "alfa romeo": "alfa romeo",
    "sauber": "sauber",
    "kick sauber": "sauber",
    "rb": "rb",
    "racing bulls": "rb",
    "alphatauri": "rb",
    "toro rosso": "rb",
    "haas": "haas",
}


def team_canon(name: str) -> str:
    n = _norm_team(name)
    return _TEAM_CANON.get(n, n)


def teams_compatible(a: str, b: str) -> bool:
    ca, cb = team_canon(a), team_canon(b)
    return ca == cb or ca in cb or cb in ca


def filter_sectors_by_team(sectors: pd.DataFrame, team: str | None) -> pd.DataFrame:
    if sectors.empty or not team or team == "All" or "Team" not in sectors.columns:
        return sectors
    mask = sectors["Team"].astype(str).map(lambda t: teams_compatible(team, t))
    return sectors.loc[mask].copy()


def filter_sectors(
    sectors: pd.DataFrame,
    event_keys: list[str],
    team: str | None = None,
    y0: int | None = None,
    y1: int | None = None,
) -> pd.DataFrame:
    """Apply circuit event match + optional year window + team filter."""
    if sectors.empty or not event_keys:
        return sectors.iloc[0:0].copy()
    sub = sectors[sectors["event_key"].isin(event_keys)].copy()
    if y0 is not None and "Year" in sub.columns:
        sub = sub[sub["Year"] >= y0]
    if y1 is not None and "Year" in sub.columns:
        sub = sub[sub["Year"] <= y1]
    return filter_sectors_by_team(sub, team)


def attach_pit_teams(pits: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Join constructor_name onto pit stops (processed pits lack team)."""
    if pits.empty:
        return pits
    if "constructor_name" in pits.columns:
        return pits
    key = results[["raceId", "driverId", "constructor_name"]].drop_duplicates()
    return pits.merge(key, on=["raceId", "driverId"], how="left")


def avg_pit_stops(pits: pd.DataFrame, circuit: str, y0: int, y1: int, team: str | None) -> float:
    if pits.empty:
        return np.nan
    sub = pits[(pits["circuit_label"] == circuit) & (pits["year"] >= y0) & (pits["year"] <= y1)]
    if team and team != "All" and "constructor_name" in sub.columns:
        sub = sub[sub["constructor_name"].astype(str).map(lambda t: teams_compatible(team, t))]
    if sub.empty:
        return np.nan
    counts = sub.groupby(["raceId", "driverId"]).size()
    return float(counts.mean())


def heatmap_figure(metrics: pd.DataFrame, circuits: list[str], metric: str, sel_circuit: str, sel_grid: int) -> go.Figure:
    """
    Scatter-square heatmap.

    Streamlit's plotly on_select does NOT return points for go.Heatmap, so each
    cell is a clickable square marker instead (same visual encoding as the mock).
    """
    grids = list(range(1, 11))
    xs, ys, colors, labels, customs, sizes, opacities = [], [], [], [], [], [], []

    vals = []
    for c in circuits:
        for g in grids:
            hit = metrics[(metrics["circuit_label"] == c) & (metrics["grid"] == g)]
            if not hit.empty:
                vals.append(metric_value(hit.iloc[0], metric))
    vmin = float(np.nanmin(vals)) if vals else 0.0
    vmax = float(np.nanmax(vals)) if vals else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    def _color(v: float) -> str:
        t = (v - vmin) / (vmax - vmin)
        if metric == "finish":
            t = 1.0 - t  # darker = better (lower finish)
        # interpolate SEQ_LO -> SEQ_HI
        lo = np.array([0xCD, 0xE2, 0xFB], dtype=float)
        hi = np.array([0x0D, 0x36, 0x6B], dtype=float)
        rgb = (lo + t * (hi - lo)).astype(int)
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    for c in circuits:
        for g in grids:
            hit = metrics[(metrics["circuit_label"] == c) & (metrics["grid"] == g)]
            xs.append(f"P{g}")
            ys.append(c)
            selected = c == sel_circuit and g == sel_grid
            sizes.append(42 if selected else 36)
            opacities.append(1.0 if selected else 0.92)
            if hit.empty:
                colors.append("#e1e0d9")
                labels.append("")
                customs.append([c, g, None, None, None, 0])
            else:
                r = hit.iloc[0]
                v = metric_value(r, metric)
                colors.append(_color(float(v)))
                labels.append(fmt_metric(v, metric))
                customs.append(
                    [
                        c,
                        int(g),
                        float(r["p1_pct"]),
                        float(r["podium_pct"]),
                        float(r["avg_finish"]) if pd.notna(r["avg_finish"]) else None,
                        int(r["n_starts"]),
                    ]
                )

    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=labels,
            textfont=dict(size=11, color="#0b0b0b"),
            marker=dict(
                symbol="square",
                size=sizes,
                color=colors,
                opacity=opacities,
                line=dict(
                    width=[3 if (c == sel_circuit and g == sel_grid) else 1 for c, g in zip(ys, [int(x[1:]) for x in xs])],
                    color=["#0b0b0b" if (c == sel_circuit and g == sel_grid) else "#ffffff" for c, g in zip(ys, [int(x[1:]) for x in xs])],
                ),
            ),
            customdata=customs,
            hovertemplate=(
                "<b>%{customdata[0]} · P%{customdata[1]}</b><br>"
                "Win %: %{customdata[2]:.0f}%<br>"
                "Podium %: %{customdata[3]:.0f}%<br>"
                "Avg finish: P%{customdata[4]:.1f}<br>"
                "Races (n): %{customdata[5]}<br>"
                "<extra>click to select</extra>"
            ),
        )
    )
    title = {
        "win": "Win probability heatmap",
        "podium": "Podium probability heatmap",
        "finish": "Average finish heatmap",
    }[metric]
    fig.update_layout(
        title=dict(
            text=(
                f"{title} · darker = higher · click a cell"
                if metric != "finish"
                else f"{title} · darker = better · click a cell"
            ),
            font=dict(size=14),
        ),
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(
            title="Starting grid position →",
            categoryorder="array",
            categoryarray=[f"P{g}" for g in grids],
            side="bottom",
        ),
        yaxis=dict(
            title=None,
            categoryorder="array",
            categoryarray=list(reversed(circuits)),
        ),
        clickmode="event+select",
        dragmode=False,
        plot_bgcolor="#fcfcfb",
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def driver_options(
    results: pd.DataFrame,
    sectors: pd.DataFrame,
    circuit: str,
    event_keys: list[str],
    team: str | None = None,
) -> list[str]:
    """Drivers for the filter: sector codes first, then race-result codes as fallback."""
    res = results[results["circuit_label"] == circuit]
    if team and team != "All" and "constructor_name" in res.columns:
        res = res[res["constructor_name"].astype(str).map(lambda t: teams_compatible(team, t))]

    from_sectors: list[str] = []
    sec = filter_sectors_by_team(
        sectors[sectors["event_key"].isin(event_keys)] if event_keys else sectors.iloc[0:0],
        team,
    )
    if not sec.empty and "Driver" in sec.columns:
        from_sectors = sec["Driver"].dropna().astype(str).unique().tolist()

    from_results: list[str] = []
    if "code" in res.columns:
        from_results = (
            res["code"].dropna().astype(str).unique().tolist()
        )
        from_results = [c for c in from_results if c and c != "\\N"]
    merged = sorted(set(from_sectors) | set(from_results))
    return ["All"] + merged


def _chart_points(chart_key: str) -> list[dict]:
    """Points from a plotly chart selection stored under chart_key (proto pattern)."""
    sel = st.session_state.get(chart_key)
    if sel is None:
        return []
    if isinstance(sel, dict):
        return ((sel.get("selection") or {}).get("points")) or []
    selection = getattr(sel, "selection", None)
    if selection is None:
        return []
    if isinstance(selection, dict):
        return selection.get("points") or []
    return getattr(selection, "points", None) or []


def _customdata0(pt: dict):
    cd = pt.get("customdata")
    if cd is None:
        return None
    if isinstance(cd, (list, tuple)):
        return cd[0] if cd else None
    return cd


def on_heatmap_click() -> None:
    """Callback: click a heatmap cell → update circuit + grid (runs before widgets)."""
    pts = _chart_points(f"heatmap_chart_{st.session_state.get('nonce', 0)}")
    if not pts:
        return
    pt = pts[0]
    cd = pt.get("customdata")
    circuit = None
    grid = None
    if isinstance(cd, (list, tuple)) and len(cd) >= 2:
        circuit = str(cd[0])
        try:
            grid = int(cd[1])
        except (TypeError, ValueError):
            grid = None
    if circuit is None and pt.get("y") is not None:
        circuit = str(pt.get("y"))
    if grid is None and pt.get("x") is not None:
        try:
            grid = int(str(pt.get("x")).replace("P", "").strip())
        except ValueError:
            grid = None
    if circuit is not None:
        st.session_state.circuit = circuit
    if grid is not None:
        st.session_state.sel_grid = int(grid)


def on_grid_point_click(chart_prefix: str) -> None:
    """Callback for small-multiples / grid-vs bars with [circuit, grid, ...] customdata."""
    pts = _chart_points(f"{chart_prefix}_{st.session_state.get('nonce', 0)}")
    if not pts:
        return
    pt = pts[0]
    cd = pt.get("customdata")
    if isinstance(cd, (list, tuple)) and len(cd) >= 2:
        try:
            st.session_state.circuit = str(cd[0])
            st.session_state.sel_grid = int(cd[1])
            return
        except (TypeError, ValueError):
            pass
    raw_x = pt.get("x")
    if raw_x is not None:
        try:
            st.session_state.sel_grid = int(str(raw_x).replace("P", "").strip())
        except ValueError:
            pass


def on_trackmap_click() -> None:
    pts = _chart_points(f"trackmap_chart_{st.session_state.get('nonce', 0)}")
    if not pts:
        return
    pt = pts[0]
    try:
        seg = int(_customdata0(pt))
        if seg in (1, 2, 3):
            st.session_state.sel_seg = seg
            return
    except (TypeError, ValueError):
        pass
    # curve 0 = faint outline; 1/2/3 = segments
    curve = pt.get("curve_number")
    if curve in (1, 2, 3):
        st.session_state.sel_seg = int(curve)


def on_fastest_click() -> None:
    pts = _chart_points(f"fastest_chart_{st.session_state.get('nonce', 0)}")
    if not pts:
        return
    clicked = pts[0].get("y")
    if clicked:
        st.session_state.driver = str(clicked)


def on_small_multiples_click() -> None:
    on_grid_point_click("small_multiples_chart")


def on_gridvs_click() -> None:
    on_grid_point_click("gridvs_chart")


def small_multiples(metrics: pd.DataFrame, metric: str, selected: str, sel_grid: int) -> go.Figure:
    """Mock encoding: mini horizontal bars for P1–P5 per circuit."""
    from plotly.subplots import make_subplots

    circuits = [c for c in SMALL_CIRCUITS if c in set(metrics["circuit_label"])]
    fig = make_subplots(rows=2, cols=2, subplot_titles=circuits, horizontal_spacing=0.12, vertical_spacing=0.18)
    for i, c in enumerate(circuits[:4]):
        r, col = divmod(i, 2)
        rows = metrics[(metrics["circuit_label"] == c) & (metrics["grid"] <= 5)].sort_values("grid")
        custom = [
            [
                c,
                int(rr["grid"]),
                float(rr["p1_pct"]),
                float(rr["podium_pct"]),
                float(rr["avg_finish"]) if pd.notna(rr["avg_finish"]) else None,
                int(rr["n_starts"]),
            ]
            for _, rr in rows.iterrows()
        ]
        fig.add_trace(
            go.Bar(
                y=[f"P{int(g)}" for g in rows["grid"]],
                x=[metric_value(rr, metric) for _, rr in rows.iterrows()],
                orientation="h",
                marker_color=[S2 if (c == selected and int(g) == sel_grid) else S1 for g in rows["grid"]],
                text=[fmt_metric(metric_value(rr, metric), metric) for _, rr in rows.iterrows()],
                textposition="outside",
                showlegend=False,
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]} · P%{customdata[1]}</b><br>"
                    "Win %: %{customdata[2]:.0f}%<br>"
                    "Podium %: %{customdata[3]:.0f}%<br>"
                    "Avg finish: P%{customdata[4]:.1f}<br>"
                    "Races (n): %{customdata[5]}<br>"
                    "<extra>click to select</extra>"
                ),
            ),
            row=r + 1,
            col=col + 1,
        )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        title_text="Small multiples by circuit · P1–P5 · click a bar",
        clickmode="event+select",
        dragmode=False,
    )
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def grid_vs_figure(metrics: pd.DataFrame, circuit: str, metric: str, sel_grid: int) -> go.Figure:
    rows = metrics[metrics["circuit_label"] == circuit].sort_values("grid")
    all_avg = (
        metrics.groupby("grid")
        .apply(lambda d: np.nanmean([metric_value(r, metric) for _, r in d.iterrows()]), include_groups=False)
        .reindex(range(1, 11))
    )
    vals = [metric_value(r, metric) for _, r in rows.iterrows()]
    grids = [int(g) for g in rows["grid"].tolist()]
    colors = [S1 if g == sel_grid else S1_SOFT for g in grids]
    custom = []
    for _, r in rows.iterrows():
        g = int(r["grid"])
        avg_all = all_avg.get(g, np.nan)
        custom.append(
            [
                circuit,
                g,
                float(r["p1_pct"]),
                float(r["podium_pct"]),
                float(r["avg_finish"]) if pd.notna(r["avg_finish"]) else None,
                int(r["n_starts"]),
                float(avg_all) if pd.notna(avg_all) else None,
            ]
        )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"P{g}" for g in grids],
            y=vals,
            marker_color=colors,
            name=circuit,
            customdata=custom,
            text=[fmt_metric(v, metric) if g == sel_grid else "" for g, v in zip(grids, vals)],
            textposition="outside",
            hovertemplate=(
                "<b>%{customdata[0]} · P%{customdata[1]}</b><br>"
                "Win %: %{customdata[2]:.0f}%<br>"
                "Podium %: %{customdata[3]:.0f}%<br>"
                "Avg finish: P%{customdata[4]:.1f}<br>"
                "Races (n): %{customdata[5]}<br>"
                "All-circuit avg: %{customdata[6]:.1f}<br>"
                "<extra>click to select grid</extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[f"P{int(g)}" for g in all_avg.index],
            y=all_avg.values,
            mode="lines",
            name="All-circuit average",
            line=dict(color="#52514e", width=2, dash="dash"),
            hovertemplate="All-circuit avg · %{x}: %{y:.1f}<extra></extra>",
        )
    )
    label = {"win": "win probability", "podium": "podium probability", "finish": "avg finish"}[metric]
    fig.update_layout(
        title=f"Grid position vs {label} · {circuit}",
        height=240,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.15),
        yaxis_title=None,
        clickmode="event+select",
        dragmode=False,
    )
    return fig


def fastest_bars(
    sectors: pd.DataFrame,
    event_keys: list[str],
    seg: int,
    driver: str,
    team: str | None = None,
    y0: int | None = None,
    y1: int | None = None,
) -> go.Figure:
    col = {1: "Sector1Time", 2: "Sector2Time", 3: "Sector3Time"}[seg]
    sub = filter_sectors(sectors, event_keys, team=team, y0=y0, y1=y1)
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub[(sub[col] > 5) & (sub[col] < 120)]
    best = sub.groupby("Driver", as_index=False)[col].min().sort_values(col).head(8)
    if best.empty:
        return go.Figure()
    fastest = float(best[col].iloc[0])
    fig = go.Figure(
        go.Bar(
            y=best["Driver"].astype(str).tolist(),
            x=best[col].tolist(),
            orientation="h",
            marker_color=[S2 if d == driver else S1 for d in best["Driver"]],
            text=[f"{v:.3f}s" for v in best[col]],
            textposition="outside",
            customdata=[[float(v - fastest)] for v in best[col]],
            hovertemplate=(
                "<b>%{y}</b> · Segment " + str(seg) + "<br>"
                "Best time: %{x:.3f}s<br>"
                "Gap to fastest: +%{customdata[0]:.3f}s<br>"
                "<extra>click to select driver</extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Fastest record by drivers · Segment {seg}",
        height=280,
        margin=dict(l=10, r=50, t=40, b=10),
        yaxis=dict(autorange="reversed", title="Driver"),
        xaxis_title="seconds",
        clickmode="event+select",
        dragmode=False,
    )
    return fig


def dist_figure(
    sectors: pd.DataFrame,
    event_keys: list[str],
    seg: int,
    driver: str,
    team: str | None = None,
    y0: int | None = None,
    y1: int | None = None,
) -> go.Figure:
    col = {1: "Sector1Time", 2: "Sector2Time", 3: "Sector3Time"}[seg]
    sub = filter_sectors(sectors, event_keys, team=team, y0=y0, y1=y1)
    if "Deleted" in sub.columns:
        sub = sub[sub["Deleted"].fillna(False) == False]  # noqa: E712
    field = pd.to_numeric(sub[col], errors="coerce")
    field = field[(field > 5) & (field < 120)].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=field, nbinsx=24, name="Overall history", marker_color=S1_SOFT, opacity=0.85))
    if driver != "All":
        drv = pd.to_numeric(sub.loc[sub["Driver"] == driver, col], errors="coerce")
        drv = drv[(drv > 5) & (drv < 120)].dropna()
        fig.add_trace(go.Histogram(x=drv, nbinsx=24, name=f"{driver} specific", marker_color=S2, opacity=0.65))
    title = "Segment time distribution" + (f" · {team}" if team and team != "All" else "")
    fig.update_layout(
        barmode="overlay",
        title=title,
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="segment time (s) →",
        yaxis_title="# laps",
        legend=dict(orientation="h", y=1.15),
    )
    return fig


def line_by_year(x, y, title: str, color=S1, name="series") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7),
            name=name,
        )
    )
    fig.update_layout(title=title, height=220, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def strategy_tyre_laps(
    sectors: pd.DataFrame,
    event_keys: list[str],
    team: str | None = None,
    y0: int | None = None,
    y1: int | None = None,
) -> go.Figure:
    sub = filter_sectors(sectors, event_keys, team=team, y0=y0, y1=y1)
    sub["Compound"] = sub["Compound"].fillna("UNKNOWN").str.upper()
    counts = sub["Compound"].value_counts()
    order = [c for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"] if c in counts.index]
    fig = go.Figure(
        go.Bar(
            x=order,
            y=[int(counts[c]) for c in order],
            marker_color=[TYRE_COLORS.get(c, "#898781") for c in order],
            text=[str(int(counts[c])) for c in order],
            textposition="outside",
        )
    )
    fig.update_layout(title="Laps by tyre type", height=250, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="# of laps")
    return fig


def strategy_pit_dist(
    pits: pd.DataFrame, circuit: str, y0: int, y1: int, team: str | None = None
) -> go.Figure:
    sub = pits[(pits["circuit_label"] == circuit) & (pits["year"] >= y0) & (pits["year"] <= y1)]
    if team and team != "All" and "constructor_name" in sub.columns:
        sub = sub[sub["constructor_name"].astype(str).map(lambda t: teams_compatible(team, t))]
    if sub.empty:
        return go.Figure()
    stops = sub.groupby(["raceId", "driverId"]).size()
    freq = stops.value_counts(normalize=True).reindex(range(0, 5), fill_value=0)
    fig = go.Figure(go.Bar(x=list(freq.index), y=freq.values, marker_color=S1_SOFT))
    fig.update_layout(
        title="Pit stop distribution",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="# of pit stops",
        yaxis_title="freq.",
        yaxis_tickformat=".0%",
    )
    return fig


def strategy_tyre_comp(
    sectors: pd.DataFrame,
    event_keys: list[str],
    team: str | None = None,
    y0: int | None = None,
    y1: int | None = None,
) -> go.Figure:
    sub = filter_sectors(sectors, event_keys, team=team, y0=y0, y1=y1)
    if sub.empty:
        return go.Figure()
    sub["Compound"] = sub["Compound"].fillna("UNKNOWN").str.upper()
    counts = sub.groupby(["Year", "Compound"]).size().unstack(fill_value=0)
    year_total = counts.sum(axis=1).replace(0, np.nan)
    pivot = counts.div(year_total, axis=0).fillna(0)
    years = [str(int(y)) for y in pivot.index.tolist()]
    fig = go.Figure()
    for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        if comp in pivot.columns:
            fig.add_trace(
                go.Bar(
                    x=years,
                    y=pivot[comp].tolist(),
                    name=comp.title(),
                    marker_color=TYRE_COLORS.get(comp, "#898781"),
                )
            )
    fig.update_layout(
        barmode="stack",
        title="Tyre composition over time",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=1.15),
    )
    return fig


def strategy_avg_pits(
    pits: pd.DataFrame, circuit: str, y0: int, y1: int, team: str | None = None
) -> go.Figure:
    sub = pits[(pits["circuit_label"] == circuit) & (pits["year"] >= y0) & (pits["year"] <= y1)]
    if team and team != "All" and "constructor_name" in sub.columns:
        sub = sub[sub["constructor_name"].astype(str).map(lambda t: teams_compatible(team, t))]
    if sub.empty:
        return go.Figure()
    by_year = sub.groupby(["year", "raceId", "driverId"]).size().groupby("year").mean()
    label = team if team and team != "All" else "All teams"
    return line_by_year(by_year.index, by_year.values, "Avg pit stops over time", color=S1, name=label)


def main() -> None:
    results = load_results()
    sectors = load_sectors()
    pits = load_pits()
    status = load_status()

    circuits = (
        results.groupby("circuit_label")["raceId"].nunique().sort_values(ascending=False).index.tolist()
    )
    default_circuit = next((c for c in circuits if "Mexico" in c), circuits[0])
    y_min, y_max = int(results["year"].min()), int(results["year"].max())

    # ---------- Header / global filters (mock) ----------
    st.markdown("## F1 Circuit Explorer")
    st.caption("Grid Position · Segment Analysis · Strategy · click charts to select · hover for details")

    defaults = {
        "circuit": default_circuit,
        "sel_grid": 3,
        "sel_seg": 1,
        "driver": "All",
        "team": "All",
        "nonce": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    nonce = int(st.session_state.nonce)

    f1, f2, f3, f4, f5, f6 = st.columns([1.35, 1.15, 0.95, 1.05, 1.05, 0.65])
    with f1:
        circuit = st.selectbox("Circuit", circuits, key="circuit")
    with f2:
        # Prefer a completed-season default; 2026 rows exist in RaceData but have no sector laps yet.
        year_lo = max(2014, y_min)
        year_hi = y_max
        default_hi = min(2025, year_hi) if year_hi >= 2018 else year_hi
        default_lo = min(2018, default_hi) if default_hi >= 2018 else year_lo
        if "years" in st.session_state:
            y_a, y_b = st.session_state.years
            # Clamp stale single-year-at-max selections that wipe Segment charts
            if y_a == y_b == year_hi and year_hi > default_hi:
                st.session_state.years = (default_lo, default_hi)
        years = st.slider(
            "Years",
            min_value=year_lo,
            max_value=year_hi,
            value=(default_lo, default_hi),
            key="years",
        )
    with f3:
        metric = st.selectbox(
            "Metric",
            options=["win", "podium", "finish"],
            format_func=lambda m: {"win": "Win %", "podium": "Podium %", "finish": "Avg finish"}[m],
            key="metric",
        )

    # Team first so Driver options can narrow to that team
    teams = ["All"] + sorted(
        results.loc[results["circuit_label"] == circuit, "constructor_name"].dropna().astype(str).unique().tolist()
    )
    if st.session_state.team not in teams:
        st.session_state.team = "All"
    with f5:
        team = st.selectbox("Team", teams, key="team")

    event_keys = sector_events_for_circuit(
        sectors,
        circuit,
        str(results.loc[results["circuit_label"] == circuit, "location"].iloc[0]),
    )
    drivers = driver_options(results, sectors, circuit, event_keys, team=team)
    if st.session_state.driver not in drivers:
        st.session_state.driver = "All"

    with f4:
        driver = st.selectbox("Driver", drivers, key="driver")
        st.caption(f"{len(drivers) - 1} drivers" + (f" · {team}" if team != "All" else ""))
    with f6:
        st.write("")
        if st.button("Reset ↺", use_container_width=True):
            next_nonce = nonce + 1
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.nonce = next_nonce
            st.rerun()

    pits = attach_pit_teams(pits, results)
    y0, y1 = years
    metrics = grid_metrics(results, y0, y1, max_grid=10)
    team_arg = None if team == "All" else team
    heat_circuits = [c for c in circuits if c in set(metrics["circuit_label"])][:6]
    for c in [circuit, *SMALL_CIRCUITS]:
        if c in set(metrics["circuit_label"]) and c not in heat_circuits:
            heat_circuits.append(c)
    heat_circuits = heat_circuits[:8]

    # ---------- KPI row (mock: 3 cards) ----------
    crashes = accident_count(results, status, circuit, y0, y1)
    avg_stops = avg_pit_stops(pits, circuit, y0, y1, team_arg)
    k1, k2, k3 = st.columns(3)
    k1.metric("# Crashes", crashes, help="Accident/collision/spun statuses at this circuit in selected years")
    k2.metric("Avg pit stops", f"{avg_stops:.2f}" if pd.notna(avg_stops) else "—", help="Mean stops per driver-race")
    k3.metric(
        "Overtakes / race",
        "n/a",
        help="Mock chart exists, but public historical overtake-by-segment data is not in RaceData/tif1.",
    )

    st.divider()

    # =====================================================
    # 1. Grid Position Explorer
    # =====================================================
    st.markdown("### 1 · Grid Position Explorer")
    st.caption("Hover a cell for Win % / Podium % / Avg finish / n · click a cell to select circuit + grid")

    gtop_l, gtop_r = st.columns([1.85, 1])
    with gtop_l:
        show_chart(
            heatmap_figure(metrics, heat_circuits, metric, circuit, int(st.session_state.sel_grid)),
            on_select=on_heatmap_click,
            selection_mode="points",
            key=f"heatmap_chart_{nonce}",
        )
        sel_grid = st.slider("Selected grid position", 1, 10, key="sel_grid")
        circuit = st.session_state.circuit
    with gtop_r:
        hit = metrics[(metrics["circuit_label"] == circuit) & (metrics["grid"] == sel_grid)]
        st.markdown("**Selected info**")
        st.markdown(f"### {circuit} · P{sel_grid}")
        if hit.empty:
            st.write("No starts for this slot / years.")
        else:
            row = hit.iloc[0]
            c1, c2 = st.columns(2)
            c1.write(f"Win %\n\n**{row['p1_pct']:.0f}%**")
            c2.write(f"Podium %\n\n**{row['podium_pct']:.0f}%**")
            c1.write(
                f"Avg finish\n\n**P{row['avg_finish']:.1f}**"
                if pd.notna(row["avg_finish"])
                else "Avg finish\n\n**—**"
            )
            c2.write(f"Races (n)\n\n**{int(row['n_starts'])}**")
            st.caption(
                f"Small sample caution when n is low (n={int(row['n_starts'])}). "
                "DNFs excluded from average finish. Weather, crashes, safety cars and strategy also affect outcomes."
            )

    gbot_l, gbot_r = st.columns([1, 1.15])
    with gbot_l:
        show_chart(
            small_multiples(metrics, metric, circuit, sel_grid),
            on_select=on_small_multiples_click,
            selection_mode="points",
            key=f"small_multiples_chart_{nonce}",
        )
    with gbot_r:
        show_chart(
            grid_vs_figure(metrics, circuit, metric, sel_grid),
            on_select=on_gridvs_click,
            selection_mode="points",
            key=f"gridvs_chart_{nonce}",
        )

    st.divider()

    # =====================================================
    # 2. Segment Analysis
    # =====================================================
    st.markdown("### 2 · Segment Analysis")
    seg = int(st.session_state.sel_seg) if int(st.session_state.sel_seg) in (1, 2, 3) else 1
    st.session_state.sel_seg = seg
    st.caption(
        f"Selected **Segment {seg}** · click a coloured section on the track map to change · hover for details"
    )

    s_top_l, s_top_r = st.columns([1.35, 1])
    meta = results[results["circuit_label"] == circuit].iloc[0]
    with s_top_l:
        outline = resolve_outline(
            circuit,
            str(meta["location"]),
            str(meta.get("circuit_official", meta["location"])),
            year=y1,
        )
        if outline:
            show_chart(
                circuit_map_figure(outline, seg),
                on_select=on_trackmap_click,
                selection_mode="points",
                key=f"trackmap_chart_{nonce}",
            )
        else:
            st.info("No track outline for this circuit. Run `py -3 scripts/build_aggregates.py --only maps`.")
    with s_top_r:
        if event_keys:
            fast_fig = fastest_bars(
                sectors,
                event_keys,
                seg,
                driver if driver != "All" else "",
                team=team_arg,
                y0=y0,
                y1=y1,
            )
            if fast_fig.data:
                show_chart(
                    fast_fig,
                    on_select=on_fastest_click,
                    selection_mode="points",
                    key=f"fastest_chart_{nonce}",
                )
            else:
                st.warning("No fastest-sector rows for this selection.")
        else:
            st.warning("No sector timing matched this circuit.")

    s1, s2, s3 = st.columns([1, 1.25, 1])
    with s1:
        st.markdown("**Number of overtakes**")
        st.info("Mock uses a year line chart here. Current sources have no overtake-by-segment history.")
    with s2:
        if event_keys:
            show_chart(dist_figure(sectors, event_keys, seg, driver, team=team_arg, y0=y0, y1=y1))
        else:
            st.write("No distribution data.")
    with s3:
        sub = results[(results["circuit_label"] == circuit) & (results["year"] >= y0) & (results["year"] <= y1)]
        if team_arg:
            sub = sub[sub["constructor_name"].astype(str).map(lambda t: teams_compatible(team_arg, t))]
        dnf = sub[sub["position"].isna()].groupby("year").size()
        show_chart(line_by_year(dnf.index, dnf.values, "Number of DNF (circuit-level proxy)"))
        st.caption("Mock asks for DNF in segment; public data only supports circuit-level DNF here.")

    st.divider()

    # =====================================================
    # 3. Strategy
    # =====================================================
    st.markdown("### 3 · Strategy")
    st.caption(
        "Filtered by Team above" + (f" · showing **{team}**" if team != "All" else " · All teams")
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if event_keys:
            show_chart(strategy_tyre_laps(sectors, event_keys, team=team_arg, y0=y0, y1=y1))
        else:
            st.write("No tyre laps.")
    with c2:
        fig = strategy_pit_dist(pits, circuit, y0, y1, team=team_arg)
        if fig.data:
            show_chart(fig)
        else:
            st.write("No pit distribution.")
    with c3:
        fig = strategy_tyre_comp(sectors, event_keys, team=team_arg, y0=y0, y1=y1)
        if fig.data:
            show_chart(fig)
        else:
            st.write("No tyre composition.")
    with c4:
        fig = strategy_avg_pits(pits, circuit, y0, y1, team=team_arg)
        if fig.data:
            show_chart(fig)
        else:
            st.write("No avg pit series.")

    st.divider()
    st.caption(
        "Chart encodings follow F1_Combined_Dashboard_Mock.html. "
        "Data: processed RaceData + tif1 sectors; track outlines snapshotted for Streamlit Cloud "
        "(rebuild with `py -3 scripts/build_aggregates.py --only maps`)."
    )


if __name__ == "__main__":
    main()
