# ================= ============================================================
# IMPORTS AND SETUP
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator, DayLocator
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress, t, sem
from scipy.optimize import curve_fit
from scipy.fftpack import fft, ifft
from scipy import optimize, fftpack, stats

import os
import ast
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
# import mysql.connector
import os

import math
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib.pyplot import cm
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

import pickle
import plotly.graph_objects as go
import plotly.express as px
from pandas.tseries.offsets import DateOffset
import plotly.io as pio

# =============================================================================
# IMPORTS AND SETUP
# =============================================================================

df = pd.read_csv("./data/observation_years_by_observer.csv")


def _parse_year_list(value):
    """Safely parse list-like year strings such as \"['1844','1845']\"."""
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, (list, tuple)):
        out = []
        for item in parsed:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


name_col = None
for candidate in ["observer_name", "ALIAS", "name"]:
    if candidate in df.columns:
        name_col = candidate
        break

if name_col is None:
    raise ValueError("Could not find an observer name column in observation_years_by_observer.csv")

if "observation_years" in df.columns:
    parsed_years = df["observation_years"].apply(_parse_year_list)
    start_years = parsed_years.apply(lambda years: min(years) if years else np.nan)
    end_years = parsed_years.apply(lambda years: max(years) if years else np.nan)
    obs_year_counts = parsed_years.apply(len).astype(float)
else:
    start_years = pd.to_numeric(df.get("start_year"), errors="coerce")
    end_years = pd.to_numeric(df.get("end_year"), errors="coerce")
    span_years = (end_years - start_years + 1)
    if "year_count" in df.columns:
        obs_year_counts = pd.to_numeric(df["year_count"], errors="coerce").fillna(span_years)
    else:
        obs_year_counts = span_years

if "observation_count" in df.columns:
    total_obs = pd.to_numeric(df["observation_count"], errors="coerce").fillna(obs_year_counts)
else:
    total_obs = obs_year_counts.copy()
total_obs = total_obs.fillna(1)


def _year_to_date(year, month, day):
    if pd.isna(year):
        return None
    return f"{int(year)}-{month:02d}-{day:02d}"


observer_stats = pd.DataFrame({
    "ALIAS": df[name_col],
    "start_date": start_years.apply(lambda y: _year_to_date(y, 1, 1)),
    "end_date": end_years.apply(lambda y: _year_to_date(y, 12, 31)),
    "observation_years": obs_year_counts,
    "total_observations": total_obs,
})


def _safe_datetime(value):
    """
    Convert a YYYY-MM-DD string (or bare year) into a Python datetime object
    without relying on pandas' limited datetime64[ns] range.
    """
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    parts = text.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return datetime(year, month, day)
    except ValueError:
        return None


def save_observer_bubble_matplotlib_poster(
    observer_stats: pd.DataFrame,
    start_year: int = 1600,
    outdir: str = "figures",
    name: str = "observer_bubbles_poster",
    width_in: float = 18.0,
    height_in: float = 14.0,
    tick_year_step: int = 20,
    size_max_pt: float = 100.0,      # max bubble radius in points
    size_min_pt: float = 10.0,       # min bubble radius in points
    label_all: bool = False,
    label_top_n: int = 30,
    alpha: float = 0.85,
    cmap_name: str = "plasma",
    dpi_png: int = 400,
):
    """
    Poster-friendly Matplotlib bubble plot.
    Large fonts and thick lines for visibility at print scale.
    """

    # ---- 1) Prepare data ----
    df = observer_stats.copy()
    req = ["start_date", "end_date", "total_observations", "observation_years", "ALIAS"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse dates safely
    df["start_date"] = df["start_date"].apply(_safe_datetime)
    df["end_date"]   = df["end_date"].apply(_safe_datetime)
    df["start_date"] = df["start_date"].where(df["start_date"].notna(), df["end_date"])
    df["end_date"]   = df["end_date"].where(df["end_date"].notna(), df["start_date"])
    df = df.dropna(subset=["start_date", "end_date"]).copy()
    start_years = df["start_date"].apply(lambda dt: dt.year)
    end_years = df["end_date"].apply(lambda dt: dt.year)
    df["coverage_years"] = end_years - start_years + 1

    df["total_observations"] = pd.to_numeric(df["total_observations"], errors="coerce").fillna(1)
    df["observation_years"]  = pd.to_numeric(df["observation_years"], errors="coerce").fillna(0)
    df.loc[df["total_observations"] <= 0, "total_observations"] = 1.0

    # Bubble area follows observation duration (prefer actual span, fall back to counted years)
    bubble_years = df["coverage_years"].where(df["coverage_years"].notna(), df["observation_years"]).fillna(1)

    # ---- 2) Size scaling ----
    v = bubble_years.astype(float).values
    vmax = v.max() if np.isfinite(v.max()) else 1
    vmin = v.min() if np.isfinite(v.min()) else 1
    area_min = size_min_pt ** 2
    area_max = size_max_pt ** 2
    s = area_min + (v / vmax) * (area_max - area_min) if vmax > 0 else np.full_like(v, area_min)

    # ---- 3) Colors ----
    cvals = df["observation_years"].astype(float).values
    norm = Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))
    cmap = get_cmap(cmap_name)

    # ---- 4) Figure setup ----
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    plt.rcParams.update({
        "font.size": 22,                # base font size
        "axes.labelsize": 26,
        "axes.titlesize": 30,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 22,
        "figure.titlesize": 32,
    })

    x = mdates.date2num(df["start_date"].values)
    y = mdates.date2num(df["end_date"].values)

    sc = ax.scatter(
        x, y,
        s=s,
        c=cvals,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        edgecolors="black",
        linewidths=1.5,
    )

    # ---- 5) Labels ----
    eligible = df[df["coverage_years"] > 30]
    if label_all:
        to_label = eligible.index
    else:
        to_label = eligible.sort_values("total_observations", ascending=False).head(label_top_n).index

    for idx in to_label:
        ax.annotate(
            str(df.loc[idx, "ALIAS"]),
            (x[df.index.get_loc(idx)], y[df.index.get_loc(idx)]),
            xytext=(8, 2),
            textcoords="offset points",
            fontsize=18,      # larger label text
            weight="bold",
            ha="left", va="center",
        )

    # ---- 6) Axes ----
    min_date = datetime(start_year, 1, 1)
    right_year = max(start_years.max(), end_years.max())
    right_year = int(math.ceil(right_year / 10.0) * 10 + 10)
    max_date = datetime(right_year, 12, 31)

    ax.set_xlim(min_date, max_date)
    ax.set_ylim(min_date, max_date)

    ax.xaxis.set_major_locator(mdates.YearLocator(base=tick_year_step))
    ax.yaxis.set_major_locator(mdates.YearLocator(base=tick_year_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Start Year", labelpad=20)
    ax.set_ylabel("End Year", labelpad=20)
    ax.set_title("Observer Bubble Plot", pad=30, weight="bold")

    # ---- 7) Colorbar ----
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.9)
    cbar.set_label("Observation Years", fontsize=24, labelpad=20)
    cbar.ax.tick_params(labelsize=22)

    # ---- 8) Save ----
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=dpi_png, bbox_inches="tight")
    plt.close(fig)

    # print(f"✅ Saved {base}.pdf/.svg/.png (poster scale, large fonts)")
    #


save_observer_bubble_matplotlib_poster(
    observer_stats,
    start_year=1600,
    name="observer_years_timeline-v1",
    size_max_pt=100,    # adjust bubble scaling for posters
    size_min_pt=20,
    label_top_n=200,     # show labels for top contributors
    cmap_name="viridis",
)
