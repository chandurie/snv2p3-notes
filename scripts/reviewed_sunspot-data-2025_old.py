"""
Generate a bubble plot summarising the post-2010 reviewed sunspot observer
sources captured in Review_sunspotsources_2010_2025_pruned.csv.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_CSV = PROJECT_ROOT / "data" / "Review_merged_sorted.csv"
REQUIRED_COLUMNS = {"Observer / Source", "Period of observation", "YearStart", "YearEnd"}


def _extract_years(text: str) -> List[int]:
    if not text or not isinstance(text, str):
        return []
    return [int(y) for y in re.findall(r"\d{4}", text)]


def _coerce_year(value) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and np.isfinite(value):
        return int(value)
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        years = _extract_years(str(value))
        return years[0] if years else None


def _clean_observer_name(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    cleaned = re.sub(r"\s*\([^)]*\)", "", name).strip()
    cleaned = re.sub(r"[\s\-\u2013\u2014/]*[\d\s\-\u2013\u2014/]+$", "", cleaned).strip()
    fallback = name.strip()
    return cleaned or fallback or "Unknown"


def _read_review_csv(csv_path: Path) -> pd.DataFrame:
    """Read the review CSV trying multiple header rows if necessary."""
    for header in (1, 0):
        df = pd.read_csv(csv_path, sep=";", header=header)
        if REQUIRED_COLUMNS.issubset(df.columns):
            return df
    missing = ", ".join(sorted(REQUIRED_COLUMNS - set(df.columns)))
    raise ValueError(f"CSV at {csv_path} is missing required columns: {missing}")


def load_reviewed_observer_stats(path: os.PathLike[str] | str = REVIEW_CSV) -> pd.DataFrame:
    """
    Read the pruned review CSV and return a stats dataframe with
    start/end years and derived coverage information.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reviewed observer CSV not found: {csv_path}")
    df = _read_review_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        start_year = _coerce_year(row.get("YearStart"))
        end_year = _coerce_year(row.get("YearEnd"))
        period_years = _extract_years(row.get("Period of observation", ""))
        if start_year is None and period_years:
            start_year = min(period_years)
        if end_year is None and period_years:
            end_year = max(period_years)
        if start_year is None:
            start_year = end_year
        if end_year is None:
            end_year = start_year
        if start_year is None or end_year is None:
            continue
        if end_year < start_year:
            start_year, end_year = end_year, start_year
        coverage = end_year - start_year + 1
        raw_name = row.get("Observer / Source", "Unknown")
        records.append(
            {
                "observer": raw_name,
                "observer_clean": _clean_observer_name(raw_name),
                "start_year": start_year,
                "end_year": end_year,
                "coverage_years": coverage,
                "target_authors": row.get("Target author(s)", ""),
                "source": row.get("Source", ""),
            }
        )
    if not records:
        raise ValueError(
            "No valid observer records were parsed. "
            "Check that the CSV has YearStart/YearEnd values."
        )
    stats = pd.DataFrame(records)
    stats["start_date"] = stats["start_year"].apply(lambda y: datetime(int(y), 1, 1))
    stats["end_date"] = stats["end_year"].apply(lambda y: datetime(int(y), 12, 31))
    return stats


def save_reviewed_bubble_plot(
    observer_stats: pd.DataFrame,
    start_year: int = 1600,
    name: str = "reviewed_sources_bubbles_merged",
    outdir: str = "figures",
    width_in: float = 18,
    height_in: float = 14,
    tick_year_step: int = 25,
    size_max_pt: float = 140,
    size_min_pt: float = 24,
    cmap_name: str = "viridis",
    alpha: float = 0.9,
    label_top_n: Optional[int] = None,
    dpi_png: int = 1200,
    font_scale: float = 1.15,
    background_color: str = "#f8f8f8",
):
    """
    Bubble chart styled for large-format figures inspired by mittheilungen-plots.py.
    """
    df = observer_stats.copy()
    if df.empty:
        raise ValueError("observer_stats is empty; nothing to plot.")

    df = df.dropna(subset=["start_date", "end_date"])
    df = df.sort_values("coverage_years", ascending=False)

    bubble_vals = df["coverage_years"].astype(float).values
    vmax = bubble_vals.max() if np.isfinite(bubble_vals.max()) else 1
    area_min = size_min_pt**2
    area_max = size_max_pt**2
    sizes = (
        area_min + (bubble_vals / vmax) * (area_max - area_min)
        if vmax > 0
        else np.full_like(bubble_vals, area_min)
    )

    cvals = df["coverage_years"].astype(float).values
    norm = Normalize(vmin=np.nanmin(cvals), vmax=np.nanmax(cvals))
    cmap = colormaps.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor("white")
    x = mdates.date2num(df["start_date"].values)
    y = mdates.date2num(df["end_date"].values)
    ax.scatter(
        x,
        y,
        s=sizes,
        c=cvals,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        edgecolor="#1f1f1f",
        linewidth=1.1,
    )

    diag_min = min(x.min(), y.min())
    diag_max = max(x.max(), y.max())
    ax.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        linestyle="--",
        color="#6a6a6a",
        linewidth=1.3,
        alpha=0.35,
    )

    label_limit = label_top_n if label_top_n is not None else len(df)
    label_df = df.head(label_limit)
    label_size = 8.5 * font_scale
    for idx, (_, row) in enumerate(label_df.iterrows()):
        y_offset = 8 if idx % 2 == 0 else -12
        ax.annotate(
            row.get("observer_clean", row["observer"]),
            (mdates.date2num(row["start_date"]), mdates.date2num(row["end_date"])),
            textcoords="offset points",
            xytext=(6, y_offset),
            fontsize=label_size,
            weight="bold",
            color="#1b1b1b",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.6, linewidth=0),
        )

    min_date = datetime(start_year, 1, 1)
    right_year = int(math.ceil(df["end_year"].max() / 10.0) * 10 + 10)
    max_date = datetime(right_year, 12, 31)

    ax.set_xlim(min_date, max_date)
    ax.set_ylim(min_date, max_date)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=tick_year_step))
    ax.yaxis.set_major_locator(mdates.YearLocator(base=tick_year_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    tick_fontsize = 11 * font_scale
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=1.1, colors="#242424")
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.set_xlabel("Start year", fontsize=13 * font_scale, fontweight="bold")
    ax.set_ylabel("End year", fontsize=13 * font_scale, fontweight="bold")
    ax.set_title(
        "Reviewed Sunspot Observations",
        pad=20,
        weight="bold",
        fontsize=18 * font_scale,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.012)
    cbar.set_label("Coverage (years)", fontsize=11 * font_scale, fontweight="bold")
    cbar.ax.tick_params(labelsize=9 * font_scale)

    fig.tight_layout(pad=1.5)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=dpi_png, bbox_inches="tight")
    plt.close(fig)


def save_reviewed_timeline_plot(
    observer_stats: pd.DataFrame,
    name: str = "reviewed_sources_timeline_merged",
    outdir: str = "figures",
    top_n: Optional[int] = None,
    width_in: float = 22,
    row_spacing: float = 1.1,
    row_height_in: float = 0.2,
    cmap_name: str = "plasma",
    line_width: float = 6.0,
    tick_year_step: int = 25,
    dpi_png: int = 1200,
    font_scale: float = 1.1,
    background_color: str = "#f8f8f8",
):
    """
    Horizontal bar timeline similar to sunspot_timeline.pdf but using reviewed data.
    """
    df = observer_stats.copy()
    if df.empty:
        raise ValueError("observer_stats is empty; nothing to plot.")

    df = df.sort_values("start_year").reset_index(drop=True)
    if top_n:
        df = df.head(top_n)
    if df.empty:
        raise ValueError("No observers available after applying top_n filter.")

    rows = len(df)
    fig_height = max(10, rows * row_height_in)
    fig, ax = plt.subplots(figsize=(width_in, fig_height))
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor("white")

    durations = df["coverage_years"].astype(float)
    norm = mpl.colors.Normalize(vmin=durations.min(), vmax=durations.max())
    cmap = colormaps.get_cmap(cmap_name)

    label_offset_days = 365 * 0.4
    start_nums = mdates.date2num(df["start_date"].values)
    end_nums = mdates.date2num(df["end_date"].values)

    for idx, row in enumerate(df.itertuples()):
        y = idx * row_spacing
        color = cmap(norm(row.coverage_years))
        ax.plot(
            [start_nums[idx], end_nums[idx]],
            [y, y],
            linewidth=line_width,
            color=color,
            solid_capstyle="round",
            alpha=0.95,
        )
        label_x = end_nums[idx] + label_offset_days if idx % 2 == 0 else start_nums[idx] - label_offset_days
        ha = "left" if idx % 2 == 0 else "right"
        ax.text(
            label_x,
            y,
            row.observer_clean,
            va="center",
            ha=ha,
            fontsize=9.5 * font_scale,
            fontweight="bold",
            color="#1c1c1c",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.65, linewidth=0),
        )

    min_start = df["start_date"].min()
    max_end = df["end_date"].max()
    ax.set_xlim(min_start, max_end + pd.Timedelta(days=365))
    ax.set_ylim(-row_spacing, (rows - 1) * row_spacing + row_spacing * 1.4)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_xlabel("Year", fontsize=13 * font_scale, fontweight="bold")
    ax.set_title(
        "Reviewed Sunspot Observers",
        fontsize=18 * font_scale,
        fontweight="bold",
        pad=24,
    )

    ax.xaxis.set_major_locator(mdates.YearLocator(base=tick_year_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    tick_fontsize = 11 * font_scale
    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=True,
        labelbottom=True,
        labeltop=True,
        labelsize=tick_fontsize,
        width=1.1,
        colors="#262626",
    )
    for label in ax.get_xticklabels(which="both"):
        label.set_fontweight("bold")

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.025,
        pad=0.01,
    )
    cbar.set_label("Coverage (years)", fontsize=11 * font_scale, fontweight="bold")
    cbar.ax.tick_params(labelsize=9 * font_scale)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout(pad=1.5)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=dpi_png, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    stats = load_reviewed_observer_stats()
    save_reviewed_bubble_plot(stats)
    save_reviewed_timeline_plot(stats)
