"""
Create bubble and timeline plots for Mittheilungen observers.

- Bubble plot: start year on X, end year on Y, bubble size = span (end_year - start_year),
  bubble color = actual years observed.
- Timeline plot: horizontal bars whose length represents (end_year - start_year) and color
  represents actual years observed.

The script is intentionally small and parameterised so a new CSV with the same columns
can be dropped in and plotted via CLI flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "observation_years_by_observer.csv"
DEFAULT_OUTDIR = Path(__file__).resolve().parents[1] / "figures"


def load_observer_data(csv_path: Path) -> pd.DataFrame:
    """Load data and standardise column names and derived spans."""
    df = pd.read_csv(csv_path)
    name_col = next((c for c in ["name", "observer_name", "ALIAS"] if c in df.columns), None)
    if not name_col:
        raise ValueError(f"No name column found in {csv_path}")

    required = ["start_year", "end_year", "year_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")

    data = pd.DataFrame(
        {
            "name": df[name_col].astype(str),
            "origin": df["origin"] if "origin" in df.columns else "unknown",
            "start_year": pd.to_numeric(df["start_year"], errors="coerce"),
            "end_year": pd.to_numeric(df["end_year"], errors="coerce"),
            "year_count": pd.to_numeric(df["year_count"], errors="coerce"),
        }
    )

    data = data.dropna(subset=["start_year", "end_year", "year_count"]).copy()
    data = data[data["end_year"] >= data["start_year"]]
    data["span_years"] = (data["end_year"] - data["start_year"]).astype(float)

    # Preserve integer-like years for cleaner axes labels.
    for col in ["start_year", "end_year"]:
        data[col] = data[col].round().astype(int)
    return data


def _color_scale(values: np.ndarray, cmap_name: str):
    clean = np.asarray(values, dtype=float)
    vmin = np.nanmin(clean)
    vmax = np.nanmax(clean)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def _bubble_sizes(span_years: np.ndarray, size_min: float = 80.0, size_max: float = 2400.0) -> np.ndarray:
    """Scale span values into scatter marker areas."""
    spans = np.asarray(span_years, dtype=float)
    if len(spans) == 0:
        return spans
    s_min, s_max = np.nanmin(spans), np.nanmax(spans)
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_max == 0:
        return np.full_like(spans, size_min)
    return np.interp(spans, (s_min, s_max), (size_min, size_max))


def save_bubble_plot(
    data: pd.DataFrame,
    out_path: Path,
    cmap_name: str,
    start_year: int | None = None,
    end_year: int | None = None,
    label_top_n: int = 50,
    label_all: bool = False,
):
    """Scatter showing start vs end with bubble size = span and color = active years."""
    cmap, norm = _color_scale(data["year_count"].values, cmap_name)
    sizes = _bubble_sizes(data["span_years"].values)

    x_min = start_year if start_year is not None else int(data["start_year"].min())
    x_max = end_year if end_year is not None else int(data["end_year"].max())
    padding = max(1, int((x_max - x_min) * 0.03))
    x_min -= padding
    x_max += padding

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(
        data["start_year"],
        data["end_year"],
        s=sizes,
        c=data["year_count"],
        cmap=cmap,
        norm=norm,
        alpha=0.72,
        edgecolor="black",
        linewidth=0.8,
    )

    diag_min = min(x_min, data["start_year"].min(), data["end_year"].min())
    diag_max = max(x_max, data["start_year"].max(), data["end_year"].max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="#555555", linewidth=1, alpha=0.2)

    # Label every observer (or the requested top N) with slight offsets to reduce overlap.
    if label_all:
        label_df = data
    else:
        label_df = data.sort_values("year_count", ascending=False).head(label_top_n or len(data))
    offsets = [(3, 2), (-3, 2), (3, -2), (-3, -2)]
    for idx, (_, row) in enumerate(label_df.iterrows()):
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(
            row["name"],
            (row["start_year"], row["end_year"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.8,
            fontweight="bold",
            ha="left" if dx >= 0 else "right",
            va="center",
            clip_on=True,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("Start year")
    ax.set_ylabel("End year")
    ax.set_title("Mittheilungen observers – bubble plot")
    ax.grid(True, linestyle="--", alpha=0.35)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.015)
    cbar.set_label("Actual observation years")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_timeline_plot(
    data: pd.DataFrame,
    out_path: Path,
    cmap_name: str,
    start_year: int | None = None,
    end_year: int | None = None,
):
    """Horizontal bar timeline with bar length = span and color = active years."""
    plot_df = data.sort_values(["start_year", "end_year", "name"]).reset_index(drop=True)
    cmap, norm = _color_scale(plot_df["year_count"].values, cmap_name)
    spans = plot_df["span_years"].to_numpy()
    spans_for_plot = np.where(spans == 0, 0.35, spans)  # show zero-span entries with a sliver

    x_min = start_year if start_year is not None else int(plot_df["start_year"].min())
    x_max = end_year if end_year is not None else int(plot_df["end_year"].max())
    padding = max(1, int((x_max - x_min) * 0.04))
    x_min -= padding
    x_max += padding

    y_positions = np.arange(len(plot_df))
    height = max(10, min(24, 6 + len(plot_df) * 0.16))
    fig, ax = plt.subplots(figsize=(18, height))
    bar_colors = cmap(norm(plot_df["year_count"].values))
    bars = ax.barh(
        y_positions,
        spans_for_plot,
        left=plot_df["start_year"],
        color=bar_colors,
        edgecolor="#262626",
        linewidth=0.6,
    )

    label_pad = (x_max - x_min) * 0.006
    for y, bar, (_, row) in zip(y_positions, bars, plot_df.iterrows()):
        end_x = bar.get_x() + bar.get_width()
        place_left = end_x + label_pad > x_max
        text_x = end_x + label_pad if not place_left else bar.get_x() - label_pad
        ha = "left" if not place_left else "right"
        ax.text(
            text_x,
            y,
            row["name"],
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color="#1a1a1a",
        )

    ax.set_ylim(-0.8, len(plot_df) - 0.2)
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_title(
        "Historical Mittheilungen Observers Timeline (color = actual observation years)",
        fontsize=17,
        pad=16,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", color="#bfbfbf", alpha=0.4)
    ax.tick_params(axis="x", labelsize=10)

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        fraction=0.035,
    )
    cbar.set_label("Actual observation years", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=350, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create bubble and timeline plots for observer data.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to observation_years_by_observer.csv")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Directory for output figures")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name")
    parser.add_argument("--start-year", type=int, dest="start_year", default=None, help="Fix left bound of the plots")
    parser.add_argument("--end-year", type=int, dest="end_year", default=None, help="Fix right bound of the plots")
    parser.add_argument(
        "--label-top-n",
        type=int,
        default=None,
        help="If set, limit bubble labels to the top N by year_count. Otherwise, label all.",
    )
    parser.add_argument(
        "--no-label-all",
        dest="label_all",
        action="store_false",
        help="Disable labeling all bubbles (use label-top-n if provided).",
    )
    parser.set_defaults(label_all=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_observer_data(args.csv)
    bubble_path = args.outdir / "mitt_observer_bubble"
    timeline_path = args.outdir / "mitt_observer_timeline"

    save_bubble_plot(
        data,
        bubble_path,
        cmap_name=args.cmap,
        start_year=args.start_year,
        end_year=args.end_year,
        label_top_n=args.label_top_n,
        label_all=args.label_all,
    )
    save_timeline_plot(
        data,
        timeline_path,
        cmap_name=args.cmap,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
