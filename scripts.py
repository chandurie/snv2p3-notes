"""
Sunspot Number Reconstruction Analysis
=====================================

This script analyzes historical sunspot observations from multiple observers and reconstructs
sunspot number series using three different observational "backbones":
1. Wolfer backbone (reference observer: ID 56)
2. Weber backbone (reference observer: ID 32)
3. Schwabe backbone (reference observer: ID 1000)

The analysis involves:
- Loading and preprocessing multiple datasets
- Applying specific quality control filters and truncations as per original methodology
- Calculating k-factors (calibration factors) between observers
- Creating normalized sunspot series
- Performing jackknife validation by removing observers iteratively to assess stability
- Visualizing results and calculating stability metrics.
"""

# =============================================================================
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

# import mysql.connector
import os

import math
from datetime import timedelta
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
# DATA LOADING AND PREPROCESSING
# =============================================================================


def observer_bubble_plot_animated(observer_dict):
    # Add a new column for scaled marker size
    # Adjust the scaling factor as needed for your data
    scaling_factor = 1  # You can tweak this
    observer_dict["marker_size"] = (
        np.sqrt(observer_dict["total_observations"]) * scaling_factor
    )

    # Assume observer_dict is already prepared as in your previous code
    observer_dict["start_year"] = observer_dict["start_date"].dt.year

    # Get all years from the earliest to the latest start year
    years = np.arange(
        observer_dict["start_year"].min(), observer_dict["start_year"].max() + 1
    )

    # For each year, include all observers who started up to that year
    frames = []
    for year in years:
        frame_data = observer_dict[observer_dict["start_year"] <= year]
        frames.append(frame_data)

    # Initial data (first frame)
    init_data = frames[0]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=init_data["start_date"],
                y=init_data["end_date"],
                mode="markers+text",
                marker=dict(
                    size=init_data["total_observations"],
                    color=init_data["observation_years"],
                    colorscale="Viridis",
                    showscale=True,
                    line=dict(width=2, color="DarkSlateGrey"),
                ),
                text=init_data["ALIAS"],
                textposition="middle right",
                hovertemplate=(
                    "Observer: %{text}<br>"
                    "Start: %{x|%Y-%m-%d}<br>"
                    "End: %{y|%Y-%m-%d}<br>"
                    "Total Observations: %{marker.size}<br>"
                    "Observation Years: %{marker.color:.2f}<extra></extra>"
                ),
            )
        ],
        layout=go.Layout(
            title="Animated Observer Activity Bubble Plot",
            xaxis=dict(
                title="Start Date",
                range=[init_data["start_date"].min(), init_data["start_date"].max()],
            ),
            yaxis=dict(
                title="End Date",
                range=[init_data["end_date"].min(), init_data["end_date"].max()],
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 500, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                        },
                    ],
                }
            ],
        ),
    )

    # Prepare frames for animation
    from pandas.tseries.offsets import DateOffset

    buffer_days = 365.25 * 6  # About 2 months buffer

    animation_frames = []
    for i, year in enumerate(years):
        frame_data = frames[i]
        x_min = frame_data["start_date"].min() - pd.Timedelta(days=buffer_days)
        x_max = frame_data["start_date"].max() + pd.Timedelta(days=buffer_days)
        y_min = frame_data["end_date"].min() - pd.Timedelta(days=buffer_days)
        y_max = frame_data["end_date"].max() + pd.Timedelta(days=buffer_days)
        if i == len(years) - 1:
            # Use full range for last frame
            x_min = observer_dict["start_date"].min() - pd.Timedelta(days=buffer_days)
            x_max = observer_dict["start_date"].max() + pd.Timedelta(days=buffer_days)
            y_min = observer_dict["end_date"].min() - pd.Timedelta(days=buffer_days)
            y_max = observer_dict["end_date"].max() + pd.Timedelta(days=buffer_days)
        animation_frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=frame_data["start_date"],
                        y=frame_data["end_date"],
                        mode="markers+text",
                        marker=dict(
                            size=frame_data["marker_size"],
                            color=frame_data["observation_years"],
                            colorscale="Viridis",
                            showscale=True,
                            line=dict(width=2, color="DarkSlateGrey"),
                        ),
                        text=frame_data["ALIAS"],
                        textposition="middle right",
                        hovertemplate=(
                            "Observer: %{text}<br>"
                            "Start: %{x|%Y-%m-%d}<br>"
                            "End: %{y|%Y-%m-%d}<br>"
                            "Total Observations: %{marker.size}<br>"
                            "Observation Years: %{marker.color:.2f}<extra></extra>"
                        ),
                    )
                ],
                name=str(year),
                layout=go.Layout(
                    xaxis=dict(title="Start Date", range=[x_min, x_max]),
                    yaxis=dict(title="End Date", range=[y_min, y_max]),
                    title=f"Animated Observer Bubble Plot - Year {year}",
                ),
            )
        )

    fig.frames = animation_frames

    # Add slider for years
    sliders = [
        {
            "steps": [
                {
                    "args": [
                        [str(year)],
                        {
                            "frame": {"duration": 500, "redraw": True},
                            "mode": "immediate",
                        },
                    ],
                    "label": str(year),
                    "method": "animate",
                }
                for year in years
            ],
            "transition": {"duration": 0},
            "x": 0.1,
            "y": 0,
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Year: ",
                "visible": True,
                "xanchor": "center",
            },
            "len": 0.9,
        }
    ]

    fig.update_layout(sliders=sliders)

    fig.show()


def observer_bubble_plots(observer_dict):
    fig = px.scatter(
        observer_dict,
        x="start_date",
        y="end_date",
        size="total_observations",
        color="observation_years",
        hover_name="ALIAS",
        text="ALIAS",  # Add observer name as text
        size_max=60,
        labels={
            "start_date": "Start Date",
            "end_date": "End Date",
            "total_observations": "Total Observations",
            "observation_years": "Observation Years",
            "FK_OBSERVERS": "Observer ID",
        },
        title="Observer Bubble Plot",
    )

    # Set text position (try 'middle right', 'top center', etc.)
    fig.update_traces(
        marker=dict(line=dict(width=2, color="DarkSlateGrey")),
        textposition="middle right",
        textfont=dict(size=12),
    )

    # Save the figure to a static image file
    # pio.write_image(fig, 'my_plot.png')
    # pio.write_image(fig, 'my_plot.pdf')  # For directly saving as a PDF

    fig.show()


def plot_observer_timeline(df, observer_stats, max_observers=None):
    """
    Create a timeline plot showing actual observation days for all observers.

    Args:
        df (pd.DataFrame): Full observation data (must contain 'FK_OBSERVERS', 'Date', 'WOLF', 'ALIAS').
        observer_stats (pd.DataFrame): Stats table with 'FK_OBSERVERS', 'start_date', 'end_date', 'observation_years', 'total_observations'.
        max_observers (int or None): Limit the number of observers. If None, plots all.
    """
    # Determine number of observers to plot
    if max_observers is None:
        top_observers = observer_stats.sort_values("start_date")
    else:
        top_observers = observer_stats.nlargest(max_observers, "observation_years")
        top_observers = top_observers.sort_values("start_date")

    n_obs = len(top_observers)

    # Create dynamic color map
    cmap = plt.cm.get_cmap("tab20", n_obs)  # tab20 works well for categories

    fig, ax = plt.subplots(
        figsize=(15, max(8, n_obs * 0.3))
    )  # Taller if more observers

    for i, (_, obs) in enumerate(top_observers.iterrows()):
        obs_data = df[df["FK_OBSERVERS"] == obs["FK_OBSERVERS"]].dropna(subset=["WOLF"])

        if not obs_data.empty:
            ax.scatter(
                obs_data["Date"],
                [i] * len(obs_data),
                color=cmap(i),
                alpha=0.7,
                s=80,
                marker="|",
            )

    # Set Y-axis labels
    ax.set_yticks(range(n_obs))
    ax.set_yticklabels(
        [
            f"{obs['ALIAS']} ({obs['total_observations']} obs)"
            for _, obs in top_observers.iterrows()
        ],
        fontsize=8,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Observers")
    ax.set_title("Observation Timeline by Observer")
    ax.grid(True, axis="x", alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_observer_timeline_mitt_all(df, observer_stats_mitt_all, max_observers=40):
    """
    Create a timeline plot showing actual observation days with optimized visualization
    """
    # Select top observers by observation duration, then sort by start date
    top_observers = observer_stats_mitt_all.nlargest(max_observers, "observation_years")
    top_observers_sorted = top_observers.sort_values("start_date")

    fig, ax = plt.subplots(figsize=(15, 10))
    # colors = plt.cm.tab20(np.linspace(0, 1, len(top_observers_sorted)))
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1a5f9a",
        "#ff6f00",
        "#33a02c",
        "#e31a1c",
        "#6a3d9a",
        "#ff7f00",
        "#cab2d6",
        "#1f78b4",
        "#ffbb78",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1a5f9a",
        "#ff6f00",
        "#33a02c",
        "#e31a1c",
        "#6a3d9a",
        "#ff7f00",
        "#cab2d6",
        "#1f78b4",
        "#ffbb78",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    for i, (_, obs) in enumerate(top_observers_sorted.iterrows()):
        # Get actual observation dates for this observer
        obs_data = df[df["FK_OBSERVERS"] == obs["FK_OBSERVERS"]].copy()
        obs_data = obs_data.dropna(subset=["WOLF"])

        if len(obs_data) > 0:
            # Use scatter plot for better performance with many points
            ax.scatter(
                obs_data["Date"],
                [i] * len(obs_data),
                color=colors[i],
                alpha=0.7,
                s=100,
                marker="|",
            )

    ax.set_yticks(range(len(top_observers_sorted)))
    ax.set_yticklabels(
        [
            f"{obs['ALIAS']} ({obs['total_observations']} obs)"
            for _, obs in top_observers_sorted.iterrows()
        ]
    )
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Observers (sorted by start date)")
    ax.set_title(f"Actual Observation Days(Wolf Source Book Data)")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_observer_timeline_wolf_source_book(
    df, observer_stats_source_book, max_observers=40
):
    """
    Create a timeline plot showing actual observation days with optimized visualization
    """
    # Select top observers by observation duration, then sort by start date
    top_observers = observer_stats_source_book.nlargest(
        max_observers, "observation_years"
    )
    top_observers_sorted = top_observers.sort_values("start_date")

    fig, ax = plt.subplots(figsize=(15, 10))
    # colors = plt.cm.tab20(np.linspace(0, 1, len(top_observers_sorted)))
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1a5f9a",
        "#ff6f00",
        "#33a02c",
        "#e31a1c",
        "#6a3d9a",
        "#ff7f00",
        "#cab2d6",
        "#1f78b4",
        "#ffbb78",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#1a5f9a",
        "#ff6f00",
        "#33a02c",
        "#e31a1c",
        "#6a3d9a",
        "#ff7f00",
        "#cab2d6",
        "#1f78b4",
        "#ffbb78",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    for i, (_, obs) in enumerate(top_observers_sorted.iterrows()):
        # Get actual observation dates for this observer
        obs_data = df[df["obsID"] == obs["obsID"]].copy()
        obs_data = obs_data.dropna(subset=["wolf"])

        if len(obs_data) > 0:
            # Use scatter plot for better performance with many points
            ax.scatter(
                obs_data["Date"],
                [i] * len(obs_data),
                color=colors[i],
                alpha=0.7,
                s=100,
                marker="|",
            )

    ax.set_yticks(range(len(top_observers_sorted)))
    ax.set_yticklabels(
        [
            f"{obs['obsname']} ({obs['total_observations']} obs)"
            for _, obs in top_observers_sorted.iterrows()
        ]
    )
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Observers (sorted by start date)")
    ax.set_title(f"Actual Observation Days(Wolf Source Book Data)")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def load_and_preprocess_main_data(base_path):
    """
    Load the main sunspot observation dataset and merge with SNV1 data,
    applying specific initial filters and truncations.

    Args:
        base_path (str): Base directory path where data files are located.

    Returns:
        dict: Dictionary of DataFrames grouped by observer ID after initial processing.
        pd.DataFrame: Merged dataset with SNV1 data (before grouping).
    """
    print("Loading main sunspot observation data...")

    # Load main observation data

    main_data_path = os.path.join(base_path, "mitt_data_all.csv")
    main_data = pd.read_csv(main_data_path)
    df_main = pd.DataFrame(main_data)
    df_main.rename(columns={"DATE": "Date"}, inplace=True)
    df_main = df_main.sort_values("Date")
    df_main["Date"] = pd.to_datetime(df_main["Date"])

    # Load SNV1 reference data
    snv1_data_path = os.path.join(base_path, "snv1_full.csv")
    df_snv1 = pd.read_csv(snv1_data_path)
    df_snv1["Date"] = pd.to_datetime(df_snv1["Date"])
    df_snv1 = df_snv1.sort_values("Date")

    # Merge main data with SNV1
    df_merged = pd.merge(df_main, df_snv1, on="Date")

    # Group by observer ID for individual analysis
    observer_groups = dict(tuple(df_merged.groupby("FK_OBSERVERS")))

    # Apply specific truncations and filters from the original script
    # Note: These operations modify the observer_groups dictionary directly.

    # Filter observer 36 by FK_RUBRICS
    if 36 in observer_groups:
        observer_groups[36] = (
            observer_groups[36].loc[observer_groups[36].FK_RUBRICS == 43].copy()
        )

    # Process observer 30 (Airy) and create observer 310 (Main from Airy)
    # The code creates a new observer (ID 310, alias "Main") from a subset of Airy's (ID 30) observations between 1850 and 1853. It then modifies the original Airy data to retain only observations between 1858 and 1861 and updates the alias to "Airy (various observers)".  This process effectively splits the original Airy data into two separate observer entries in the `observer_groups` dictionary, reflecting different periods and potentially different contributors to the observations.
    # 1. **Check for Airy Data:** `if 30 in observer_groups:` checks if data for observer 30 exists in the `observer_groups` dictionary. If not, the code within the `if` block is skipped.
    # 2. **Create a Copy:** `airy_data = observer_groups[30].copy()` creates a copy of the DataFrame associated with observer 30 and stores it in the `airy_data` variable. This is crucial to avoid modifying the original data in `observer_groups` directly.
    # 3. **Create Observer 310 (Main):**
    #     - `df_310 = airy_data.set_index("Date")` sets the 'Date' column as the index of the `airy_data` copy. This is often necessary for time series operations.
    #     - `df_310 = df_310.truncate(before="1850-01-01", after="1853-01-01")` truncates the data to include only observations between January 1, 1850, and January 1, 1853. This creates a subset of Airy's data for the new observer "Main".
    #     - `df_310["FK_OBSERVERS"] = 310` assigns the observer ID 310 to this new DataFrame.
    #     - `df_310["ALIAS"] = "Main"` assigns the alias "Main" to this observer.
    #     - `observer_groups[310] = df_310.reset_index()` adds the new DataFrame `df_310` to the `observer_groups` dictionary with the key 310, and resets the index so 'Date' becomes a regular column again.
    # 4. **Modify Original Observer 30 (Airy) Data:**
    #     - `df_30 = airy_data.set_index("Date")` sets the 'Date' column as the index of the original `airy_data` copy (note that this is a separate copy from `df_310`).
    #     - `df_30 = df_30.truncate(before="1858-01-01", after="1861-01-01")` truncates the original Airy data to include only observations between January 1, 1858, and January 1, 1861. This modifies the data for the original Airy observer (ID 30).
    #     - `df_30["FK_OBSERVERS"] = 30` reassigns the observer ID 30 (which might have been lost during index manipulation).
    #     - `df_30["ALIAS"] = "Airy (various observers)"` changes the alias of observer 30 to "Airy (various observers)". This reflects the fact that the remaining data after truncation might represent observations from various individuals under Airy's direction.
    #     - `observer_groups[30] = df_30.reset_index()` updates the `observer_groups` dictionary with the modified Airy data, resetting the index.

    if 30 in observer_groups:
        airy_data = observer_groups[
            30
        ].copy()  # Keep a copy of original Airy data for later

        # Create observer 310 (Main) from Airy data, truncated
        df_310 = airy_data.set_index("Date")
        df_310 = df_310.truncate(before="1850-01-01", after="1853-01-01")
        df_310["FK_OBSERVERS"] = 310
        df_310["ALIAS"] = "Main"
        observer_groups[310] = df_310.reset_index()

        # Truncate original observer 30 (Airy) data
        df_30 = airy_data.set_index("Date")
        df_30 = df_30.truncate(before="1858-01-01", after="1861-01-01")
        df_30["FK_OBSERVERS"] = 30
        df_30["ALIAS"] = "Airy (various observers)"
        observer_groups[30] = df_30.reset_index()

    # Truncate observer 177 data
    if 177 in observer_groups:
        df_177 = observer_groups[177].copy()
        df_177 = df_177.sort_values("Date").set_index("Date")
        df_177 = df_177.truncate(before="1925-01-01", after="1946-01-01")
        observer_groups[177] = df_177.reset_index()

    return observer_groups, df_merged


def print_observer_aliases(observer_groups):
    """Prints the observer ID and alias for all observers."""
    print("Observer ID | Alias")
    print("-" * 20)  # Separator line
    for obs_id, obs_data in observer_groups.items():
        try:
            alias = obs_data["ALIAS"].iloc[0]
        except (
            IndexError,
            KeyError,
        ):  # Handle cases where 'ALIAS' is missing or DataFrame is empty
            alias = "N/A"  # Or any other placeholder
        print(f"{obs_id} | {alias}")


# Example usage (assuming observer_groups is already populated):
# print_observer_aliases(observer_groups)


def load_supplementary_datasets(base_path):
    """
    Load additional observational datasets from various sources.

    Args:
        base_path (str): Base directory path where data files are located.

    Returns:
        dict: Dictionary containing supplementary datasets
    """
    print("Loading supplementary datasets...")

    supplementary_data = {}

    # Define file paths and corresponding names
    datasets = {
        "stempell": "Stempell.csv",
        "quimby": "Quimby1.csv",
        "ventosa": "Ventosa.csv",
        "schwab": "Schwab.csv",
        "woinoff": "Woinoff.csv",
    }

    for name, filename in datasets.items():
        filepath = os.path.join(base_path, filename)
        df = pd.read_csv(filepath)
        df.rename(columns={"DATE": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        supplementary_data[name] = df

    return supplementary_data


def load_specialized_datasets(base_path):
    """
    Load specialized datasets including Misawa, Arlt, Tevel, Spoerer, and Teague data.

    Args:
        base_path (str): Base directory path where data files are located.

    Returns:
        dict: Dictionary containing specialized datasets with observer IDs
    """
    print("Loading specialized datasets...")

    specialized_data = {}

    # Load Misawa data (Excel format with multiple sheets)
    misawa_path = os.path.join(base_path, "misawa_data.xlsx")
    misawa_excel = pd.read_excel(
        misawa_path,
        sheet_name=None,
        engine="openpyxl",
    )

    # Concatenate all Misawa sheets
    misawa_combined = pd.concat(misawa_excel.values(), sort=True)
    misawa_combined = misawa_combined.dropna(subset=["Year"])
    misawa_combined["Date"] = pd.to_datetime(
        misawa_combined[["Year", "Month", "Day"]], errors="coerce"
    )
    misawa_combined = misawa_combined[["Date", "Group", "Spot"]]
    misawa_combined = misawa_combined.replace("?", np.nan)
    misawa_combined["WOLF"] = 10 * misawa_combined["Group"].astype(
        float
    ) + misawa_combined["Spot"].astype(float)
    misawa_combined.columns = ["Date", "GROUPS", "SUNSPOTS", "WOLF"]
    misawa_combined["ALIAS"] = "Misawa"
    misawa_combined["FK_OBSERVERS"] = 1004
    specialized_data[1004] = misawa_combined

    # Load Schwabe-Arlt data
    arlt_path = os.path.join(base_path, "arlt1.csv")
    arlt_data = pd.read_csv(arlt_path)
    arlt_data.rename(
        columns={"Date_1": "Date", "Group": "GROUPS", "Spot": "SUNSPOTS", "SN": "WOLF"},
        inplace=True,
    )
    arlt_data["Date"] = pd.to_datetime(arlt_data["Date"])
    arlt_data["ALIAS"] = "Schwabe_Arlt"
    arlt_data["FK_OBSERVERS"] = 1000
    specialized_data[1000] = arlt_data

    # Load Tevel data
    tevel_path = os.path.join(base_path, "Tevel_s.txt")
    tevel_data = pd.read_csv(
        tevel_path,
        sep="\t",
        skiprows=13,
    )
    tevel_data["Date"] = pd.to_datetime(tevel_data[["YEAR", "MONTH", "DAY"]])
    tevel_data["WOLF"] = tevel_data["SUNSPOTS"] + 10 * tevel_data["GROUPS"]
    tevel_data["ALIAS"] = "Tevel(Carassco)"
    tevel_data["FK_OBSERVERS"] = 1001
    specialized_data[1001] = tevel_data

    # Load Spoerer data
    spoerer_path = os.path.join(base_path, "Spoerer_counts.csv")
    spoerer_data = pd.read_csv(spoerer_path)
    spoerer_data.rename(
        columns={
            "Date_1": "Date",
            "Groups": "GROUPS",
            "Sunspots": "SUNSPOTS",
            "Wolf": "WOLF",
        },
        inplace=True,
    )
    spoerer_data["Date"] = pd.to_datetime(spoerer_data["Date"])
    spoerer_data["ALIAS"] = "Spoerer (recounted)"
    spoerer_data["FK_OBSERVERS"] = 1002
    specialized_data[1002] = spoerer_data

    # Load Teague-Carrington data
    teague_path = os.path.join(base_path, "Teague_consolidated1 (1).csv")
    teague_data = pd.read_csv(teague_path)
    teague_data.rename(
        columns={
            "Date_1": "Date",
            "SN_Tot": "WOLF",
            "G_Tot": "GROUPS",
            "S_Tot": "SUNSPOTS",
        },
        inplace=True,
    )
    teague_data["Date"] = pd.to_datetime(teague_data["Date"])
    teague_data["FK_OBSERVERS"] = 1003
    teague_data["ALIAS"] = "Carrington Teague"
    specialized_data[1003] = teague_data

    return specialized_data


def load_wolf_source_book_data(base_path):
    """Load and process Wolf source book data.

    Args:
        base_path (str): Base directory path where data files are located.

    Returns:
        pd.DataFrame: Processed Wolf source book data.
                   Returns an empty DataFrame if the file is not found or if an error occurs during loading.
    """
    print("\033[91mLoading Wolf source book data...\033[0m")
    wolf_source_book_path = os.path.join(base_path, "wolf_source_book.csv")

    try:
        df = pd.read_csv(wolf_source_book_path, sep=",")
    except FileNotFoundError:
        print(
            f"Wolf source book data file not found at {wolf_source_book_path} - returning empty DataFrame"
        )
        return pd.DataFrame()  # Return empty DataFrame if file not found
    except Exception as e:  # Catch other potential errors during file loading
        print(f"Error loading Wolf source book {e} - returning empty DataFrame")
        return pd.DataFrame()

    df.rename(columns={"date": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    dd = dict(tuple(df.groupby("obsID")))

    wolf_s = dd[2].copy()
    # wolf_p = dd[4].copy()  # This line is not used later, so it's removed

    wolf_s.rename(
        columns={"wolf_norm": "WOLF", "groups": "GROUPS", "sunspots": "SUNSPOTS"},
        inplace=True,
    )
    wolf_s["ALIAS"] = "WOLF_SM"
    wolf_s["FK_OBSERVERS"] = 2

    return wolf_s  # Return the processed DataFrame directly


# =============================================================================
# DATA PROCESSING AND QUALITY CONTROL
# =============================================================================


def apply_quality_filters(observer_groups):
    """
    Apply general quality filters to observer data and add plotting flags.
    Specific truncations and filters are handled in load_and_preprocess_main_data.

    Args:
        observer_groups (dict): Dictionary of observer DataFrames.

    Returns:
        dict: Filtered and flagged observer data.
    """
    print("\033[91mApplying general quality filters...\033[0m")

    filtered_groups = {}
    observer_ids = list(observer_groups.keys())

    # Remove mixed Wolf data (observer ID 2)
    # This was originally handled by `del dd[2]`
    if 2 in observer_ids:
        observer_ids.remove(2)

    for i, obs_id in enumerate(observer_ids):
        df = observer_groups[obs_id].copy()

        # Apply general quality filters
        # Filter out unrealistic group counts
        df = df[df["GROUPS"] < 15]
        df["flagged"] = i  # Add plotting flag for visualization

        filtered_groups[obs_id] = df
        # print(f"Observer {obs_id}: {len(df)} observations after general filtering")

    return filtered_groups


def merge_supplementary_data(filtered_groups, supplementary_data, specialized_data):
    """
    Merge supplementary datasets with main observer groups.

    Args:
        filtered_groups (dict): Main observer data.
        supplementary_data (dict): Supplementary datasets.
        specialized_data (dict): Specialized datasets.

    Returns:
        dict: Combined dataset with all observations.
    """

    # Merge supplementary data with existing observers based on original logic
    merge_mapping = {
        139: "stempell",
        114: "woinoff",
        78: "schwab",
        77: "quimby",
        57: "ventosa",
    }

    for obs_id, data_key in merge_mapping.items():
        if obs_id in filtered_groups and data_key in supplementary_data:
            filtered_groups[obs_id] = pd.concat(
                [filtered_groups[obs_id], supplementary_data[data_key]],
                axis=0,
                sort=True,
            )
            print(f"\033[91mMerged {data_key} into observer {obs_id}.\033[0m")

    # Handle Williston Observatory (ID 150) merger into ID 142
    if 150 in filtered_groups:
        print("\033[91mMerging Williston Observatory (ID 150) into ID 142...\033[0m")
        # Ensure we're modifying a copy if it might be used elsewhere
        williston_data = filtered_groups[150].copy()
        williston_data["ALIAS"] = "Williston Observatory"
        williston_data["FK_OBSERVERS"] = 142

        if 142 in filtered_groups:
            filtered_groups[142] = pd.concat(
                [filtered_groups[142], williston_data], axis=0, sort=True
            )
        else:  # If 142 doesn't exist, create it with 150's data
            filtered_groups[142] = williston_data

        del filtered_groups[150]  # Remove original ID 150

    # Add specialized datasets
    filtered_groups.update(specialized_data)
    print(f"Added {len(specialized_data)} specialized datasets.")

    return filtered_groups


# =============================================================================
# BACKBONE ASSIGNMENT AND K-FACTOR CALCULATION
# =============================================================================


def assign_observers_to_backbones(observer_groups):
    """
    Assign observers to the most appropriate backbone based on overlap.

    The three backbones are:
    - Wolfer (observer 56): Primary reference
    - Weber (observer 32): Secondary reference
    - Schwabe (observer 1000): Historical reference

    Args:
        observer_groups (dict): Dictionary of observer DataFrames.

    Returns:
        tuple: Three dictionaries for Wolfer, Weber, and Schwabe backbones.
    """
    print("Assigning observers to backbones...")

    # Define backbone reference observers
    # Ensure backbone observers exist before proceeding
    wolfer_ref = observer_groups.get(56)
    weber_ref = observer_groups.get(32)
    schwabe_ref = observer_groups.get(1000)

    if not all(
        [wolfer_ref is not None, weber_ref is not None, schwabe_ref is not None]
    ):
        print(
            "Warning: One or more backbone reference observers are missing. Some assignments may be incomplete."
        )
        # Proceed with available backbones, but functions relying on all three might fail.

    # Initialize backbone dictionaries
    wolfer_backbone = {}
    weber_backbone = {}
    schwabe_backbone = {}

    observer_ids = list(observer_groups.keys())

    for obs_id in observer_ids:
        # Add backbone observers to their own backbone
        if obs_id == 56 and wolfer_ref is not None:
            wolfer_backbone[obs_id] = observer_groups[obs_id]
            continue
        if obs_id == 32 and weber_ref is not None:
            weber_backbone[obs_id] = observer_groups[obs_id]
            continue
        if obs_id == 1000 and schwabe_ref is not None:
            schwabe_backbone[obs_id] = observer_groups[obs_id]
            continue

        # Skip processing if backbone ref is missing for overlap calculation
        if wolfer_ref is None and weber_ref is None and schwabe_ref is None:
            continue

        observer_data = observer_groups[obs_id]

        wolfer_overlap = 0
        if wolfer_ref is not None:
            wolfer_overlap = len(
                pd.merge(wolfer_ref, observer_data, on="Date", how="inner")
            )

        weber_overlap = 0
        if weber_ref is not None:
            weber_overlap = len(
                pd.merge(weber_ref, observer_data, on="Date", how="inner")
            )

        schwabe_overlap = 0
        if schwabe_ref is not None:
            schwabe_overlap = len(
                pd.merge(schwabe_ref, observer_data, on="Date", how="inner")
            )

        # Assign to backbone with minimum overlap threshold of 20
        min_overlap = 20
        if (
            wolfer_overlap < min_overlap
            and weber_overlap < min_overlap
            and schwabe_overlap < min_overlap
        ):
            # print(f"Observer {obs_id} excluded due to insufficient overlap ({wolfer_overlap}, {weber_overlap}, {schwabe_overlap}).")
            continue

        # Assign to backbone with maximum overlap, prioritizing Wolfer, then Weber, then Schwabe
        if (
            wolfer_overlap > weber_overlap
            and wolfer_overlap > schwabe_overlap
            and wolfer_ref is not None
        ):
            wolfer_backbone[obs_id] = observer_data
        elif (
            weber_overlap > wolfer_overlap
            and weber_overlap > schwabe_overlap
            and weber_ref is not None
        ):
            weber_backbone[obs_id] = observer_data
        elif schwabe_ref is not None:  # Catch all if schwabe is the only one or highest
            schwabe_backbone[obs_id] = observer_data
        else:
            # print(f"Observer {obs_id} could not be assigned to a backbone.")
            pass

    print(f"Wolfer backbone: {len(wolfer_backbone)} observers")
    print(f"Weber backbone: {len(weber_backbone)} observers")
    print(f"Schwabe backbone: {len(schwabe_backbone)} observers")

    return wolfer_backbone, weber_backbone, schwabe_backbone


def plot_observer_coverage(
    backbone_dict,
    backbone_name,
    output_dir="plots",
    width_in_inches=10,
    height_in_inches=15,
    dots_per_inch=100,
    color_palette=None,
):
    """
    Plots observer coverage for a given backbone and saves the figure.

    Args:
        backbone_dict (dict): Dictionary where keys are observer IDs and values are DataFrames.
        backbone_name (str): Name of the backbone (e.g., "Wolfer", "Weber", "Schwabe").
        output_dir (str): Directory to save the plots. Defaults to "plots".
        width_in_inches (int): Width of the figure in inches. Defaults to 10.
        height_in_inches (int): Height of the figure in inches. Defaults to 15.
        dots_per_inch (int): DPI of the figure. Defaults to 100.
        color_palette (list, optional): List of colors to use for the lines. If None, a default palette is used.
    """

    n_observers = len(backbone_dict)
    if color_palette is None:
        # Use a colormap to generate a list of colors
        cmap = plt.get_cmap("viridis")  # Or any other colormap you prefer
        color_palette = [cmap(i / n_observers) for i in range(n_observers)]
    elif len(color_palette) < n_observers:
        print(
            f"Warning: Provided color palette has fewer colors ({len(color_palette)}) than observers ({n_observers}). Colors will be cycled."
        )

    plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dots_per_inch)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)  # Keep the grid

    ind = np.linspace(
        0, n_observers - 1, num=n_observers
    )  # Modified to match number of observers
    li = []  # List to store observer aliases or IDs for y-axis ticks

    for i, (obs_id, obs_data) in enumerate(backbone_dict.items()):
        start_date = obs_data["Date"].min()
        end_date = obs_data["Date"].max()
        c = color_palette[i % len(color_palette)]  # Cycle through colors if needed
        plt.hlines(
            ind[i],  # Use ind for y-position
            start_date,
            end_date,
            colors=c,  # Use color from palette
            lw=2,
            label=obs_data["ALIAS"].iloc[0]
            if "ALIAS" in obs_data.columns
            else f"Observer {obs_id}",
        )
        li.append(
            obs_data["ALIAS"].iloc[0]
            if "ALIAS" in obs_data.columns
            else f"Observer {obs_id}"
        )

    plt.yticks(ind, li, rotation="horizontal", fontsize=8)
    plt.xlim(
        pd.to_datetime("1815-01-01"), pd.to_datetime("1950-01-01")
    )  # Wider time range
    plt.xticks(fontsize=15)
    plt.xlabel("Time (in years)", fontsize=15)
    plt.title(f"Observer Coverage for {backbone_name} Backbone", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure in PDF and PNG formats
    pdf_filename = os.path.join(output_dir, f"observer_coverage_{backbone_name}.pdf")
    png_filename = os.path.join(output_dir, f"observer_coverage_{backbone_name}.png")

    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.savefig(png_filename, bbox_inches="tight")
    print(
        f"Saved {backbone_name} observer coverage plot to {pdf_filename} and {png_filename}"
    )

    plt.show()


def plot_observer_coverageold(backbone_dict, backbone_name, output_dir="plots"):
    """Plots observer coverage for a given backbone and saves the figure."""
    plt.figure(figsize=(15, 6))  # Adjust size as needed
    for i, (obs_id, obs_data) in enumerate(backbone_dict.items()):
        start_date = obs_data["Date"].min()
        end_date = obs_data["Date"].max()
        plt.hlines(
            i,
            start_date,
            end_date,
            colors="steelblue",
            lw=2,
            label=obs_data["ALIAS"].iloc[0]
            if "ALIAS" in obs_data.columns
            else f"Observer {obs_id}",
        )  # Use alias if available

    plt.yticks(
        range(len(backbone_dict)),
        [
            obs_data["ALIAS"].iloc[0]
            if "ALIAS" in obs_data.columns
            else f"Observer {obs_id}"
            for obs_id, obs_data in backbone_dict.items()
        ],
        rotation="horizontal",
        fontsize=8,
    )  # Use alias if available
    plt.xlabel("Time (in years)", fontsize=15)
    plt.title(f"Observer Coverage for {backbone_name} Backbone", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure in PDF and PNG formats
    pdf_filename = os.path.join(output_dir, f"observer_coverage_{backbone_name}.pdf")
    png_filename = os.path.join(output_dir, f"observer_coverage_{backbone_name}.png")

    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.savefig(png_filename, bbox_inches="tight")
    print(
        f"Saved {backbone_name} observer coverage plot to {pdf_filename} and {png_filename}"
    )

    plt.show()


# Call this function after assigning observers to backbones
# Ensure wolfer_backbone_full, weber_backbone_full, and schwabe_backbone_full
# are defined and populated with your observer data before calling these functions.


# Example usage (assuming your backbone data structures are ready):
# plot_observer_coverage(wolfer_backbone_full, "Wolfer")
# plot_observer_coverage(weber_backbone_full, "Weber")
# plot_observer_coverage(schwabe_backbone_full, "Schwabe")
#


def plot_backbone_time_series(
    backbone_df_normalized, backbone_name, output_dir="plots"
):
    """
    Plots the time series for a backbone observer and saves the figure
    in PDF and PNG formats.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(
        backbone_df_normalized["Date"],
        backbone_df_normalized["WOLF"],
        ".",
        label="Raw Sunspot Number",
    )  # Assuming 'WOLF' is still available after normalization
    plt.plot(
        backbone_df_normalized["Date"],
        backbone_df_normalized["WOLF_n"],
        ".",
        label="Normalized Sunspot Number",
    )
    plt.xlabel("Time (in years)", fontsize=15)
    plt.ylabel("Sunspot Number", fontsize=15)
    plt.title(f"{backbone_name} Backbone Time Series", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define filenames for saving
    pdf_filename = os.path.join(output_dir, f"time_series_{backbone_name}.pdf")
    png_filename = os.path.join(output_dir, f"time_series_{backbone_name}.png")

    # Save the figure in PDF and PNG formats
    plt.savefig(pdf_filename, bbox_inches="tight")
    plt.savefig(png_filename, bbox_inches="tight")
    print(
        f"Saved {backbone_name} backbone time series plot to {pdf_filename} and {png_filename}"
    )

    plt.show()


# Example usage (after constructing normalized series and assuming all_combined_data is available):
# Ensure all_combined_data is a dictionary where keys are observer IDs
# and values are pandas DataFrames containing 'Date', 'WOLF', and 'WOLF_n' columns.
# plot_backbone_time_series(all_combined_data[56], "Wolfer") # Assuming 56 is Wolfer's ID
# plot_backbone_time_series(all_combined_data[32], "Weber") # Assuming 32 is Weber's ID
# plot_backbone_time_series(all_combined_data[1000], "Schwabe") # Assuming 1000 is Schwabe's ID


def summarize_tevel_backbone_observers(
    all_combined_data, tevel_backbone_full, tevel_k_factors_initial_df
):
    """
    Generates a summary table of observers overlapping with the Tevel backbone.

    Args:
        all_combined_data (dict): Dictionary of all combined observer data.
        tevel_backbone_full (dict): Dictionary of observers in the Tevel backbone.
        tevel_k_factors_initial_df (pd.DataFrame): DataFrame of initial k-factors for the Tevel backbone.

    Returns:
        pd.DataFrame: Summary table of Tevel backbone observers.
    """

    if not tevel_backbone_full or tevel_k_factors_initial_df.empty:
        print(
            "Tevel backbone data or k-factors are missing. Returning empty DataFrame."
        )
        return pd.DataFrame()

    summary_data = []
    for obs_id, obs_data in tevel_backbone_full.items():
        k_factor_row = tevel_k_factors_initial_df[
            tevel_k_factors_initial_df["FK_OBSERVERS"] == obs_id
        ]
        if k_factor_row.empty:
            print(
                f"No k-factor found for observer {obs_id} in Tevel backbone. Skipping."
            )
            continue

        k = k_factor_row["k"].iloc[0]
        dk = k_factor_row["dk"].iloc[0]

        name = (
            all_combined_data.get(obs_id, pd.DataFrame())["ALIAS"].iloc[0]
            if all_combined_data.get(obs_id) is not None
            and not all_combined_data.get(obs_id).empty
            else f"ID {obs_id}"
        )
        no_obs = len(obs_data)
        start = obs_data["Date"].min()
        end = obs_data["Date"].max()
        overlapped_obs = k_factor_row["Overlapped obs"].iloc[0]
        zero_obs = k_factor_row["Zero obs"].iloc[0]

        summary_data.append(
            {
                "ID": obs_id,
                "Name": name,
                "No.of Obs": no_obs,
                "Overlapped obs": overlapped_obs,
                "Zero obs": zero_obs,
                "Start": start,
                "End": end,
                "k": k,
                "δk": dk,
                "Slope (TLS)": np.nan,  # Placeholder for TLS slope
                "Err": np.nan,  # Placeholder for TLS error
            }
        )

    return pd.DataFrame(summary_data)


def summarize_schwabe_backbone_observers(
    all_combined_data, schwabe_backbone_full, schwabe_k_factors_initial_df
):
    """
    Generates a summary table of observers overlapping with the Schwabe backbone.

    Args:
        all_combined_data (dict): Dictionary of all combined observer data.
        schwabe_backbone_full (dict): Dictionary of observers in the Schwabe backbone.
        schwabe_k_factors_initial_df (pd.DataFrame): DataFrame of initial k-factors for the Schwabe backbone.

    Returns:
        pd.DataFrame: Summary table of Schwabe backbone observers.
    """

    if not schwabe_backbone_full or schwabe_k_factors_initial_df.empty:
        print(
            "Schwabe backbone data or k-factors are missing. Returning empty DataFrame."
        )
        return pd.DataFrame()

    summary_data = []
    for obs_id, obs_data in schwabe_backbone_full.items():
        k_factor_row = schwabe_k_factors_initial_df[
            schwabe_k_factors_initial_df["FK_OBSERVERS"] == obs_id
        ]
        if k_factor_row.empty:
            print(
                f"No k-factor found for observer {obs_id} in Schwabe backbone. Skipping."
            )
            continue

        k = k_factor_row["k"].iloc[0]
        dk = k_factor_row["dk"].iloc[0]
        name = (
            all_combined_data.get(obs_id, pd.DataFrame())["ALIAS"].iloc[0]
            if all_combined_data.get(obs_id) is not None
            and not all_combined_data.get(obs_id).empty
            else f"ID {obs_id}"
        )
        no_obs = len(obs_data)
        start = obs_data["Date"].min()
        end = obs_data["Date"].max()
        overlapped_obs = k_factor_row["Overlapped obs"].iloc[0]
        zero_obs = k_factor_row["Zero obs"].iloc[0]

        summary_data.append(
            {
                "ID": obs_id,
                "Name": name,
                "No.of Obs": no_obs,
                "Overlapped obs": overlapped_obs,
                "Zero obs": zero_obs,
                "Start": start,
                "End": end,
                "k": k,
                "δk": dk,
                # "Slope (TLS)": np.nan,  # Placeholder for TLS slope (not available)
                # "Err": np.nan,        # Placeholder for TLS error (not available)
            }
        )

    return pd.DataFrame(summary_data)


def calculate_k_factors(backbone_data, observer_data_dict, backbone_id):
    """
    Calculate k-factors (scaling factors) for observers relative to a backbone observer.

    Parameters:
    - backbone_data (pd.DataFrame): DataFrame containing the backbone observer's data.
    - observer_data_dict (dict): Dictionary of observer DataFrames (including backbone).
    - backbone_id (int): ID of the backbone observer.

    Returns:
    - pd.DataFrame: DataFrame with k-factors and statistics for each observer.
    """
    if backbone_data is None or backbone_data.empty:
        # print(f"Cannot calculate k-factors for backbone ID {backbone_id}: Backbone data is empty or None.")
        return pd.DataFrame()  # Return empty DataFrame if backbone data is missing

    # print(f"Calculating k-factors for backbone ID {backbone_id}...")
    results = []

    for obs_id, obs_data in observer_data_dict.items():
        if obs_id == backbone_id:
            # Backbone's k-factor is 1.0 with itself
            results.append(
                {
                    "FK_OBSERVERS": obs_id,
                    "Name": obs_data["ALIAS"].iloc[0]
                    if not obs_data.empty and "ALIAS" in obs_data.columns
                    else f"Observer {obs_id}",
                    "No.of Obs": len(obs_data),
                    "Overlapped obs": len(obs_data),
                    "Zero obs": (obs_data["WOLF"] == 0).sum(),
                    "Start": obs_data["Date"].min() if not obs_data.empty else None,
                    "End": obs_data["Date"].max() if not obs_data.empty else None,
                    "k": 1.0,
                    "dk": 0.0,
                }
            )
            continue

        # Ensure both dataframes have 'WOLF' column
        if "WOLF" not in backbone_data.columns or "WOLF" not in obs_data.columns:
            # print(f"Skipping k-factor calculation for {obs_id}: Missing 'WOLF' column.")
            continue

        # Merge backbone data with observer data on Date
        merged_data = pd.merge(
            backbone_data,
            obs_data,
            on="Date",
            how="inner",
            suffixes=("_backbone", "_obs"),
        )

        if merged_data.empty or len(merged_data) < 20:  # Skip if insufficient overlap
            # print(f"Skipping k-factor for {obs_id}: Insufficient overlap or empty merged data.")
            continue

        # Calculate basic statistics
        start_date = obs_data["Date"].min()
        end_date = obs_data["Date"].max()
        total_obs = len(obs_data)
        overlapped_obs = len(merged_data)
        zero_obs = (obs_data["WOLF"] == 0).sum()
        observer_name = (
            obs_data["ALIAS"].iloc[0]
            if not obs_data.empty and "ALIAS" in obs_data.columns
            else f"Observer {obs_id}"
        )

        # Calculate ratio (backbone/observer) and handle infinities
        # Avoid division by zero by filtering out zero values in 'WOLF_obs'
        ratio_data = merged_data[merged_data["WOLF_obs"] != 0].copy()
        if ratio_data.empty:
            # print(f"Skipping k-factor for {obs_id}: No non-zero 'WOLF_obs' for ratio calculation.")
            continue
        ratio_data["Ratio"] = ratio_data["WOLF_backbone"] / ratio_data["WOLF_obs"]
        ratio_data["Ratio"] = ratio_data["Ratio"].replace([np.inf, -np.inf], np.nan)
        ratio_data = ratio_data.dropna(subset=["Ratio"])

        # Resample to monthly means to reduce noise
        monthly_ratios = ratio_data.set_index("Date").resample("M")["Ratio"].mean()
        monthly_ratios = monthly_ratios.dropna()

        if len(monthly_ratios) == 0:
            # print(f"Skipping k-factor for {obs_id}: No valid monthly ratios after resampling.")
            continue

        # Calculate k-factor statistics
        k_factor = monthly_ratios.mean()
        k_factor_std = monthly_ratios.std()

        # Store results
        results.append(
            {
                "FK_OBSERVERS": obs_id,
                "Name": observer_name,
                "No.of Obs": total_obs,
                "Overlapped obs": overlapped_obs,
                "Zero obs": zero_obs,
                "Start": start_date,
                "End": end_date,
                "k": k_factor,
                "dk": k_factor_std,
            }
        )
    return pd.DataFrame(results)


def construct_normalized_series(observer_df, k_factors_df, max_wolf=300):
    """
    Construct normalized sunspot series using k-factors.

    Parameters:
    - observer_df (pd.DataFrame): Combined DataFrame with all observations for a backbone.
    - k_factors_df (pd.DataFrame): DataFrame with k-factors for each observer in observer_df.
    - max_wolf (int): Maximum allowed Wolf number (for filtering outliers).

    Returns:
    - pd.DataFrame: Normalized individual observations.
    - pd.DataFrame: Daily aggregated normalized values.
    """
    if observer_df.empty or k_factors_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Ensure 'FK_OBSERVERS' is in both DataFrames
    if (
        "FK_OBSERVERS" not in observer_df.columns
        or "FK_OBSERVERS" not in k_factors_df.columns
    ):
        # print("Both observer_df and k_factors_df must contain 'FK_OBSERVERS' column for normalization.")
        return pd.DataFrame(), pd.DataFrame()

    # Ensure 'Date' and 'WOLF' are in observer_df
    if "Date" not in observer_df.columns or "WOLF" not in observer_df.columns:
        # print("observer_df must contain 'Date' and 'WOLF' columns for normalization.")
        return pd.DataFrame(), pd.DataFrame()

    # Merge observer data with k-factors
    normalized_data = pd.merge(
        observer_df, k_factors_df[["FK_OBSERVERS", "k"]], on="FK_OBSERVERS", how="left"
    )
    normalized_data.rename(
        columns={"k_y": "k"}, inplace=True
    )  # Or k_x to k if that's the name
    # Drop observations for which no k-factor was found (i.e., 'k' is NaN)
    normalized_data = normalized_data.dropna(subset=["k"])

    # Remove null Wolf numbers and apply k-factor normalization
    normalized_data = normalized_data[normalized_data["WOLF"].notnull()]
    normalized_data["WOLF_n"] = normalized_data["WOLF"] * normalized_data["k"]

    # Filter out extreme values
    normalized_data = normalized_data[normalized_data["WOLF_n"] < max_wolf]

    # Aggregate to daily values (mean, std, count)
    daily_aggregated = (
        normalized_data.groupby("Date")["WOLF_n"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return normalized_data, daily_aggregated


def calculate_backbone_scaling_factors(backbone1_norm_data, backbone2_norm_data):
    """
    Calculate scaling factor between two backbone observers' normalized series.

    Parameters:
    - backbone1_norm_data (pd.DataFrame): First backbone observer's normalized data (e.g., Wolfer).
    - backbone2_norm_data (pd.DataFrame): Second backbone observer's normalized data (e.g., Weber).

    Returns:
    - Tuple of (scaling_factor, std_deviation)
    """
    if backbone1_norm_data.empty or backbone2_norm_data.empty:
        # print("One or both backbone normalized dataframes are empty for scaling factor calculation.")
        return np.nan, np.nan

    # Merge the two backbone datasets
    merged = pd.merge(
        backbone1_norm_data,
        backbone2_norm_data,
        on="Date",
        how="inner",
        suffixes=("_1", "_2"),
    )

    if merged.empty:
        # print("No overlap between backbones for scaling factor calculation.")
        return np.nan, np.nan

    # Calculate ratio and handle infinities
    # Filter out zero values in the denominator to avoid division by zero
    ratio_data = merged[
        merged["WOLF_n_2"] != 0
    ].copy()  # Using WOLF_n for already normalized data
    if ratio_data.empty:
        # print("No non-zero WOLF_n_2 values for ratio calculation.")
        return np.nan, np.nan

    ratio_data["Ratio"] = ratio_data["WOLF_n_1"] / ratio_data["WOLF_n_2"]
    ratio_data["Ratio"] = ratio_data["Ratio"].replace([np.inf, -np.inf], np.nan)
    ratio_data = ratio_data.dropna(subset=["Ratio"])

    if ratio_data.empty:
        # print("No valid ratios after dropping NaNs and infinities.")
        return np.nan, np.nan

    # Calculate statistics
    scaling_factor = ratio_data["Ratio"].mean()
    std_deviation = ratio_data["Ratio"].std()
    return scaling_factor, std_deviation


# ============================================================================
# JACKKNIFE ANALYSIS (LEAVE-ONE-OUT VALIDATION)
# ============================================================================


def run_jackknife_analysis(
    all_combined_data,
    initial_wolfer_backbone_observers,
    initial_weber_backbone_observers,
    initial_schwabe_backbone_observers,
    annual_unified_original,
):
    """
    Orchestrates the complete jackknife analysis.
    Performs leave-one-out validation by systematically removing observers from each backbone,
    recalculating k-factors, normalizing, scaling, and combining the series for each iteration.

    Parameters:
    - all_combined_data (dict): The complete dictionary of observer DataFrames.
    - initial_wolfer_backbone_observers (dict): Initial assignment of observers to Wolfer backbone.
    - initial_weber_backbone_observers (dict): Initial assignment of observers to Weber backbone.
    - initial_schwabe_backbone_observers (dict): Initial assignment of observers to Schwabe backbone.
    - annual_unified_original (pd.DataFrame): The original, non-jackknifed annual unified series.

    Returns:
    - tuple: (jackknife_results_dict, jackknife_metadata_df, all_k_factors_iterations_dict)
        - jackknife_results_dict (dict): Annual unified series for each jackknife iteration.
        - jackknife_metadata_df (pd.DataFrame): Metadata about excluded observers in each iteration.
        - all_k_factors_iterations_dict (dict): K-factors for all observers in each iteration.
    """
    print("\n=== PERFORMING JACKKNIFE ANALYSIS ===")

    jackknife_results = {}  # Stores annual series for each iteration
    jackknife_metadata = []  # Stores info about excluded observers
    all_k_factors_iterations = {}  # Stores k-factors for all observers in each iteration

    # Get unique observer IDs from the initial backbone assignments
    wolfer_obs_ids = list(initial_wolfer_backbone_observers.keys())
    weber_obs_ids = list(initial_weber_backbone_observers.keys())
    schwabe_obs_ids = list(initial_schwabe_backbone_observers.keys())

    # Ensure backbone reference IDs are always in the list to avoid errors if they're the only one
    if 56 not in wolfer_obs_ids and 56 in all_combined_data:
        wolfer_obs_ids.append(56)
    if 32 not in weber_obs_ids and 32 in all_combined_data:
        weber_obs_ids.append(32)
    if 1000 not in schwabe_obs_ids and 1000 in all_combined_data:
        schwabe_obs_ids.append(1000)

    # Determine the maximum number of iterations based on the longest backbone list
    max_iterations = max(len(wolfer_obs_ids), len(weber_obs_ids), len(schwabe_obs_ids))

    for i in range(max_iterations):
        # Determine which observer to exclude from each backbone for this iteration
        # Use modulo to cycle through observers if backbone lists have different lengths
        excluded_wolfer_id = wolfer_obs_ids[i % len(wolfer_obs_ids)]
        excluded_weber_id = weber_obs_ids[i % len(weber_obs_ids)]
        excluded_schwabe_id = schwabe_obs_ids[i % len(schwabe_obs_ids)]

        print(
            f"Iteration {i + 1}/{max_iterations}: Excluding IDs - Wolfer: {excluded_wolfer_id}, Weber: {excluded_weber_id}, Schwabe: {excluded_schwabe_id}"
        )

        # Create a temporary copy of all observer data for this iteration
        current_iter_data = all_combined_data.copy()

        # Remove the 'left-out' observer's data from this iteration's set
        # IMPORTANT: Do NOT remove the backbone reference observer (ID 56, 32, 1000) themselves if they are selected for exclusion.
        # This ensures there's always a reference to calculate k-factors against.
        if excluded_wolfer_id != 56 and excluded_wolfer_id in current_iter_data:
            current_iter_data.pop(excluded_wolfer_id)
        if excluded_weber_id != 32 and excluded_weber_id in current_iter_data:
            current_iter_data.pop(excluded_weber_id)
        if excluded_schwabe_id != 1000 and excluded_schwabe_id in current_iter_data:
            current_iter_data.pop(excluded_schwabe_id)

        # Re-assign remaining observers to backbones based on the current_iter_data
        iter_wolfer_backbone, iter_weber_backbone, iter_schwabe_backbone = (
            assign_observers_to_backbones(current_iter_data)
        )

        # Recalculate k-factors for ALL remaining observers against their respective backbones for this iteration
        wolfer_k_factors_iter_df = calculate_k_factors(
            current_iter_data.get(56), iter_wolfer_backbone, 56
        )
        weber_k_factors_iter_df = calculate_k_factors(
            current_iter_data.get(32), iter_weber_backbone, 32
        )
        schwabe_k_factors_iter_df = calculate_k_factors(
            current_iter_data.get(1000), iter_schwabe_backbone, 1000
        )

        # Store these k-factors for `all_k_factors_iterations` (keyed by observer alias)
        current_iter_k_factors_map = {}
        for df_k in [
            wolfer_k_factors_iter_df,
            weber_k_factors_iter_df,
            schwabe_k_factors_iter_df,
        ]:
            if not df_k.empty:
                for _, row in df_k.iterrows():
                    current_iter_k_factors_map[row["Name"]] = row["k"]
        all_k_factors_iterations[i] = current_iter_k_factors_map

        # Construct normalized series for each backbone using the *recalculated* k-factors for this iteration
        wolfer_norm_iter, _ = construct_normalized_series(
            pd.concat(iter_wolfer_backbone.values())
            if iter_wolfer_backbone
            else pd.DataFrame(),
            wolfer_k_factors_iter_df,
            max_wolf=300,
        )
        weber_norm_iter, _ = construct_normalized_series(
            pd.concat(iter_weber_backbone.values())
            if iter_weber_backbone
            else pd.DataFrame(),
            weber_k_factors_iter_df,
            max_wolf=300,
        )
        schwabe_norm_iter, _ = construct_normalized_series(
            pd.concat(iter_schwabe_backbone.values())
            if iter_schwabe_backbone
            else pd.DataFrame(),
            schwabe_k_factors_iter_df,
            max_wolf=300,
        )

        # Ensure 'Date' and 'WOLF_n' are present in normalized dataframes, if not, create empty ones
        for df_n in [wolfer_norm_iter, weber_norm_iter, schwabe_norm_iter]:
            if "Date" not in df_n.columns:
                df_n["Date"] = pd.NaT
            if "WOLF_n" not in df_n.columns:
                df_n["WOLF_n"] = np.nan

        # Recalculate inter-backbone scaling factors (k1_iter, k2_iter) for this iteration's normalized series
        k1_iter, _ = calculate_backbone_scaling_factors(
            wolfer_norm_iter, weber_norm_iter
        )
        k2_iter, _ = calculate_backbone_scaling_factors(
            weber_norm_iter, schwabe_norm_iter
        )

        # Apply scaling and combine series (logic adapted from user's `perform_jackknife_analysis`)
        weber_scaled_iter = weber_norm_iter.copy()
        if not pd.isna(k1_iter):
            weber_scaled_iter["WOLF_n"] = weber_scaled_iter["WOLF_n"] * k1_iter
        else:
            weber_scaled_iter["WOLF_n"] = np.nan  # If no valid k1, set to NaN

        schwabe_scaled_iter = schwabe_norm_iter.copy()
        if not pd.isna(k1_iter) and not pd.isna(k2_iter):
            schwabe_scaled_iter["WOLF_n"] = (
                schwabe_scaled_iter["WOLF_n"] * k1_iter * k2_iter
            )
        else:
            schwabe_scaled_iter["WOLF_n"] = np.nan  # If no valid k1 or k2, set to NaN

        combined_filtered_iter = pd.concat(
            [
                wolfer_norm_iter[["Date", "WOLF_n"]],
                weber_scaled_iter[["Date", "WOLF_n"]],
                schwabe_scaled_iter[["Date", "WOLF_n"]],
            ],
            axis=0,
            sort=True,
        )

        # Remove NaNs before resampling, especially if `WOLF_n` became NaN due to missing k-factors
        combined_filtered_iter = combined_filtered_iter.dropna(subset=["WOLF_n"])

        annual_filtered_iter = (
            combined_filtered_iter.set_index("Date")
            .resample("Y")["WOLF_n"]
            .mean()
            .reset_index()
        )
        annual_filtered_iter = annual_filtered_iter.dropna(
            subset=["WOLF_n"]
        )  # Drop years with no data

        # Store results for this iteration
        jackknife_results[i] = annual_filtered_iter

        # Get names for metadata using the original all_combined_data aliases
        wolfer_name = (
            all_combined_data[excluded_wolfer_id]["ALIAS"].iloc[0]
            if excluded_wolfer_id in all_combined_data
            and not all_combined_data[excluded_wolfer_id].empty
            and "ALIAS" in all_combined_data[excluded_wolfer_id].columns
            else f"ID {excluded_wolfer_id}"
        )
        weber_name = (
            all_combined_data[excluded_weber_id]["ALIAS"].iloc[0]
            if excluded_weber_id in all_combined_data
            and not all_combined_data[excluded_weber_id].empty
            and "ALIAS" in all_combined_data[excluded_weber_id].columns
            else f"ID {excluded_weber_id}"
        )
        schwabe_name = (
            all_combined_data[excluded_schwabe_id]["ALIAS"].iloc[0]
            if excluded_schwabe_id in all_combined_data
            and not all_combined_data[excluded_schwabe_id].empty
            and "ALIAS" in all_combined_data[excluded_schwabe_id].columns
            else f"ID {excluded_schwabe_id}"
        )

        jackknife_metadata.append(
            {
                "iteration": i,
                "Wolfer backbone excluded": wolfer_name,
                "Weber backbone excluded": weber_name,
                "Schwabe backbone excluded": schwabe_name,
            }
        )

    print("\nJackknife analysis complete.")
    return jackknife_results, pd.DataFrame(jackknife_metadata), all_k_factors_iterations


# ============================================================================
# VISUALIZATION AND COMPARISON
# ============================================================================


def create_comparison_plots(jackknife_results, annual_unified, base_path):
    """Create comprehensive comparison plots."""

    # Load SNV1 data for comparison
    snv1_annual = None
    try:
        snv1_data_path = os.path.join(base_path, "snv1_full.csv")
        snv1_data = pd.read_csv(snv1_data_path)
        snv1_data["Date"] = pd.to_datetime(snv1_data["Date"])
        snv1_data = snv1_data[snv1_data["Sn"] != -1]
        snv1_annual = (
            snv1_data.set_index("Date").resample("Y")["Sn"].mean().reset_index()
        )
        snv1_annual = snv1_annual[
            (snv1_annual["Date"] >= "1818-01-01")
            & (snv1_annual["Date"] <= "1945-01-01")
        ]
    except FileNotFoundError:  # Changed from FileError for robustness
        print(
            f"SNV1 data file not found at {snv1_data_path} - skipping comparison with SNV1"
        )
    except Exception as e:
        print(f"Error loading SNV1 data: {e} - skipping comparison with SNV1")

    # Create main comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot 1: All jackknife iterations
    ax1.set_title(
        "Jackknife Analysis: Leave-One-Out Validation", fontsize=14, fontweight="bold"
    )
    for i, annual_data in jackknife_results.items():
        ax1.plot(
            annual_data["Date"],
            annual_data["WOLF_n"],
            alpha=0.3,
            color="gray",
            linewidth=0.5,
        )

    # Overlay original unified series
    ax1.plot(
        annual_unified["Date"],
        annual_unified["WOLF_n"],
        color="red",
        linewidth=2,
        label="Original SNV2.2",
    )

    if snv1_annual is not None:
        ax1.plot(
            snv1_annual["Date"],
            snv1_annual["Sn"]
            * 1.6,  # Scaling factor might need adjustment based on SNV2.2 scale
            color="black",
            linewidth=2,
            label="SNV1 (scaled)",
        )

    ax1.set_ylabel("Sunspot Number", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Uncertainty bands
    ax2.set_title(
        "SNV2.2 Reconstruction with Uncertainty Bands", fontsize=14, fontweight="bold"
    )

    # Calculate percentiles across jackknife results
    # Use dates from the original unified series to ensure consistent time axis
    all_dates = annual_unified["Date"]
    percentiles = []

    for date in all_dates:
        values = []
        for annual_data in jackknife_results.values():
            matching_rows = annual_data[annual_data["Date"] == date]
            if not matching_rows.empty:
                values.append(matching_rows["WOLF_n"].iloc[0])

        if values:
            percentiles.append(
                {
                    "Date": date,
                    "p10": np.percentile(values, 10),
                    "p25": np.percentile(values, 25),
                    "p50": np.percentile(values, 50),
                    "p75": np.percentile(values, 75),
                    "p90": np.percentile(values, 90),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
            )
        else:
            # Append NaNs for dates where no jackknife results are available
            percentiles.append(
                {
                    "Date": date,
                    "p10": np.nan,
                    "p25": np.nan,
                    "p50": np.nan,
                    "p75": np.nan,
                    "p90": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                }
            )

    percentiles_df = pd.DataFrame(percentiles)
    percentiles_df = percentiles_df.dropna(
        subset=["p50"]
    )  # Drop rows where no percentile could be calculated

    # Plot uncertainty bands
    ax2.fill_between(
        percentiles_df["Date"],
        percentiles_df["p10"],
        percentiles_df["p90"],
        alpha=0.2,
        color="blue",
        label="80% Confidence Band",
    )
    ax2.fill_between(
        percentiles_df["Date"],
        percentiles_df["p25"],
        percentiles_df["p75"],
        alpha=0.3,
        color="blue",
        label="50% Confidence Band",
    )

    # Plot median and original
    ax2.plot(
        percentiles_df["Date"],
        percentiles_df["p50"],
        color="blue",
        linewidth=2,
        label="Jackknife Median",
    )
    ax2.plot(
        annual_unified["Date"],
        annual_unified["WOLF_n"],
        color="red",
        linewidth=2,
        label="Original SNV2.2",
    )

    if snv1_annual is not None:
        ax2.plot(
            snv1_annual["Date"],
            snv1_annual["Sn"] * 1.6,
            color="black",
            linewidth=2,
            label="SNV1 (scaled)",
        )

    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Sunspot Number", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("jackknife_comparison.png", dpi=300, bbox_inches="tight")
    # plt.show() # Commented out to prevent blocking in non-interactive environments

    return percentiles_df


def create_k_factor_evolution_plot(all_k_factors_iterations):
    """Create plot showing evolution of k-factors over time."""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Evolution of K-factors in Jackknife Analysis", fontsize=16, fontweight="bold"
    )

    if not all_k_factors_iterations:
        print("No k-factor iterations available for plotting.")
        plt.close(fig)  # Close the empty figure
        return

    # Get all unique observer aliases that have k-factors in any iteration
    all_observer_aliases = set()
    for k_factors_map in all_k_factors_iterations.values():
        all_observer_aliases.update(k_factors_map.keys())

    # Define a list of important observers for plotting, prioritize these
    important_observers_plot = [
        "WOLF",
        "RUDOLPH",
        "SCHWABE",
        "CARRINGTON",
        "AIRY",
    ]  # Added AIRY based on common aliases

    # Filter to only include observers that are actually present in the data
    available_observers_for_plot = [
        obs for obs in important_observers_plot if obs in all_observer_aliases
    ]

    # If fewer than 4 important observers, add other available observers
    if len(available_observers_for_plot) < 4:
        remaining_observers = sorted(
            list(all_observer_aliases - set(available_observers_for_plot))
        )
        available_observers_for_plot.extend(
            remaining_observers[: 4 - len(available_observers_for_plot)]
        )

    # Limit to top 4 for the 2x2 grid
    available_observers_for_plot = available_observers_for_plot[:4]

    for idx, observer_alias in enumerate(available_observers_for_plot):
        ax = axes[idx // 2, idx % 2]

        k_values = []
        iteration_labels = []  # To store descriptive labels for each iteration

        for iteration_idx, k_factors_map in all_k_factors_iterations.items():
            if observer_alias in k_factors_map:
                k_values.append(k_factors_map[observer_alias])
                iteration_labels.append(
                    f"Iter {iteration_idx}"
                )  # Simple label for x-axis

        if k_values:
            ax.bar(range(len(k_values)), k_values, alpha=0.7)
            ax.set_title(f"K-factor for {observer_alias}")
            ax.set_ylabel("K-factor")
            ax.set_xlabel("Jackknife Iteration")
            ax.tick_params(axis="x", rotation=45)

            # Add horizontal line for mean
            mean_k = np.mean(k_values)
            ax.axhline(
                y=mean_k, color="red", linestyle="--", label=f"Mean: {mean_k:.3f}"
            )
            ax.legend()

            # Set x-tick labels (can be simplified if too many iterations)
            if len(iteration_labels) > 10:
                step = len(iteration_labels) // 10
                ax.set_xticks(range(0, len(iteration_labels), step))
                ax.set_xticklabels(
                    [iteration_labels[i] for i in range(0, len(iteration_labels), step)]
                )
            else:
                ax.set_xticks(range(len(iteration_labels)))
                ax.set_xticklabels(iteration_labels)
        else:
            ax.set_title(f"No K-factor data for {observer_alias}")
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("k_factor_evolution.png", dpi=300, bbox_inches="tight")
    # plt.show() # Commented out


def calculate_stability_metrics(
    jackknife_results, annual_unified, all_k_factors_iterations
):
    """Calculate stability metrics for the jackknife analysis."""

    print("\n" + "=" * 60)
    print("JACKKNIFE STABILITY ANALYSIS")
    print("=" * 60)

    if not jackknife_results:
        print("No jackknife results available for stability metrics.")
        return pd.DataFrame()

    # Calculate annual sunspot number statistics
    all_annual_values = []
    # Use dates from the original unified series for consistent time axis
    dates = annual_unified["Date"].unique()

    for date in dates:
        values_for_date = []
        for annual_data in jackknife_results.values():
            matching_rows = annual_data[annual_data["Date"] == date]
            if not matching_rows.empty:
                values_for_date.append(matching_rows["WOLF_n"].iloc[0])

        if values_for_date:
            all_annual_values.append(
                {
                    "Date": date,
                    "mean": np.mean(values_for_date),
                    "std": np.std(values_for_date),
                    "cv": (np.std(values_for_date) / np.mean(values_for_date) * 100)
                    if np.mean(values_for_date) != 0
                    else 0,  # Handle division by zero
                    "min": np.min(values_for_date),
                    "max": np.max(values_for_date),
                    "range": np.max(values_for_date) - np.min(values_for_date),
                }
            )

    stability_df = pd.DataFrame(all_annual_values)
    stability_df = stability_df.dropna(
        subset=["mean"]
    )  # Drop years where no mean could be calculated

    # Overall statistics
    print(f"Number of jackknife iterations: {len(jackknife_results)}")
    if not stability_df.empty:
        print(
            f"Time period analyzed: {stability_df['Date'].min().year} - {stability_df['Date'].max().year}"
        )
        print(f"\nOverall Stability Metrics (based on annual series):")
        print(f"Mean coefficient of variation: {stability_df['cv'].mean():.2f}%")
        print(f"Median coefficient of variation: {stability_df['cv'].median():.2f}%")
        print(f"Max coefficient of variation: {stability_df['cv'].max():.2f}%")
        print(f"Years with CV > 10%: {(stability_df['cv'] > 10).sum()}")
        print(f"Years with CV > 20%: {(stability_df['cv'] > 20).sum()}")
    else:
        print("No stability metrics could be calculated from jackknife results.")

    # K-factor stability
    print(f"\nK-factor Stability:")
    if all_k_factors_iterations:
        # Get all unique observer aliases from the k-factor iterations
        all_observer_aliases = set()
        for k_factors_map in all_k_factors_iterations.values():
            all_observer_aliases.update(k_factors_map.keys())

        # Iterate through common observers or those you want to highlight
        # Use aliases like "WOLF", "RUDOLPH", "SCHWABE", "CARRINGTON", "AIRY"
        for observer_alias in sorted(
            list(all_observer_aliases)
        ):  # Iterate through all found aliases
            k_values = [
                k_factors_map.get(observer_alias, np.nan)
                for k_factors_map in all_k_factors_iterations.values()
            ]
            k_values = [k for k in k_values if not np.isnan(k)]

            if k_values:
                print(
                    f"{observer_alias}: mean={np.mean(k_values):.3f}, "
                    f"std={np.std(k_values):.3f}, "
                    f"CV={np.std(k_values) / np.mean(k_values) * 100:.1f}%"
                    if np.mean(k_values) != 0
                    else "CV=0%"
                )
    else:
        print("No k-factor iteration data available for stability metrics.")

    # Identify most unstable periods
    if not stability_df.empty:
        print(f"\nMost Unstable Periods (highest CV):")
        unstable_periods = stability_df.nlargest(5, "cv")
        for _, row in unstable_periods.iterrows():
            print(f"{row['Date'].year}: CV={row['cv']:.1f}%, range={row['range']:.1f}")
    else:
        print("Cannot identify unstable periods: Stability DataFrame is empty.")

    return stability_df


def save_results(
    jackknife_results,
    jackknife_metadata,
    all_k_factors_iterations,
    annual_unified,
    percentiles_df,
    base_path,
):
    """Save all results to files."""

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    output_dir = os.path.join(
        base_path, "jackknife_results"
    )  # Save results in a subfolder within BASE_DATA_PATH
    os.makedirs(output_dir, exist_ok=True)

    # Save individual jackknife iterations
    for iteration_name, data in jackknife_results.items():
        filename = f"{output_dir}/jackknife_iteration_{iteration_name}.csv"
        data.to_csv(filename, index=False)
        # print(f"Saved: {filename}") # Commented out to reduce verbose output during loop

    print(
        f"Saved individual jackknife iteration series to '{output_dir}/jackknife_iteration_X.csv'"
    )

    # Save k-factors for all iterations
    # Convert all_k_factors_iterations (dict of dicts: {iter_idx: {alias: k_val}}) to DataFrame
    k_factors_df_list = []
    for iter_idx, k_map in all_k_factors_iterations.items():
        temp_df = pd.DataFrame([k_map])
        temp_df["iteration"] = iter_idx
        k_factors_df_list.append(temp_df)

    if k_factors_df_list:
        all_k_factors_df_combined = pd.concat(k_factors_df_list).set_index("iteration")
        k_factors_filename = os.path.join(output_dir, "k_factors_all_iterations.csv")
        all_k_factors_df_combined.to_csv(k_factors_filename)
        print(f"Saved: {k_factors_filename}")
    else:
        print("No k-factor data to save.")

    # Save stability metrics
    stability_metrics = calculate_stability_metrics(
        jackknife_results, annual_unified, all_k_factors_iterations
    )  # Recalculate if not passed
    if not stability_metrics.empty:
        stability_metrics_filename = os.path.join(output_dir, "stability_metrics.csv")
        stability_metrics.to_csv(stability_metrics_filename, index=False)
        print(f"Saved: {stability_metrics_filename}")
    else:
        print("No stability metrics to save.")

    # Save comparison with uncertainty bands
    if not percentiles_df.empty:
        uncertainty_filename = os.path.join(output_dir, "uncertainty_bands.csv")
        percentiles_df.to_csv(uncertainty_filename, index=False)
        print(f"Saved: {uncertainty_filename}")
    else:
        print("No uncertainty band data to save.")

    print(f"\nAll results saved to '{output_dir}' directory")
