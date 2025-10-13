import plotly.express as px
from scripts import *
import pandas as pd
import os


def observer_bubble_plots_force_start_year_new1(
    observer_stats, start_year=1800, outdir="figures", name="observer_bubble"
):
    fig = px.scatter(
        observer_stats,
        x="start_date",
        y="end_date",
        size="total_observations",
        color="observation_years",
        hover_name="ALIAS",
        text="ALIAS",
        size_max=60,
        title="Observer Bubble Plot",
    )

    fig.update_traces(
        marker=dict(line=dict(width=2, color="DarkSlateGrey")),
        textposition="middle right",
        textfont=dict(size=12),
    )

    min_date = pd.Timestamp(f"{start_year}-01-01")
    max_x = observer_stats["start_date"].max() + pd.Timedelta(days=20 * 365)
    max_y = observer_stats["end_date"].max() + pd.Timedelta(days=20 * 365)
    fig.update_xaxes(range=[min_date, max_x])
    fig.update_yaxes(range=[min_date, max_y])

    # --- Static PNG export ---
    outdir = os.path.join(os.getcwd(), outdir)  # ensure relative to project root
    os.makedirs(outdir, exist_ok=True)
    png_path = os.path.join(outdir, f"{name}.png")

    try:
        fig.write_image(png_path, format="png", scale=3, width=1600, height=1200)
        print(f"✅ Saved static snapshot to {png_path}")
    except Exception as e:
        print(f"⚠️ Could not save static snapshot: {e}")

    return fig


BASE_DATA_PATH = "./data/"

observer_groups, combined_data_main = load_and_preprocess_main_data(BASE_DATA_PATH)
supplementary_data = load_supplementary_datasets(BASE_DATA_PATH)
specialized_data = load_specialized_datasets(BASE_DATA_PATH)
wolf_source_book_data = load_wolf_source_book_data(BASE_DATA_PATH)

# --- Data Quality Control and Merging ---
filtered_observer_groups = apply_quality_filters(observer_groups)
all_combined_data = merge_supplementary_data(
    filtered_observer_groups, supplementary_data, specialized_data
)

# Merge all observer DataFrames into one
df_all_combined = pd.concat(all_combined_data.values(), ignore_index=True)

# Optional: sort by Date
df_all_combined = df_all_combined.sort_values("Date").reset_index(drop=True)

observer_stats_all_combined = (
    df_all_combined.groupby(["FK_OBSERVERS", "ALIAS"])
    .agg(
        {
            "Date": ["min", "max", "count"],
            "WOLF": ["count", "mean", "std"],
            "GROUPS": ["count", "mean"],
            "SUNSPOTS": ["count", "mean"],
        }
    )
    .round(2)
)

observer_stats_all_combined.columns = [
    "_".join(col).strip() for col in observer_stats_all_combined.columns
]
observer_stats_all_combined = observer_stats_all_combined.reset_index()

# Then rename columns and calculate duration
observer_stats_all_combined = observer_stats_all_combined.rename(
    columns={
        "Date_min": "start_date",
        "Date_max": "end_date",
        "Date_count": "total_observations",
        "WOLF_count": "wolf_observations",
        "WOLF_mean": "avg_wolf_number",
        "WOLF_std": "wolf_std",
        "GROUPS_count": "group_observations",
        "GROUPS_mean": "avg_groups",
        "SUNSPOTS_count": "sunspot_observations",
        "SUNSPOTS_mean": "avg_sunspots",
    }
)

observer_stats_all_combined["observation_years"] = (
    (
        observer_stats_all_combined["end_date"]
        - observer_stats_all_combined["start_date"]
    ).dt.days
    / 365.25
).round(2)


fig = observer_bubble_plots_force_start_year_new1(observer_stats_all_combined)

fig
