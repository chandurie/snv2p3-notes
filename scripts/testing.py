import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load dataset
df = pd.read_csv("./observation_years_by_observer.csv")

# Compute duration
df["duration"] = df["end_year"] - df["start_year"] + 1

# Sort by total observation contribution
df = df.sort_values("observation_count", ascending=False)

def select_top_n(df, n=100):
    return df.head(n).copy()

# Choose your top-N here:
df_top = select_top_n(df, n=120)

# Sort by start year to produce historical timeline flow
df_top = df_top.sort_values("start_year")

# Normalize duration for color mapping
norm = mpl.colors.Normalize(vmin=df_top["duration"].min(),
                            vmax=df_top["duration"].max())

# Color palette options (choose one):
cmap = mpl.cm.plasma      # vivid, modern, high contrast
# cmap = mpl.cm.turbo     # super vibrant, poster flashy
# cmap = mpl.cm.cividis   # smooth, color-blind optimized

plt.figure(figsize=(20, 14))
ax = plt.gca()

for i, row in enumerate(df_top.itertuples()):
    color = cmap(norm(row.duration))
    
    # Draw segment
    ax.plot([row.start_year, row.end_year], [i, i],
            linewidth=6, color=color, solid_capstyle="round")

    # Alternate label positions for legibility
    if i % 2 == 0:
        ax.text(row.end_year + 1, i, row.name,
                va='center', fontsize=9.5, fontfamily="Georgia")
    else:
        ax.text(row.start_year - 1, i, row.name,
                va='center', ha='right', fontsize=9.5, fontfamily="Georgia")

# Remove y ticks (makes it clean)
ax.set_yticks([])

# Titles & Labels
ax.set_xlabel("Year", fontsize=16, fontfamily="Georgia")
ax.set_title("Historical Sunspot Observers Timeline (Color = Observation Duration)",
             fontsize=24, fontfamily="Georgia", pad=25)

# Gentle grid lines for time scale readability
ax.grid(axis="x", linestyle="--", alpha=0.25)

# Add colorbar legend
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("Duration of Observations (Years)", fontsize=14, fontfamily="Georgia")

plt.tight_layout()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np
#
# # Load dataset
# df = pd.read_csv("./observation_years_by_observer.csv")
# df["duration"] = df["end_year"] - df["start_year"] + 1
# df = df.sort_values("observation_count", ascending=False)
#
# # Select top N contributors
# def select_top_n(df, n=50):
#     return df.head(n).copy()
#
# df_top = select_top_n(df, n=120)   # adjust N freely
#
# # Sort by start year for visual narrative flow
# df_top = df_top.sort_values("start_year")
#
# # Normalize observation count for color mapping
# norm = mpl.colors.Normalize(vmin=df_top["observation_count"].min(), 
#                             vmax=df_top["observation_count"].max())
# cmap = mpl.cm.magma  # or plasma / viridis / inferno / turbo
#
# plt.figure(figsize=(18, 14))
# ax = plt.gca()
#
# for i, row in enumerate(df_top.itertuples()):
#
#     # Line color = mapped from observation count
#     color = cmap(norm(row.observation_count))
#
#     # Draw timeline segment
#     ax.plot([row.start_year, row.end_year], [i, i], 
#             linewidth=6, color=color, solid_capstyle="round")
#
#     # Alternate label side to avoid crowding
#     if i % 2 == 0:
#         ax.text(row.end_year + 1, i, row.name, 
#                 va='center', fontsize=9, fontfamily="Georgia")
#     else:
#         ax.text(row.start_year - 1, i, row.name, 
#                 va='center', ha='right', fontsize=9, fontfamily="Georgia")
#
# # y-axis: no ticks (clean)
# ax.set_yticks([])
#
# # Labels & Title
# ax.set_xlabel("Year", fontsize=16, fontfamily="Georgia")
# ax.set_title("Historical Sunspot Observers Timeline (Top Contributors)", 
#              fontsize=22, fontfamily="Georgia", pad=20)
#
# # Nice grid styling
# ax.grid(axis="x", linestyle="--", alpha=0.3)
#
# # Colorbar legend
# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, pad=0.02)
# cbar.set_label("Number of Observations", fontsize=14)
#
# plt.tight_layout()
# plt.show()

# import pandas as pd
#
# # Load your full dataset
# df = pd.read_csv("./observation_years_by_observer.csv")
#
# # Compute midpoints and durations
# df["mid_year"] = (df["start_year"] + df["end_year"]) / 2
# df["duration"] = df["end_year"] - df["start_year"] + 1
#
# # Sort by total observation contribution to identify major observers
# df = df.sort_values("observation_count", ascending=False)
#
# def select_top_n(df, n=30):
#     """Return top N contributors based on observation_count."""
#     return df.head(n).copy()
#
# df_top = select_top_n(df, n=200)  # You can change n at any time
#
# # import plotly.express as px
# #
# # fig = px.timeline(
# #     df_top,
# #     x_start="start_year",
# #     x_end="end_year",
# #     y="name",
# #     color="observation_count",
# #     color_continuous_scale="Viridis",
# # )
# #
# # fig.update_yaxes(autorange="reversed")
# # fig.update_layout(
# #     title="Sunspot Observation Timelines (Top Contributors)",
# #     xaxis_title="Year",
# #     yaxis_title="Observer / Observatory",
# #     font=dict(family="Arial", size=14),
# # )
# #
# # fig.show()
# #
# #
# # import plotly.express as px
# #
# # fig = px.scatter(
# #     df_top,
# #     x="mid_year",
# #     y="name",
# #     size="observation_count",
# #     color="duration",
# #     color_continuous_scale=px.colors.sequential.Sunset,
# #     hover_data=["start_year", "end_year", "observation_count", "duration"],
# # )
# #
# # fig.update_layout(
# #     title="Sunspot Observers: Volume & Duration",
# #     xaxis_title="Year (mid-point of activity)",
# #     yaxis_title="Observer",
# #     font=dict(family="Helvetica Neue", size=15),
# #     plot_bgcolor="white"
# # )
# #
# # fig.show()
#
# # this is working fine
# import matplotlib.pyplot as plt
#
# df_top = df_top.sort_values("start_year")
#
# plt.figure(figsize=(12, 10))
#
# for i, row in enumerate(df_top.itertuples()):
#     plt.plot([row.start_year, row.end_year], [i, i], linewidth=4, color="#6b4f3f")
#     plt.text(row.start_year, i + 0.1, row.name, fontsize=10, fontfamily="Georgia")
#
# plt.yticks([])
# plt.xlabel("Year", fontsize=14, fontfamily="Georgia")
# plt.title("Historical Sunspot Observers Timeline", fontsize=18, fontfamily="Georgia")
# plt.grid(axis="x", linestyle="--", alpha=0.4)
#
# plt.show()
#
