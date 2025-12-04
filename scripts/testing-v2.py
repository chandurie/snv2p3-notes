import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

df = pd.read_csv("./observation_years_by_observer.csv")
df["duration"] = df["end_year"] - df["start_year"] + 1
df = df.sort_values("observation_count", ascending=False)

df_top = df.head(200).copy().sort_values("start_year")
rows = len(df_top)

# Visual spacing controls
row_spacing = 1.0          # spacing in data coordinates (keep at 1.0)
row_height_in = 0.15       # inches of figure height per observer row
fig_height = max(12, rows * row_height_in)

# Custom year compression so pre-1750 timeline is visually shorter
min_year = df_top.start_year.min()
max_year = df_top.end_year.max()
pivot_year = 1750
pre_scale = .25   # fraction of normal scale before pivot
# pre_scale = 0.35   # fraction of normal scale before pivot


def compress_year(year: float) -> float:
    """Piecewise-linear time compression centered on pivot_year."""
    if year <= pivot_year:
        return (year - min_year) * pre_scale
    compressed_pre = (pivot_year - min_year) * pre_scale
    return compressed_pre + (year - pivot_year)


df_top["start_plot"] = df_top["start_year"].apply(compress_year)
df_top["end_plot"] = df_top["end_year"].apply(compress_year)
x_min_plot = df_top["start_plot"].min()
x_max_plot = df_top["end_plot"].max()

norm = mpl.colors.Normalize(vmin=df_top["duration"].min(), vmax=df_top["duration"].max())
cmap = mpl.cm.plasma

fig = plt.figure(figsize=(20, fig_height))
ax = fig.add_axes([0.08, 0.08, 0.88, 0.86])

label_offset = (x_max_plot - x_min_plot) * 0.01

for i, row in enumerate(df_top.itertuples()):
    y = i * row_spacing
    color = cmap(norm(row.duration))

    # Thicker bars
    ax.plot([row.start_plot, row.end_plot], [y, y],
            linewidth=8, color=color, solid_capstyle="round")

    # Clean labels (no glow, no bold)
    if i % 2 == 0:
        ax.text(row.end_plot + label_offset, y, row.name,
                va='center', fontsize=10, fontfamily="Georgia")
    else:
        ax.text(row.start_plot - label_offset, y, row.name,
                va='center', ha='right', fontsize=10, fontfamily="Georgia")

ax.set_yticks([])
ax.set_xlabel("Year (pre-1750 compressed)", fontsize=16, fontfamily="Georgia")
ax.set_title("Historical Sunspot Observers Timeline (Color = Duration, skewed axis)",
             fontsize=24, fontfamily="Georgia", pad=25)
ax.grid(axis="x", linestyle="--", alpha=0.25)

# Period shading (Maunder Minimum, kept)
ax.axvspan(compress_year(1645), compress_year(1715), color="gray", alpha=0.10)
ax.text(compress_year(1650), rows * row_spacing * 1.02,
        "Maunder Minimum", fontsize=14, fontfamily="Georgia", alpha=0.6)

# Horizontal colorbar inside
x_target = 1630
x_fr = (compress_year(x_target) - x_min_plot) / (x_max_plot - x_min_plot)
cb_ax = ax.inset_axes([x_fr, 1.04, 0.25, 0.025], transform=ax.transAxes)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cb_ax, orientation='horizontal')
cbar.set_label("Duration (Years)", fontsize=12, fontfamily="Georgia")
cbar.ax.tick_params(labelsize=10)

ax.set_xlim(x_min_plot, x_max_plot)
ax.set_ylim(-row_spacing, rows * row_spacing * 1.02)
tick_years = [1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950]
ax.set_xticks([compress_year(y) for y in tick_years])
ax.set_xticklabels([str(y) for y in tick_years])
ax.tick_params(axis="x", which="both", bottom=True, top=True,
               labelbottom=True, labeltop=True)
fig = plt.gcf()

fig.savefig("sunspot_timeline.svg", format="svg", dpi=300)   # A — vector
fig.savefig("sunspot_timeline.pdf", format="pdf")            # B — print-ready
fig.savefig("sunspot_timeline_600dpi.png", dpi=600)          # C — poster raster

plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import patheffects
# import matplotlib as mpl
# import numpy as np
#
# df = pd.read_csv("./observation_years_by_observer.csv")
# df["duration"] = df["end_year"] - df["start_year"] + 1
# df = df.sort_values("observation_count", ascending=False)
#
# df_top = df.head(120).copy().sort_values("start_year")
#
# # Compute observers per year
# years = np.arange(df_top.start_year.min(), df_top.end_year.max()+1)
# obs_per_year = pd.DataFrame({
#     "year": years,
#     "observer_count": [(df_top.start_year <= y).sum() - (df_top.end_year < y).sum() for y in years]
# })
#
# # Highlight longest duration observer(s) (H)
# max_duration = df_top.duration.max()
# highlight_names = set(df_top.loc[df_top.duration == max_duration, "name"])
#
# norm = mpl.colors.Normalize(vmin=df_top.duration.min(), vmax=df_top.duration.max())
# cmap = mpl.cm.plasma
#
# fig = plt.figure(figsize=(20, 16))
#
# # ---------------- TOP PANEL (timeline) ----------------
# ax = fig.add_axes([0.1, 0.32, 0.85, 0.63])
#
# for i, row in enumerate(df_top.itertuples()):
#     color = cmap(norm(row.duration))
#
#     # H — bold + glow for longest-duration
#     lw = 7 if row.name in highlight_names else 6
#
#     ax.plot([row.start_year, row.end_year], [i, i],
#             linewidth=lw, color=color, solid_capstyle="round")
#
#     # G — glow (white halo behind text)
#     txt = ax.text(row.end_year + 1, i, row.name,
#                   va='center', fontsize=9.5, fontfamily="Georgia")
#     txt.set_path_effects([
#         patheffects.Stroke(linewidth=3, foreground='white'),
#         patheffects.Normal()
#     ])
#
# # Remove y ticks
# ax.set_yticks([])
# ax.set_xlabel("")
# ax.set_title("Historical Sunspot Observers Timeline (Color = Duration)", fontsize=24, fontfamily="Georgia", pad=25)
# ax.grid(axis="x", linestyle="--", alpha=0.25)
#
# # E — Maunder Minimum shading (1645–1715 approx.)
# ax.axvspan(1645, 1715, color="gray", alpha=0.10)
# ax.text(1650, len(df_top)*1.02, "Maunder Minimum", fontsize=14, fontfamily="Georgia", alpha=0.6)
#
# # Add horizontal colorbar inside plot near x ≈ 1630
# min_year, max_year = df_top.start_year.min(), df_top.end_year.max()
# x_target = 1630
# x_fr = (x_target - min_year) / (max_year - min_year)
# cb_ax = ax.inset_axes([x_fr, 1.04, 0.25, 0.025], transform=ax.transAxes)
#
# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, cax=cb_ax, orientation='horizontal')
# cbar.set_label("Duration (Years)", fontsize=12, fontfamily="Georgia")
# cbar.ax.tick_params(labelsize=10)
#
# # ---------------- LOWER PANEL (Observers per year) ----------------
# ax2 = fig.add_axes([0.1, 0.08, 0.85, 0.18])
# ax2.plot(obs_per_year.year, obs_per_year.observer_count, color="#444", linewidth=2.5)
#
# ax2.set_xlabel("Year", fontsize=16, fontfamily="Georgia")
# ax2.set_ylabel("Number of Observers", fontsize=14, fontfamily="Georgia")
# ax2.grid(axis="y", linestyle="--", alpha=0.4)
#
# # Match timeline horizontal extent
# ax2.set_xlim(min_year, max_year)
# fig = plt.gcf()
#
# fig.savefig("sunspot_timeline.svg", format="svg", dpi=300)   # A — vector
# fig.savefig("sunspot_timeline.pdf", format="pdf")            # B — print-ready
# fig.savefig("sunspot_timeline_600dpi.png", dpi=600)          # C — poster raster
#
# # plt.show()
#
