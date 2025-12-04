import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load dataset
df = pd.read_csv("./observation_years_by_observer.csv")
df["duration"] = df["end_year"] - df["start_year"] + 1
df = df.sort_values("observation_count", ascending=False)

def select_top_n(df, n=120):
    return df.head(n).copy()

df_top = select_top_n(df)
df_top = df_top.sort_values("start_year")

norm = mpl.colors.Normalize(vmin=df_top["duration"].min(), vmax=df_top["duration"].max())
cmap = mpl.cm.plasma

plt.figure(figsize=(20, 14))
ax = plt.gca()

for i, row in enumerate(df_top.itertuples()):
    color = cmap(norm(row.duration))
    ax.plot([row.start_year, row.end_year], [i, i],
            linewidth=6, color=color, solid_capstyle="round")
    
    if i % 2 == 0:
        ax.text(row.end_year + 1, i, row.name,
                va='center', fontsize=9.5, fontfamily="Georgia")
    else:
        ax.text(row.start_year - 1, i, row.name,
                va='center', ha='right', fontsize=9.5, fontfamily="Georgia")

ax.set_yticks([])
ax.set_xlabel("Year", fontsize=16, fontfamily="Georgia")
ax.set_title("Historical Sunspot Observers Timeline (Color = Observation Duration)",
             fontsize=24, fontfamily="Georgia", pad=25)
ax.grid(axis="x", linestyle="--", alpha=0.25)

# ---- Horizontal Colorbar placed at x ≈ 1630 ----
min_year = df_top.start_year.min()
max_year = df_top.end_year.max()

# Convert x=1630 to axis coordinates (0 to 1)
x_target = 1630
x_fr = (x_target - min_year) / (max_year - min_year)

# Create inset colorbar axis in axis coordinate space
cb_ax = ax.inset_axes([x_fr, .90, 0.25, 0.025], transform=ax.transAxes)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(sm, cax=cb_ax, orientation='horizontal')
cbar.set_label("Duration of Observations (Years)", fontsize=12, fontfamily="Georgia")
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
fig = plt.gcf()

fig.savefig("sunspot_timeline.svg", format="svg", dpi=300)   # A — vector
fig.savefig("sunspot_timeline.pdf", format="pdf")            # B — print-ready
fig.savefig("sunspot_timeline_600dpi.png", dpi=600)          # C — poster raster

plt.show()

