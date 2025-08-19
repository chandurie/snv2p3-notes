# Create summary statistics
print("\n" + "=" * 50)
print("OBSERVER STATISTICS")
print("=" * 50)


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

# print(observer_stats_all_combined.nlargest(10, 'observation_years')[
#    ['ALIAS', 'start_date', 'end_date', 'observation_years', 'total_observations']
#    #['FK_OBSERVERS', 'ALIAS', 'start_date', 'end_date', 'observation_years', 'total_observations']
# ])

print(f"\nTotal unique observers: {len(observer_stats_all_combined)}")
print(
    f"Observation period: {df_all_combined['Date'].min()} to {df_all_combined['Date'].max()}"
)
print(
    f"Total observation span: {((df_all_combined['Date'].max() - df_all_combined['Date'].min()).days / 365.25):.1f} years"
)


print(
    f"Number of Observers by FK_OBSERVERS: {len(df_all_combined['FK_OBSERVERS'].unique())}"
)
print(f"Number of observers by ALIAS: {len(df_all_combined['ALIAS'].unique())}")
print(
    f"Number of observes by ALIAS({len(df_all_combined['ALIAS'].unique())}) is different from \n number of observers by FK_OBSERVERS({len(df_all_combined['FK_OBSERVERS'].unique())}). \n"
)
print(f"One of the ALIAS has multiple FK_OBSERVERS IDs.")
print(f"Checking ALIAS'es with multiple FK_OBSERVERS IDs.")
fk_observers_counts = df_all_combined.groupby("ALIAS")["FK_OBSERVERS"].nunique()
multiple_fk_observers = fk_observers_counts[fk_observers_counts > 1]
alias_with_multiple_fk_observers = multiple_fk_observers.index[0]
fk_observer_ids_with_alias_with_multiple_observers = df_all_combined[
    df_all_combined["ALIAS"] == alias_with_multiple_fk_observers
]["FK_OBSERVERS"].unique()
print(f"{alias_with_multiple_fk_observers} has multiple FK_OBSERVERS IDs.")
print(
    f"FK_OBSERVERS IDs associated with {alias_with_multiple_fk_observers} are {fk_observer_ids_with_alias_with_multiple_observers} \n"
)


```{python}
#| label: fig-plot_observer_bubble_all_combined
#| fig-cap: Bubble plot showing all observers in all combined data

fig = px.scatter(
    observer_stats_all_combined,
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
        "FK_OBSERVERS": "Observer ID"
    },
    title="Observer Bubble Plot"
)

# Set text position (try 'middle right', 'top center', etc.)
fig.update_traces(
    marker=dict(line=dict(width=2, color='DarkSlateGrey')),
    textposition='middle right',
    textfont=dict(size=12)
)

# Save the figure to a static image file
# pio.write_image(fig, 'my_plot.png')
# pio.write_image(fig, 'my_plot.pdf')  # For directly saving as a PDF

fig.show()

```



```{python}
#| label: fig-plot_observer_bubble_all_combined_animated
#| fig-cap: Animated bubble plot showing all observers in all combined data.

# Add a new column for scaled marker size
# Adjust the scaling factor as needed for your data
scaling_factor = 1  # You can tweak this
observer_stats_all_combined['marker_size'] = np.sqrt(observer_stats_all_combined['total_observations']) * scaling_factor

# Assume observer_stats_all_combined is already prepared as in your previous code
observer_stats_all_combined['start_year'] = observer_stats_all_combined['start_date'].dt.year

# Get all years from the earliest to the latest start year
years = np.arange(observer_stats_all_combined['start_year'].min(), observer_stats_all_combined['start_year'].max() + 1)

# For each year, include all observers who started up to that year
frames = []
for year in years:
    frame_data = observer_stats_all_combined[observer_stats_all_combined['start_year'] <= year]
    frames.append(frame_data)


# Initial data (first frame)
init_data = frames[0]

fig = go.Figure(
    data=[
        go.Scatter(
            x=init_data['start_date'],
            y=init_data['end_date'],
            mode='markers+text',
            marker=dict(
                size=init_data['total_observations'],
                color=init_data['observation_years'],
                colorscale='Viridis',
                showscale=True,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=init_data['ALIAS'],
            textposition='middle right',
            hovertemplate=(
                'Observer: %{text}<br>'
                'Start: %{x|%Y-%m-%d}<br>'
                'End: %{y|%Y-%m-%d}<br>'
                'Total Observations: %{marker.size}<br>'
                'Observation Years: %{marker.color:.2f}<extra></extra>'
            ),
        )
    ],
    layout=go.Layout(
        title="Animated Observer Activity Bubble Plot",
        xaxis=dict(title='Start Date', range=[init_data['start_date'].min(), init_data['start_date'].max()]),
        yaxis=dict(title='End Date', range=[init_data['end_date'].min(), init_data['end_date'].max()]),
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }
            ]
        }]
    )
)


# Prepare frames for animation
from pandas.tseries.offsets import DateOffset

buffer_days = 365.25 * 6  # About 2 months buffer

animation_frames = []
for i, year in enumerate(years):
    frame_data = frames[i]
    x_min = frame_data['start_date'].min() - pd.Timedelta(days=buffer_days)
    x_max = frame_data['start_date'].max() + pd.Timedelta(days=buffer_days)
    y_min = frame_data['end_date'].min() - pd.Timedelta(days=buffer_days)
    y_max = frame_data['end_date'].max() + pd.Timedelta(days=buffer_days)
    if i == len(years) - 1:
        # Use full range for last frame
        x_min = observer_stats_all_combined['start_date'].min() - pd.Timedelta(days=buffer_days)
        x_max = observer_stats_all_combined['start_date'].max() + pd.Timedelta(days=buffer_days)
        y_min = observer_stats_all_combined['end_date'].min() - pd.Timedelta(days=buffer_days)
        y_max = observer_stats_all_combined['end_date'].max() + pd.Timedelta(days=buffer_days)
    animation_frames.append(
        go.Frame(
            data=[
                go.Scatter(
                    x=frame_data['start_date'],
                    y=frame_data['end_date'],
                    mode='markers+text',
                    marker=dict(
                        size=frame_data['marker_size'],
                        color=frame_data['observation_years'],
                        colorscale='Viridis',
                        showscale=True,
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    text=frame_data['ALIAS'],
                    textposition='middle right',
                    hovertemplate=(
                        'Observer: %{text}<br>'
                        'Start: %{x|%Y-%m-%d}<br>'
                        'End: %{y|%Y-%m-%d}<br>'
                        'Total Observations: %{marker.size}<br>'
                        'Observation Years: %{marker.color:.2f}<extra></extra>'
                    ),
                )
            ],
            name=str(year),
            layout=go.Layout(
                xaxis=dict(title='Start Date', range=[x_min, x_max]),
                yaxis=dict(title='End Date', range=[y_min, y_max]),
                title=f"Animated Observer Bubble Plot - Year {year}"
            )
        )
    )


fig.frames = animation_frames

# Add slider for years
sliders = [{
    "steps": [
        {
            "args": [
                [str(year)],
                {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}
            ],
            "label": str(year),
            "method": "animate"
        }
        for year in years
    ],
    "transition": {"duration": 0},
    "x": 0.1,
    "y": 0,
    "currentvalue": {"font": {"size": 16}, "prefix": "Year: ", "visible": True, "xanchor": "center"},
    "len": 0.9
}]

fig.update_layout(sliders=sliders)

fig.show()
```



def observer_bubble_plot_animated(observer_dict):
# Add a new column for scaled marker size
# Adjust the scaling factor as needed for your data
    scaling_factor = 1  # You can tweak this
    observer_dict['marker_size'] = np.sqrt(observer_dict['total_observations']) * scaling_factor

# Assume observer_dict is already prepared as in your previous code
    observer_dict['start_year'] = observer_dict['start_date'].dt.year

# Get all years from the earliest to the latest start year
    years = np.arange(observer_dict['start_year'].min(), observer_dict['start_year'].max() + 1)

# For each year, include all observers who started up to that year
    frames = []
    for year in years:
        frame_data = observer_dict[observer_dict['start_year'] <= year]
        frames.append(frame_data)


# Initial data (first frame)
    init_data = frames[0]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=init_data['start_date'],
                y=init_data['end_date'],
                mode='markers+text',
                marker=dict(
                    size=init_data['total_observations'],
                    color=init_data['observation_years'],
                    colorscale='Viridis',
                    showscale=True,
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=init_data['ALIAS'],
                textposition='middle right',
                hovertemplate=(
                    'Observer: %{text}<br>'
                    'Start: %{x|%Y-%m-%d}<br>'
                    'End: %{y|%Y-%m-%d}<br>'
                    'Total Observations: %{marker.size}<br>'
                    'Observation Years: %{marker.color:.2f}<extra></extra>'
                ),
            )
        ],
        layout=go.Layout(
            title="Animated Observer Activity Bubble Plot",
            xaxis=dict(title='Start Date', range=[init_data['start_date'].min(), init_data['start_date'].max()]),
            yaxis=dict(title='End Date', range=[init_data['end_date'].min(), init_data['end_date'].max()]),
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    }
                ]
            }]
        )
    )


# Prepare frames for animation
    from pandas.tseries.offsets import DateOffset

    buffer_days = 365.25 * 6  # About 2 months buffer

    animation_frames = []
    for i, year in enumerate(years):
        frame_data = frames[i]
        x_min = frame_data['start_date'].min() - pd.Timedelta(days=buffer_days)
        x_max = frame_data['start_date'].max() + pd.Timedelta(days=buffer_days)
        y_min = frame_data['end_date'].min() - pd.Timedelta(days=buffer_days)
        y_max = frame_data['end_date'].max() + pd.Timedelta(days=buffer_days)
        if i == len(years) - 1:
            # Use full range for last frame
            x_min = observer_dict['start_date'].min() - pd.Timedelta(days=buffer_days)
            x_max = observer_dict['start_date'].max() + pd.Timedelta(days=buffer_days)
            y_min = observer_dict['end_date'].min() - pd.Timedelta(days=buffer_days)
            y_max = observer_dict['end_date'].max() + pd.Timedelta(days=buffer_days)
        animation_frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=frame_data['start_date'],
                        y=frame_data['end_date'],
                        mode='markers+text',
                        marker=dict(
                            size=frame_data['marker_size'],
                            color=frame_data['observation_years'],
                            colorscale='Viridis',
                            showscale=True,
                            line=dict(width=2, color='DarkSlateGrey')
                        ),
                        text=frame_data['ALIAS'],
                        textposition='middle right',
                        hovertemplate=(
                            'Observer: %{text}<br>'
                            'Start: %{x|%Y-%m-%d}<br>'
                            'End: %{y|%Y-%m-%d}<br>'
                            'Total Observations: %{marker.size}<br>'
                            'Observation Years: %{marker.color:.2f}<extra></extra>'
                        ),
                    )
                ],
                name=str(year),
                layout=go.Layout(
                    xaxis=dict(title='Start Date', range=[x_min, x_max]),
                    yaxis=dict(title='End Date', range=[y_min, y_max]),
                    title=f"Animated Observer Bubble Plot - Year {year}"
                )
            )
        )


    fig.frames = animation_frames

# Add slider for years
    sliders = [{
        "steps": [
            {
                "args": [
                    [str(year)],
                    {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}
                ],
                "label": str(year),
                "method": "animate"
            }
            for year in years
        ],
        "transition": {"duration": 0},
        "x": 0.1,
        "y": 0,
        "currentvalue": {"font": {"size": 16}, "prefix": "Year: ", "visible": True, "xanchor": "center"},
        "len": 0.9
    }]

    fig.update_layout(sliders=sliders)

    fig.show()
