import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ============================================================
# 1. CONFIG — Path to your raw CSV FILE
# ============================================================

INPUT_CSV = "../data/review_sources-2010-2025.csv"      # <-- your input CSV
OUTPUT_CSV = "../data/clean_sources-2010-2025.csv"

# ============================================================
# 2. Helper functions
# ============================================================

def remove_parentheses(text):
    """Remove anything in (...) to match Option C."""
    if not isinstance(text, str):
        return text
    return re.sub(r'\s*\(.*?\)', '', text).strip()

def parse_years(text):
    """Extract YYYY or YYYY–YYYY from strings."""
    if not isinstance(text, str):
        return None, None
    t = text.replace("–", "-").replace("—", "-")
    years = re.findall(r'\d{4}', t)
    if len(years) == 1:
        return int(years[0]), int(years[0])
    elif len(years) >= 2:
        return int(years[0]), int(years[1])
    return None, None

def parse_actual_days(value):
    """
    Extract numeric days only.
    If not purely numeric, return blank so user fills manually.
    """
    value = str(value)
    digits = "".join([c for c in value if c.isdigit()])
    if digits:
        return int(digits)
    return ""   # Leave blank


# ============================================================
# 3. Load user CSV (raw_data)
# ============================================================

# df_raw = pd.read_csv(INPUT_CSV)

# Load CSV safely (semicolon-separated, skip malformed rows)
df_raw = pd.read_csv(
    INPUT_CSV,
    sep=';',               # your file uses semicolons
    engine='python',       # more forgiving parser
    on_bad_lines='skip'    # skip lines that break formatting
)
# Expected columns:
# RecoveredDataset, TemporalCoverage, ObsDays
# But we accept ANY column order.

col_map = {}
for col in df_raw.columns:
    col_l = col.lower()
    if "recover" in col_l:
        col_map["RecoveredDataset"] = col
    elif "tempor" in col_l:
        col_map["TemporalCoverage"] = col
    elif "obs" in col_l:
        col_map["ObsDays"] = col

df = df_raw[[col_map["RecoveredDataset"],
             col_map["TemporalCoverage"],
             col_map["ObsDays"]]].copy()

df.columns = ["RecoveredDataset", "TemporalCoverage", "ObsDays"]


# ============================================================
# 4. Clean the dataset
# ============================================================

# Remove parentheses (Option C)
df["RecoveredDataset"] = df["RecoveredDataset"].apply(remove_parentheses)

# Parse years
df["StartDate"], df["EndDate"] = zip(*df["TemporalCoverage"].apply(parse_years))

# Parse numeric observation days
df["ActualObservationDays"] = df["ObsDays"].apply(parse_actual_days)

# Prepare final columns
df_clean = df[["RecoveredDataset", "StartDate", "EndDate", "ActualObservationDays"]]

# Save CSV
df_clean.to_csv(OUTPUT_CSV, sep=";", index=False)
print(f"Saved cleaned CSV to {OUTPUT_CSV}")


# ============================================================
# 5. Bubble Plot (skip blanks)
# ============================================================

df_plot = df_clean[df_clean["ActualObservationDays"] != ""]

plt.figure(figsize=(12, 8))
plt.scatter(
    df_plot["StartDate"],
    df_plot["EndDate"],
    s=df_plot["ActualObservationDays"].astype(int) / 3,
    alpha=0.5,
    edgecolor='k'
)

plt.xlabel("Start Year")
plt.ylabel("End Year")
plt.title("Sunspot Dataset Bubble Plot")
plt.grid(True)
plt.savefig("bubble_plot_manual.png", dpi=300)
plt.close()

# ============================================================
# 6. Timeline Plot
# ============================================================

plt.figure(figsize=(14, 14))
for i, row in df_clean.iterrows():
    plt.plot([row.StartDate, row.EndDate], [i, i], linewidth=3)

plt.yticks(range(len(df_clean)), df_clean["RecoveredDataset"])
plt.xlabel("Year")
plt.title("Sunspot Dataset Timeline")
plt.grid(axis="x")
plt.tight_layout()
plt.savefig("timeline_plot_manual.png", dpi=300)
plt.close()

print("Generated bubble_plot_manual.png and timeline_plot_manual.png")

