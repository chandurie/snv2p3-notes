#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Copy-paste your sessionid from browser devtools
COOKIES = {
    "sessionid": "516oouehqzs7s6q4q4kteolq53zp3bju"
}

url = "https://silso.observatory.be/historical_sunspot_observations/historical_sunspot_observations/observation_years_by_observer/"

print("🔗 Downloading webpage with session cookie...")
resp = requests.get(url, cookies=COOKIES)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

# Find the table that has headers "Observer" and "Observation years"
obs_table = None
for table in soup.find_all("table"):
    headers = [th.text.strip().lower() for th in table.find_all("th")]
    if "observer" in headers and "observation years" in headers:
        obs_table = table
        break

if obs_table is None:
    raise RuntimeError("❌ Could not find Observation Years table — check sessionid cookie")

print("✅ Table located — extracting...")

data = []
rows = obs_table.find_all("tr")[1:]  # skip header

for row in rows:
    cols = row.find_all("td")
    if len(cols) != 2:
        continue
    observer = cols[0].text.strip()
    years = [a.text.strip() for a in cols[1].find_all("a")]
    data.append([observer, years])

df = pd.DataFrame(data, columns=["observer_name", "observation_years"])
df.to_csv("observation_years_by_observer.csv", index=False)

print("✅ Saved CSV: observation_years_by_observer.csv")
print(df.head())

