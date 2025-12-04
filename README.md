# FARSuN: Findability and Accessibility of historical Raw Sunspot Numbers

This repository hosts the working files for the FARSuN programme, which is rebuilding the historical International Sunspot Number (SN) record in preparation for SN V3.0. It combines data-rescue notebooks, quality-control scripts, Quarto articles, plotting utilities, and conference-ready posters used by the Royal Observatory of Belgium/SILSO team. The goal is to replay the observing chain from the original manuscripts onward—**gather sources**, **process and verify the data**, **validate inter-observer calibration**, and **publish machine-readable products** suitable for reproducible science.

The project currently focuses on:

- Digitising and harmonising legacy corpora such as Wolf’s *Mittheilungen*, the Zürich observer tables (1945–1979), Gruithuisen and Adams notebooks, and Stark’s printed ledgers.
- Integrating community-recovered telescopic datasets (2010–2025) to provide wider temporal coverage and redundant observers for scale transfer.
- Producing QA dashboards, bubble/timeline plots, AGU posters, and briefing decks that explain how the recovered material feeds into SN V3 calibration work.

---

## Repository map

| Path | Purpose |
| --- | --- |
| `data/` | Source CSV, XLSX, and markdown descriptions for each observer set (Mittheilungen, Zürich tables, Gruithuisen, Adams, Stark, etc.). Most files are semicolon-delimited. |
| `scripts/` | Python utilities for cleaning the review spreadsheets and generating figures (`reviewed_sunspot-data-2025.py`, `mittheilungen-plots.py`, exploratory notebooks). |
| `figures/` | Plot exports (PDF/PNG/SVG) referenced by the Quarto pages, posters, and talks. |
| `*.qmd` | Quarto documents for the public SNV3 reconstruction site (see `_quarto.yml` for navigation). |
| `_site/` | Rendered HTML/PDF/DOCX artifacts produced by `quarto render`. Commit only the files you intend to publish. |
| `posters/` | XeLaTeX sources for AGU posters (`agu-poster-v3.tex`, data extraction variants, QR codes). |
| `presentations/` | Reveal.js slides and LaTeX talks derived from the same data narratives. |
| `docs/` | Static assets shared to collaborators (e.g., exported figures for reports). |
| `environment.yml` | Conda specification for the `astro` environment used throughout the scripts. |
| `LICENSE` | MIT License covering the code and accompanying documentation. Data contributors may impose additional terms—see the relevant `data/*.md` files. |

---

## Data coverage at a glance

| Dataset | Description | Status (early 2025) | Immediate next steps |
| --- | --- | --- | --- |
| **Zürich Tables (1945–1979)** | Daily counts, Wolf numbers, observing conditions for the Zürich network. | 100 % of >2 000 tables digitised; metadata complete for 1945–1959; observer scaling dashboards active. | Finish post-1960 metadata, confirm 1980 holdings, publish VO/EPN‑TAP with uncertainties. |
| **Mittheilungen (1610–1918)** | Wolf’s journals consolidating the earliest telescopic observations. | Fully integrated into FARSuN schema with NG > NS and duplication checks. | Keep observer/instrument vocabularies aligned and link to new extractions. |
| **C. H. Adams (1819–1823)** | 338 drawings and 1 056 entries with spotless-day coverage. | Orientation/dewarping workflows in progress; comparing direct vs. reflection methods. | Extract positions/areas, finish calibration with contemporaries, release w/ uncertainties. |
| **Gruithuisen (1817–1849)** | Kurrent manuscripts noting sunspots, weather, and context. | Images/text collected; two-pass QC and citizen-science workflows ready. | Finalise NG/NS tables, harmonise metadata, expose FAIR catalogues. |
| **Augustin Stark (1813–1835)** | Printed descriptions of sunspot groups/chains. | OCR scripts and metadata linkage underway. | Tighten QC flags and connect to the observer registry. |
| **Community recoveries (2010–2025)** |  Review_sunspotsources CSV summarising modern data-rescue work. | Used to produce the “reviewed sources” bubble/timeline figures. | Keep spreadsheet updated as new publications appear. |

---

## Getting started

### 1. Create the Python environment

```bash
conda env create -f environment.yml
conda activate astro
```

The environment includes `numpy`, `pandas`, `matplotlib`, `sunpy`, `astropy`, `plotly`, and `mysql-connector-python`. Install `quarto` (>=1.5) and a TeX distribution with XeLaTeX when building the documentation or posters.

### 2. Render the Quarto knowledge base

```bash
quarto render            # builds HTML/PDF/DOCX into _site/
quarto preview           # live-reloads during editing
```

The navigation, sidebar, and output formats are defined in `_quarto.yml`. Each `.qmd` file (e.g., `farsun.qmd`, `mittheilungen_data-1800.qmd`) can include executable Python chunks; chunk caching is enabled via `execute.freeze: auto`.

### 3. Recreate the main figures

#### Reviewed observer bubble & timeline plots

```bash
python scripts/reviewed_sunspot-data-2025.py \
  --help  # run with defaults to regenerate figures/reviewed_sources_*.{pdf,png,svg}
```

The script reads `data/Review_merged_sorted.csv`, normalises the observer spans, and outputs publication-ready bubble and timeline plots in `figures/`. Adjust parameters for alternate colour maps, label counts, or size scaling as needed.

#### Mittheilungen bubble/timeline

```bash
python scripts/mittheilungen-plots.py \
  --csv data/observation_years_by_observer.csv \
  --outdir figures \
  --start-year 1600 --end-year 1950
```

This tool powers the large-format bubble/timeline plots used on posters. It handles label de-confliction, dataset tabs, and A0-friendly sizing.

#### Additional helpers

- `scripts/reviewed_sunspot-data.py` – legacy cleaning pipeline that parses reviewer spreadsheets and emits `clean_sources-2010-2025.csv`.
- `scripts/webscraping-mitt-db.py` – experimental scraper for digitising Mittheilungen table of contents.
- `scripts/testing*.py` – scratch pads for pipeline experiments.

Each script assumes relative paths from the repository root; prefer `python scripts/<file>.py` to keep imports working.

### 4. Build posters and presentations

The `posters/` directory contains XeLaTeX beamerposter layouts styled for AGU:

```bash
cd posters
xelatex -interaction=nonstopmode agu-poster-v3.tex
```

The template pulls assets from `figures/` and `logos/`. For Reveal.js or Quarto presentations located in `presentations/`, run `quarto render presentations/<file>.qmd`.

---

## Coding standards & testing

- **Python style** – favour modern typing hints, `pathlib` paths, and vectorised pandas/numpy operations. Scripts should exit with actionable errors when input CSVs are missing or malformed.
- **Plotting** – keep a separation between data-access functions and plotting functions (`save_*` helpers). Parameters should be exposed at the function signature so posters can tweak dimensions/colour maps without editing internals.
- **Testing/run checks** – there is no formal test suite yet; the recommended smoke tests are:
  - `python scripts/reviewed_sunspot-data-2025.py`
  - `python scripts/mittheilungen-plots.py`
  - `quarto render`
  - `xelatex posters/agu-poster-v3.tex`

Please run the ones relevant to your change before opening a pull request.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Keep changes focused (plots, data ingestion, documentation, etc.).
3. Document new datasets in `data/Review-of-Data-Recover-post-SNV2.md` or the relevant `.qmd` file so the provenance remains clear.
4. Regenerate affected figures/posters and stage the updated files alongside the code.
5. Submit a pull request describing the motivation, datasets touched, and verification steps.

Discussions, bug reports, and suggestions are always welcome via GitHub issues or by contacting the FARSuN/SILSO team.

---

## License

The code and documentation are distributed under the [MIT License](LICENSE). Third-party scans or datasets may retain additional copyright or usage restrictions; please consult the metadata in `data/` before redistributing raw observations.
