# LeagueSim — Streamlit App

This bundle lets you tune a generated baseball league, export CSVs, and run season sims — all from a Streamlit UI.

## Quick Start

### Windows
```bat
cd LeagueSim_Package
run.bat
```
Then open http://localhost:8501

### macOS / Linux
```bash
cd LeagueSim_Package
./run.sh
```
Then open http://localhost:8501

> The scripts create a `.venv` virtual environment, install requirements, and launch Streamlit.

## What’s inside

- `streamlit_app.py` — the UI (Tabs: Tuning, League Build & Preview, Run Simulation, README).
- `make_league.py`, `simulate_seasons_roster_style.py`, `sim_utils.py`, etc. — your generation and sim logic.
- `rules_pitch_by_pitch.yaml` — priors for pitch-by-pitch model (used by the sim).
- `league_out/` — where the app writes organizations/teams/rosters/schedule.
- `season_out/` — where game CSVs are saved after running the sim.
- `requirements.txt`, `run.sh`, `run.bat` — setup helpers.
- `.streamlit/config.toml` — Streamlit settings.

## Usage in the App

1. **Sidebar → Script folder**: leave as the current folder (auto-detected) or point to it.
2. **Tab 1 — League Tuning**: adjust demographics, command/velo, schedule size, fatigue/TTO/injuries/pulls/extras.
3. **Tab 2 — League Build & Preview**:
   - *Quick Preview* to sanity-check distributions in-memory.
   - *Write CSVs to Folder* to generate official `league_out/*.csv` (+ `config_used.json` snapshot).
4. **Tab 3 — Run Simulation**:
   - Decide whether to **Regenerate league** (from current tuning) or use the existing `league_out`.
   - Provide path to `rules_pitch_by_pitch.yaml` (it's in the same folder by default).
   - Choose output folder (default `season_out`) and run.
5. **Tab 4 — README**: in-app docs if provided.

## Tips

- Keep a fixed **Seed** for reproducible roster distributions.
- After a build, check the "Realized pitcher throws %" and "Realized batter side %" readouts.
- Share only this folder; your tester does not need any other files.

## Python Version

Tested with Python 3.10–3.11. If you have 3.12+ and see issues, try 3.11.

## Troubleshooting

- If the browser blocks downloading `.py`, grab the ZIP.
- On Windows PowerShell, you may need: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` to allow venv activation.
