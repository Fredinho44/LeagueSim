# League Simulator — Install & Use (Streamlit)

This project builds synthetic YakkerTech-style or Trackman-Style(SOON) pitch-by-pitch CSVs and lets you tune **league demographics + game model weights** through a **Streamlit UI**.

---

## 1. Install

**Requirements**
- Python 3.9+
- Works on Windows, macOS, Linux
- Recommended: a virtual environment

```bash
# Windows (PowerShell)
py -3 -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install streamlit pandas numpy pyyaml faker
```

---

## 2. Launch Streamlit

From the folder containing your scripts (`streamlit_app.py`, `make_league.py`, `simulate_seasons_roster_style.py`, `sim_utils.py`):

```bash
streamlit run streamlit_app.py
```

This will open the app in your browser at http://localhost:8501.

---

## 3. Workflow in the App

### **Tab 1 — League Tuning**
- Adjust **roster composition** (handedness, two-way rate, command tier μ/σ, velo priors, durability, games per team).
- Tweak **game model weights** (SP/RP recovery days, fatigue penalties, TTO penalty, injury chance, mid-game pull thresholds, extra-inning penalties).  
  *All of these knobs are pushed directly into `sim_utils.py` when you run a sim—no manual editing needed.*
- Save/load configs as JSON for reuse.

### **Tab 2 — Build League**
- Click **Generate League CSVs**.  
  Outputs to `league_out/`:
  - `organizations.csv`
  - `teams.csv`
  - `rosters.csv`
  - `schedule.csv`

### **Tab 3 — Roster Overview**
- Preview rosters or load `rosters.csv` from disk.
- Get summary stats (Count / Mean / Std / Percentiles).
- Export a summary table.

### **Tab 4 — Run Simulation**
- Point to:
  - `league_out/rosters.csv`
  - `league_out/schedule.csv`
  - `rules_pitch_by_pitch.yaml` (priors YAML)
  - A YakkerTech **template CSV** (any real game, just for column order)
- Choose output folder (default: `season_out/`).
- Run the simulator.  
  You’ll get one YakkerTech-style CSV per scheduled game:
  ```
  season_out/
    Season01/
      GAME_0001-Home@Away.csv
      GAME_0002-...
      training_labels.csv
  ```

---

## 4. Minimal Run-Through

```bash
# 1. Install deps
pip install streamlit pandas numpy pyyaml faker

# 2. Launch app
streamlit run streamlit_app.py

# 3. In the app
#    Tab 1: tune league + sim knobs
#    Tab 2: build league → writes league_out/
#    Tab 4: run sim → writes season_out/
```

---

## 5. Expected Folder Layout

```
your_project/
  streamlit_app.py
  make_league.py
  simulate_seasons_roster_style.py
  sim_utils.py
  rules_pitch_by_pitch.yaml
  league_out/
    organizations.csv
    teams.csv
    rosters.csv
    schedule.csv
  season_out/
    Season01/
      M0001-TeamA@TeamB.csv
      ...
```

---

# ModelCA League Builder & Season Simulator — README

This project generates a **synthetic baseball/softball league** and (optionally) runs **season simulations** that produce YakkerTech‑style per‑game CSVs. It includes:
- `make_league.py` — creates organizations, teams, **rosters** with bio/geometry/scouting/durability, and a balanced **schedule**.
- `simulate_seasons_roster_style.py` — uses your rosters + priors YAML + a real YakkerTech template to simulate games and emit per‑game CSVs.
- `streamlit_app.py` — a UI to **tune league parameters**, generate CSVs, and review roster stats like the “legend” screenshot (Count/Mean/Std/Percentiles).

---

## Quick Start

### 1) Environment
```bash
cd C:\Users\User\Desktop\ModelCA
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install streamlit pandas pyyaml numpy faker
```

### 2) Run the app
```bash
streamlit run .\streamlit_app.py
```
Open the browser (usually http://localhost:8501).

### 3) Generate a league
1. **1) League Tuning** — adjust handedness, two‑way rate, command tier μ/σ, velo priors, durability, games per tier, etc. (Advanced JSON editors available.)
2. **2) Build League** — click **Generate League CSVs using current tuning**. This writes:
   - `league_out/organizations.csv`
   - `league_out/teams.csv`
   - `league_out/rosters.csv`  ← full roster with all attributes
   - `league_out/schedule.csv`
3. **3) Roster Overview** — load `rosters.csv` and view the summary table (Count / Mean / Std / Min / P05 / P25 / P50 / P75 / P95 / Max).

### 4) (Optional) Run the season simulator
From CLI or wire it into the app (future tab):
```bash
python .\simulate_seasons_roster_style.py ^
  --roster_csv   ".\league_out\rosters.csv" ^
  --schedule_csv ".\league_out\schedule.csv" ^
  --priors       ".\rules_pitch_by_pitch.yaml" ^
  --template_csv ".\05_28_2024 11_43_19 AM-Florida@Florida.csv" ^
  --out_dir      ".\season_out" ^
  --seed 7
```
Outputs (per season): per‑game YakkerTech‑style CSVs, team summaries, standings (if enabled in that script version).

---

## File & Folder Structure

```
ModelCA/
├─ make_league.py
├─ simulate_seasons_roster_style.py
├─ streamlit_app.py
├─ rules_pitch_by_pitch.yaml        # priors for simulator (pitch mixes, location, outcomes)
├─ 05_28_2024 ... Florida@Florida.csv  # real YakkerTech template (column names/order)
├─ league_out/
│  ├─ organizations.csv
│  ├─ teams.csv
│  ├─ rosters.csv
│  └─ schedule.csv
└─ season_out/
   └─ Season01/ ... (game CSVs)
```

---

## League Tuning (what each knob does)

### Handedness & Batting Side
- **PITCH_HAND_P**: ratio of pitcher throwing hands (`Right`, `Left`).
- **BAT_SIDE_P**: batting side mix (`Right`, `Left`, `Switch`).

### Two‑Way Players
- **TWO_WAY_RATE** (per tier): probability hitters are also pitchers. Two‑ways get an additional **PitcherSecondaryRole** (`SP`/`RP`) plus pitching grades, usage, and geometry.

### Command Tier (CommandTier)
- **_CMD_TIER_PARAMS** per tier: `{mu, sd, clip}` for a normalized command multiplier (~0.80–1.20 typical).
- **_CMD_MIX**: team‑level minimum counts above `high` and below `low` to guarantee “a couple command guys and a couple wild ones.”

### Velo Priors (AvgFBVelo)
- **_VELO_PRIORS**: mean/sd (mph) per `(Tier, Cluster)`. App’s global Δ slider can nudge all means at once.

### Durability / Workload
- **DURABILITY_PRIORS**: stamina mean/sd, pitch‑count mean/sd, previous season IP, etc. The script derives **PitchCountLimit**, **AvgPitchesPerOuting**, **ExpectedBattersFaced**, **RecoveryDaysNeeded** from stamina, command, age, and velo.

### Cluster Mix & Pitch Usage
- **CLUSTER_WEIGHTS_R/L**: how RHP/LHP are distributed across pitching clusters (e.g., `PowerFB`, `SinkerSlider`, `CutterMix`, `ChangeCmd`, `BreakHeavy`).
- **PITCH_CLUSTERS**: baseline pitch mixes by cluster (e.g., PowerFB might be 55% FB, 25% SL, …). The generator then biases per‑pitch **command** and **usage** by the scout grades.

### Arm Slot / Geometry
- **ARM_PRIORS**: distribution of arm‑slot buckets by cluster (`Overhand`, `High34`, `Low34`, `Sidearm`, `Submarine`).
- Release metrics are derived from body metrics + arm slot:
  - **RelHeight_ft** (Z), **RelSide_ft** (X; sign follows throws), **Extension_ft**.

### Games per Team
- **GAMES_PER_TEAM**: per tier (Majors/AAA/Rookie). The scheduler builds balanced home/away slates and adds extra rounds to hit targets.

---

## Roster CSV — Column Legend

### Identity & Assignment
- **TeamID, TeamName, OrgID, OrgName, Tier**: team/org identity; Tier ∈ {Majors, AAA, Rookie}.
- **PlayerID, FirstName, LastName, FullName**: player identity.
- **Role**: `"SP"`, `"RP"`, or `"BAT"` (non‑two‑way hitters are `"BAT"`).
- **Position**: primary defensive position for hitters; empty for pitchers.
- **SecondaryPosition, IsTwoWay, PitcherSecondaryRole**: two‑way fields (0/1 for IsTwoWay).

### Handedness & Archetypes
- **Throws, Bats**: `"Right"` / `"Left"` (and `"Switch"` for Bats).
- **Cluster**: pitcher cluster (`PowerFB`, `SinkerSlider`, `CutterMix`, `ChangeCmd`, `BreakHeavy`).
- **HitterArchetype**: `"Balanced"`, `"Contact"`, `"Power"`, `"Speed"`, `"OBP"`.

### Pitch Usage & Command
- **UsageJSON**: normalized dict of pitch usage (e.g., `{"Fastball":0.54,"Slider":0.27,...}`).
- **CommandTier**: normalized command multiplier (≈0.8–1.2 typical).
- **CommandByPitchJSON**: per‑pitch command scalars (CommandTier × pitch skill/grade boosts).

### Bio / Body / Geometry
- **AgeYears, HeightIn, WingspanIn**: integers (inches for height/wingspan).
- **ArmSlotDeg, ArmSlotBucket**: release arm‑slot estimate and bucket label.
- **RelHeight_ft (Z)**, **RelSide_ft (X)**, **Extension_ft**: release geometry in feet.

### Durability / Workload
- **AvgFBVelo**: avg FB velo (mph) for pitchers (and two‑way pitching persona).
- **PrevSeasonIP**: prior season innings.
- **StaminaScore**: aggregate 30–95 index used to derive workload.
- **PitchCountLimit**: per‑outing cap target.
- **AvgPitchesPerOuting**: expected average per appearance.
- **ExpectedBattersFaced**: expected BF per outing.
- **RecoveryDaysNeeded**: suggested days of rest after an outing.
- **InjuryFlag**: 0/1 heuristic risk flag (high velo/age add risk).


## Simulator Inputs & Outputs (high level)

**Inputs**
- `rosters.csv` — generated above.
- `schedule.csv` — generated above.
- `rules_pitch_by_pitch.yaml` — prior distributions for pitch types, locations, calls, in‑play splits, etc.
- One **real game** YakkerTech CSV to clone column order/names.

**Outputs**
- One **YakkerTech‑style CSV per game** in `season_out/SeasonXX/`.
- Optionally, roll‑ups: team_boxscores, standings, player pitching lines (depends on simulator version).

---

## Common Issues & Fixes

- `UnboundLocalError: p_role` — only call `sample_present_role` for pitchers. For non–two‑way hitters set `present_role=None` and format role as future‑only (“55”).
- `ModuleNotFoundError` on reload (Streamlit) — reload by module name; see the `_reload_module` helper in the app.
- “None” strings in CSV — the CSV writer should coerce `None` → `""` (see `write_csv()` implementation).


## License / Credits

This is a synthetic data generation and simulation toolkit intended for analytics prototyping and educational use.
YakkerTech & Trackman are a separate product — CSVs here only mimic column names/order for workflow compatibility.
© 2025 Alfredo Caraballo.