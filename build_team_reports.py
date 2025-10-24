#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build team reports per season and a combined longitudinal report.

Inputs (per SeasonN directory under --root):
- team_standings.csv
- team_batting.csv
- team_pitching.csv

Outputs:
- <SeasonN>/team_season_report.csv  (joined batting + pitching + W/L/T/WinPct)
- <root>/team_season_combined.csv   (all seasons stacked with Season column)
- <root>/team_season_improvements.csv (per team, season-over-season deltas)

Usage:
  python build_team_reports.py --root ./season_out
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import pandas as pd


def _read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


essential_bat_cols = [
    "Team", "PA","AB","R","H","1B","2B","3B","HR","RBI","BB","K","HBP","SF",
    "TB","AVG","OBP","SLG","OPS"
]

essential_pit_cols = [
    "Team","G","GS","IP","BF","Pitches","H","2B","3B","HR","BB","K","R","ER","K9","BB9","WHIP","ERA"
]


def build_for_season(season_dir: Path) -> Path | None:
    ts = _read_csv(season_dir / "team_standings.csv")
    tb = _read_csv(season_dir / "team_batting.csv")
    tp = _read_csv(season_dir / "team_pitching.csv")
    if tb.empty and tp.empty:
        return None

    # Trim to essentials if present
    if not tb.empty:
        cols = [c for c in essential_bat_cols if c in tb.columns]
        tb = tb[cols]
    if not tp.empty:
        cols = [c for c in essential_pit_cols if c in tp.columns]
        tp = tp[cols]

    # Merge batting and pitching on Team
    if tb.empty:
        merged = tp.copy()
    elif tp.empty:
        merged = tb.copy()
    else:
        merged = pd.merge(tb, tp, on="Team", how="outer", suffixes=("_bat","_pit"))

    # Add W/L/T/WinPct if standings available
    if not ts.empty:
        ts2 = ts.copy()
        if "WinPct" not in ts2.columns:
            try:
                ts2["WinPct"] = ts2.apply(lambda r: (float(r.get("W",0)) + 0.5*float(r.get("T",0))) / max(1.0, (float(r.get("W",0))+float(r.get("L",0))+float(r.get("T",0)))), axis=1)
            except Exception:
                pass
        keep = [c for c in ["Team","W","L","T","RS","RA","WinPct"] if c in ts2.columns]
        merged = pd.merge(merged, ts2[keep], on="Team", how="left")

    out_path = season_dir / "team_season_report.csv"
    merged.to_csv(out_path, index=False)
    return out_path


def build_combined(root: Path) -> tuple[Path | None, Path | None]:
    seasons: List[Path] = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("season")])
    if not seasons:
        return None, None

    frames = []
    for sd in seasons:
        rpt = sd / "team_season_report.csv"
        if not rpt.exists():
            built = build_for_season(sd)
            rpt = built if built else rpt
        if rpt.exists():
            df = pd.read_csv(rpt)
            df.insert(0, "Season", sd.name)
            frames.append(df)
    if not frames:
        return None, None

    combined = pd.concat(frames, ignore_index=True)
    combined_path = root / "team_season_combined.csv"
    combined.to_csv(combined_path, index=False)

    # Compute deltas per team sorted by Season order
    try:
        import re
        def season_key(s: str) -> int:
            m = re.search(r"(\d+)$", s)
            return int(m.group(1)) if m else 0
        combined_sorted = combined.sort_values(by=["Team","Season"], key=lambda s: s.map(season_key))
    except Exception:
        combined_sorted = combined.sort_values(["Team","Season"])  # fallback

    # Choose a few rate/summary columns to compare
    rate_cols = [c for c in ["WinPct","OPS","ERA","WHIP","K9","BB9"] if c in combined_sorted.columns]
    deltas = []
    for team, grp in combined_sorted.groupby("Team", sort=False):
        prev = None
        for _, row in grp.iterrows():
            rec = {"Team": team, "Season": row["Season"]}
            for c in rate_cols:
                rec[c] = row.get(c, None)
                if prev is not None and (c in prev) and pd.notna(prev[c]) and pd.notna(row.get(c)):
                    rec[c+"_Delta"] = float(row.get(c)) - float(prev[c])
            deltas.append(rec)
            prev = {c: row.get(c) for c in rate_cols}
    improvements = pd.DataFrame(deltas)
    improvements_path = root / "team_season_improvements.csv"
    improvements.to_csv(improvements_path, index=False)

    return combined_path, improvements_path


def main():
    ap = argparse.ArgumentParser("Build team season reports and combined improvements")
    ap.add_argument("--root", default=str(Path(__file__).parent / "season_out"), help="Root directory containing Season* folders")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    # Per-season reports
    seasons = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("season")])
    any_built = False
    for sd in seasons:
        out = build_for_season(sd)
        if out:
            print(f"Built: {out}")
            any_built = True

    # Combined
    cb, imp = build_combined(root)
    if cb:
        print(f"Combined report: {cb}")
    if imp:
        print(f"Improvements report: {imp}")
    if not any_built and not cb:
        print("No season data found under root.")


if __name__ == "__main__":
    main()

