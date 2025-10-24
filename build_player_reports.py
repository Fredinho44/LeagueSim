#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build player-by-player season reports (batting and pitching) and combined, multi-season views.

Per-season outputs (under SeasonN/):
- player_batting.csv     (derived from per-game YakkerTech CSVs)
- player_pitching.csv    (already produced by season_runner; copied/enhanced if present)

Combined outputs (under --root):
- player_batting_combined.csv
- player_pitching_combined.csv
- player_batting_improvements.csv (season-over-season deltas for AVG/OBP/SLG/OPS)
- player_pitching_improvements.csv (season-over-season deltas for ERA/WHIP/K9/BB9)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd


def _safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _iter_game_csvs(season_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in season_dir.glob("*.csv"):
        name = p.name.lower()
        if name.endswith("_trackman.csv"):
            continue
        if name.startswith("team_") or name in ("team_standings.csv","team_batting.csv","team_pitching.csv",
                                                "player_pitching.csv","team_season_report.csv"):
            continue
        files.append(p)
    return files


def build_batting_for_season(season_dir: Path) -> Path | None:
    games = _iter_game_csvs(season_dir)
    if not games:
        return None
    parts = []
    for g in games:
        df = _safe_read_csv(g)
        if df.empty:
            continue
        # We need at minimum BatterId and PlayResult
        if "BatterId" not in df.columns or "PlayResult" not in df.columns:
            continue
        d = pd.DataFrame({
            "BatterId": df["BatterId"].astype(str),
            "Batter": df.get("Batter", "").astype(str),
            "PlayResult": df["PlayResult"].astype(str).str.strip(),
            "KorBB": df.get("KorBB", "").astype(str).str.strip(),
        })
        parts.append(d)
    if not parts:
        return None
    allp = pd.concat(parts, ignore_index=True)
    # Normalize events
    pr = allp["PlayResult"].str.lower()
    kb = allp["KorBB"].str.lower()
    is_walk = (pr.eq("walk") | kb.eq("walk"))
    is_k = pr.isin(["strikeoutswinging","strikeoutlooking"]) | kb.eq("strikeout")
    is_hbp = pr.eq("hitbypitch")
    is_sf = pr.eq("sacrifice")
    is_out = pr.isin(["out","fielderchoice","error"])  # error counts as AB, ROE
    is_single = pr.eq("single")
    is_double = pr.eq("double")
    is_triple = pr.eq("triple")
    is_hr = pr.eq("homerun")

    # Vectorized boolean indicators, then groupby-sum (avoids deprecated apply on groups)
    allp = allp.assign(
        BB=is_walk.astype(int),
        K=is_k.astype(int),
        HBP=is_hbp.astype(int),
        SF=is_sf.astype(int),
        **{
            "1B": is_single.astype(int),
            "2B": is_double.astype(int),
            "3B": is_triple.astype(int),
        },
        HR=is_hr.astype(int),
        OUT=is_out.astype(int),
    )
    grp = allp.groupby(["BatterId","Batter"], dropna=False)
    base = grp.size().rename("PA").to_frame()
    sums = grp[["BB","K","HBP","SF","1B","2B","3B","HR","OUT"]].sum()
    stats = base.join(sums).reset_index()
    # Derive AB, H, TB and rates
    stats["H"] = stats["1B"] + stats["2B"] + stats["3B"] + stats["HR"]
    stats["AB"] = stats["OUT"] + stats["H"]  # excludes BB, HBP, SF
    stats["TB"] = stats["1B"] + 2*stats["2B"] + 3*stats["3B"] + 4*stats["HR"]
    def _safe_div(a, b):
        try:
            a = float(a); b = float(b)
            return (a / b) if (b and b != 0.0) else float("nan")
        except Exception:
            return float("nan")
    stats["AVG"] = stats.apply(lambda r: round(_safe_div(r["H"], r["AB"]), 3), axis=1)
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    stats["OBP"] = stats.apply(lambda r: round(_safe_div(r["H"] + r["BB"] + r["HBP"], r["AB"] + r["BB"] + r["HBP"] + r["SF"]), 3), axis=1)
    stats["SLG"] = stats.apply(lambda r: round(_safe_div(r["TB"], r["AB"]), 3), axis=1)
    stats["OPS"] = stats.apply(lambda r: round((r["OBP"] + r["SLG"]) if pd.notna(r["OBP"]) and pd.notna(r["SLG"]) else float("nan"), 3), axis=1)
    # ISO and BABIP
    stats["ISO"] = stats.apply(lambda r: round((r["SLG"] - r["AVG"]) if pd.notna(r["SLG"]) and pd.notna(r["AVG"]) else float("nan"), 3), axis=1)
    def _babip(r):
        ab = float(r["AB"]); k = float(r["K"]); hr = float(r["HR"]); sf = float(r["SF"])
        denom = ab - k - hr + sf
        num = float(r["H"]) - hr
        return round(_safe_div(num, denom), 3)
    stats["BABIP"] = stats.apply(_babip, axis=1)

    # wOBA and wRC+ (using standard MLB-ish weights; IBB not tracked so included in BB)
    wBB, wHBP, w1B, w2B, w3B, wHR = 0.69, 0.72, 0.89, 1.27, 1.62, 2.10
    def _woba_num(r):
        return (wBB * float(r["BB"]) + wHBP * float(r["HBP"]) +
                w1B * float(r["1B"]) + w2B * float(r["2B"]) + w3B * float(r["3B"]) + wHR * float(r["HR"]))
    def _woba_den(r):
        # AB + BB + HBP + SF (no IBB split)
        return float(r["AB"]) + float(r["BB"]) + float(r["HBP"]) + float(r["SF"]) 
    stats["wOBA"] = stats.apply(lambda r: round(_safe_div(_woba_num(r), _woba_den(r)), 3), axis=1)

    # wRC+ requires league context; try to read season team totals if available
    try:
        tb = pd.read_csv(season_dir / "team_batting.csv")
        lg_PA = float(tb["PA"].sum()) if "PA" in tb.columns else 0.0
        lg_R = float(tb["R"].sum()) if "R" in tb.columns else 0.0
        # League wOBA from our player stats aggregation
        lg_num = stats.apply(_woba_num, axis=1).sum()
        lg_den = stats.apply(_woba_den, axis=1).sum()
        lg_woba = (lg_num / lg_den) if lg_den else float("nan")
        woba_scale = 1.15  # typical scale; can be refined
        lg_r_pa = (lg_R / lg_PA) if lg_PA else float("nan")
        def _wrc_plus(r):
            pa = float(r["PA"]) or 0.0
            woba = float(r["wOBA"]) if pd.notna(r["wOBA"]) else float("nan")
            if pa <= 0 or not pd.notna(woba) or not pd.notna(lg_woba) or not pd.notna(lg_r_pa):
                return ""
            wraa = (woba - lg_woba) / woba_scale * pa
            try:
                val = 100.0 * (((wraa / pa) + lg_r_pa) / lg_r_pa)
                return int(round(val))
            except Exception:
                return ""
        stats["wRC+"] = stats.apply(_wrc_plus, axis=1)
    except Exception:
        # If we cannot compute league context, leave wRC+ blank
        stats["wRC+"] = ""
    outp = season_dir / "player_batting.csv"
    stats.rename(columns={"BatterId":"PlayerId","Batter":"Name"}).to_csv(outp, index=False)
    return outp


def build_pitching_for_season(season_dir: Path) -> Path | None:
    """Normalize and emit player_pitching.csv with PlayerId column.
    If an existing file uses PitcherId, rename it to PlayerId for consistency.
    """
    pp = season_dir / "player_pitching.csv"
    if not pp.exists():
        return None
    try:
        df = pd.read_csv(pp)
    except Exception:
        return pp
    # Normalize ID col
    if "PlayerId" not in df.columns:
        if "PitcherId" in df.columns:
            df = df.rename(columns={"PitcherId": "PlayerId"})
        elif "ID" in df.columns:
            df = df.rename(columns={"ID": "PlayerId"})
    # Ensure Name col exists
    if "Name" not in df.columns:
        # Try common alternatives
        for alt in ("Pitcher","FullName"):
            if alt in df.columns:
                df = df.rename(columns={alt: "Name"})
                break
    try:
        df.to_csv(pp, index=False)
    except Exception:
        pass
    return pp


def build_combined(root: Path, which: str) -> tuple[Path | None, Path | None]:
    assert which in ("batting","pitching")
    seasons = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("season")])
    frames = []
    for sd in seasons:
        f = sd / ("player_batting.csv" if which == "batting" else "player_pitching.csv")
        if not f.exists():
            continue
        df = pd.read_csv(f)
        # Normalize PlayerId for pitching if needed
        if which == "pitching" and "PlayerId" not in df.columns and "PitcherId" in df.columns:
            df = df.rename(columns={"PitcherId": "PlayerId"})
        if "PlayerId" not in df.columns:
            # Skip files without an ID
            continue
        df.insert(0, "Season", sd.name)
        frames.append(df)
    if not frames:
        return None, None
    combined = pd.concat(frames, ignore_index=True)
    comb_path = root / (f"player_{which}_combined.csv")
    combined.to_csv(comb_path, index=False)

    # Improvements (season-over-season deltas)
    try:
        import re
        def season_key(s: str) -> int:
            m = re.search(r"(\d+)$", s)
            return int(m.group(1)) if m else 0
        combined_sorted = combined.sort_values(by=["PlayerId","Season"], key=lambda s: s.map(season_key))
    except Exception:
        combined_sorted = combined.sort_values(["PlayerId","Season"])  # fallback

    if which == "batting":
        rate_cols = [c for c in ["AVG","OBP","SLG","OPS"] if c in combined_sorted.columns]
    else:
        rate_cols = [c for c in ["ERA","WHIP","K9","BB9"] if c in combined_sorted.columns]

    rows = []
    for pid, grp in combined_sorted.groupby("PlayerId", sort=False):
        prev = None
        for _, row in grp.iterrows():
            rec = {"PlayerId": pid, "Season": row["Season"], "Name": row.get("Name", "")}
            for c in rate_cols:
                rec[c] = row.get(c, None)
                if prev is not None and (c in prev) and pd.notna(prev[c]) and pd.notna(row.get(c)):
                    rec[c+"_Delta"] = float(row.get(c)) - float(prev[c])
            rows.append(rec)
            prev = {c: row.get(c) for c in rate_cols}
    imp = pd.DataFrame(rows)
    imp_path = root / (f"player_{which}_improvements.csv")
    imp.to_csv(imp_path, index=False)
    return comb_path, imp_path


def main():
    ap = argparse.ArgumentParser("Build player-by-player season reports")
    ap.add_argument("--root", default=str(Path(__file__).parent / "season_out"))
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    seasons = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower().startswith("season")])
    any_built = False
    for sd in seasons:
        b = build_batting_for_season(sd)
        if b: print(f"Built batting: {b}"); any_built = True
        p = build_pitching_for_season(sd)
        if p: print(f"Found pitching: {p}")

    cb_bat, imp_bat = build_combined(root, "batting")
    cb_pit, imp_pit = build_combined(root, "pitching")
    if cb_bat: print(f"Combined batting: {cb_bat}")
    if imp_bat: print(f"Improvements batting: {imp_bat}")
    if cb_pit: print(f"Combined pitching: {cb_pit}")
    if imp_pit: print(f"Improvements pitching: {imp_pit}")
    if not any_built and not (cb_bat or cb_pit):
        print("No season game CSVs found to build player reports.")


if __name__ == "__main__":
    main()
