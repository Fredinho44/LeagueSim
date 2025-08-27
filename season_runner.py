#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, yaml, random
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

from plate_loc_model import PlateLocSampler, default_mixtures_demo, load_mixtures_json
from sim_utils import (
    PULL_RUNS, PULL_STRESS_PITCHES, INJURY_CHANCE_HEAVY_OVER, EXTRA_INNING_RECOVERY_BONUS_DAYS,
    read_csv_rows, parse_date,
    staff_from_roster_rows, lineup_from_roster_rows, choose_sp_for_date, is_available,
    mark_recovery, maybe_injure, _safe_float
)
try:
    from sim_utils import EXTRA_INNING_RECOVERY_BONUS_DAYS  # noqa: F401
except Exception:
    EXTRA_INNING_RECOVERY_BONUS_DAYS = 1
import sim_utils  # to adjust global knobs if needed
from game_sim import simulate_one_game

def main():
    ap = argparse.ArgumentParser("Season Runner — Synthetic YakkerTech (roster-driven)")
    ap.add_argument("--roster_csv",   required=True)
    ap.add_argument("--schedule_csv", required=True)
    ap.add_argument("--priors",       required=True)
    ap.add_argument("--template_csv", required=False,default="", help="Optional YakkerTech template CSV")
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--seed",         type=int, default=7)
    ap.add_argument("--pull_runs_threshold", type=int, default=PULL_RUNS)
    ap.add_argument("--pull_high_stress_inning_pitches", type=int, default=PULL_STRESS_PITCHES)
    ap.add_argument("--injury_chance_heavy_over", type=float, default=INJURY_CHANCE_HEAVY_OVER)
    ap.add_argument("--plate_mixtures_json", default="", help="JSON file with mixture params for plate location")
    ap.add_argument("--plate_ar1_rho", type=float, default=0.30, help="AR(1) persistence for plate location")
    ap.add_argument("--plate_noise_x", type=float, default=0.03, help="x noise (ft)")
    ap.add_argument("--plate_noise_y", type=float, default=0.04, help="y noise (ft)")
    args = ap.parse_args()

    rng = random.Random(args.seed); np.random.seed(args.seed)

    # plate location sampler
    np_rng = np.random.default_rng(args.seed or 7)
    if args.plate_mixtures_json and Path(args.plate_mixtures_json).is_file():
        mixtures = load_mixtures_json(args.plate_mixtures_json)
    else:
        mixtures = default_mixtures_demo()
    plate_sampler = PlateLocSampler(
        rng=np_rng, mixtures=mixtures,
        rho=args.plate_ar1_rho,
        noise_x=args.plate_noise_x,
        noise_y=args.plate_noise_y,
    )

    # adjust global injury chance
    sim_utils.INJURY_CHANCE_HEAVY_OVER = float(args.injury_chance_heavy_over)

    # priors & template
    with open(args.priors, "r", encoding="utf-8") as f:
        priors = yaml.safe_load(f) or {}
        if args.template_csv and Path(args.template_csv).is_file():
             template_cols = list(pd.read_csv(args.template_csv, nrows=1, low_memory=False).columns)
        else:template_cols = []

    
    for must in ["Throws","Top/Bottom","BatterSide","PlayResult","KorBB","RunsScored","OutsOnPlay","HitType"]:
        if must not in template_cols:
            template_cols.append(must)

    season_root = Path(args.out_dir)
    out_dir = season_root / "Season01"
    out_dir.mkdir(parents=True, exist_ok=True)
    byteam_dir = out_dir / "ByTeam"
    byteam_dir.mkdir(parents=True, exist_ok=True)
    team_results: List[Dict[str,Any]] = []
    standings = defaultdict(lambda: {"W":0,"L":0,"T":0,"RS":0,"RA":0})
    season_pitchers = defaultdict(lambda: {"Name":"", "Team":"", "Role":"",
                                           "G":0,"GS":0,"IPOuts":0,"BF":0,"Pitches":0,
                                           "H":0,"B2":0,"B3":0,"HR":0,"BB":0,"K":0,"R":0,"ER":0})
    season_team_off = defaultdict(lambda: {"PA":0,"AB":0,"R":0,"H":0,"B2":0,"B3":0,"HR":0,"RBI":0,"BB":0,"K":0})

    roster_rows = read_csv_rows(Path(args.roster_csv))
    teams = set()
    for r in roster_rows:
        key = r.get("TeamID") or r.get("TeamName")
        if key: teams.add(str(key))
    team_staffs = {key: staff_from_roster_rows(roster_rows, key) for key in teams}

    roster_by_team = defaultdict(list)
    for r in roster_rows:
        key = r.get("TeamID") or r.get("TeamName")
        if key: roster_by_team[str(key)].append(r)

    sched = pd.read_csv(args.schedule_csv)
    def team_key_from_row(prefix: str, row) -> str:
        return str(row.get(prefix+"TeamID") or row.get(prefix+"Team") or row.get(prefix+"TeamName"))
    sched["__DateObj__"] = sched["Date"].apply(parse_date)
    sched = sched.sort_values(["__DateObj__"]).reset_index(drop=True)

    for idx, row in sched.iterrows():
        game_id = str(row.get("GameID") or f"G{idx+1:04d}")
        gdt: datetime = row["__DateObj__"]
        home_key = team_key_from_row("Home", row); away_key = team_key_from_row("Away", row)
        if not home_key or not away_key: continue
        if home_key not in team_staffs or away_key not in team_staffs: continue

        home_staff = team_staffs[home_key]; away_staff = team_staffs[away_key]
        home_sp = choose_sp_for_date(home_staff, gdt); away_sp = choose_sp_for_date(away_staff, gdt)
        if not home_sp or not away_sp:
            print(f"⚠️  Skipping {game_id} {gdt.date()} — no available SP"); continue

        home_pen = [p for p in home_staff["pen"] if is_available(p, gdt)][:4]
        away_pen = [p for p in away_staff["pen"] if is_available(p, gdt)][:4]
        knobs = {
            "pull_runs_threshold": int(args.pull_runs_threshold),
            "pull_high_stress_inning_pitches": int(args.pull_high_stress_inning_pitches),
        }

        df, score, usage, pbox_game, team_off_game = simulate_one_game(
            rng, priors, template_cols, gdt, home_key, away_key, home_sp, away_sp, home_pen, away_pen,
            roster_by_team, knobs, plate_sampler
        )

        # save per-game CSV
        fname = f"{gdt.strftime('%Y%m%d')}-{away_key}@{home_key}-{game_id}.csv"
        game_path = out_dir / fname
        df.to_csv(game_path, index=False)
        for team_key in (home_key, away_key):
            td = byteam_dir / str(team_key); td.mkdir(parents=True, exist_ok=True)
            (td / fname).write_text(game_path.read_text(encoding="utf-8"), encoding="utf-8")

        # results/standings
        rs, ra = score["home"], score["away"]
        team_results.append({"GameID": game_id, "Date": gdt.strftime("%Y-%m-%d"),
                             "HomeTeam": home_key, "AwayTeam": away_key,
                             "HomeScore": rs, "AwayScore": ra,
                             "Winner": home_key if rs > ra else (away_key if ra > rs else "Tie"),
                             "Loser":  away_key if rs > ra else (home_key if ra > rs else "Tie")})
        standings[home_key]["RS"] += rs; standings[home_key]["RA"] += ra
        standings[away_key]["RS"] += ra; standings[away_key]["RA"] += rs
        if rs > ra: standings[home_key]["W"] += 1; standings[away_key]["L"] += 1
        elif ra > rs: standings[away_key]["W"] += 1; standings[home_key]["L"] += 1
        else: standings[home_key]["T"] += 1; standings[away_key]["T"] += 1

        # season rollups
        for pid, g in pbox_game.items():
            s = season_pitchers[pid]
            if not s["Name"]: s["Name"] = g["Name"]
            if not s["Team"]: s["Team"] = g["Team"]
            if not s["Role"]: s["Role"] = g["Role"]
            for k in ("G","GS","IPOuts","BF","Pitches","H","B2","B3","HR","BB","K","R","ER"):
                s[k] += g[k]
        for tkey, g in team_off_game.items():
            s = season_team_off[tkey]
            for k in ("PA","AB","R","H","B2","B3","HR","RBI","BB","K"):
                s[k] += g[k]

        # recovery & injuries
        mark_recovery(home_sp, gdt, usage["home_sp_pitches"], "SP")
        mark_recovery(away_sp, gdt, usage["away_sp_pitches"], "SP")
        for p in home_pen:
            if p.get("_PitchCount",0) > 0: mark_recovery(p, gdt, p["_PitchCount"], "RP")
        for p in away_pen:
            if p.get("_PitchCount",0) > 0: mark_recovery(p, gdt, p["_PitchCount"], "RP")

        # extra-inning bonus day for anyone who appeared in extras
        if usage.get("extra_innings", 0) > 0:
            extras_home = set(usage.get("home_pids_in_extras", []))
            extras_away = set(usage.get("away_pids_in_extras", []))
            def _add_day_if_extras(p):
                if p["PitcherId"] in extras_home.union(extras_away) and p.get("_NextOK") is not None:
                    p["_NextOK"] = p["_NextOK"] + timedelta(days=1)
            _add_day_if_extras(home_sp); _add_day_if_extras(away_sp)
            for p in home_pen: _add_day_if_extras(p)
            for p in away_pen: _add_day_if_extras(p)

        from random import random as _r
        def inj(p, ct):
            if ct > p["_Limit"] + 25 and _r() < sim_utils.INJURY_CHANCE_HEAVY_OVER:
                maybe_injure(p, rng, ct)
        inj(home_sp, usage["home_sp_pitches"]); inj(away_sp, usage["away_sp_pitches"])
        for penp in home_pen: inj(penp, penp.get("_PitchCount",0))
        for penp in away_pen: inj(penp, penp.get("_PitchCount",0))

        print(f"✅ {gdt.date()} {away_key}@{home_key}  Final: {score['away']}-{score['home']}  -> {fname}")

    # write summaries
    pd.DataFrame(team_results).to_csv(out_dir / "team_game_results.csv", index=False)

    std_rows = []
    for team, s in standings.items():
        g = s["W"]+s["L"]+s["T"]
        std_rows.append({
            "Team": team, "W": s["W"], "L": s["L"], "T": s["T"],
            "GP": g, "RS": s["RS"], "RA": s["RA"], "RD": s["RS"]-s["RA"],
            "WinPct": round((s["W"] + 0.5*s["T"]) / g, 3) if g else 0.0
        })
    pd.DataFrame(std_rows).sort_values(["WinPct","RD"], ascending=[False,False]).to_csv(out_dir / "team_standings.csv", index=False)

    def outs_to_ip_str(outs: int) -> str:
        return f"{outs//3}.{outs%3}"

    pitchers_rows = []
    for pid, s in season_pitchers.items():
        pitchers_rows.append({
            "PitcherId": pid, "Name": s["Name"], "Team": s["Team"], "Role": s["Role"],
            "G": s["G"], "GS": s["GS"], "IP": outs_to_ip_str(s["IPOuts"]),
            "BF": s["BF"], "Pitches": s["Pitches"], "H": s["H"], "2B": s["B2"], "3B": s["B3"],
            "HR": s["HR"], "BB": s["BB"], "K": s["K"], "R": s["R"], "ER": s["ER"]
        })
    pd.DataFrame(pitchers_rows).sort_values(["Team","Name"]).to_csv(out_dir / "player_pitching.csv", index=False)

    tbat = []
    for t, s in season_team_off.items():
        tbat.append({"Team": t, **s})
    pd.DataFrame(tbat).sort_values(["Team"]).to_csv(out_dir / "team_batting.csv", index=False)

    tpit = defaultdict(lambda: {"G":0,"GS":0,"IPOuts":0,"BF":0,"Pitches":0,"H":0,"B2":0,"B3":0,"HR":0,"BB":0,"K":0,"R":0,"ER":0})
    for s in season_pitchers.values():
        tp = tpit[s["Team"]]
        for k in tp.keys():
            tp[k] += s[k] if k in s else 0
    tpit_rows = []
    for team, s in tpit.items():
        tpit_rows.append({"Team": team, "G": s["G"], "GS": s["GS"], "IP": f"{s['IPOuts']//3}.{s['IPOuts']%3}",
                          "BF": s["BF"], "Pitches": s["Pitches"], "H": s["H"], "2B": s["B2"], "3B": s["B3"],
                          "HR": s["HR"], "BB": s["BB"], "K": s["K"], "R": s["R"], "ER": s["ER"]})
    pd.DataFrame(tpit_rows).sort_values(["Team"]).to_csv(out_dir / "team_pitching.csv", index=False)

    print(f"\n🎯 Done. Wrote season outputs to {str(out_dir)}")
    print("   - team_game_results.csv")
    print("   - team_standings.csv")
    print("   - player_pitching.csv")
    print("   - team_batting.csv")
    print("   - team_pitching.csv")
    print("   - ByTeam/<Team>/ copies of each game")

if __name__ == "__main__":
    main()
