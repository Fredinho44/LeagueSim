#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, yaml, random, hashlib, json
from dataclasses import replace
import sys, builtins
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import math

from plate_loc_model import PlateLocSampler, default_mixtures_demo, load_location_bundle, PlateLocBundle
from csv_format_manager import CSVFormatManager
from sim_utils import (
    PULL_RUNS, PULL_STRESS_PITCHES, INJURY_CHANCE_HEAVY_OVER, EXTRA_INNING_RECOVERY_BONUS_DAYS,
    read_csv_rows, parse_date,
    staff_from_roster_rows, choose_sp_for_date, is_available,
    mark_recovery, maybe_injure
)
try:
    from sim_utils import EXTRA_INNING_RECOVERY_BONUS_DAYS  # noqa: F401
except Exception:
    EXTRA_INNING_RECOVERY_BONUS_DAYS = 1
import sim_utils  # to adjust global knobs if needed
from game_sim import simulate_one_game

# Ensure console prints do not crash on Windows cp1252. Prefer UTF-8; otherwise drop unencodable chars.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def _print_ascii_safe(*args, **kwargs):
    end = kwargs.get("end", "\n")
    sep = kwargs.get("sep", " ")
    s = sep.join(str(a) for a in args)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        sys.stdout.write(s + end)
    except Exception:
        # Fallback: remove/ignore characters that cannot be encoded in current codepage
        try:
            sys.stdout.write(s.encode(enc, errors="ignore").decode(enc, errors="ignore") + end)
        except Exception:
            # Last resort
            sys.stdout.write((s.encode(errors="ignore").decode(errors="ignore")) + end)

builtins.print = _print_ascii_safe

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
    ap.add_argument("--ball_bias", type=float, default=1.00, help="Ball probability multiplier (1.00 = realistic walk rates)")
    ap.add_argument("--plate_mixtures_json", default="", help="JSON file with mixture params for plate location")
    ap.add_argument("--plate_zone_targets_json", default="", help="Optional JSON mapping of context -> target in-zone rates to calibrate plate mixtures")
    ap.add_argument("--plate_ar1_rho", type=float, default=0.20, help="AR(1) persistence for plate location")
    ap.add_argument("--plate_noise_x", type=float, default=0.15, help="x noise (ft)")
    ap.add_argument("--plate_noise_y", type=float, default=0.18, help="y noise (ft)")
    ap.add_argument("--output_formats", nargs="+", default=["yakkertech"], 
                    choices=["yakkertech", "trackman"], 
                    help="Output formats (yakkertech, trackman, or both)")
    ap.add_argument("--trackman_template_csv", default="", help="Optional TrackMan template CSV for column order")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging (conversion/sim)")
    ap.add_argument("--checkpoint_every", type=int, default=500,
                    help="Write incremental rollups every N games (0=disable)")
    args = ap.parse_args()

    rng = random.Random(args.seed); np.random.seed(args.seed)

    # plate location sampler
    np_rng = np.random.default_rng(args.seed or 7)
    plate_bundle: PlateLocBundle | None = None
    if args.plate_mixtures_json and Path(args.plate_mixtures_json).is_file():
        plate_bundle = load_location_bundle(args.plate_mixtures_json)
        mixtures = plate_bundle.mixtures
    else:
        mixtures = default_mixtures_demo()
        plate_bundle = PlateLocBundle(
            mixtures=mixtures,
            region_usage={},
            pitch_call_by_region={},
            tilt_distribution={},
            mixture_info={},
            meta={"source": "default_demo"}
        )
    # Keep mixtures/params to build a per-game sampler for determinism
    base_mixtures = mixtures
    plate_params = dict(rho=args.plate_ar1_rho, noise_x=args.plate_noise_x, noise_y=args.plate_noise_y)

    # Optional calibration of mixtures to target in-zone fractions
    try:
        if args.plate_zone_targets_json:
            p = Path(args.plate_zone_targets_json)
            if p.is_file():
                with p.open("r", encoding="utf-8") as f:
                    targets = json.load(f)
                from plate_loc_model import calibrate_mixtures
                base_mixtures = calibrate_mixtures(base_mixtures, targets)
                if plate_bundle is not None:
                    plate_bundle = replace(plate_bundle, mixtures=base_mixtures)
    except Exception as _e:
        # Non-fatal: proceed with uncalibrated mixtures
        pass

    # adjust global injury chance
    sim_utils.INJURY_CHANCE_HEAVY_OVER = float(args.injury_chance_heavy_over)

    # priors & template
    with open(args.priors, "r", encoding="utf-8") as f:
        priors = yaml.safe_load(f) or {}
        if args.template_csv and Path(args.template_csv).is_file():
             template_cols = list(pd.read_csv(args.template_csv, nrows=1, low_memory=False).columns)
        else:template_cols = []

    
    # Do not force-add generic 'Throws'; we already output 'PitcherThrows'
    for must in ["PAofInning", "PitchofPA", "Top/Bottom","BatterSide","PlayResult","KorBB","RunsScored","OutsOnPlay","HitType"]:
        if must not in template_cols:
            template_cols.append(must)

    # Initialize CSV format manager (honor verbose toggle)
    csv_manager = CSVFormatManager(verbose=bool(args.verbose))
    output_formats = args.output_formats
    
    # Write outputs directly to the provided out_dir (no extra Season01 nesting)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    byteam_dir = out_dir / "ByTeam"
    byteam_dir.mkdir(parents=True, exist_ok=True)
    team_results: List[Dict[str,Any]] = []
    injuries_log: List[Dict[str,Any]] = []
    standings = defaultdict(lambda: {"W":0,"L":0,"T":0,"RS":0,"RA":0})
    season_pitchers = defaultdict(lambda: {"Name":"", "Team":"", "Role":"",
                                           "G":0,"GS":0,"IPOuts":0,"BF":0,"Pitches":0,
                                           "H":0,"B2":0,"B3":0,"HR":0,"BB":0,"K":0,"R":0,"ER":0})
    season_team_off = defaultdict(lambda: {"PA":0,"AB":0,"R":0,"H":0,"B2":0,"B3":0,"HR":0,"RBI":0,"BB":0,"K":0,"HBP":0,"SF":0})

    # Helpers for rollup snapshots

    def _outs_to_ip_str(outs: int) -> str:
        return f"{outs//3}.{outs%3}"

    def _ip_to_decimal_from_outs(outs: int) -> float:
        return (outs // 3) + (outs % 3) / 3.0

    def _safe_div(num, den):
        return (float(num) / float(den)) if den and float(den) != 0.0 else float("nan")

    def _write_rollups_snapshot():
        try:
            # team_game_results
            pd.DataFrame(team_results).to_csv(out_dir / "team_game_results.csv", index=False)

            # standings with Pythagorean
            std_rows = []
            for team, s in standings.items():
                g = s["W"] + s["L"] + s["T"]
                winpct = round((s["W"] + 0.5 * s["T"]) / g, 3) if g else 0.0
                try:
                    exp = 1.83
                    rs_exp = (s["RS"] ** exp)
                    ra_exp = (s["RA"] ** exp)
                    py_pct = (rs_exp / (rs_exp + ra_exp)) if (rs_exp + ra_exp) > 0 else 0.0
                    py_wins = round(py_pct * g, 1)
                    py_diff = round((s["W"] - py_wins), 1)
                except Exception:
                    py_pct, py_wins, py_diff = 0.0, 0.0, 0.0
                std_rows.append({
                    "Team": team, "W": s["W"], "L": s["L"], "T": s["T"],
                    "GP": g, "RS": s["RS"], "RA": s["RA"], "RD": s["RS"] - s["RA"],
                    "WinPct": winpct,
                    "PyWinPct": round(py_pct, 3), "PyWins": py_wins, "PyDiff": py_diff
                })
            (pd.DataFrame(std_rows)
             .sort_values(["WinPct","RD"], ascending=[False, False], na_position="last")
             .to_csv(out_dir / "team_standings.csv", index=False))

            # player_pitching snapshot with FIP and KBB%
            pitchers_rows = []
            lg_K = 0; lg_BB = 0; lg_HR = 0; lg_HBP = 0; lg_IP_outs = 0
            for pid, s in season_pitchers.items():
                def safe_int(val):
                    if isinstance(val, str):
                        try: return int(val)
                        except ValueError: return 0
                    return int(val) if val is not None else 0
                outs_val = safe_int(s["IPOuts"])
                ip_dec = _ip_to_decimal_from_outs(outs_val)
                k_val = safe_int(s["K"]); bb_val = safe_int(s["BB"]); h_val = safe_int(s["H"]); er_val = safe_int(s["ER"])
                k9 = round(_safe_div(9.0 * k_val, ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
                bb9 = round(_safe_div(9.0 * bb_val, ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
                whip = round(_safe_div(bb_val + h_val, ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
                era  = round(_safe_div(9.0 * er_val, ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
                ip_str_val = s["IPOuts"]
                if isinstance(ip_str_val, str):
                    try: ip_str_val = int(ip_str_val)
                    except ValueError: ip_str_val = 0
                lg_K += int(k_val); lg_BB += int(bb_val); lg_HR += int(safe_int(s["HR"])); lg_HBP += 0; lg_IP_outs += int(ip_str_val)
                pitchers_rows.append({
                  "PitcherId": pid,
                  "Name": s["Name"], "Team": s["Team"], "Role": s["Role"],
                  "G": s["G"], "GS": s["GS"], "IP": _outs_to_ip_str(ip_str_val),
                  "BF": s["BF"], "Pitches": s["Pitches"], "H": s["H"],
                  "2B": s["B2"], "3B": s["B3"], "HR": s["HR"], "BB": s["BB"], "K": s["K"],
                  "R": s["R"], "ER": s["ER"],
                  "K9": k9, "BB9": bb9, "WHIP": whip, "ERA": era
                })
            try:
                lg_IP = (lg_IP_outs // 3) + (lg_IP_outs % 3) / 3.0
                lg_ERA = 0.0
                if lg_IP > 0:
                    lg_ER = sum(int(v.get("ER",0) or 0) for v in season_pitchers.values())
                    lg_ERA = 9.0 * lg_ER / lg_IP
                const = lg_ERA - ((13.0 * lg_HR + 3.0 * (lg_BB + lg_HBP) - 2.0 * lg_K) / max(1e-6, lg_IP)) if lg_IP > 0 else 0.0
            except Exception:
                const = 0.0
            out_rows = []
            for r in pitchers_rows:
                try:
                    bf = float(r.get("BF", 0) or 0)
                    k = float(r.get("K", 0) or 0); bb = float(r.get("BB", 0) or 0); hr = float(r.get("HR", 0) or 0)
                    ip_txt = r.get("IP", "0.0")
                    if isinstance(ip_txt, str) and "." in ip_txt:
                        parts = ip_txt.split("."); ip_outs = int(parts[0]) * 3 + int(parts[1])
                    elif isinstance(ip_txt, str):
                        ip_outs = int(float(ip_txt) * 3.0)
                    else:
                        ip_outs = int(ip_txt) * 3
                    ip = (ip_outs // 3) + (ip_outs % 3) / 3.0
                    kbb_pct = round(((k - bb) / bf) * 100.0, 1) if bf > 0 else ""
                    fip = round(((13.0 * hr + 3.0 * (bb) - 2.0 * k) / ip) + const, 2) if ip > 0 else ""
                except Exception:
                    kbb_pct = ""; fip = ""
                r.update({"KBB%": kbb_pct, "FIP": fip}); out_rows.append(r)
            (pd.DataFrame(out_rows).sort_values(["Team","Name"]).to_csv(out_dir / "player_pitching.csv", index=False))

            # team_batting snapshot
            tbat_rows = []
            for t, s in season_team_off.items():
                HBP = int(s.get("HBP", 0) or 0); SF  = int(s.get("SF", 0) or 0)
                singles = max(0, int(s["H"]) - int(s["B2"]) - int(s["B3"]) - int(s["HR"]))
                TB = singles + 2 * int(s["B2"]) + 3 * int(s["B3"]) + 4 * int(s["HR"])
                AB = int(s["AB"]); H = int(s["H"]); BB = int(s["BB"])
                AVG = _safe_div(H, AB)
                obp_den_full = AB + BB + HBP + SF
                OBP = _safe_div(H + BB + HBP, obp_den_full) if obp_den_full > 0 else _safe_div(H + BB, AB + BB)
                SLG = _safe_div(TB, AB)
                OPS = (OBP + SLG) if (not math.isnan(OBP) and not math.isnan(SLG)) else float("nan")
                ISO = (round(SLG - AVG, 3) if (not math.isnan(SLG) and not math.isnan(AVG)) else "")
                try:
                    babip_den = AB - int(s["K"]) - int(s["HR"]) + SF
                    babip_num = int(s["H"]) - int(s["HR"])
                    BABIP = round(_safe_div(babip_num, babip_den), 3)
                except Exception:
                    BABIP = ""
                tbat_rows.append({
                 "Team": t,
                 "PA": s["PA"], "AB": s["AB"], "R": s["R"], "H": s["H"],
                 "1B": singles, "2B": s["B2"], "3B": s["B3"], "HR": s["HR"],
                 "RBI": s["RBI"], "BB": s["BB"], "K": s["K"],
                 "HBP": HBP, "SF": SF,
                 "TB": TB,
                 "AVG": round(AVG, 3) if not math.isnan(AVG) else "",
                 "OBP": round(OBP, 3) if not math.isnan(OBP) else "",
                 "SLG": round(SLG, 3) if not math.isnan(SLG) else "",
                 "OPS": round(OPS, 3) if not math.isnan(OPS) else "",
                 "ISO": ISO, "BABIP": BABIP,
                })
            (pd.DataFrame(tbat_rows).sort_values(["Team"]).to_csv(out_dir / "team_batting.csv", index=False))

            # team_pitching snapshot
            tpit = defaultdict(lambda: {"G":0,"GS":0,"IPOuts":0,"BF":0,"Pitches":0,"H":0,"B2":0,"B3":0,"HR":0,"BB":0,"K":0,"R":0,"ER":0})
            for s in season_pitchers.values():
               tp = tpit[s["Team"]]
               for k in tp.keys(): tp[k] += int(s.get(k, 0) or 0)
            tpit_rows = []
            for team, s in tpit.items():
              ip_dec = _ip_to_decimal_from_outs(s["IPOuts"])
              k9 = round(_safe_div(9.0 * s["K"], ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
              bb9= round(_safe_div(9.0 * s["BB"], ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
              whip= round(_safe_div(s["BB"] + s["H"], ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
              era = round(_safe_div(9.0 * s["ER"], ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
              try: kbb_pct = round(((float(s["K"]) - float(s["BB"])) / float(s["BF"])) * 100.0, 1) if float(s["BF"]) > 0 else ""
              except Exception: kbb_pct = ""
              try: fip_team = round(((13.0 * float(s["HR"]) + 3.0 * float(s["BB"]) - 2.0 * float(s["K"])) / ip_dec) + 0.0, 2) if ip_dec > 0 else ""
              except Exception: fip_team = ""
              tpit_rows.append({
                "Team": team,
                "G": s["G"], "GS": s["GS"], "IP": _outs_to_ip_str(s["IPOuts"]),
                "BF": s["BF"], "Pitches": s["Pitches"], "H": s["H"], "2B": s["B2"], "3B": s["B3"],
                "HR": s["HR"], "BB": s["BB"], "K": s["K"], "R": s["R"], "ER": s["ER"],
                "K9": k9, "BB9": bb9, "WHIP": whip, "ERA": era, "KBB%": kbb_pct, "FIP": fip_team
              })
            (pd.DataFrame(tpit_rows).sort_values(["Team"]).to_csv(out_dir / "team_pitching.csv", index=False))

            if injuries_log:
                pd.DataFrame(injuries_log).to_csv(out_dir / "injuries.csv", index=False)
        except Exception:
            pass

    games_done = 0

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
        gdt = row["__DateObj__"]
        # Ensure gdt is a datetime object, not DatetimeIndex
        if hasattr(gdt, 'iloc'):
            gdt = gdt.iloc[0] if len(gdt) > 0 else None
        elif hasattr(gdt, 'values'):
            gdt = gdt.values[0] if len(gdt.values) > 0 else None
        if not isinstance(gdt, datetime):
            try:
                gdt = pd.to_datetime(gdt)
            except Exception:
                continue
        # Additional check to ensure gdt is a proper datetime object
        if pd.isna(gdt):
            continue
        home_key = team_key_from_row("Home", row); away_key = team_key_from_row("Away", row)
        if not home_key or not away_key: continue
        if home_key not in team_staffs or away_key not in team_staffs: continue

        home_staff = team_staffs[home_key]; away_staff = team_staffs[away_key]
        home_sp = choose_sp_for_date(home_staff, gdt) if isinstance(gdt, datetime) else None
        away_sp = choose_sp_for_date(away_staff, gdt) if isinstance(gdt, datetime) else None
        if not home_sp or not away_sp:
            if isinstance(gdt, datetime):
                print(f"⚠️  Skipping {game_id} {gdt.date()} — no available SP")
            else:
                print(f"⚠️  Skipping {game_id} — no available SP")
            continue

        home_pen = [p for p in home_staff["pen"] if isinstance(gdt, datetime) and is_available(p, gdt)][:4]
        away_pen = [p for p in away_staff["pen"] if isinstance(gdt, datetime) and is_available(p, gdt)][:4]
        knobs = {
            "pull_runs_threshold": int(args.pull_runs_threshold),
            "pull_high_stress_inning_pitches": int(args.pull_high_stress_inning_pitches),
            "ball_bias": float(args.ball_bias),  # Ensure realistic BallCalled probabilities for walk generation
        }

        # Derive per-game deterministic seeds from base seed + game identifiers
        seed_str = f"{int(args.seed)}|{str(gdt.date() if isinstance(gdt, datetime) else gdt)}|{game_id}|{home_key}|{away_key}"
        seed_bytes = hashlib.md5(seed_str.encode("utf-8")).digest()
        per_game_seed = int.from_bytes(seed_bytes[:4], byteorder="big", signed=False)
        rng_game = random.Random(per_game_seed)
        try:
            import numpy as _np
            _np.random.seed(per_game_seed % (2**32 - 1))
            np_rng_game = _np.random.default_rng(per_game_seed % (2**32 - 1))
        except Exception:
            np_rng_game = None
        # Build a per-game plate sampler for repeatable single-game reruns
        pg_sampler = PlateLocSampler(
            rng=(np_rng_game if np_rng_game is not None else np.random.default_rng(per_game_seed % (2**32 - 1))),
            mixtures=base_mixtures,
            **plate_params
        )

        df, score, usage, pbox_game, team_off_game = simulate_one_game(
            rng_game, priors, template_cols, gdt if isinstance(gdt, datetime) else datetime.now(), home_key, away_key, home_sp, away_sp, home_pen, away_pen,
            roster_by_team, knobs, pg_sampler, plate_bundle=plate_bundle
        )

        # Stamp core metadata and ensure critical columns exist for downstream formatters
        try:
            df["Date"] = (gdt.strftime("%Y-%m-%d") if isinstance(gdt, datetime) else str(gdt))
            df["HomeTeam"] = str(home_key)
            df["AwayTeam"] = str(away_key)
            df["GameID"] = str(game_id)
        except Exception:
            pass

        # Best-effort: populate per-pitch Pitcher name and teams if missing
        try:
            home_name = home_sp.get("Pitcher") or home_sp.get("Name") or ""
            away_name = away_sp.get("Pitcher") or away_sp.get("Name") or ""

            if "Top/Bottom" in df.columns:
                # Pitcher name by half-inning (home pitches on Top, away on Bottom)
                if "Pitcher" not in df.columns or df["Pitcher"].isna().all() or (df["Pitcher"] == "").all():
                    df["Pitcher"] = df["Top/Bottom"].apply(lambda tb: home_name if str(tb).strip().lower()=="top" else away_name)

                # Teams for each half-inning
                df["PitcherTeam"] = df["Top/Bottom"].apply(lambda tb: str(home_key) if str(tb).strip().lower()=="top" else str(away_key))
                df["BatterTeam"]  = df["Top/Bottom"].apply(lambda tb: str(away_key) if str(tb).strip().lower()=="top" else str(home_key))
        except Exception:
            pass

        for col in [
            "RelSpeed","SpinRate","SpinAxis",
            "RelHeight","RelSide","Extension",
            "PlateLocHeight","PlateLocSide",
            "VertBreak","HorzBreak",
            "VertRelAngle","HorzRelAngle",
            "VertApprAngle","HorzApprAngle",
        ]:
            if col not in df.columns:
                df[col] = ""

        _P_KEYS = ("G","GS","IPOuts","BF","Pitches","H","B2","B3","HR","BB","K","R","ER")
        _T_KEYS = ("PA","AB","R","H","B2","B3","HR","RBI","BB","K","HBP","SF") 

        for pid, g in pbox_game.items():
          for k in _P_KEYS:
           g[k] = int(g.get(k, 0) or 0)

        for tkey, g in team_off_game.items():
          for k in _T_KEYS:
           g[k] = int(g.get(k, 0) or 0)

        # save per-game CSV in specified format(s)
        fname = f"{gdt.strftime('%Y%m%d')}-{away_key}@{home_key}-{game_id}.csv"
        game_path = out_dir / fname
        
        # Save in all requested formats
        saved_paths = {}
        # Sanitize Notes: drop SB2/SB3 tokens and tidy separators
        try:
            if "Notes" in df.columns:
                def _clean_notes(x):
                    try:
                        s = str(x)
                    except Exception:
                        return ""
                    parts = [t.strip() for t in s.split(";")]
                    parts = [t for t in parts if t and t not in ("SB2", "SB3")]
                    return ";".join(parts)
                df["Notes"] = df["Notes"].apply(_clean_notes)
        except Exception:
            pass
        if "yakkertech" in output_formats:
            df.to_csv(game_path, index=False)
            saved_paths["yakkertech"] = game_path
        
        if "trackman" in output_formats:
            trackman_fname = f"{gdt.strftime('%Y%m%d')}-{away_key}@{home_key}-{game_id}_trackman.csv"
            trackman_path = out_dir / trackman_fname
            csv_manager.save_game_in_format(
                df, trackman_path, "trackman",
                yakkertech_template_path="",
                trackman_template_path=(args.trackman_template_csv or None)
            )
            saved_paths["trackman"] = trackman_path
        
        # Copy to team directories
        for team_key in (home_key, away_key):
            td = byteam_dir / str(team_key)
            td.mkdir(parents=True, exist_ok=True)
            
            # Copy all format files
            for format_name, path in saved_paths.items():
                if path.exists():
                    (td / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

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

        # Incremental checkpoint every N games
        games_done += 1
        if int(args.checkpoint_every or 0) > 0 and (games_done % int(args.checkpoint_every) == 0):
            _write_rollups_snapshot()

        # season rollups
        for pid, g in pbox_game.items():
            s = season_pitchers[pid]
            if not s["Name"]: s["Name"] = g["Name"]
            if not s["Team"]: s["Team"] = g["Team"]
            if not s["Role"]: s["Role"] = g["Role"]
            for k in ("G","GS","IPOuts","BF","Pitches","H","B2","B3","HR","BB","K","R","ER"):
                s[k] += int(g.get(k, 0) or 0)
        for tkey, g in team_off_game.items():
            s = season_team_off[tkey]
            for k in ("PA","AB","R","H","B2","B3","HR","RBI","BB","K"):
                s[k] += int(g.get(k, 0) or 0)

        # recovery & injuries
        if isinstance(gdt, datetime):
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
                    p["_NextOK"] = p["_NextOK"] + timedelta(days=EXTRA_INNING_RECOVERY_BONUS_DAYS)
            _add_day_if_extras(home_sp); _add_day_if_extras(away_sp)
            for p in home_pen: _add_day_if_extras(p)
            for p in away_pen: _add_day_if_extras(p)

        from random import random as _r
        def inj(p, ct, role: str):
            was_inj = bool(p.get("_Injured", False))
            if ct > p["_Limit"] + 25 and _r() < sim_utils.INJURY_CHANCE_HEAVY_OVER:
                maybe_injure(p, rng, ct)
            if (not was_inj) and bool(p.get("_Injured", False)):
                ret = p.get("_NextOK")
                days = None; ret_iso = ""
                try:
                    if isinstance(ret, datetime) and isinstance(gdt, datetime):
                        days = max(0, (ret.date() - gdt.date()).days)
                        ret_iso = ret.strftime("%Y-%m-%d")
                except Exception:
                    pass
                injuries_log.append({
                    "Date": (gdt.strftime("%Y-%m-%d") if isinstance(gdt, datetime) else str(gdt)),
                    "GameID": str(game_id),
                    "Team": str(p.get("TeamKey", "")),
                    "Opponent": str(away_key if str(p.get("TeamKey","")) == str(home_key) else home_key),
                    "PitcherId": str(p.get("PitcherId", "")),
                    "Name": str(p.get("Pitcher", "")),
                    "Role": str(role),
                    "PitchesThisGame": int(ct or 0),
                    "Limit": int(p.get("_Limit", 0) or 0),
                    "OverBy": int(max(0, int(ct or 0) - int(p.get("_Limit", 0) or 0))),
                    "InjuryChance": float(sim_utils.INJURY_CHANCE_HEAVY_OVER),
                    "InjuryDays": (int(days) if days is not None else ""),
                    "ReturnDate": ret_iso,
                })
        inj(home_sp, usage["home_sp_pitches"], "SP"); inj(away_sp, usage["away_sp_pitches"], "SP")
        for penp in home_pen: inj(penp, penp.get("_PitchCount",0), "RP")
        for penp in away_pen: inj(penp, penp.get("_PitchCount",0), "RP")

        if isinstance(gdt, datetime):
            print(f"✅ {gdt.date()} {away_key}@{home_key}  Final: {score['away']}-{score['home']}  -> {fname}")
        else:
            print(f"✅ {away_key}@{home_key}  Final: {score['away']}-{score['home']}  -> {fname}")

    def _outs_to_ip_str(outs: int) -> str:
    # CSV display in 10ths: .1 = 1 out, .2 = 2 outs
        return f"{outs//3}.{outs%3}"

    def _ip_to_decimal_from_outs(outs: int) -> float:
     # Real math in 1/3 increments for ERA/WHIP/K9/BB9
       return (outs // 3) + (outs % 3) / 3.0

    def _safe_div(num, den):
        return (float(num) / float(den)) if den and float(den) != 0.0 else float("nan")

    # 1) team_game_results.csv
    pd.DataFrame(team_results).to_csv(out_dir / "team_game_results.csv", index=False)

    # 2) team_standings.csv (RS/RA/W/L/T already tallied during loop)
    std_rows = []
    for team, s in standings.items():
      g = s["W"] + s["L"] + s["T"]
      winpct = round((s["W"] + 0.5 * s["T"]) / g, 3) if g else 0.0
      # Pythagorean expectation
      try:
          exp = 1.83
          rs_exp = (s["RS"] ** exp)
          ra_exp = (s["RA"] ** exp)
          py_pct = (rs_exp / (rs_exp + ra_exp)) if (rs_exp + ra_exp) > 0 else 0.0
          py_wins = round(py_pct * g, 1)
          py_diff = round((s["W"] - py_wins), 1)
      except Exception:
          py_pct, py_wins, py_diff = 0.0, 0.0, 0.0
      std_rows.append({
        "Team": team, "W": s["W"], "L": s["L"], "T": s["T"],
        "GP": g, "RS": s["RS"], "RA": s["RA"], "RD": s["RS"] - s["RA"],
        "WinPct": winpct,
        "PyWinPct": round(py_pct, 3),
        "PyWins": py_wins,
        "PyDiff": py_diff
    })
    (pd.DataFrame(std_rows)
     .sort_values(["WinPct","RD"], ascending=[False, False], na_position="last")
    .to_csv(out_dir / "team_standings.csv", index=False))

    # 3) player_pitching.csv (add K9/BB9/WHIP/ERA)
    pitchers_rows = []
    # League totals for FIP constant computation
    lg_K = 0; lg_BB = 0; lg_HR = 0; lg_HBP = 0; lg_IP_outs = 0
    for pid, s in season_pitchers.items():
        # Safe conversion of values to numbers
        def safe_int(val):
            if isinstance(val, str):
                try:
                    return int(val)
                except ValueError:
                    return 0
            return int(val) if val is not None else 0
        
        outs_val = safe_int(s["IPOuts"])
        ip_dec = _ip_to_decimal_from_outs(outs_val)
        k_val = safe_int(s["K"])
        bb_val = safe_int(s["BB"])
        h_val = safe_int(s["H"])
        er_val = safe_int(s["ER"])
        
        k9   = round(_safe_div(9.0 * k_val, ip_dec), 2)    if not math.isnan(ip_dec) else float("nan")
        bb9  = round(_safe_div(9.0 * bb_val, ip_dec), 2)   if not math.isnan(ip_dec) else float("nan")
        
        whip = round(_safe_div(bb_val + h_val, ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
        era  = round(_safe_div(9.0 * er_val, ip_dec), 2)   if not math.isnan(ip_dec) else float("nan")

        ip_str_val = s["IPOuts"]
        if isinstance(ip_str_val, str):
            try:
                ip_str_val = int(ip_str_val)
            except ValueError:
                ip_str_val = 0
        
        lg_K += int(safe_int(s["K"]))
        lg_BB += int(safe_int(s["BB"]))
        lg_HR += int(safe_int(s["HR"]))
        lg_HBP += 0  # not tracked; assume 0
        lg_IP_outs += int(ip_str_val)

        pitchers_rows.append({
          "PitcherId": pid,
          "Name": s["Name"], "Team": s["Team"], "Role": s["Role"],
          "G": s["G"], "GS": s["GS"], "IP": _outs_to_ip_str(ip_str_val),
          "BF": s["BF"], "Pitches": s["Pitches"], "H": s["H"],
          "2B": s["B2"], "3B": s["B3"], "HR": s["HR"], "BB": s["BB"], "K": s["K"],
          "R": s["R"], "ER": s["ER"],
          "K9": k9, "BB9": bb9, "WHIP": whip, "ERA": era
        })

    # FIP constant so league FIP ~= league ERA
    try:
        lg_IP = (lg_IP_outs // 3) + (lg_IP_outs % 3) / 3.0
        lg_ERA = 0.0
        if lg_IP > 0:
            # compute league ERA from season_pitchers ER totals
            lg_ER = sum(int(v.get("ER",0) or 0) for v in season_pitchers.values())
            lg_ERA = 9.0 * lg_ER / lg_IP
        const = lg_ERA - ((13.0 * lg_HR + 3.0 * (lg_BB + lg_HBP) - 2.0 * lg_K) / max(1e-6, lg_IP)) if lg_IP > 0 else 0.0
    except Exception:
        const = 0.0

    # Now attach K-BB% and FIP per pitcher
    out_rows = []
    for r in pitchers_rows:
        try:
            bf = float(r.get("BF", 0) or 0)
            k = float(r.get("K", 0) or 0)
            bb = float(r.get("BB", 0) or 0)
            hr = float(r.get("HR", 0) or 0)
            ip_txt = r.get("IP", "0.0")
            ip_outs = 0
            if isinstance(ip_txt, str) and "." in ip_txt:
                parts = ip_txt.split(".")
                ip_outs = int(parts[0]) * 3 + int(parts[1])
            elif isinstance(ip_txt, str):
                ip_outs = int(float(ip_txt) * 3.0)
            else:
                ip_outs = int(ip_txt) * 3
            ip = (ip_outs // 3) + (ip_outs % 3) / 3.0
            kbb_pct = round(((k - bb) / bf) * 100.0, 1) if bf > 0 else ""
            fip = round(((13.0 * hr + 3.0 * (bb) - 2.0 * k) / ip) + const, 2) if ip > 0 else ""
        except Exception:
            kbb_pct = ""; fip = ""
        r.update({"KBB%": kbb_pct, "FIP": fip})
        out_rows.append(r)

    (pd.DataFrame(out_rows)
     .sort_values(["Team","Name"])
     .to_csv(out_dir / "player_pitching.csv", index=False))

    # 4) team_batting.csv (compute 1B/TB/AVG/OBP/SLG/OPS; use HBP/SF if present)
    tbat_rows = []
    for t, s in season_team_off.items():
        HBP = int(s.get("HBP", 0) or 0)
        SF  = int(s.get("SF", 0) or 0)

        singles = int(s["H"]) - int(s["B2"]) - int(s["B3"]) - int(s["HR"])
        singles = max(0, singles)

        TB = singles + 2 * int(s["B2"]) + 3 * int(s["B3"]) + 4 * int(s["HR"])

        AB = int(s["AB"]); H = int(s["H"]); BB = int(s["BB"])

        AVG = _safe_div(H, AB)
        obp_den_full = AB + BB + HBP + SF
        OBP = _safe_div(H + BB + HBP, obp_den_full) if obp_den_full > 0 else _safe_div(H + BB, AB + BB)
        SLG = _safe_div(TB, AB)
        OPS = (OBP + SLG) if (not math.isnan(OBP) and not math.isnan(SLG)) else float("nan")

        # ISO and BABIP
        ISO = (round(SLG - AVG, 3) if (not math.isnan(SLG) and not math.isnan(AVG)) else "")
        denom_babip = AB - k_val if False else None
        # compute BABIP with available fields
        try:
            babip_den = AB - int(s["K"]) - int(s["HR"]) + SF
            babip_num = int(s["H"]) - int(s["HR"])
            BABIP = round(_safe_div(babip_num, babip_den), 3)
        except Exception:
            BABIP = ""

        tbat_rows.append({
         "Team": t,
         "PA": s["PA"], "AB": s["AB"], "R": s["R"], "H": s["H"],
         "1B": singles, "2B": s["B2"], "3B": s["B3"], "HR": s["HR"],
         "RBI": s["RBI"], "BB": s["BB"], "K": s["K"],
         "HBP": HBP, "SF": SF,
         "TB": TB,
         "AVG": round(AVG, 3) if not math.isnan(AVG) else "",
         "OBP": round(OBP, 3) if not math.isnan(OBP) else "",
         "SLG": round(SLG, 3) if not math.isnan(SLG) else "",
         "OPS": round(OPS, 3) if not math.isnan(OPS) else "",
         "ISO": ISO,
         "BABIP": BABIP,
        })

    (pd.DataFrame(tbat_rows)
     .sort_values(["Team"])
     .to_csv(out_dir / "team_batting.csv", index=False))

    # 5) team_pitching.csv (sum pitcher counts + rate stats)
    tpit = defaultdict(lambda: {"G":0,"GS":0,"IPOuts":0,"BF":0,"Pitches":0,"H":0,"B2":0,"B3":0,"HR":0,"BB":0,"K":0,"R":0,"ER":0})
    for s in season_pitchers.values():
       tp = tpit[s["Team"]]
       for k in tp.keys():
           tp[k] += int(s.get(k, 0) or 0)

    tpit_rows = []
    # Reuse FIP constant for teams
    for team, s in tpit.items():
      ip_dec = _ip_to_decimal_from_outs(s["IPOuts"])
      k9   = round(_safe_div(9.0 * s["K"], ip_dec), 2)    if not math.isnan(ip_dec) else float("nan")
      bb9  = round(_safe_div(9.0 * s["BB"], ip_dec), 2)   if not math.isnan(ip_dec) else float("nan")
      whip = round(_safe_div(s["BB"] + s["H"], ip_dec), 2) if not math.isnan(ip_dec) else float("nan")
      era  = round(_safe_div(9.0 * s["ER"], ip_dec), 2)   if not math.isnan(ip_dec) else float("nan")
      try:
          kbb_pct = round(((float(s["K"]) - float(s["BB"])) / float(s["BF"])) * 100.0, 1) if float(s["BF"]) > 0 else ""
      except Exception:
          kbb_pct = ""
      try:
          fip_team = round(((13.0 * float(s["HR"]) + 3.0 * float(s["BB"]) - 2.0 * float(s["K"])) / ip_dec) + const, 2) if ip_dec > 0 else ""
      except Exception:
          fip_team = ""

      tpit_rows.append({
        "Team": team,
        "G": s["G"], "GS": s["GS"], "IP": _outs_to_ip_str(s["IPOuts"]),
        "BF": s["BF"], "Pitches": s["Pitches"], "H": s["H"], "2B": s["B2"], "3B": s["B3"],
        "HR": s["HR"], "BB": s["BB"], "K": s["K"], "R": s["R"], "ER": s["ER"],
        "K9": k9, "BB9": bb9, "WHIP": whip, "ERA": era, "KBB%": kbb_pct, "FIP": fip_team
      })

    (pd.DataFrame(tpit_rows)
     .sort_values(["Team"])
     .to_csv(out_dir / "team_pitching.csv", index=False))

    # 6) injuries.csv (if any)
    try:
        if injuries_log:
            pd.DataFrame(injuries_log).to_csv(out_dir / "injuries.csv", index=False)
    except Exception:
        pass

    print(f"\n🎯 Done. Wrote season outputs to {str(out_dir)}")
    print("   - team_game_results.csv")
    print("   - team_standings.csv")
    print("   - player_pitching.csv")
    print("   - team_batting.csv")
    print("   - team_pitching.csv")
    if injuries_log:
        print("   - injuries.csv")
    print("   - ByTeam/<Team>/ copies of each game")
