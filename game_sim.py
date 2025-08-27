#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid, random
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from faker import Faker

from plate_loc_model import PlateLocSampler, count_bucket
from sim_utils import (
    DEFAULT_SP_RECOVERY, DEFAULT_RP_RECOVERY,
    INJURY_CHANCE_HEAVY_OVER, INPLAY_RESULTS, VALID_HITTYPE_BY_RESULT, ANGLE_BINS, EV_BOUNDS,
    HITTYPE_DISTANCE_BOUNDS, PULL_RUNS, PULL_STRESS_PITCHES, FATIGUE_PER_PITCH_OVER,
    FATIGUE_PER_BF_OVER, VELO_LOSS_PER_OVER10, SPIN_LOSS_PER_OVER10,
    TTO_PENALTY, EXTRA_INNING_FATIGUE_SCALE, EXTRA_INNING_CMD_FLAT_PENALTY,
    canon_hand, clamp, sample_categorical, sample_statpack, sample_feature_value,
    mvn_sample, safe_get, infer_hittype_from_angle, repair_batted_ball_fields, _safe_float,
    _grade_to_weight, _extract_present_from_role, _platoon_batter_bonus, _adjust_pitchcall,
    _adjust_inplay_split, lineup_from_roster_rows
)

fake = Faker()

def simulate_one_game(
    rng: random.Random,
    priors: dict,
    template_cols: List[str],
    date_time: datetime,
    home_team_key: str,
    away_team_key: str,
    home_sp: Dict[str,Any],
    away_sp: Dict[str,Any],
    home_pen: List[Dict[str,Any]],
    away_pen: List[Dict[str,Any]],
    roster_by_team: Dict[str, List[Dict[str,str]]],
    knobs: Dict[str,Any],
    plate_sampler: Optional[PlateLocSampler] = None
) -> Tuple[pd.DataFrame, Dict[str,int], Dict[str,Any], Dict[str,Dict[str,int]], Dict[str,Dict[str,int]]]:

    pitches_cfg: Dict[str, Any] = priors.get("pitches", {}) or {}
    pitch_types = list(pitches_cfg.keys())
    out_cols = template_cols[:]
    game_over = False

    home_line = lineup_from_roster_rows(roster_by_team.get(str(home_team_key), []), home_team_key, rng)
    away_line = lineup_from_roster_rows(roster_by_team.get(str(away_team_key), []), away_team_key, rng)

    def _write_pitcher_throws(row, outcols, raw_value, prefer_long=True):
        canon = canon_hand(raw_value)
        long  = "Right" if canon == "R" else "Left"
        wrote_any = False
        long_aliases = {"Throws","PitcherThrows","PitcherThrow","PitcherHand","PitcherHanded","PitcherHandedness","PitcherThr"}
        abbrev_aliases = {"PitcherThrowsAbbrev","PitcherThrAbbrev"}
        for c in list(outcols):
            lc = c.lower().replace("_","").replace(" ","")
            if (c in long_aliases) or (("pitcher" in lc) and ("throw" in lc or "hand" in lc)):
                row[c] = long if prefer_long else canon
                wrote_any = True
            if c in abbrev_aliases:
                row[c] = canon
        if not wrote_any:
            if "PitcherThrows" not in outcols:
                outcols.append("PitcherThrows")
            row["PitcherThrows"] = long

    def attach_prof(p: dict):
        p["_Profile"] = {
            "command_tier": float(p.get("_CmdBase") or 1.0),
            "usage": (p.get("_Usage") or {}),
            "cmd_by_pitch": (p.get("_CmdByPitch") or {}),
            "stamina": float(p.get("_Stamina") or 50.0),
            "pitch_limit": int(p.get("_Limit") or 85),
            "expected_bf": float(p.get("_ExpBF") or 18.0),
            "avg_pitches": float(p.get("_AvgOut") or 80.0),
            "weight_pitch": float(p.get("_PitchingWeight") or 1.0),
            "avg_fb_velo": _safe_float(p.get("_AvgFBVelo"), None),
            "rel_h": _safe_float(p.get("_RelHeight_ft"), None),
            "rel_x": _safe_float(p.get("_RelSide_ft"), None),
            "ext":   _safe_float(p.get("_Extension_ft"), None),
        }
        p["_PitchCount"] = 0; p["_BF"] = 0; p["_R"] = 0; p["_PitchesThisHalf"] = 0

    for obj in [home_sp, *home_pen, away_sp, *away_pen]:
        attach_prof(obj)

    def _blend_geom(model_val, roster_val, w_roster=0.6, jitter_sd=0.05, lo=3.5, hi=7.0):
        mv = None if model_val in ("", None) else float(model_val)
        rv = None if roster_val in ("", None) else float(roster_val)
        if mv is None and rv is None: return None
        if mv is None: base = rv
        elif rv is None: base = mv
        else: base = (1 - w_roster) * mv + w_roster * rv
        base = base + rng.normalvariate(0.0, jitter_sd)
        return round(clamp(base, lo, hi), 3)

    def sp_ip_goal(p):
        sta = p["_Profile"]["stamina"]; ebf = p["_Profile"]["expected_bf"]
        ip_from_bf = ebf / 4.3
        base_from_sta = 4.0 + 3.0 * (sta / 100.0)
        mu = 0.5*ip_from_bf + 0.5*base_from_sta
        return int(max(3, min(8, round(rng.normalvariate(mu, 0.75)))))

    def should_pull(p):
        if p["_R"] >= knobs["pull_runs_threshold"]: return True
        if p["_PitchesThisHalf"] > knobs["pull_high_stress_inning_pitches"]: return True
        if p["_PitchCount"] > p["_Profile"]["pitch_limit"] + 20: return True
        return False

    used_in_extras_home = set()
    used_in_extras_away = set()

    def get_next_rp(is_top: bool, inning_num: int):
        pen = home_pen if is_top else away_pen
        avail = [x for x in pen if not x.get("_UsedUp", False)]
        if not avail: return None
        p = avail[0]
        p["_UsedUp"] = True
        attach_prof(p)
        if inning_num > 9:
            # if is_top: fielding team is HOME
            # if bottom: fielding team is AWAY
            (used_in_extras_home if is_top else used_in_extras_away).add(p["PitcherId"])
        return p

    def pick_pitch(p):
        usage = {k:v for k,v in p["_Profile"]["usage"].items() if k in pitch_types}
        if usage: return sample_categorical(usage, rng)
        mix = {pt: float(pitches_cfg[pt].get("mix_pct", 0.0) or 0.0) for pt in pitch_types}
        s = sum(mix.values()) or 1.0; mix = {pt: mix[pt]/s for pt in pitch_types}
        return sample_categorical(mix, rng)

    def tto_pen(line_slot_obj: dict) -> float:
        try:
            return TTO_PENALTY if int(line_slot_obj.get("_TimesFaced", 0)) >= 2 else 0.0
        except Exception:
            return 0.0

    def cmd_for_pitch(p, pt, batter_slot_obj, inning_num: int):
        prof = p["_Profile"]; base = float(prof.get("command_tier") or 1.0)
        cmd = float((prof.get("cmd_by_pitch") or {}).get(pt, base))
        cmd *= float(prof.get("weight_pitch") or 1.0)

        limit = int(prof.get("pitch_limit") or 85)
        over_p = max(0, int(p.get("_PitchCount", 0)) - limit)
        ebf = float(prof.get("expected_bf") or 18.0)
        over_bf = max(0.0, float(p.get("_BF", 0)) - ebf)

        extra_inn = max(0, int(inning_num) - 9)
        fatigue_scale = 1.0 + EXTRA_INNING_FATIGUE_SCALE * extra_inn
        if over_p > 0: cmd /= (1.0 + FATIGUE_PER_PITCH_OVER * over_p * fatigue_scale)
        if over_bf > 0: cmd /= (1.0 + FATIGUE_PER_BF_OVER * over_bf * fatigue_scale)
        if int(p.get("_PitchesThisHalf", 0)) >= PULL_STRESS_PITCHES:
            cmd *= (0.9 ** max(1.0, fatigue_scale))
        cmd *= (1.0 - tto_pen(batter_slot_obj))
        if extra_inn > 0: cmd /= (1.0 + EXTRA_INNING_CMD_FLAT_PENALTY * extra_inn)
        return max(0.25, cmd)

    def _pick_hittype_for_result(play_result: str, pt: str, hand_code: str, rng: random.Random) -> Optional[str]:
        ht_dist = safe_get(pitches_cfg, pt, "hands", hand_code, "hit_types", default={}) or {}
        def _sample(dist: Dict[str, float]) -> str:
            d = {k: max(0.0, float(v)) for k,v in dist.items()}
            z = sum(d.values()) or 1.0
            d = {k: v/z for k,v in d.items()}
            return sample_categorical(d, rng)
        if play_result in ("Single","Double","Triple","HomeRun"):
            valid = VALID_HITTYPE_BY_RESULT.get(play_result, [])
            filt = {k:v for k,v in ht_dist.items() if (not valid or k in valid)}
            if filt: return _sample(filt)
            if play_result == "Single":  return _sample({"GroundBall":0.55, "LineDrive":0.30, "FlyBall":0.15})
            if play_result == "Double":  return _sample({"LineDrive":0.65, "FlyBall":0.35})
            if play_result == "Triple":  return "FlyBall"
            if play_result == "HomeRun": return _sample({"FlyBall":0.75, "LineDrive":0.25})
        if play_result == "Sacrifice":      return _sample({"FlyBall":0.75, "LineDrive":0.25})
        if play_result == "FielderChoice":  return "GroundBall"
        if play_result in ("Out","Error"):
            base = ht_dist or {"GroundBall":0.45, "FlyBall":0.35, "LineDrive":0.15, "Popup":0.05}
            return _sample(base)
        return None

    def sample_loc(pt, hand_code, command):
        hb = safe_get(pitches_cfg, pt, "hands", hand_code, default={})
        loc = hb.get("location_model") or {}; mean, cov = loc.get("mean"), loc.get("cov")
        if not (isinstance(mean, (list,tuple)) and isinstance(cov,(list,tuple))): return None
        try: cov_arr = np.array(cov, dtype=float) / max(1e-6, float(command))
        except Exception: cov_arr = np.array(cov, dtype=float)
        x,z = mvn_sample(mean, cov_arr)
        return (float(clamp(x, -0.95, 0.95)), float(clamp(z, 1.0, 4.0)))

    def load_zone_tbl(pt, hand_code): return safe_get(pitches_cfg, pt, "hands", hand_code, "pitch_call_by_zone", default=None)

    def classify_zone(tbl, x, z):
        if not isinstance(tbl, dict): return (None, None)
        grid = tbl.get("grid") or {}; outside = tbl.get("outside") or {}
        xe, ze = grid.get("x_edges"), grid.get("z_edges")
        if isinstance(xe, list) and isinstance(ze, list) and len(xe)==4 and len(ze)==4:
            if (x >= xe[0] and x <= xe[-1] and z >= ze[0] and z <= ze[-1]):
                ci = min(max(np.digitize([x], xe)[0] - 1, 0), 2)
                ri = min(max(np.digitize([z], ze)[0] - 1, 0), 2)
                key = f"r{ri}c{ci}"
                if grid.get("cells", {}).get(key): return ("grid", key)
        xl, xr, zl, zr = -0.708, 0.708, 1.5, 3.5
        dx = 0.0 if (x >= xl and x <= xr) else (xl - x if x < xl else x - xr)
        dz = 0.0 if (z >= zl and z <= zr) else (zl - z if z < zl else z - zr)
        target = "edge" if (dx*dx + dz*dz) ** 0.5 <= 0.15 else "chase"
        if outside.get(target): return ("outside", target)
        return (None, None)

    def sample_pitch_call_from_zone(tbl, where_tuple):
        kind, key = where_tuple
        if kind is None: return None
        dist = (tbl.get("grid") or {}).get("cells", {}).get(key, {}) if kind=="grid" \
               else (tbl.get("outside") or {}).get(key, {})
        if not dist: return None
        return sample_categorical(dist, rng)

    def runner(owner_pid):
     return {"pid": owner_pid}

    def adv_walk(bases, owner_pid):
     scored = []
     if bases[0] and bases[1] and bases[2]:
        scored.append(bases[2])
     return scored, [runner(owner_pid), bases[0], bases[1]]

    def adv_1(bases, owner_pid):
     scored = []
     if bases[2]:
        scored.append(bases[2])
     return scored, [runner(owner_pid), bases[0], bases[1]]
 
    def adv_2(bases, owner_pid):
     scored = []
     if bases[2]:
        scored.append(bases[2])
     if bases[1]:
        scored.append(bases[1])
     return scored, [None, runner(owner_pid), bases[0]]

    def adv_3(bases, owner_pid):
     scored = [b for b in bases if b]
     return scored, [None, None, runner(owner_pid)]

    def adv_hr(bases, owner_pid): 
     scored = [b for b in bases if b] + [runner(owner_pid)]
     return scored, [None, None, None]

    def adv_error(bases, owner_pid):
     scored = []
     if bases[2]:
        scored.append(bases[2])
     return scored, [runner(owner_pid), bases[0], bases[1]]

    def adv_fc(bases, owner_pid):
     b1, b2, b3 = bases
     if b1:
        return [], [runner(owner_pid), None, b3]
     if b2:
        return [], [None, b2, None]
     if b3:
        return [], [None, b2, b3]
     return [], bases

    def adv_sac(bases, owner_pid):
     scored = []
     if bases[2]:
        scored.append(bases[2])
     return scored, [bases[0], runner(owner_pid), bases[1]]


    pbox = defaultdict(lambda: {"Name":"", "Team":"", "Role":"", "G":0, "GS":0, "IPOuts":0, "BF":0, "Pitches":0,
                                "H":0, "B2":0, "B3":0, "HR":0, "BB":0, "K":0, "R":0, "ER":0})
    team_off = defaultdict(lambda: {"PA":0,"AB":0,"R":0,"H":0,"B2":0,"B3":0,"HR":0,"RBI":0,"BB":0,"K":0})

    def ensure_pbox(p):
        pid = p["PitcherId"]; box = pbox[pid]
        if not box["Name"]: box["Name"] = p["Pitcher"]
        if not box["Team"]: box["Team"] = p["TeamKey"]
        if not box["Role"]: box["Role"] = p["Role"]
        return box

    game_id = f"SYN-{uuid.uuid4().hex[:8].upper()}"; stadium = f"{fake.city()} Park"
    current_dt = date_time.replace(microsecond=0); rows = []; score = {"home":0, "away":0}

    _final_innings = 9

    def half_inning(is_top: bool, inn: int, cur_pitcher: Dict[str, Any], lineup_idx: int):
        nonlocal current_dt, rows, score, game_over

        MAX_PITCHES_PER_PA = 25
        MAX_PITCHES_PER_HALF = 400

        outs, balls, strikes = 0, 0, 0
        paofinning, pitchofpa = 1, 1
        bases = [None, None, None]
        runs_this_half = 0

        # mark SP as used in extras (for recovery bonuses) if applicable
        if inn > 9:
            (used_in_extras_home if is_top else used_in_extras_away).add(cur_pitcher["PitcherId"])

        batting_team  = away_team_key if is_top else home_team_key
        fielding_team = home_team_key if is_top else away_team_key
        lineup = away_line if is_top else home_line
        li = int(lineup_idx)

        cur_pitcher["_PitchesThisHalf"] = 0
        ensure_pbox(cur_pitcher)
        pbox[cur_pitcher["PitcherId"]]["G"] += 1
        if (is_top and cur_pitcher is away_sp) or ((not is_top) and cur_pitcher is home_sp):
            pbox[cur_pitcher["PitcherId"]]["GS"] += 1

        while outs < 3:
            if cur_pitcher["_PitchesThisHalf"] >= MAX_PITCHES_PER_HALF: break

            batter = lineup[li]
            phand_code = cur_pitcher["_ThrowsCanon"]
            pt = pick_pitch(cur_pitcher)

            row = {c: "" for c in out_cols}
            def setc(k, v):
                if k in out_cols: row[k] = v

            setc("GameID", game_id); setc("Stadium", stadium)
            setc("Date", current_dt.strftime("%m/%d/%Y")); setc("Time", current_dt.strftime("%I:%M:%S %p"))
            setc("HomeTeam", home_team_key); setc("AwayTeam", away_team_key)
            setc("Top/Bottom", "Top" if is_top else "Bottom"); setc("Inning", inn)
            setc("PAofInning", paofinning); setc("PitchofPA", pitchofpa)
            setc("PitchNo", len(rows) + 1); setc("League", "Synthetic")
            setc("Balls", balls); setc("Strikes", strikes); setc("Outs", outs)
            setc("OutsOnPlay", 0); setc("KorBB", ""); setc("RunsScored", 0)

            setc("Pitcher",  cur_pitcher.get("Pitcher", "")); setc("PitcherId", cur_pitcher.get("PitcherId", ""))
            _write_pitcher_throws(row, out_cols, cur_pitcher.get("Throws", "R"), prefer_long=True)

            setc("PitcherTeam", fielding_team); setc("CatcherTeam", fielding_team)
            setc("Batter", batter["Batter"]); setc("BatterId", batter["BatterId"])
            setc("BatterSide", batter["BatterSide"]); setc("BatterTeam", batting_team)
            setc("TaggedPitchType", pt)

            hb = safe_get(pitches_cfg, pt, "hands", phand_code, default={})
            core = hb.get("core", {}) or {}
            core_map = {"velo": "RelSpeed", "spin": "SpinRate", "ivb": "InducedVertBreak", "hb": "HorzBreak"}
            for friendly, col in core_map.items():
                if col in out_cols:
                    rto = 2 if col in ("RelSpeed", "InducedVertBreak", "HorzBreak") else 0
                    row[col] = sample_statpack(core.get(friendly), rng, round_to=rto)

            features = hb.get("features", {}) or {}
            for col in (
                "RelHeight", "RelSide", "Extension", "VertApprAngle", "HorzApprAngle", "ZoneSpeed", "ZoneTime",
                "SpinAxis", "Tilt", "VertRelAngle", "HorzRelAngle", "VertBreak",
                "pfxx", "pfxz", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"
            ):
                if col in out_cols and col in features:
                    rto = 2 if col in ("VertApprAngle","HorzApprAngle","ZoneSpeed","SpinAxis","VertRelAngle","HorzRelAngle") else 3
                    val = sample_feature_value(features[col], rng, round_to=rto)
                    if val is not None: row[col] = val
            if "ZoneSpeed" in out_cols and row.get("ZoneSpeed","")=="" and row.get("RelSpeed","")!="":
                row["ZoneSpeed"] = row["RelSpeed"]

            if pt.lower().startswith("fast") and "RelSpeed" in out_cols:
                roster_mu = _safe_float(cur_pitcher["_Profile"].get("avg_fb_velo"))
                model_mu  = _safe_float(row.get("RelSpeed"))
                if (roster_mu is not None) or (model_mu is not None):
                    base = (0.6 * (roster_mu if roster_mu is not None else model_mu) \
                            + 0.4 * (model_mu  if model_mu  is not None else roster_mu))
                    row["RelSpeed"] = round(base + rng.normalvariate(0.0, 0.15), 2)

            if "RelHeight" in out_cols:
                row["RelHeight"] = _blend_geom(row.get("RelHeight"), cur_pitcher["_Profile"].get("rel_h"), 0.6, 0.04, 4.0, 7.0)
            if "RelSide" in out_cols:
                row["RelSide"] = _blend_geom(row.get("RelSide"),   cur_pitcher["_Profile"].get("rel_x"), 0.6, 0.03, -3.5, 3.5)
            if "Extension" in out_cols:
                row["Extension"] = _blend_geom(row.get("Extension"),cur_pitcher["_Profile"].get("ext"),  0.6, 0.05, 4.0, 8.5)

            command = cmd_for_pitch(cur_pitcher, pt, batter, inn)
            bq = float(batter.get("_Quality", 1.0))
            bq *= (1.0 + _platoon_batter_bonus(batter.get("BatterSide"), phand_code))
            bq = max(0.85, min(1.20, bq))

            loc = None
            try:
                if plate_sampler is not None:
                    pthrows = "Right" if str(phand_code).upper().startswith("R") else "Left"
                    bats = (row.get("BatterSide") or batter.get("BatterSide") or "Right")
                    ctx = {"pitch_type": pt, "pthrows": pthrows, "bats": bats,
                           "count_bucket": count_bucket(balls, strikes),
                           "pitcher_id": cur_pitcher.get("PitcherId", "")}
                    x, y = plate_sampler.sample(ctx)
                    loc = (float(x), float(y))
            except Exception:
                loc = None
            if loc is None:
                loc = sample_loc(pt, phand_code, command)
            if loc is not None:
                setc("PlateLocSide",   round(loc[0], 3))
                setc("PlateLocHeight", round(loc[1], 3))

            pc = None
            if loc is not None:
                zt = load_zone_tbl(pt, phand_code)
                if zt:
                    pc = sample_pitch_call_from_zone(zt, classify_zone(zt, loc[0], loc[1]))
            if pc is None:
                base_pc = safe_get(pitches_cfg, pt, "hands", phand_code, "outcomes",
                                   default={"BallCalled": 0.42, "StrikeCalled": 0.15, "StrikeSwinging": 0.16, "Foul": 0.15, "InPlay": 0.12})
                pc_dist = _adjust_pitchcall(base_pc, batter_q=bq, pitch_cmd=command)
                pc = sample_categorical(pc_dist, rng)
            setc("PitchCall", pc)

            if (pc in ("Foul","BallCalled")) and (pitchofpa > MAX_PITCHES_PER_PA):
                pc = "InPlay"; setc("PitchCall", pc)
                setc("PlayResult","Out")
                terminal = True; play_result = "Out"; hit_type = "GroundBall"
                outs += 1; setc("OutsOnPlay", 1)
                team_off[batting_team]["AB"] += 1
            else:
                terminal = False; play_result = None; hit_type = None

            runs_scored = 0; rbi_scored = 0; scored_runners = []
            pid = cur_pitcher["PitcherId"]; pstat = ensure_pbox(cur_pitcher)

            if pc in ("StrikeSwinging","StrikeCalled"):
                if strikes < 2:
                    strikes += 1
                else:
                    strikes = 3; outs += 1; terminal = True
                    setc("KorBB","Strikeout")
                    setc("PlayResult","StrikeoutSwinging" if pc=="StrikeSwinging" else "StrikeoutLooking")
                    pstat["K"] += 1; team_off[batting_team]["K"] += 1; team_off[batting_team]["AB"] += 1

            elif pc == "BallCalled":
                if balls < 3:
                    balls += 1
                else:
                    balls = 4; terminal = True
                    setc("KorBB","Walk"); setc("PlayResult","Walk")
                    scored_runners, bases = adv_walk(bases, owner_pid=pid)
                    if scored_runners:
                        runs_scored = len(scored_runners); rbi_scored = 1
                    pstat["BB"] += 1; team_off[batting_team]["BB"] += 1

            elif pc == "Foul":
                if strikes < 2: strikes += 1

            elif pc == "InPlay":
                terminal = True
                split = safe_get(pitches_cfg, pt, "hands", phand_code, "inplay_split",
                                 default={"Out":0.7,"Single":0.2,"Double":0.07,"Triple":0.01,"HomeRun":0.02})
                split = _adjust_inplay_split(split, batter_q=bq)
                play_result = sample_categorical(split, rng)
                setc("PlayResult", play_result)
                hit_type = _pick_hittype_for_result(play_result, pt, phand_code, rng)

                detail = safe_get(pitches_cfg, pt, "hands", phand_code, "outcomes_detail", play_result, default={})
                if isinstance(detail, dict) and detail:
                    for c, sp in detail.items():
                        if c in out_cols:
                            rto = 2 if c in ("ExitSpeed","Angle","Direction","Bearing","HangTime",
                                             "yt_HitVelocityX","yt_HitVelocityY","yt_HitVelocityZ",
                                             "yt_HitBreakX","yt_HitBreakY","yt_HitBreakT") else 3
                            val = sample_statpack(sp, rng, round_to=rto)
                            if val is not None: row[c] = val

                if play_result == "Out":
                    outs += 1; setc("OutsOnPlay", 1); team_off[batting_team]["AB"] += 1
                    if outs <= 2 and bases[0] and hit_type == "GroundBall":
                        dp_chance = 0.36 if outs == 1 else 0.28
                        if rng.random() < dp_chance:
                            outs += 1; setc("OutsOnPlay", 2); bases = [None, bases[1], bases[2]]; terminal = True
                    ang = _safe_float(row.get("Angle"), None)
                    dist = _safe_float(row.get("Distance"), None)
                    deep = (dist and dist >= 280) or (ang and ang >= 28)
                    if outs <= 2 and bases[2] and (hit_type in ("FlyBall","LineDrive")):
                        sac_prob = 0.62 if deep else 0.35
                        if rng.random() < sac_prob:
                            setc("PlayResult","Sacrifice")
                            team_off[batting_team]["AB"] = max(0, team_off[batting_team]["AB"] - 1)
                            scored_runners = [bases[2]]; bases[2] = None
                            runs_scored = 1; rbi_scored = 1
                            team_off[batting_team]["R"] += 1; runs_this_half += 1
                            setc("RunsScored", 1); setc("OutsOnPlay", 1); terminal = True

                elif play_result == "Single":
                    pstat["H"] += 1; team_off[batting_team]["H"] += 1; team_off[batting_team]["AB"] += 1
                    scored_runners, bases = adv_1(bases, owner_pid=pid)

                elif play_result == "Double":
                    pstat["H"] += 1; pstat["B2"] += 1
                    team_off[batting_team]["H"] += 1; team_off[batting_team]["B2"] += 1; team_off[batting_team]["AB"] += 1
                    scored_runners, bases = adv_2(bases, owner_pid=pid)

                elif play_result == "Triple":
                    pstat["H"] += 1; pstat["B3"] += 1
                    team_off[batting_team]["H"] += 1; team_off[batting_team]["B3"] += 1; team_off[batting_team]["AB"] += 1
                    scored_runners, bases = adv_3(bases, owner_pid=pid)

                elif play_result == "HomeRun":
                    pstat["H"] += 1; pstat["HR"] += 1
                    team_off[batting_team]["H"] += 1; team_off[batting_team]["HR"] += 1; team_off[batting_team]["AB"] += 1
                    scored_runners, bases = adv_hr(bases, owner_pid=pid)

                elif play_result == "Error":
                    scored_runners, bases = adv_error(bases, owner_pid=pid)
                    team_off[batting_team]["AB"] += 1

                elif play_result == "FielderChoice":
                    scored_runners, bases = adv_fc(bases, owner_pid=pid)
                    outs += 1; setc("OutsOnPlay", 1); team_off[batting_team]["AB"] += 1

                elif play_result == "Sacrifice":
                    if bases[2]:
                        outs += 1; setc("OutsOnPlay", 1)
                        scored_runners = [bases[2]]; bases[2] = None
                        runs_scored = 1; rbi_scored = 1
                        team_off[batting_team]["R"] += 1; runs_this_half += 1
                        setc("RunsScored", 1); terminal = True
                    else:
                        setc("PlayResult","Out")
                        outs += 1; setc("OutsOnPlay", 1); team_off[batting_team]["AB"] += 1
                        terminal = True

            if play_result in ("Single","Double","Triple","HomeRun"):
                ht_dist = safe_get(pitches_cfg, pt, "hands", phand_code, "hit_types", default={})
                if ht_dist:
                    valid = VALID_HITTYPE_BY_RESULT.get(play_result, [])
                    filt = {k: v for k, v in ht_dist.items() if (not valid or k in valid)}
                    if filt: hit_type = sample_categorical(filt, rng)
            hit_type = repair_batted_ball_fields(row, play_result, hit_type, rng)
            if "HitType" in out_cols and hit_type is not None:
                row["HitType"] = hit_type

            if scored_runners:
                for r in scored_runners:
                    if not r: continue
                    r_owner = r["pid"]
                    pbox[r_owner]["R"]  += 1
                    pbox[r_owner]["ER"] += 1
                    if r_owner == pid: cur_pitcher["_R"] += 1
                runs_scored = len(scored_runners)
                if play_result == "HomeRun": rbi_scored = len(scored_runners)
                elif row.get("KorBB") == "Walk": rbi_scored = 1
                elif play_result == "Sacrifice": rbi_scored = len(scored_runners)
                team_off[batting_team]["R"] += runs_scored
                runs_this_half += runs_scored
            setc("RunsScored", runs_scored)
            team_off[batting_team]["RBI"] += rbi_scored

            over10 = max(0, cur_pitcher["_PitchCount"] - cur_pitcher["_Profile"]["pitch_limit"]) / 10.0
            val_rel = row.get("RelSpeed", None)
            if "RelSpeed" in row and (val_rel is not None) and (str(val_rel) != ""):
                row["RelSpeed"] = round(float(val_rel) - VELO_LOSS_PER_OVER10 * over10, 2)
            val_spin = row.get("SpinRate", None)
            if "SpinRate" in row and (val_spin is not None) and (str(val_spin) != ""):
                row["SpinRate"] = round(max(0.0, float(val_spin) - SPIN_LOSS_PER_OVER10 * over10), 0)

            rows.append(row)
            current_dt += timedelta(seconds=5)

            end_now = False
            if (not is_top) and inn >= 9 and (score["home"] + runs_this_half) > score["away"]:
                end_now = True
                if "WalkOff" in out_cols: row["WalkOff"] = 1

            cur_pitcher["_PitchCount"] += 1
            cur_pitcher["_PitchesThisHalf"] += 1
            pstat["Pitches"] += 1

            if end_now:
                game_over = True
                break

            terminal_by_rule = (
                (row.get("KorBB") == "Walk") or
                (pc == "InPlay") or
                (pc in ("StrikeSwinging","StrikeCalled") and (strikes == 3))
            )
            if terminal_by_rule:
                pstat["BF"] += 1
                cur_pitcher["_BF"] += 1
                team_off[batting_team]["PA"] += 1

            if terminal:
                outs_awarded = 0
                if row.get("KorBB") == "Strikeout":
                    outs_awarded = 1
                elif row.get("PlayResult") in ("Out","Sacrifice","FielderChoice"):
                    try: outs_awarded = int(row.get("OutsOnPlay") or 1)
                    except Exception: outs_awarded = 1
                pstat["IPOuts"] += outs_awarded

                balls, strikes = 0, 0
                paofinning += 1
                pitchofpa = 1
                batter["_TimesFaced"] = int(batter["_TimesFaced"]) + 1

                if should_pull(cur_pitcher) and outs < 3:
                    nxt = get_next_rp(is_top, inn)
                    if nxt is not None:
                        ensure_pbox(nxt)
                        nxt["G_used_here"] = True
                        cur_pitcher = nxt

                li = (li + 1) % 9
            else:
                pitchofpa += 1

        if is_top: score["away"] += runs_this_half
        else:      score["home"] += runs_this_half

        return cur_pitcher, li

    cur_home = home_sp; cur_away = away_sp
    home_li  = 0;       away_li  = 0
    inn = 1
    while True:
        cur_away, away_li = half_inning(True,  inn, cur_away, away_li)
        if inn >= _final_innings and score["home"] > score["away"]: break
        if game_over: break

        cur_home, home_li = half_inning(False, inn, cur_home, home_li)
        if game_over: break

        if inn >= 9 and score["home"] != score["away"]: break
        inn += 1

    usage = {
        "home_sp_pitches": home_sp["_PitchCount"], "home_sp_bf": 0, "home_sp_r": pbox[home_sp["PitcherId"]]["R"],
        "away_sp_pitches": away_sp["_PitchCount"], "away_sp_bf": 0, "away_sp_r": pbox[away_sp["PitcherId"]]["R"],
        "home_pen_pitches": sum(p.get("_PitchCount",0) for p in home_pen),
        "away_pen_pitches": sum(p.get("_PitchCount",0) for p in away_pen),
        "total_innings": inn,
        "extra_innings": max(0, inn - 9),
        "home_pids_in_extras": list(used_in_extras_home),
        "away_pids_in_extras": list(used_in_extras_away),
    }

    df = pd.DataFrame(rows)
    for c in out_cols:
        if c not in df.columns: df[c] = pd.NA
    df = df[out_cols]
    return df, score, usage, dict(pbox), dict(team_off)
