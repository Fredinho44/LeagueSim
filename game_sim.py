#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid, random
from typing import Dict, Any, Optional, List, Tuple, Literal
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from faker import Faker

from plate_loc_model import PlateLocSampler, count_bucket, PlateLocBundle
from sim_utils import (
    DEFAULT_SP_RECOVERY, DEFAULT_RP_RECOVERY,
    INJURY_CHANCE_HEAVY_OVER, INPLAY_RESULTS, VALID_HITTYPE_BY_RESULT, ANGLE_BINS, EV_BOUNDS,
    HITTYPE_DISTANCE_BOUNDS, PULL_RUNS, PULL_STRESS_PITCHES, FATIGUE_PER_PITCH_OVER,
    FATIGUE_PER_BF_OVER, VELO_LOSS_PER_OVER10, SPIN_LOSS_PER_OVER10,
    TTO_PENALTY, EXTRA_INNING_FATIGUE_SCALE, EXTRA_INNING_CMD_FLAT_PENALTY,
    COUNT_MIX_SCALE, TTO_MIX_SCALE,
    SB_ATTEMPT_R1_BASE, SB_ATTEMPT_R2_BASE, SB_SUCCESS_BASE, SB_CATCHER_R_BONUS, ERROR_RATE,
    canon_hand, clamp, sample_categorical, sample_statpack, sample_feature_value,
    mvn_sample, safe_get, infer_hittype_from_angle, repair_batted_ball_fields, _safe_float,
    _grade_to_weight, _extract_present_from_role, _platoon_batter_bonus, _adjust_pitchcall,
    _adjust_inplay_split, lineup_from_roster_rows, lineup_by_positions
)

fake = Faker()

# Strike zone dimensions (feet)
PLATE_HALF_WIDTH = 0.708
ZONE_Z_LOW = 1.55
ZONE_Z_HIGH = 3.45

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
    plate_sampler: Optional[PlateLocSampler] = None,
    plate_bundle: Optional[PlateLocBundle] = None
) -> Tuple[pd.DataFrame, Dict[str,int], Dict[str,Any], Dict[str,Dict[str,Any]], Dict[str,Dict[str,Any]]]:

    pitches_cfg: Dict[str, Any] = priors.get("pitches", {}) or {}
    pitch_types = list(pitches_cfg.keys())
    bundle_calls_map = plate_bundle.pitch_call_by_region if plate_bundle and getattr(plate_bundle, "pitch_call_by_region", None) else {}
    bundle_usage_map = plate_bundle.region_usage if plate_bundle and getattr(plate_bundle, "region_usage", None) else {}
    default_x_edges = [float(v) for v in np.linspace(-PLATE_HALF_WIDTH, PLATE_HALF_WIDTH, 4)]
    default_z_edges = [float(v) for v in np.linspace(ZONE_Z_LOW, ZONE_Z_HIGH, 4)]

    out_cols = template_cols[:]
    game_over = False
    knobs = dict(knobs or {})
    knobs.setdefault("ball_bias", 1.0)

    home_line, home_c = lineup_by_positions(roster_by_team.get(str(home_team_key), []), home_team_key, rng)
    away_line, away_c = lineup_by_positions(roster_by_team.get(str(away_team_key), []), away_team_key, rng)

    def _canon_key_part(val: Any) -> str:
        if val is None:
            return "__"
        s = str(val).strip()
        return s if s else "__"

    def _lookup_bundle(table: Dict[tuple, Any], pitch_type: str, hand_code: str, bats_code: str, count_tag: str):
        if not table:
            return None
        keys = [
            (pitch_type, hand_code, bats_code, count_tag),
            (pitch_type, hand_code, bats_code, "__"),
            (pitch_type, hand_code, "__", "__"),
            (pitch_type, "__", "__", "__"),
            ("__default__", "__", "__", "__"),
        ]
        for key in keys:
            norm_key = tuple(_canon_key_part(part) for part in key)
            if norm_key in table:
                return table[norm_key]
        return None

    def _normalize_call_dist(dist: Any) -> Dict[str, float]:
        if not isinstance(dist, dict):
            return {}
        cleaned: Dict[str, float] = {}
        for k, v in dist.items():
            try:
                cleaned[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        total = sum(max(0.0, val) for val in cleaned.values())
        if total <= 0.0:
            return {}
        return {k: max(0.0, val) / total for k, val in cleaned.items()}

    def _weighted_mix(call_map: Any, weight_map: Any) -> Dict[str, float]:
        if not isinstance(call_map, dict) or not call_map:
            return {}
        accum: Dict[str, float] = {}
        total = 0.0
        if isinstance(weight_map, dict):
            for region, dist in call_map.items():
                try:
                    weight = float(weight_map.get(region, 0.0))
                except (TypeError, ValueError):
                    weight = 0.0
                if weight <= 0.0:
                    continue
                norm = _normalize_call_dist(dist)
                if not norm:
                    continue
                total += weight
                for outcome, prob in norm.items():
                    accum[outcome] = accum.get(outcome, 0.0) + weight * prob
        if total <= 0.0:
            accum.clear()
            count = 0
            for dist in call_map.values():
                norm = _normalize_call_dist(dist)
                if not norm:
                    continue
                count += 1
                for outcome, prob in norm.items():
                    accum[outcome] = accum.get(outcome, 0.0) + prob
            if count == 0:
                return {}
            return {k: v / count for k, v in accum.items()}
        return {k: v / total for k, v in accum.items()}

    def _apply_ball_bias_to_dist(dist: Dict[str, float], factor: float) -> Dict[str, float]:
        if factor == 1.0 or "BallCalled" not in dist:
            return dist
        out = {k: max(0.0, float(v)) for k, v in dist.items()}
        out["BallCalled"] = out.get("BallCalled", 0.0) * factor
        total = sum(max(0.0, v) for v in out.values()) or 1.0
        return {k: max(0.0, v) / total for k, v in out.items()}

    for must in ["Top/Bottom", "BatterSide", "PlayResult", "KorBB", "RunsScored", "OutsOnPlay", "HitType", "PitchCall"]:
        if must not in out_cols:
            out_cols.append(must)

    core_cols = [
        "Date", "Time", "GameID", "HomeTeam", "AwayTeam", "Stadium", "League",
        "Inning", "PAofInning", "PitchofPA", "Outs", "Balls", "Strikes",
        "Pitcher", "PitcherId", "PitcherThrows", "PitcherTeam", "PitcherSet",
        "Batter", "BatterId", "BatterSide", "BatterTeam",
        "RelSpeed", "SpinRate", "SpinAxis", "RelHeight", "RelSide", "Extension",
        "PlateLocHeight", "PlateLocSide", "VertBreak", "InducedVertBreak", "HorzBreak",
        "VertRelAngle", "HorzRelAngle", "VertApprAngle", "HorzApprAngle", "ZoneSpeed", "ZoneTime",
        "ExitSpeed", "Angle", "Direction", "HitSpinRate", "Distance", "LastTrackedDistance", "Bearing", "HangTime",
        "pfxx", "pfxz", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0",
    ]
    for col in core_cols:
        if col not in out_cols:
            out_cols.append(col)

    for c in ("Catcher", "CatcherId", "CatcherTeam"):
        if c not in out_cols:
            out_cols.append(c)

    def _write_pitcher_throws(row, outcols, raw_value, prefer_long=True):
        canon = canon_hand(raw_value)
        long: Literal["Right", "Left"] = "Right" if canon == "R" else "Left"
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
        import json

        numeric_map = {
            "CommandTier": "_CmdBase",
            "StaminaScore": "_Stamina",
            "PitchCountLimit": "_Limit",
            "BF_Max": "_ExpBF",
            "ExpectedBattersFaced": "_ExpBF",
            "AvgPitchesPerOuting": "_AvgOut",
            "PitchingWeight": "_PitchingWeight",
            "AvgFBVelo": "_AvgFBVelo",
            "RelHeight_ft": "_RelHeight_ft",
            "RelSide_ft": "_RelSide_ft",
            "Extension_ft": "_Extension_ft",
        }
        for src, dst in numeric_map.items():
            if (dst not in p or p[dst] in (None, "")) and src in p:
                p[dst] = _safe_float(p.get(src), None)

        json_map = {
            "UsageJSON": "_Usage",
            "CommandByPitchJSON": "_CmdByPitch",
        }
        for src, dst in json_map.items():
            if dst in p and isinstance(p[dst], dict):
                continue
            raw = p.get(src)
            if isinstance(raw, dict):
                p[dst] = raw
            elif isinstance(raw, str) and raw.strip():
                try:
                    p[dst] = json.loads(raw)
                except Exception:
                    p[dst] = {}
            else:
                p[dst] = {}

        if "_ThrowsCanon" not in p or not p.get("_ThrowsCanon"):
            p["_ThrowsCanon"] = canon_hand(p.get("Throws", "R"))

        p["_Profile"] = {
            "command_tier": float(p.get("_CmdBase") or 1.0),
            "usage": p.get("_Usage") or {},
            "cmd_by_pitch": p.get("_CmdByPitch") or {},
            "stamina": float(p.get("_Stamina") or 50.0),
            "pitch_limit": int(p.get("_Limit") or 85),
            "expected_bf": float(p.get("_ExpBF") or 18.0),
            "avg_pitches": float(p.get("_AvgOut") or 80.0),
            "weight_pitch": float(p.get("_PitchingWeight") or 1.0),
            "avg_fb_velo": _safe_float(p.get("_AvgFBVelo"), None),
            "rel_h": _safe_float(p.get("_RelHeight_ft"), None),
            "rel_x": _safe_float(p.get("_RelSide_ft"), None),
            "ext": _safe_float(p.get("_Extension_ft"), None),
        }

        p["_PitchCount"] = 0
        p["_BF"] = 0
        p["_R"] = 0
        p["_PitchesThisHalf"] = 0

        try:
            sta = float(p["_Profile"].get("stamina", 50.0) or 50.0)
            limit = int(p["_Profile"].get("pitch_limit", 85) or 85)
            scale_lim = 1.0 + max(-0.15, min(0.15, (sta - 50.0) * 0.003))
            p["_Profile"]["pitch_limit"] = int(max(60, min(140, round(limit * scale_lim))))

            ebf = float(p["_Profile"].get("expected_bf", 18.0) or 18.0)
            scale_bf = 1.0 + max(-0.10, min(0.10, (sta - 50.0) * 0.002))
            p["_Profile"]["expected_bf"] = max(9.0, min(30.0, ebf * scale_bf))
        except Exception:
            pass

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

    def _classify_pen_roles(pen_list: List[Dict[str, Any]]):
        if not pen_list:
            return
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rp in pen_list:
            cmd = float(rp.get("_CmdBase", 1.0) or 1.0)
            velo = _safe_float(rp.get("_AvgFBVelo"), None)
            v_adj = 0.0 if velo is None else max(-5.0, min(5.0, (velo - 90.0) / 2.0))
            scored.append((cmd * 10.0 + v_adj, rp))
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored:
            scored[0][1]["_BP_Role"] = "Closer"
        if len(scored) > 1:
            scored[1][1]["_BP_Role"] = scored[1][1].get("_BP_Role", "Setup") or "Setup"
        for _, rp in scored[2:]:
            if rp.get("_BP_Role"):
                continue
            stam = float(rp.get("_Stamina", 50) or 50)
            throws = (rp.get("_ThrowsCanon") or "R").upper()
            if stam >= 60:
                rp["_BP_Role"] = "Long"
            elif throws == "L" and stam <= 55:
                rp["_BP_Role"] = "Lefty"
            else:
                rp["_BP_Role"] = "Middle"

    _classify_pen_roles(home_pen)
    _classify_pen_roles(away_pen)

    def get_next_rp(is_top: bool, inning_num: int, lineup_here: List[Dict[str, Any]], li_idx: int):
        pen = home_pen if is_top else away_pen
        avail = [x for x in pen if not x.get("_UsedUp", False)]
        if not avail:
            return None

        fielding_side = "home" if is_top else "away"
        batting_side = "away" if is_top else "home"
        lead = int(score[fielding_side]) - int(score[batting_side])
        save_situation = (inning_num >= 9) and (lead > 0) and (lead <= 3)
        high_lev = (inning_num >= 7) and (abs(lead) <= 2)
        need_long = (inning_num <= 5 and lead <= -3) or (inning_num <= 4)

        upcoming = []
        for offset in range(3):
            try:
                upcoming.append((lineup_here[(li_idx + offset) % 9].get("BatterSide") or "R").strip().upper())
            except Exception:
                upcoming.append("R")
        left_cnt = sum(1 for side in upcoming if side.startswith("L"))
        prefer_left = left_cnt > (len(upcoming) - left_cnt)

        def has_role(rp: Dict[str, Any], roles: set[str]) -> bool:
            role = (rp.get("_BP_Role") or "").capitalize()
            return role in roles

        candidates = avail[:]
        if save_situation:
            primary = [p for p in candidates if has_role(p, {"Closer"})]
            if not primary:
                primary = [p for p in candidates if has_role(p, {"Setup", "Middle"})]
            candidates = primary or candidates
        elif high_lev:
            primary = [p for p in candidates if has_role(p, {"Setup", "Closer"})]
            if not primary:
                primary = [p for p in candidates if has_role(p, {"Middle"})]
            candidates = primary or candidates
        elif need_long:
            primary = [p for p in candidates if has_role(p, {"Long"})]
            candidates = primary or candidates
        else:
            primary = [p for p in candidates if has_role(p, {"Middle", "Long"})]
            candidates = primary or candidates

        def side_score(rp: Dict[str, Any]) -> int:
            throws = (rp.get("_ThrowsCanon") or "R").upper()
            if prefer_left:
                return 2 if throws == "L" else 1
            return 2 if throws == "R" else 1

        def tiebreak(rp: Dict[str, Any]) -> float:
            if save_situation or high_lev:
                return float(rp.get("_CmdBase", 1.0) or 1.0)
            if need_long:
                return float(rp.get("_Stamina", 50) or 50)
            return float(rp.get("_CmdBase", 1.0) or 1.0)

        candidates.sort(key=lambda rp: (side_score(rp), tiebreak(rp)), reverse=True)
        choice = candidates[0]
        choice["_UsedUp"] = True
        attach_prof(choice)
        flag = (choice.get("InjuryProne") or "").strip().lower()
        if flag in {"y", "yes", "true", "1"}:
            choice["_Profile"]["pitch_limit"] = max(60, int(choice["_Profile"].get("pitch_limit", 85) * 0.9))
            choice["_Profile"]["stamina"] = max(30.0, float(choice["_Profile"].get("stamina", 50.0)) * 0.9)
        if inning_num > 9:
            (used_in_extras_home if is_top else used_in_extras_away).add(choice.get("PitcherId"))
        return choice

    def pick_pitch(p: Dict[str, Any], balls: int, strikes: int, batter_side: Optional[str], times_faced: int) -> str:
        prof_usage = {k: v for k, v in (p["_Profile"].get("usage") or {}).items() if k in pitch_types}
        if not prof_usage:
            base = {pt: float(pitches_cfg[pt].get("mix_pct", 0.0) or 0.0) for pt in pitch_types}
            total = sum(base.values()) or 1.0
            prof_usage = {pt: base[pt] / total for pt in pitch_types}

        bucket = count_bucket(int(balls or 0), int(strikes or 0))
        fb = {"Fastball", "FourSeamFastball", "TwoSeamFastball", "Sinker", "Cutter"}
        br = {"Slider", "Curveball", "KnuckleCurve", "Knuckleball", "Sweeper"}
        off = {"Changeup", "Splitter", "Forkball"}

        def _scale(val: float) -> float:
            return 1.0 + float(COUNT_MIX_SCALE) * (val - 1.0)

        count_mult = {"fb": 1.0, "br": 1.0, "off": 1.0}
        if bucket == "behind":
            count_mult = {"fb": _scale(1.35), "br": _scale(0.80), "off": _scale(0.90)}
        elif bucket == "ahead":
            count_mult = {"fb": _scale(0.88), "br": _scale(1.22), "off": _scale(1.10)}

        pthrows = (p.get("_ThrowsCanon") or "R").upper()
        bside = (batter_side or "R").upper()
        opp_side = (bside.startswith("L") and pthrows == "R") or (bside.startswith("R") and pthrows == "L")
        platoon_mult = {"fb": 1.0, "br": 1.0, "off": 1.0}
        if opp_side:
            platoon_mult = {"fb": 0.98, "br": 1.06, "off": 1.08}

        tto_mult = {"fb": 1.0, "br": 1.0, "off": 1.0}
        if times_faced >= 2:
            base_factor = 0.95 if times_faced == 2 else 0.90
            factor = 1.0 - float(TTO_MIX_SCALE) * (1.0 - base_factor)
            factor = max(0.80, min(1.0, factor))
            inv = 1.0 / max(1e-6, factor)
            tto_mult = {"fb": factor, "br": min(inv, 1.20), "off": min(inv, 1.15)}

        weights: Dict[str, float] = {}
        for pt, w in prof_usage.items():
            group = "fb" if pt in fb else ("br" if pt in br else ("off" if pt in off else None))
            mult = 1.0
            if group:
                mult *= count_mult.get(group, 1.0)
                mult *= platoon_mult.get(group, 1.0)
                mult *= tto_mult.get(group, 1.0)
            weights[pt] = max(0.0, float(w)) * mult

        total = sum(weights.values())
        if total <= 0.0:
            total = float(len(pitch_types) or 1)
            weights = {pt: 1.0 / total for pt in (pitch_types or ["Fastball"])}
        else:
            weights = {pt: w / total for pt, w in weights.items()}

        return sample_categorical(weights, rng)

    def tto_pen(line_slot_obj: dict) -> float:
        try:
            faced = int(line_slot_obj.get("_TimesFaced", 0))
        except Exception:
            faced = 0
        if faced <= 1:
            return 0.0
        if faced == 2:
            return 0.6 * TTO_PENALTY
        extra = max(0, faced - 3)
        return min(2.0 * TTO_PENALTY, TTO_PENALTY * (1.0 + 0.10 * extra))

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

    def load_zone_tbl(pt, hand_code, bats_code="__", count_tag="__"):
        tbl = safe_get(pitches_cfg, pt, "hands", hand_code, "pitch_call_by_zone", default=None)
        if tbl:
            return tbl
        call_map = _lookup_bundle(bundle_calls_map, pt, hand_code, bats_code, count_tag)
        if not call_map:
            return None
        outside: Dict[str, Dict[str, float]] = {}
        for region, dist in call_map.items():
            norm = _normalize_call_dist(dist)
            if norm:
                outside[str(region)] = norm
        return {"grid": {"x_edges": default_x_edges, "z_edges": default_z_edges}, "outside": outside} if outside else None

    def classify_zone(tbl, x, z):
        if not isinstance(tbl, dict):
            return (None, None)
        grid = tbl.get("grid") or {}
        outside = tbl.get("outside") or {}
        xe = grid.get("x_edges") or default_x_edges
        ze = grid.get("z_edges") or default_z_edges
        row_labels = grid.get("row_labels") or []
        col_labels = grid.get("col_labels") or []
        if isinstance(xe, list) and isinstance(ze, list) and len(xe) >= 2 and len(ze) >= 2:
            if xe[0] <= x <= xe[-1] and ze[0] <= z <= ze[-1]:
                col_idx = min(max(np.digitize([x], xe)[0] - 1, 0), len(xe) - 2)
                row_idx = min(max(np.digitize([z], ze)[0] - 1, 0), len(ze) - 2)
                if row_labels and col_labels and len(row_labels) == len(ze) - 1 and len(col_labels) == len(xe) - 1:
                    key = f"{row_labels[row_idx]}_{col_labels[col_idx]}"
                else:
                    key = f"r{row_idx}c{col_idx}"
                if (grid.get("cells") or {}).get(key):
                    return ("grid", key)
        xl, xr = (xe[0], xe[-1]) if xe else (-PLATE_HALF_WIDTH, PLATE_HALF_WIDTH)
        zl, zr = (ze[0], ze[-1]) if ze else (ZONE_Z_LOW, ZONE_Z_HIGH)
        dx = 0.0 if xl <= x <= xr else min(abs(x - xl), abs(x - xr))
        dz = 0.0 if zl <= z <= zr else min(abs(z - zl), abs(z - zr))
        edge_pad_x = float(grid.get("edge_pad_x", 0.15))
        edge_pad_z = float(grid.get("edge_pad_z", 0.15))
        target = "edge" if (dx <= edge_pad_x and dz <= edge_pad_z) else "chase"
        if outside.get(target):
            return ("outside", target)
        return (None, None)

    def sample_pitch_call_from_zone(tbl, where_tuple):
        kind, key = where_tuple
        if kind is None:
            return None
        dist = (tbl.get("grid") or {}).get("cells", {}).get(key, {}) if kind == "grid" else (tbl.get("outside") or {}).get(key, {})
        if not dist:
            return None
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
            times_faced = int(str(batter.get("_TimesFaced", 0) or 0))
            pt = pick_pitch(cur_pitcher, balls, strikes, batter.get("BatterSide"), times_faced)

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
            try:
                def _grade_adj(val: Any, mid: float = 50.0, span: float = 30.0, cap: float = 0.10) -> float:
                    num = _safe_float(val, None)
                    if num is None:
                        return 0.0
                    return max(-cap, min(cap, (num - mid) / span * cap))

                bq *= (1.0 + _grade_adj(batter.get("Contact")))
                bq *= (1.0 + _grade_adj(batter.get("Power")))
                bq *= (1.0 + _grade_adj(batter.get("Discipline")))
                bq = max(0.80, min(1.25, bq))
            except Exception:
                pass

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

            ball_bias = float(knobs.get("ball_bias", 1.0))
            if loc is not None and ball_bias != 1.0:
                zone_x_min, zone_x_max = -PLATE_HALF_WIDTH, PLATE_HALF_WIDTH
                zone_z_min, zone_z_max = ZONE_Z_LOW, ZONE_Z_HIGH
                x_loc, z_loc = loc
                in_zone = zone_x_min <= x_loc <= zone_x_max and zone_z_min <= z_loc <= zone_z_max
                if ball_bias > 1.0 and in_zone:
                    move_prob = min(0.8, (ball_bias - 1.0) * 0.5)
                    if rng.random() < move_prob:
                        direction = rng.choice(["left", "right", "low", "high"])
                        margin = rng.uniform(0.2, 0.8)
                        if direction == "left":
                            x_loc = zone_x_min - margin
                        elif direction == "right":
                            x_loc = zone_x_max + margin
                        elif direction == "low":
                            z_loc = zone_z_min - margin
                        else:
                            z_loc = zone_z_max + margin
                        loc = (x_loc, z_loc)
                elif ball_bias < 1.0 and not in_zone:
                    move_prob = min(0.8, (1.0 - ball_bias) * 0.5)
                    if rng.random() < move_prob:
                        x_loc = max(zone_x_min, min(zone_x_max, x_loc))
                        z_loc = max(zone_z_min, min(zone_z_max, z_loc))
                        loc = (x_loc, z_loc)

            if loc is not None:
                setc("PlateLocSide", round(loc[0], 3))
                setc("PlateLocHeight", round(loc[1], 3))

            bats_val = row.get("BatterSide") or batter.get("BatterSide") or "R"
            bats_short = "R" if str(bats_val).upper().startswith("R") else "L"

            pc = None
            bundle_calls_ctx = _lookup_bundle(bundle_calls_map, pt, phand_code, bats_short, count_bucket(balls, strikes))
            bundle_usage_ctx = _lookup_bundle(bundle_usage_map, pt, phand_code, bats_short, count_bucket(balls, strikes))

            def _apply_count_bias(dist: Dict[str, float], b: int, s: int) -> Dict[str, float]:
                try:
                    b = int(b)
                    s = int(s)
                except Exception:
                    return dist
                modifiers = {k: 1.0 for k in ("BallCalled", "StrikeCalled", "StrikeSwinging", "Foul", "InPlay")}
                if (b, s) == (3, 0):
                    modifiers.update({"BallCalled": 1.25, "StrikeCalled": 0.85, "StrikeSwinging": 0.85, "Foul": 0.90, "InPlay": 0.90})
                elif (b, s) == (3, 1):
                    modifiers.update({"BallCalled": 1.12, "StrikeSwinging": 0.95, "Foul": 0.97})
                elif (b, s) == (0, 2):
                    modifiers.update({"BallCalled": 0.85, "StrikeSwinging": 1.15, "Foul": 1.08, "InPlay": 0.95})
                elif (b, s) == (1, 2):
                    modifiers.update({"BallCalled": 0.90, "StrikeSwinging": 1.10, "Foul": 1.05, "InPlay": 0.97})
                elif (b, s) == (2, 0):
                    modifiers.update({"BallCalled": 1.08, "StrikeCalled": 0.92})
                elif (b, s) == (2, 2):
                    modifiers.update({"Foul": 1.05, "StrikeSwinging": 1.02, "InPlay": 0.98})
                elif (b, s) == (3, 2):
                    modifiers.update({"BallCalled": 0.95, "Foul": 1.05, "InPlay": 1.03})
                out = {k: max(0.0, float(dist.get(k, 0.0))) for k in dist.keys()}
                for key, mult in modifiers.items():
                    if key in out:
                        out[key] = max(0.0, out[key] * mult)
                total = sum(out.values()) or 1.0
                return {k: v / total for k, v in out.items() if v > 0.0}

            zone_table = load_zone_tbl(pt, phand_code, bats_short, count_bucket(balls, strikes))
            zone_where = (None, None)
            if loc is not None and zone_table:
                zone_where = classify_zone(zone_table, loc[0], loc[1])
                pc = sample_pitch_call_from_zone(zone_table, zone_where)

            if pc is None and zone_table and zone_where[0] is not None:
                kind, key = zone_where
                raw_zone = ((zone_table.get("grid") or {}).get("cells", {}).get(key, {})
                            if kind == "grid" else (zone_table.get("outside") or {}).get(key, {}))
                zone_dist = _normalize_call_dist(raw_zone)
                if zone_dist:
                    zone_dist = _apply_ball_bias_to_dist(zone_dist, ball_bias)
                    zone_dist = _apply_count_bias(zone_dist, balls, strikes)
                    if zone_dist:
                        pc = sample_categorical(zone_dist, rng)

            if pc is None and bundle_calls_ctx:
                zone_dist = None
                if zone_where[1] is not None:
                    zone_dist = _normalize_call_dist(bundle_calls_ctx.get(zone_where[1], {}))
                if not zone_dist:
                    zone_dist = _weighted_mix(bundle_calls_ctx, bundle_usage_ctx)
                if zone_dist:
                    zone_dist = _apply_ball_bias_to_dist(zone_dist, ball_bias)
                    zone_dist = _apply_count_bias(zone_dist, balls, strikes)
                    if zone_dist:
                        pc = sample_categorical(zone_dist, rng)

            if pc is None:
                base_pc = safe_get(pitches_cfg, pt, "hands", phand_code, "outcomes",
                                   default={"BallCalled": 0.42, "StrikeCalled": 0.15, "StrikeSwinging": 0.16, "Foul": 0.15, "InPlay": 0.12})
                pc_dist = _adjust_pitchcall(base_pc or {}, batter_q=bq, pitch_cmd=command)
                avg_velo = _safe_float(cur_pitcher["_Profile"].get("avg_fb_velo"), None)
                if avg_velo is not None and str(pt).lower().startswith("fast"):
                    delta = avg_velo - 88.0
                    whiff = max(0.90, min(1.10, 1.03 + 0.03 * (delta / 2.0)))
                    inplay = max(0.90, min(1.10, 0.97 - 0.03 * (delta / 2.0)))
                    pc_dist = dict(pc_dist)
                    if "StrikeSwinging" in pc_dist:
                        pc_dist["StrikeSwinging"] *= whiff
                    if "InPlay" in pc_dist:
                        pc_dist["InPlay"] *= inplay
                    total = sum(max(0.0, v) for v in pc_dist.values()) or 1.0
                    pc_dist = {k: max(0.0, v) / total for k, v in pc_dist.items()}
                pc_dist = _apply_ball_bias_to_dist(pc_dist, ball_bias)
                pc_dist = _apply_count_bias(pc_dist, balls, strikes)
                if not pc_dist:
                    pc_dist = {"BallCalled": 0.45, "StrikeCalled": 0.20, "StrikeSwinging": 0.15, "Foul": 0.10, "InPlay": 0.10}
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
                try:
                    error_prob = float(ERROR_RATE)
                except Exception:
                    error_prob = 0.0
                if error_prob > 0.0:
                    split = dict(split)
                    if "Out" in split:
                        split["Out"] = max(0.0, float(split["Out"]) - error_prob)
                    split["Error"] = float(split.get("Error", 0.0)) + error_prob
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

            if not terminal:
                def _attempt_steal_second(cur_bases: List[Optional[dict]]):
                    nonlocal outs
                    if not cur_bases[0]:
                        return cur_bases
                    if outs >= 2:
                        return cur_bases
                    if rng.random() < SB_ATTEMPT_R1_BASE:
                        catcher = home_c if fielding_team == home_team_key else away_c
                        throws = (catcher.get("CatcherThrows") or "").strip().upper()
                        success = SB_SUCCESS_BASE + (SB_CATCHER_R_BONUS if throws.startswith("R") else 0.0)
                        if rng.random() < success:
                            cur_bases = [None, cur_bases[0], cur_bases[2]]
                            if "Notes" in out_cols:
                                prev = row.get("Notes", "")
                                row["Notes"] = (prev + ";" if prev else "") + "SB2"
                        else:
                            outs += 1
                            cur_bases = [None, cur_bases[1], cur_bases[2]]
                            if "Notes" in out_cols:
                                prev = row.get("Notes", "")
                                row["Notes"] = (prev + ";" if prev else "") + "CS2"
                    return cur_bases

                def _attempt_steal_third(cur_bases: List[Optional[dict]]):
                    nonlocal outs
                    if not cur_bases[1]:
                        return cur_bases
                    if outs >= 2:
                        return cur_bases
                    if rng.random() < SB_ATTEMPT_R2_BASE:
                        catcher = home_c if fielding_team == home_team_key else away_c
                        throws = (catcher.get("CatcherThrows") or "").strip().upper()
                        success = (SB_SUCCESS_BASE - 0.05) + (SB_CATCHER_R_BONUS if throws.startswith("R") else 0.0)
                        if rng.random() < success:
                            cur_bases = [cur_bases[0], None, cur_bases[1]]
                            if "Notes" in out_cols:
                                prev = row.get("Notes", "")
                                row["Notes"] = (prev + ";" if prev else "") + "SB3"
                        else:
                            outs += 1
                            cur_bases = [cur_bases[0], None, cur_bases[2]]
                            if "Notes" in out_cols:
                                prev = row.get("Notes", "")
                                row["Notes"] = (prev + ";" if prev else "") + "CS3"
                    return cur_bases

                outs_before = outs
                bases = _attempt_steal_second(bases)
                bases = _attempt_steal_third(bases)

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
                    nxt = get_next_rp(is_top, inn, lineup, li)
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
