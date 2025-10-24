#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas.core.frame import DataFrame
import uuid, random
from typing import Dict, Any, Literal, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from faker import Faker
from aim_engine import PitchAimEngine
from loc_params_builder import build_params_from

# Strike zone dimensions (in feet)
PLATE_HALF_WIDTH = 0.708  # ~17" / 2
ZONE_Z_LOW = 1.55
ZONE_Z_HIGH = 3.45

from plate_loc_model import PlateLocSampler, default_mixtures_demo, count_bucket, PlateLocBundle
from attr_location_sampler import sample_loc_from_roster_attrs
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
# put near the top of game_sim.py (or a shared constants module that’s imported first)
DEFAULT_ZONE_COL_LABELS = ["in", "mid", "out"]     # x from left→right
DEFAULT_ZONE_ROW_LABELS = ["low", "mid", "high"]   # z from bottom→top


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

    bundle_calls_map = plate_bundle.pitch_call_by_region if plate_bundle and getattr(plate_bundle, 'pitch_call_by_region', None) else {}
    bundle_usage_map = plate_bundle.region_usage if plate_bundle and getattr(plate_bundle, 'region_usage', None) else {}
    default_x_edges = [float(v) for v in np.linspace(-PLATE_HALF_WIDTH, PLATE_HALF_WIDTH, 4)]
    default_z_edges = [float(v) for v in np.linspace(ZONE_Z_LOW, ZONE_Z_HIGH, 4)]
    rules_yaml = priors.get("RulesPitchByPitch") or {}
    profiles_csv_path = knobs.get("pitcher_profiles_csv") if isinstance(knobs, dict) else None

    aim_params = build_params_from(priors, rules_yaml, profiles_csv_path)
    aim_engine= PitchAimEngine(aim_params, rng)
    def _canon_key_part(val) -> str:
        if val is None:
            return "__"
        s = str(val).strip()
        return s if s else "__"

    def _lookup_bundle(table: dict, pitch_type: str, hand_code: str, bats_code: str, count_tag: str):
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
            value = table.get(norm_key)
            if value:
                return value
        return None

    def _normalize_call_dist(dist) -> dict[str, float]:
        if not isinstance(dist, dict):
            return {}
        cleaned: dict[str, float] = {}
        for k, v in dist.items():
            try:
                cleaned[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        total = sum(max(0.0, val) for val in cleaned.values())
        if total <= 0.0:
            return {}
        return {k: max(0.0, val) / total for k, val in cleaned.items()}

    def _weighted_mix(call_map: dict, weight_map: dict | None) -> dict[str, float]:
        if not isinstance(call_map, dict) or not call_map:
            return {}
        accum: dict[str, float] = {}
        total = 0.0
        if isinstance(weight_map, dict):
            for region, dist in call_map.items():
                w_raw = weight_map.get(region, None)
                try:
                    w = float(w_raw)
                except (TypeError, ValueError):
                    w = None
                if w is None or w <= 0.0:
                    continue
                norm = _normalize_call_dist(dist)
                if not norm:
                    continue
                total += w
                for outcome, prob in norm.items():
                    accum[outcome] = accum.get(outcome, 0.0) + w * prob
        if total <= 0.0:
            # fall back to simple average of available regions
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

    def _edge_pads_from_cfg(priors, pt, hand_code, default_x=0.15, default_z=0.15):
        """Resolve edge_pad_x/z priority:
        1) grid.edge_pad_* if already present (handled elsewhere)
        2) PitchCallZones[pt].hands[hand_code].edge_pad_*
        3) PitchCallZones[pt].edge_pad_*
        4) provided defaults
        """
        pc = safe_get(priors, "PitchCallZones", pt) or {}
        by_hand = (pc.get("hands") or {}).get(hand_code) or {}
        pad_x = by_hand.get("edge_pad_x", pc.get("edge_pad_x", default_x))
        pad_z = by_hand.get("edge_pad_z", pc.get("edge_pad_z", default_z))
        try:
            return float(pad_x), float(pad_z)
        except Exception:
            return float(default_x), float(default_z)

    def _apply_ball_bias_to_dist(dist: dict[str, float], factor: float) -> dict[str, float]:
        if factor == 1.0 or "BallCalled" not in dist:
            return dist
        out = {k: max(0.0, float(v)) for k, v in dist.items()}
        out["BallCalled"] = out.get("BallCalled", 0.0) * float(factor)
        total = sum(max(0.0, v) for v in out.values()) or 1.0
        return {k: max(0.0, v) / total for k, v in out.items()}

    if plate_sampler is None:
        _sampler_rng = np.random.default_rng(10)
        plate_sampler = PlateLocSampler(
            rng=_sampler_rng,
            mixtures=default_mixtures_demo(),
            rho=0.20,
            noise_x=0.15,
            noise_y=0.18
        )

    out_cols: list[str] = template_cols[:]
    # Ensure critical columns exist
    for must in ["Top/Bottom","BatterSide","PlayResult","KorBB","RunsScored","OutsOnPlay","HitType","PitchCall"]:
        if must not in out_cols:
            out_cols.append(must)
    # Expand required columns so per-pitch values are always written (improves YT/TM population)
    _must_core = [
        # context/meta
        "Date","Time","GameID","HomeTeam","AwayTeam","Stadium","League",
        # inning state
        "Inning","PAofInning","PitchofPA","Outs","Balls","Strikes",
        # participants
        "Pitcher","PitcherId","PitcherThrows","PitcherTeam","PitcherSet","Batter","BatterId","BatterSide","BatterTeam",
        # release/flight metrics
        "RelSpeed","SpinRate","SpinAxis","RelHeight","RelSide","Extension",
        "PlateLocHeight","PlateLocSide","VertBreak","InducedVertBreak","HorzBreak",
        "VertRelAngle","HorzRelAngle","VertApprAngle","HorzApprAngle","ZoneSpeed","ZoneTime",
        # batted ball essentials
        "ExitSpeed","Angle","Direction","HitSpinRate","Distance","LastTrackedDistance","Bearing","HangTime",
        # simple kinematic proxies
        "pfxx","pfxz","x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0",
    ]
    for _c in _must_core:
        if _c not in out_cols:
            out_cols.append(_c)

    # Ensure catcher-related columns exist so we can populate from roster
    # Do NOT force-add Throws/CatcherThrows – only include if template has them
    for c in ("Catcher", "CatcherId", "CatcherTeam"):
        if c not in out_cols:
            out_cols.append(c)

    game_state = {"over": False}
    knobs.setdefault("ball_bias", 1.00)
    home_line, home_c = lineup_by_positions(roster_by_team.get(str(home_team_key), []), home_team_key, rng)
    away_line, away_c = lineup_by_positions(roster_by_team.get(str(away_team_key), []), away_team_key, rng)

    def _write_pitcher_throws(row, outcols, raw_value, prefer_long=True):
        canon: str = canon_hand(raw_value)
        long: Literal['Right', 'Left']  = "Right" if canon == "R" else "Left"
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
        """
        Hydrate pitcher profile from roster CSV columns (if present)
        and build underscore fields used by the sim.
        """
        import json

        # 1) Map numeric roster columns → underscore fields
        NUM_MAP: dict[str, str] = {
            "CommandTier":           "_CmdBase",
            "StaminaScore":          "_Stamina",
            "PitchCountLimit":       "_Limit",
            "BF_Max":                "_ExpBF",
            "AvgPitchesPerOuting":   "_AvgOut",
            "PitchingWeight":        "_PitchingWeight",
            "AvgFBVelo":             "_AvgFBVelo",
            "RelHeight_ft":          "_RelHeight_ft",
            "RelSide_ft":            "_RelSide_ft",
            "Extension_ft":            "_Extension_ft",
        }
        for src, dst in NUM_MAP.items():
            if (dst not in p) or (p[dst] in ("", None)):
                if src in p:
                    p[dst] = _safe_float(p.get(src), None)

        # 2) Map JSON roster blobs (usage/cmd) → underscore fields
        JSON_MAP = {
            "UsageJSON":            "_Usage",
            "CommandByPitchJSON":   "_CmdByPitch",
        }
        for src, dst in JSON_MAP.items():
            if (dst not in p) or (p[dst] in ("", None, {})):
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

        # 3) Canonical throws
        if "_ThrowsCanon" not in p or not p["_ThrowsCanon"]:
            p["_ThrowsCanon"] = canon_hand(p.get("Throws", "R"))

        # 4) Build runtime profile
        p["_Profile"] = {
            "command_tier": float(p.get("_CmdBase") or 1.0),
            "usage":         (p.get("_Usage") or {}),
            "cmd_by_pitch":  (p.get("_CmdByPitch") or {}),

            "stamina":     float(p.get("_Stamina") or 50.0),
            "pitch_limit": int(p.get("_Limit") or 85),
            "expected_bf": float(p.get("_ExpBF") or 18.0),
            "avg_pitches": float(p.get("_AvgOut") or 80.0),

            "weight_pitch": float(p.get("_PitchingWeight") or 1.0),
            "avg_fb_velo":  _safe_float(p.get("_AvgFBVelo"), None),

            "rel_h": _safe_float(p.get("_RelHeight_ft"), None),
            "rel_x": _safe_float(p.get("_RelSide_ft"), None),
            "ext":   _safe_float(p.get("_Extension_ft"), None),
        }

        # 5) Per-game counters
        p["_PitchCount"] = 0
        p["_BF"] = 0
        p["_R"] = 0
        p["_PitchesThisHalf"] = 0

        # 6) Apply stamina scaling to effective pitch limit and expected BF
        try:
            sta = float(p["_Profile"].get("stamina", 50.0) or 50.0)
            # Scale limit within ±15% around 50 stamina
            limit = int(p["_Profile"].get("pitch_limit", 85) or 85)
            scale_lim = 1.0 + max(-0.15, min(0.15, (sta - 50.0) * 0.003))
            p["_Profile"]["pitch_limit"] = int(max(60, min(140, round(limit * scale_lim))))

            # Scale expected batters faced within ±10%
            ebf = float(p["_Profile"].get("expected_bf", 18.0) or 18.0)
            scale_bf = 1.0 + max(-0.10, min(0.10, (sta - 50.0) * 0.002))
            p["_Profile"]["expected_bf"] = max(9.0, min(30.0, ebf * scale_bf))
        except Exception:
            pass

    for obj in [home_sp, *home_pen, away_sp, *away_pen]:
        attach_prof(obj)

    def _blend_geom(model_val, roster_val, w_roster=0.6, jitter_sd=0.05, lo=3.5, hi=7.0):
        mv: float | None = None if model_val in ("", None) else float(model_val)
        rv = None if roster_val in ("", None) else float(roster_val)
        if mv is None and rv is None: return None
        if mv is None: base = rv
        elif rv is None: base = mv
        else: base = (1 - w_roster) * mv + w_roster * rv
        if base is not None:
            base = base + rng.normalvariate(0.0, jitter_sd)
            return round(clamp(base, lo, hi), 3)
        return None

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

    def _classify_pen_roles(pen_list: list[dict]):
        # Assign simple bullpen roles for selection heuristics
        if not pen_list:
            return
        # Base scores for closer/setup selection
        scored = []
        for rp in pen_list:
            cmd = float(rp.get("_CmdBase", 1.0) or 1.0)
            velo = _safe_float(rp.get("_AvgFBVelo"), None)
            v_adj = 0.0 if velo is None else max(-5.0, min(5.0, (float(velo) - 90.0) / 2.0))
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
            thr = (rp.get("_ThrowsCanon") or "R").upper()
            if stam >= 60:
                rp["_BP_Role"] = "Long"
            elif thr == "L" and stam <= 55:
                rp["_BP_Role"] = "Lefty"
            else:
                rp["_BP_Role"] = "Middle"

    _classify_pen_roles(home_pen)
    _classify_pen_roles(away_pen)

    def get_next_rp(is_top: bool, inning_num: int, lineup_here: list[dict], li_idx: int):
        pen = home_pen if is_top else away_pen
        avail = [x for x in pen if not x.get("_UsedUp", False)]
        if not avail:
            return None

        # Leverage context
        fielding_side = "home" if is_top else "away"
        batting_side = "away" if is_top else "home"
        lead = int(score[fielding_side]) - int(score[batting_side])
        save_situation = (inning_num >= 9) and (lead > 0) and (lead <= 3)
        high_lev = (inning_num >= 7) and (abs(lead) <= 2)
        need_long = (inning_num <= 5 and lead <= -3) or (inning_num <= 4)

        # Upcoming hitters handedness (next 3)
        upsides = []
        for k in range(3):
            try:
                upsides.append((lineup_here[(li_idx + k) % 9].get("BatterSide") or "R").strip().upper())
            except Exception:
                upsides.append("R")
        l_cnt = sum(1 for s in upsides if s.startswith("L"))
        r_cnt = sum(1 for s in upsides if s.startswith("R"))
        prefer_L = l_cnt > r_cnt

        # Role filters
        def by_role(rp, roles):
            return (rp.get("_BP_Role") or "").capitalize() in roles

        candidates = avail[:]
        if save_situation:
            pri = [x for x in candidates if by_role(x, {"Closer"})]
            if not pri:
                pri = [x for x in candidates if by_role(x, {"Setup","Middle"})]
            candidates = pri or candidates
        elif high_lev:
            pri = [x for x in candidates if by_role(x, {"Setup","Closer"})]
            if not pri:
                pri = [x for x in candidates if by_role(x, {"Middle"})]
            candidates = pri or candidates
        elif need_long:
            pri = [x for x in candidates if by_role(x, {"Long"})]
            candidates = pri or candidates
        else:
            pri = [x for x in candidates if by_role(x, {"Middle","Long"})]
            candidates = pri or candidates

        # Handedness preference
        def side_score(rp):
            thr = (rp.get("_ThrowsCanon") or "R").upper()
            if prefer_L:
                return 2 if thr == "L" else 1
            else:
                return 2 if thr == "R" else 1

        # Tie-breaker by command for high leverage; by stamina for long
        def tiebreak(rp):
            if save_situation or high_lev:
                return float(rp.get("_CmdBase", 1.0) or 1.0)
            if need_long:
                return float(rp.get("_Stamina", 50) or 50)
            return float(rp.get("_CmdBase", 1.0) or 1.0)

        candidates.sort(key=lambda x: (side_score(x), tiebreak(x)), reverse=True)
        p = candidates[0]
        p["_UsedUp"] = True
        attach_prof(p)
        # Inj-prone example tweak, if you carry this flag in roster
        flag = (p.get("InjuryProne") or "").strip().lower()
        if flag in ("y","yes","true","1"):
            p["_Profile"]["pitch_limit"] = max(60, int(p["_Profile"]["pitch_limit"] * 0.9))
            p["_Profile"]["stamina"] = max(30.0, float(p["_Profile"]["stamina"]) * 0.9)
        if inning_num > 9:
            (used_in_extras_home if is_top else used_in_extras_away).add(p["PitcherId"])
        return p

    def pick_pitch(p, balls: int, strikes: int, batter_side: str | None, times_faced: int) -> str:
        """Select a pitch type from pitcher's usage, modulated by count, platoon and TTO."""
        prof_usage = {k: v for k, v in p["_Profile"].get("usage", {}).items() if k in pitch_types}
        if not prof_usage:
            base = {pt: float(pitches_cfg[pt].get("mix_pct", 0.0) or 0.0) for pt in pitch_types}
            s = sum(base.values()) or 1.0
            prof_usage = {pt: base[pt] / s for pt in pitch_types}

        # Heuristic multipliers by count from PITCHER perspective
        cb = count_bucket(int(balls or 0), int(strikes or 0))
        # Group pitch types into buckets for adjustment
        fb = {"Fastball","FourSeamFastball","TwoSeamFastball","Sinker","Cutter"}
        br = {"Slider","Curveball","KnuckleCurve","Knuckleball","Sweeper"}
        off = {"Changeup","Splitter","Forkball"}

        mult_count = {"fb": 1.0, "br": 1.0, "off": 1.0}
        def _scale(v: float) -> float:
            return 1.0 + float(COUNT_MIX_SCALE) * (float(v) - 1.0)
        if cb == "behind":
            base = {"fb": 1.35, "br": 0.80, "off": 0.90}
            mult_count = {k: _scale(v) for k, v in base.items()}
        elif cb == "ahead":
            base = {"fb": 0.88, "br": 1.22, "off": 1.10}
            mult_count = {k: _scale(v) for k, v in base.items()}

        # Platoon tweak: vs opposite-side hitters, small boost to CH/SL
        bside = (batter_side or "R").strip().upper()
        pthrows = (p.get("_ThrowsCanon") or "R").strip().upper()
        opp_side = (bside.startswith("L") and pthrows == "R") or (bside.startswith("R") and pthrows == "L")
        mult_platoon = {"fb": 1.0, "br": 1.0, "off": 1.0}
        if opp_side:
            mult_platoon = {"fb": 0.98, "br": 1.06, "off": 1.08}

        # Times-through-order mix shift: lean a bit more on secondaries 3rd time+
        mult_tto = {"fb": 1.0, "br": 1.0, "off": 1.0}
        if times_faced >= 2:
            base_factor = 0.95 if times_faced == 2 else 0.90
            factor = 1.0 - float(TTO_MIX_SCALE) * (1.0 - base_factor)
            factor = max(0.80, min(1.0, factor))
            inv = 1.0 / max(1e-6, factor)
            mult_tto = {"fb": factor, "br": min(inv, 1.20), "off": min(inv, 1.15)}

        # Combine multipliers and reweight usage
        weights: dict[str, float] = {}
        for pt, w in prof_usage.items():
            g = "fb" if pt in fb else ("br" if pt in br else ("off" if pt in off else "other"))
            m = (mult_count.get(g, 1.0) * mult_platoon.get(g, 1.0) * mult_tto.get(g, 1.0))
            weights[pt] = max(0.0, float(w)) * float(m)

        total = sum(weights.values())
        if total <= 0:
            # fallback to uniform if something went wrong
            n = max(1, len(pitch_types))
            weights = {pt: 1.0 / n for pt in pitch_types}
        else:
            weights = {pt: w / total for pt, w in weights.items()}

        return sample_categorical(weights, rng)

    def tto_pen(line_slot_obj: dict) -> float:
        """Graduated times-through-order penalty applied to command.
        1st PA: 0, 2nd PA: 0.6*TTO_PENALTY, 3rd+: 1.0*TTO_PENALTY (+10% each PA after 3rd, capped).
        """
        try:
            tf = int(line_slot_obj.get("_TimesFaced", 0))
        except Exception:
            tf = 0
        if tf <= 0:
            return 0.0
        if tf == 1:
            return 0.0
        if tf == 2:
            return 0.6 * TTO_PENALTY
        extra = max(0, tf - 3)
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
        hb = safe_get(pitches_cfg, pt, "hands", hand_code, default={}) or {}
        loc = hb.get("location_model") or {} if hb else {}
        mean, cov = loc.get("mean"), loc.get("cov")
        if not (isinstance(mean, (list,tuple)) and isinstance(cov,(list,tuple))):
            return None
        try:
            cov_arr = np.array(cov, dtype=float) / max(1e-6, float(command))
        except Exception:
            cov_arr = np.array(cov, dtype=float)
        x,z = mvn_sample(list(mean), cov_arr.tolist())
        return (float(clamp(x, -0.95, 0.95)), float(clamp(z, 1.0, 4.0)))


    def load_zone_tbl(pt, hand_code, bats_code="__", count_tag="__"):
        pitches_cfg = safe_get(priors, "PitchCallZones")
        if not isinstance(pitches_cfg, dict):
            return None

        # 1) Direct table
        tbl = safe_get(pitches_cfg, pt, "hands", hand_code, "pitch_call_by_zone", default=None)
        if tbl:
            grid = tbl.get("grid") or {}
            if ("edge_pad_x" not in grid) or ("edge_pad_z" not in grid):
                pad_x, pad_z = _edge_pads_from_cfg(priors, pt, hand_code)
                grid = dict(grid, edge_pad_x=pad_x, edge_pad_z=pad_z)
                tbl = dict(tbl, grid=grid)
            return tbl

        # 2) Construct from bundles
        bundle_calls_map = safe_get(pitches_cfg, pt, "hands", hand_code, "bundle_calls", default=None)
        bundle_usage_map = safe_get(pitches_cfg, pt, "hands", hand_code, "bundle_usage", default=None)
        if not bundle_calls_map:
            return None

        bundle_calls_ctx = _lookup_bundle(bundle_calls_map, pt, hand_code, bats_code, count_tag) or {}
        bundle_usage_ctx = _lookup_bundle(bundle_usage_map, pt, hand_code, bats_code, count_tag) or {}

        grid_cells: dict[str, dict[str, float]] = {}
        outside: dict[str, dict[str, float]] = {}

        col_labels = list(globals().get('DEFAULT_ZONE_COL_LABELS', []))
        row_labels = list(globals().get('DEFAULT_ZONE_ROW_LABELS', []))

        for region, dist in (bundle_calls_ctx or {}).items():
            norm = _normalize_call_dist(dist)
            if not norm:
                continue
            region_key = str(region)
            added = False
            if '_' in region_key and row_labels and col_labels:
                row_part, col_part = region_key.split('_', 1)
                if row_part in row_labels and col_part in col_labels:
                    grid_cells[f"{row_part}_{col_part}"] = norm
                    added = True
            if (not added) and region_key.startswith('r') and 'c' in region_key:
                grid_cells[region_key] = norm
                added = True
            if not added:
                outside[region_key] = norm

        tbl = {}
        if grid_cells:
            pad_x, pad_z = _edge_pads_from_cfg(priors, pt, hand_code)
            tbl["grid"] = {
                "x_edges": list(default_x_edges),
                "z_edges": list(default_z_edges),
                "col_labels": col_labels,
                "row_labels": row_labels,
                "edge_pad_x": pad_x,
                "edge_pad_z": pad_z,
                "cells": grid_cells
            }
        if outside:
            tbl["outside"] = outside
        return tbl if tbl else None

    def classify_zone(tbl, x, z):
        import math
        if not isinstance(tbl, dict):
            return (None, None, None)
        grid = tbl.get("grid") or {}; outside = tbl.get("outside") or {}
        xe, ze = grid.get("x_edges"), grid.get("z_edges")
        col_labels = grid.get("col_labels") or []
        row_labels = grid.get("row_labels") or []
        if isinstance(xe, list) and isinstance(ze, list) and len(xe) >= 2 and len(ze) >= 2:
            if (x >= xe[0] and x <= xe[-1] and z >= ze[0] and z <= ze[-1]):
                ci = min(max(np.digitize([x], xe)[0] - 1, 0), len(xe) - 2)
                ri = min(max(np.digitize([z], ze)[0] - 1, 0), len(ze) - 2)
                if col_labels and row_labels and len(col_labels) == len(xe) - 1 and len(row_labels) == len(ze) - 1:
                    key = f"{row_labels[ri]}_{col_labels[ci]}"
                else:
                    key = f"r{ri}c{ci}"
                if grid.get("cells", {}).get(key):
                    return ("grid", key, 0.0)
        if isinstance(xe, list) and isinstance(ze, list) and len(xe) >= 2 and len(ze) >= 2:
            xl, xr, zl, zr = xe[0], xe[-1], ze[0], ze[-1]
        else:
            xl, xr, zl, zr = -0.708, 0.708, 1.5, 3.5
        dx = 0.0 if xl <= x <= xr else min(abs(x - xl), abs(x - xr))
        dz = 0.0 if zl <= z <= zr else min(abs(z - zl), abs(z - zr))
        d  = math.hypot(dx, dz)
        edge_pad_x = float(grid.get("edge_pad_x", 0.15))
        edge_pad_z = float(grid.get("edge_pad_z", 0.15))
        target = "edge" if (dx <= edge_pad_x and dz <= edge_pad_z) else "chase"
        if outside.get(target):
            return ("outside", target, d)
        return (None, None, None)

    def sample_pitch_call_from_zone(tbl, where_tuple):
        import math
        if not where_tuple:
            return None
        if len(where_tuple) == 2:
            kind, key = where_tuple
            d = None
        else:
            kind, key, d = where_tuple
        if kind is None:
            return None
        if kind=="grid":
            dist = (tbl.get("grid") or {}).get("cells", {}).get(key, {})
            if not dist:
                return None
            return sample_categorical(dist, rng)
        dist = (tbl.get("outside") or {}).get(key, {})
        if not dist:
            return None
        if d is not None:
            g = math.exp(-float(d) / 0.25)
            weighted = {k: (v if k in ("BallCalled", "Ball") else v * g) for k, v in dist.items()}
            z = sum(max(0.0, float(v)) for v in weighted.values()) or 1.0
            dist = {k: max(0.0, float(v)) / z for k, v in weighted.items()}
        return sample_categorical(dist, rng)

    def runner(owner_pid): return {"pid": owner_pid}
    
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
        nonlocal current_dt, rows, score
    
        MAX_PITCHES_PER_PA = 25
        MAX_PITCHES_PER_HALF = 400
    
        outs, balls, strikes = 0, 0, 0
        paofinning, pitchofpa = 1, 1
        bases = [None, None, None]
        runs_this_half = 0
    
        if inn > 9:
            (used_in_extras_home if is_top else used_in_extras_away).add(cur_pitcher["PitcherId"])
    
        batting_team  = away_team_key if is_top else home_team_key
        fielding_team = home_team_key if is_top else away_team_key
        lineup = away_line if is_top else home_line
        li = int(lineup_idx)
    
        cur_pitcher["_PitchesThisHalf"] = 0
        ensure_pbox(cur_pitcher)
        pbox[cur_pitcher["PitcherId"]]["G"] = int(pbox[cur_pitcher["PitcherId"]]["G"]) + 1
        if (is_top and cur_pitcher is away_sp) or ((not is_top) and cur_pitcher is home_sp):
            pbox[cur_pitcher["PitcherId"]]["GS"] = int(pbox[cur_pitcher["PitcherId"]]["GS"]) + 1
    
        while outs < 3:
            if cur_pitcher["_PitchesThisHalf"] >= MAX_PITCHES_PER_HALF: break
    
            batter = lineup[li]
            phand_code = cur_pitcher["_ThrowsCanon"]  # 'R'/'L'
            pt = pick_pitch(cur_pitcher, balls, strikes, batter.get("BatterSide"), int(str(batter.get("_TimesFaced", 0) or 0)))
    
            row = {c: "" for c in out_cols}
            def setc(k, v):
                if k in out_cols:
                    row[k] = v
    
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
            # Populate catcher identity/throws from starting lineup by fielding team
            _c = home_c if fielding_team == home_team_key else away_c
            setc("Catcher", _c.get("Catcher", ""))
            setc("CatcherId", _c.get("CatcherId", ""))
            setc("CatcherThrows", _c.get("CatcherThrows", ""))
            # PitcherSet: Stretch if any runner is on, else Windup
            setc("PitcherSet", "Stretch" if any(bases) else "Windup")
            setc("Batter", batter["Batter"]); setc("BatterId", batter["BatterId"])
            setc("BatterSide", batter["BatterSide"]); setc("BatterTeam", batting_team)
            setc("TaggedPitchType", pt)
    
            hb = safe_get(pitches_cfg, pt, "hands", phand_code, default={}) or {}
            core = hb.get("core", {}) or {} if hb else {}
            core_map = {"velo": "RelSpeed", "spin": "SpinRate", "ivb": "InducedVertBreak", "hb": "HorzBreak"}
            for friendly, col in core_map.items():
                if col in out_cols:
                    rto = 2 if col in ("RelSpeed", "InducedVertBreak", "HorzBreak") else 0
                    row[col] = str(sample_statpack(core.get(friendly), rng, round_to=rto))
    
            features = hb.get("features", {}) or {} if hb else {}
            for col in (
                "RelHeight", "RelSide", "Extension", "VertApprAngle", "HorzApprAngle", "ZoneSpeed", "ZoneTime",
                "SpinAxis", "Tilt", "VertRelAngle", "HorzRelAngle", "VertBreak",
                "pfxx", "pfxz", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0"
            ):
                if col in out_cols and col in features:
                    rto = 2 if col in ("VertApprAngle","HorzApprAngle","ZoneSpeed","SpinAxis","VertRelAngle","HorzRelAngle") else 3
                    val = sample_feature_value(features[col], rng, round_to=rto)
                    if val is not None: row[col] = str(val)
            if "ZoneSpeed" in out_cols and row.get("ZoneSpeed","")=="" and row.get("RelSpeed","")!="":
                row["ZoneSpeed"] = row["RelSpeed"]
    
            # Blend roster geometry & FB velo influence
            if pt.lower().startswith("fast") and "RelSpeed" in out_cols:
                roster_mu: Any | float | None = _safe_float(cur_pitcher["_Profile"].get("avg_fb_velo"))
                model_mu: Any | float | None  = _safe_float(row.get("RelSpeed"))
                if (roster_mu is not None) or (model_mu is not None):
                    # Ensure we have valid values for calculation
                    roster_val = roster_mu if roster_mu is not None else (model_mu or 88.0)
                    model_val = model_mu if model_mu is not None else (roster_mu or 88.0)
                    base: float = (0.6 * roster_val + 0.4 * model_val)
                    row["RelSpeed"] = str(round(base + rng.normalvariate(0.0, 0.15), 2))
    
            if "RelHeight" in out_cols:
                rel_height_val = _blend_geom(row.get("RelHeight"), cur_pitcher["_Profile"].get("rel_h"), 0.6, 0.04, 4.0, 7.0)
                row["RelHeight"] = str(rel_height_val) if rel_height_val is not None else ""
            if "RelSide" in out_cols:
                rel_side_val = _blend_geom(row.get("RelSide"),   cur_pitcher["_Profile"].get("rel_x"), 0.6, 0.03, -3.5, 3.5)
                row["RelSide"] = str(rel_side_val) if rel_side_val is not None else ""
            if "Extension" in out_cols:
                extension_val = _blend_geom(row.get("Extension"),cur_pitcher["_Profile"].get("ext"),  0.6, 0.05, 4.0, 8.5)
                row["Extension"] = str(extension_val) if extension_val is not None else ""
    
            command = cmd_for_pitch(cur_pitcher, pt, batter, inn)
    
            # Batter quality with platoon + optional roster grades (Contact/Power/Discipline 0–100)
            bq = float(batter.get("_Quality", 1.0))
            bq *= (1.0 + _platoon_batter_bonus(batter.get("BatterSide", "Right"), phand_code))
            bq = max(0.85, min(1.20, bq))
            try:
                def _grade(val, mid=50.0, span=30.0, cap=0.10):
                    x = _safe_float(val, None)
                    if x is None: return 0.0
                    return max(-cap, min(cap, (x - mid) / span * cap))
                bq *= (1.0 + _grade(batter.get("Contact")))
                bq *= (1.0 + _grade(batter.get("Power")))
                bq *= (1.0 + _grade(batter.get("Discipline")))
                bq = max(0.80, min(1.25, bq))
            except Exception:
                pass
    
            bats_val = (row.get("BatterSide") or batter.get("BatterSide") or "R")
            bats_short = "R" if str(bats_val).upper().startswith("R") else "L"
            count_tag = count_bucket(balls, strikes) or "even"
            loc = None
            try: 
                pthrows_short = (phand_code or "R")[0].upper()
                ctx_loc = {
                    "pt": pt,
                    "pthrows": pthrows_short,
                    "bats": bats_short,
                    "balls": balls,
                    "strikes": strikes,
                    "command": command,
                    "pitcher_id": cur_pitcher.get("PitcherId", "")
                }

                loc = aim_engine.sample(ctx_loc)
            except Exception:
                loc = None
            
            if loc is None:
                try:
                    
                    _roster_row = cur_pitcher.get('_RosterRow', cur_pitcher)
                    loc = sample_loc_from_roster_attrs(
                        pitch_type=pt,
                        roster_row=_roster_row,
                        batter_side=bats_short,
                        count_bucket= count_tag
                    )
                except Exception:
                    loc = None

            if loc is None:
                loc = sample_loc(pt, phand_code, command)
        
    
            ball_bias: float = float(knobs.get("ball_bias", 1.00))
    
            # Apply ball bias to adjust location distribution
            if loc is not None and ball_bias != 1.0:
                zone_x_min, zone_x_max = -PLATE_HALF_WIDTH, PLATE_HALF_WIDTH
                zone_z_min, zone_z_max = ZONE_Z_LOW, ZONE_Z_HIGH
                x_loc, z_loc = loc[0], loc[1]
                in_zone = (zone_x_min <= x_loc <= zone_x_max) and (zone_z_min <= z_loc <= zone_z_max)
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
                xw = max(-2.0, min(2.0, loc[0]))
                zw = max(0.0, min(5.0, loc[1]))
                setc("PlateLocSide", round(xw, 3))
                setc("PlateLocHeight", round(zw, 3))
    
            # --- PITCH CALL ---
            pc = None
            bundle_calls_ctx = _lookup_bundle(bundle_calls_map, pt, phand_code, bats_short, count_tag)
            bundle_usage_ctx = _lookup_bundle(bundle_usage_map, pt, phand_code, bats_short, count_tag)
            zone_table = load_zone_tbl(pt, phand_code, bats_short, count_tag)
    
            def _apply_count_bias(dist: dict[str, float], b: int, s: int) -> dict[str, float]:
                """Nudge outcome distribution by exact count."""
                try:
                    b = int(b); s = int(s)
                except Exception:
                    return dist
                m = {k: 1.0 for k in ("BallCalled","StrikeCalled","StrikeSwinging","Foul","InPlay")}
                if (b, s) == (3, 0):
                    m.update({"BallCalled": 1.25, "StrikeCalled": 0.85, "StrikeSwinging": 0.85, "Foul": 0.90, "InPlay": 0.90})
                elif (b, s) == (3, 1):
                    m.update({"BallCalled": 1.12, "StrikeSwinging": 0.95, "Foul": 0.97})
                elif (b, s) == (0, 2):
                    m.update({"BallCalled": 0.85, "StrikeSwinging": 1.15, "Foul": 1.08, "InPlay": 0.95})
                elif (b, s) == (1, 2):
                    m.update({"BallCalled": 0.90, "StrikeSwinging": 1.10, "Foul": 1.05, "InPlay": 0.97})
                elif (b, s) == (2, 0):
                    m.update({"BallCalled": 1.08, "StrikeCalled": 0.92})
                elif (b, s) == (2, 2):
                    m.update({"Foul": 1.05, "StrikeSwinging": 1.02, "InPlay": 0.98})
                elif (b, s) == (3, 2):
                    m.update({"BallCalled": 0.95, "Foul": 1.05, "InPlay": 1.03})
                out = {k: max(0.0, float(dist.get(k, 0.0))) for k in dist.keys()}
                for k, mult in m.items():
                    if k in out:
                        out[k] = max(0.0, float(out[k]) * float(mult))
                z = sum(max(0.0, float(v)) for v in out.values()) or 1.0
                return {k: (max(0.0, float(v)) / z) for k, v in out.items()}
    
            # Determine zone classification and safely unpack
            zone_table = load_zone_tbl(pt, phand_code, bats_short, count_tag)
            zone_kind = zone_key = None
            zone_d = None
            if loc is not None and zone_table:
                zk, zkey, zd = classify_zone(zone_table, loc[0], loc[1])
                zone_kind, zone_key, zone_d = zk, zkey, zd
    
            pc_dist = None
            zone_dist = None
    
            # 1) Use explicit zone table if we have a classification
            if zone_table and (zone_kind is not None) and (zone_key is not None):
                if zone_kind == "grid":
                    raw_zone = (zone_table.get("grid") or {}).get("cells", {}).get(zone_key, {})
                else:
                    raw_zone = (zone_table.get("outside") or {}).get(zone_key, {})
                pc_dist = _normalize_call_dist(raw_zone)
                if pc_dist and (zone_kind == "outside") and (zone_d is not None):
                    import math
                    g = math.exp(-float(zone_d) / 0.25)
                    pc_dist = _normalize_call_dist({k: (v if k in ("BallCalled","Ball") else v * g) for k, v in pc_dist.items()})
    
            # 2) Fallback to bundle distributions
            if (not pc_dist) and bundle_calls_ctx:
                if zone_key is not None:
                    pc_dist = _normalize_call_dist(bundle_calls_ctx.get(zone_key, {}))
                if not pc_dist:
                    pc_dist = _weighted_mix(bundle_calls_ctx, bundle_usage_ctx)
    
            # 3) Final fallback to base outcomes
            if not pc_dist:
                base_pc = safe_get(pitches_cfg, pt, "hands", phand_code, "outcomes",
                                   default={"BallCalled" : 0.42, "StrikeCalled": 0.15,
                                            "StrikeSwinging": 0.16, "Foul": 0.15, "InPlay": 0.12})
                pc_dist = _adjust_pitchcall(base_pc, batter_q=bq, pitch_cmd=command)
    
            # Also build zone_dist for sampling if we had a zone
            if zone_table and (zone_kind is not None) and (zone_key is not None):
                raw_zone = (zone_table.get("grid") or {}).get("cells", {}).get(zone_key, {}) if zone_kind == "grid" else (zone_table.get("outside") or {}).get(zone_key, {})
                zone_dist = _normalize_call_dist(raw_zone)
                if zone_kind == "outside" and (zone_d is not None) and zone_dist:
                    g = __import__("math").exp(-float(zone_d) / 0.25)
                    zone_dist = _normalize_call_dist({k: (v if k in ("BallCalled","Ball") else v * g) for k, v in zone_dist.items()})
    
            if (not zone_dist) and bundle_calls_ctx:
                if zone_key is not None:
                    zone_dist = _normalize_call_dist(bundle_calls_ctx.get(zone_key, {}))
                if not zone_dist:
                    zone_dist = _weighted_mix(bundle_calls_ctx, bundle_usage_ctx)
    
            # Apply global biases and sample
            dist_to_sample = zone_dist if zone_dist else pc_dist
            dist_to_sample = _apply_ball_bias_to_dist(dist_to_sample, ball_bias)
            dist_to_sample = _apply_count_bias(dist_to_sample, balls, strikes)
            pc = sample_categorical(dist_to_sample, rng)

            if zone_kind == "grid" and isinstance(dist_to_sample, dict):
                s_ball = float(knobs.get("in_zone_ball_scale", 0.03))
                s_strk = float(knobs.get("in_zone_strike_boost", 1.10))
                tmp = {}
                for k, v in dist_to_sample.items():
                    if k in ("BallCalled", "Ball"):
                        tmp[k] = v * s_ball
                    elif k in ("StrikeCalled",):
                        tmp[k] = v * s_strk
                    else:
                        tmp[k] = v
                z = sum(max(0.0, x) for x in tmp.values()) or 1.0
                dist_to_sample = {k: max(0.0, x) / z for k, x in tmp.items()}
    
            setc("PitchCall", pc)
    
            # Terminal / outcome path
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
                    pstat["K"] = int(pstat["K"]) + 1; team_off[batting_team]["K"] = int(team_off[batting_team]["K"]) + 1; team_off[batting_team]["AB"] = int(team_off[batting_team]["AB"]) + 1
    
            elif pc == "BallCalled":
                if balls < 3:
                    balls += 1
                else:
                    balls = 4; terminal = True
                    setc("KorBB","Walk"); setc("PlayResult","Walk")
                    scored_runners, bases = adv_walk(bases, owner_pid=pid)
                    if scored_runners:
                        runs_scored = len(scored_runners); rbi_scored = 1
                    pstat["BB"] = int(pstat["BB"]) + 1; team_off[batting_team]["BB"] = int(team_off[batting_team]["BB"]) + 1
    
            elif pc == "Foul":
                if strikes < 2: strikes += 1
    
            elif pc == "InPlay":
                terminal = True
                split = safe_get(pitches_cfg, pt, "hands", phand_code, "inplay_split",
                                 default={"Out":0.7,"Single":0.2,"Double":0.07,"Triple":0.01,"HomeRun":0.02})
                if split is None:
                    split = {"Out":0.7,"Single":0.2,"Double":0.07,"Triple":0.01,"HomeRun":0.02}
                split = _adjust_inplay_split(split, batter_q=bq)
                # Inject small error chance (ROE) by shaving from Outs
                try:
                    e = float(ERROR_RATE)
                except Exception:
                    e = 0.0
                if e > 0.0:
                    if "Out" in split:
                        split["Out"] = max(0.0, float(split["Out"]) - e)
                    split["Error"] = float(split.get("Error", 0.0)) + e
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
                            if val is not None: row[c] = str(val)
    
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
                    pstat["H"] = int(pstat["H"]) + 1; team_off[batting_team]["H"] = int(team_off[batting_team]["H"]) + 1; team_off[batting_team]["AB"] = int(team_off[batting_team]["AB"]) + 1
                    scored_runners, bases = adv_1(bases, owner_pid=pid)
    
                elif play_result == "Double":
                    pstat["H"] = int(pstat["H"]) + 1; pstat["B2"] = int(pstat["B2"]) + 1
                    team_off[batting_team]["H"] = int(team_off[batting_team]["H"]) + 1; team_off[batting_team]["B2"] = int(team_off[batting_team]["B2"]) + 1; team_off[batting_team]["AB"] = int(team_off[batting_team]["AB"]) + 1
                    scored_runners, bases = adv_2(bases, owner_pid=pid)
    
                elif play_result == "Triple":
                    pstat["H"] = int(pstat["H"]) + 1; pstat["B3"] = int(pstat["B3"]) + 1
                    team_off[batting_team]["H"] = int(team_off[batting_team]["H"]) + 1; team_off[batting_team]["B3"] = int(team_off[batting_team]["B3"]) + 1; team_off[batting_team]["AB"] = int(team_off[batting_team]["AB"]) + 1
                    scored_runners, bases = adv_3(bases, owner_pid=pid)
    
                elif play_result == "HomeRun":
                    pstat["H"] = int(pstat["H"]) + 1; pstat["HR"] = int(pstat["HR"]) + 1
                    team_off[batting_team]["H"] = int(team_off[batting_team]["H"]) + 1; team_off[batting_team]["HR"] = int(team_off[batting_team]["HR"]) + 1; team_off[batting_team]["AB"] = int(team_off[batting_team]["AB"]) + 1
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
            hit_type = repair_batted_ball_fields(row, play_result or "Out", hit_type, rng)
            if "HitType" in out_cols and hit_type is not None:
                row["HitType"] = hit_type
    
            if scored_runners:
                for r in scored_runners:
                    if not r: continue
                    r_owner = r["pid"]
                    pbox[r_owner]["R"] = int(pbox[r_owner]["R"]) + 1
                    pbox[r_owner]["ER"] = int(pbox[r_owner]["ER"]) + 1
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
                row["RelSpeed"] = str(round(float(val_rel) - VELO_LOSS_PER_OVER10 * over10, 2))
            val_spin = row.get("SpinRate", None)
            if "SpinRate" in row and (val_spin is not None) and (str(val_spin) != ""):
                row["SpinRate"] = str(round(max(0.0, float(val_spin) - SPIN_LOSS_PER_OVER10 * over10), 0))
    
            # Opportunistic stolen base attempts on non-terminal pitches
            if not terminal:
                def _attempt_steal_second(bases_local):
                    nonlocal outs
                    if not bases_local[0]:
                        return bases_local
                    if outs >= 2:
                        return bases_local
                    # Per-pitch attempt
                    if rng.random() < SB_ATTEMPT_R1_BASE:
                        cinfo = home_c if fielding_team == home_team_key else away_c
                        thr = (cinfo.get("CatcherThrows") or "").strip().upper()
                        p_succ = SB_SUCCESS_BASE + (SB_CATCHER_R_BONUS if thr.startswith("R") else 0.0)
                        if rng.random() < p_succ:
                            # SB of 2nd
                            bases_local = [None, bases_local[0], bases_local[2]]
                            if "Notes" in out_cols:
                                prev = row.get("Notes", "")
                                row["Notes"] = (prev + ";" if prev else "") + "SB2"
                        else:
                            # Caught stealing second
                            outs += 1
                            bases_local = [None, bases_local[1], bases_local[2]]
                            if "Notes" in out_cols:
                                prev = row.get("Notes", "")
                                row["Notes"] = (prev + ";" if prev else "") + "CS2"
                    return bases_local
    
                def _attempt_steal_third(bases_local):
                    nonlocal outs
                    if not bases_local[1]:
                        return bases_local
                    if outs >= 2:
                        return bases_local
                    if rng.random() < SB_ATTEMPT_R2_BASE:
                        cinfo = home_c if fielding_team == home_team_key else away_c
                        thr = (cinfo.get("CatcherThrows") or "").strip().upper()
                        p_succ = (SB_SUCCESS_BASE - 0.05) + (SB_CATCHER_R_BONUS if thr.startswith("R") else 0.0)
                        if rng.random() < p_succ:
                            bases_local = [bases_local[0], None, bases_local[1]]
                            if "Notes" in out_cols:
                                prev = row.get("Notes", "")
                                row["Notes"] = (prev + ";" if prev else "") + "SB3"
                        else:
                            outs += 1
                            bases_local = [bases_local[0], None, bases_local[2]]
                            if "Notes" in out_cols:
                                prev = row.get("Notes", "")
                                row["Notes"] = (prev + ";" if prev else "") + "CS3"
                    return bases_local
    
                # Try steals in a fixed order to avoid cascading attempts same pitch
                bases = _attempt_steal_second(bases)
                bases = _attempt_steal_third(bases)
    
            rows.append(row)
            current_dt += timedelta(seconds=5)
    
            end_now = False
            if (not is_top) and inn >= 9 and (score["home"] + runs_this_half) > score["away"]:
                end_now = True
                if "WalkOff" in out_cols: row["WalkOff"] = "1"
    
            cur_pitcher["_PitchCount"] += 1
            cur_pitcher["_PitchesThisHalf"] += 1
            pstat["Pitches"] = int(pstat["Pitches"]) + 1
    
            if end_now:
                game_state["over"] = True
                break
    
            terminal_by_rule = (
                (row.get("KorBB") == "Walk") or
                (pc == "InPlay") or
                (pc in ("StrikeSwinging","StrikeCalled") and (strikes == 3))
            )
            if terminal_by_rule:
                pstat["BF"] = int(pstat["BF"]) + 1
                cur_pitcher["_BF"] += 1
                team_off[batting_team]["PA"] += 1
    
            if terminal:
                outs_awarded = 0
                if row.get("KorBB") == "Strikeout":
                    outs_awarded = 1
                elif row.get("PlayResult") in ("Out","Sacrifice","FielderChoice"):
                    try: outs_awarded = int(row.get("OutsOnPlay") or 1)
                    except Exception: outs_awarded = 1
                pstat["IPOuts"] = int(pstat["IPOuts"]) + outs_awarded
    
                balls, strikes = 0, 0
                paofinning += 1
                pitchofpa = 1
                batter["_TimesFaced"] = str(int(batter["_TimesFaced"]) + 1)
    
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
        if game_state["over"]: break
    
        cur_home, home_li = half_inning(False, inn, cur_home, home_li)
        if game_state["over"]: break
    
        if inn >= 9 and score["home"] != score["away"]: break
        inn += 1
    
    usage: dict[str, Any] = {
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
        if c not in df.columns:
            df[c] = pd.NA
    # Reorder columns and ensure it's a DataFrame
    result_df: pd.DataFrame = df.reindex(columns=out_cols)
    
    return result_df, score, usage, dict(pbox), dict(team_off)
