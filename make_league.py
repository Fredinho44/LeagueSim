#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_league.py
--------------
Generate a 13-organization league with 65 athletes per org, split into:
- Majors (26 players),
- AAA (26 players),
- Rookie (13 players).

Outputs CSVs:
- organizations.csv
- teams.csv
- rosters.csv
- schedule.csv

Schedules:
- Majors: 30 games/team (13 teams, odd → byes handled)
- AAA:    30 games/team
- Rookie: 20 games/team

No external deps. Python 3.8+.

ADDED:
- Height/Wingspan for all players
- Pitchers get RelHeight_ft, RelSide_ft, Extension_ft derived from body dims

This version also adds:
- 20–80 scouting grades (pitchers & hitters)
- Derived FV/Role (e.g., 40/45/50/55/60/70/80; Role 4..8 style tags)
- PitchingWeight / HittingWeight scalars for downstream model weighting
- Usage/command/velo biasing from scouting grades
"""

import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Sequence
from collections import defaultdict

# -----------------------------
# Config defaults (CLI overrides)
# -----------------------------
DEFAULT_NUM_ORGS = 13
ATHLETES_PER_ORG = 65
SPLIT_PER_TIER = {"Majors": 26, "AAA": 26, "Rookie": 13}
GAMES_PER_TEAM = {"Majors": 30, "AAA": 30, "Rookie": 20}
START_DATE = date(2025, 4, 1)
SEED = 7

# Player mix targets
PITCH_HAND_P = {"Right": 0.70, "Left": 0.30}
BAT_SIDE_P   = {"Right": 0.55, "Left": 0.35, "Switch": 0.10}

# Pitcher clusters → rough pitch-usage priors (illustrative)
PITCH_CLUSTERS = {
    "PowerFB":      {"Fastball": 0.55, "Slider": 0.25, "Curveball": 0.10, "Changeup": 0.10},
    "SinkerSlider": {"Sinker": 0.45, "Slider": 0.35, "Changeup": 0.15, "Curveball": 0.05},
    "CutterMix":    {"Cutter": 0.45, "Fastball": 0.25, "Slider": 0.20, "Changeup": 0.10},
    "ChangeCmd":    {"Fastball": 0.35, "Changeup": 0.35, "Slider": 0.20, "Curveball": 0.10},
    "BreakHeavy":   {"Curveball": 0.35, "Slider": 0.35, "Fastball": 0.20, "Changeup": 0.10},
}
CLUSTER_WEIGHTS_R = {"PowerFB": 0.34, "SinkerSlider": 0.28, "CutterMix": 0.18, "ChangeCmd": 0.10, "BreakHeavy": 0.10}
CLUSTER_WEIGHTS_L = {"PowerFB": 0.22, "SinkerSlider": 0.22, "CutterMix": 0.18, "ChangeCmd": 0.18, "BreakHeavy": 0.20}

# Simple name banks (deterministic w/ seed)
CITIES = [
    "Orion", "Atlas", "Sierra", "Zenith", "Vertex", "Peregrine", "Summit",
    "Thunder", "Phoenix", "Apex", "Liberty", "Frontier", "Mariner", "Cascade",
    "Pioneer", "Cobalt", "Aurora", "Sterling", "Falcon", "Horizon"
]
MASCOTS = [
    "Aces","Pilots","Comets","Voyagers","Rangers","Captains","Cyclones",
    "Admirals","Sentinels","Titans","Pioneers","Stallions","Tempest","Anchors",
    "Lynx","Hawks","Wolves","Spartans","Knights","Panthers"
]
FIRSTS = [
    "Alex","Jordan","Chris","Taylor","Drew","Riley","Jesse","Logan","Cameron","Morgan",
    "Hayden","Parker","Quinn","Reese","Rowan","Avery","Casey","Elliot","Finley","Dakota",
    "Charlie","Sawyer","Micah","Kai","Archer","Nolan","Mason","Evan","Cole","Luca"
]
LASTS = [
    "Carter","Brooks","Hayes","Bennett","Jensen","Carson","Reed","Holland","Manning","Moss",
    "Spencer","Cooper","Hudson","Kennedy","Marshall","Porter","Lawson","Parker","Quinn","Ramsey",
    "Sawyer","Walker","Young","Bishop","Clayton","Douglas","Ellis","Foster","Griffin","Hunter"
]
POSITIONS = ["C","1B","2B","3B","SS","LF","CF","RF","DH"]  # 9 starters; bench will be utilities

# -----------------------------
# 20–80 Scouting utilities (NEW)
# -----------------------------
def _gmult(grade: float, per10: float = 0.08, lo: float = 0.85, hi: float = 1.25) -> float:
    """Convert a 20–80 grade to a multiplicative weight centered at 50 → 1.0."""
    if grade is None:
        return 1.0
    w = 1.0 + ((float(grade) - 50.0) / 10.0) * per10
    return max(lo, min(hi, w))

def _round5(x: float) -> int:
    return int(round(x / 5.0) * 5)

def _fv_to_role(fv: int) -> str:
    return {80:"8",70:"7",60:"6",55:"5/55",50:"5",45:"4.5",40:"4"}.get(int(fv), str(fv))

# Cluster pitch bumps in grade points
_CLUSTER_PITCH_BUMPS = {
    "PowerFB":      {"FB": +6, "SL": +4, "CB": +1, "CH":  0, "CT": +1, "SI":  0, "SPL": 0},
    "SinkerSlider": {"FB": +1, "SL": +5, "CB":  0, "CH": +1, "CT":  0, "SI": +6, "SPL": 0},
    "CutterMix":    {"FB": +1, "SL": +3, "CB":  0, "CH":  0, "CT": +6, "SI": +1, "SPL": 0},
    "ChangeCmd":    {"FB": -1, "SL":  0, "CB":  0, "CH": +7, "CT":  0, "SI":  0, "SPL": 0},
    "BreakHeavy":   {"FB":  0, "SL": +4, "CB": +7, "CH":  0, "CT":  0, "SI":  0, "SPL": 0},
}

def _tier_grade_center(tier: str) -> int:
    return {"Rookie": 45, "AAA": 50, "Majors": 55}.get(tier, 50)

def _sample_pitcher_grades(tier: str, cluster: str, role: str, cmd_tier: float, rng: random.Random) -> dict:
    base = _tier_grade_center(tier)
    sd   = 5.0
    bumps = _CLUSTER_PITCH_BUMPS.get(cluster, {})

    def g(mean, bump=0):
        return int(round(max(30, min(80, rng.normalvariate(mean + bump, sd)))))

    grades = {
        "FB":  g(base, bumps.get("FB", 0)),
        "SL":  g(base, bumps.get("SL", 0)),
        "CB":  g(base, bumps.get("CB", 0)),
        "CH":  g(base, bumps.get("CH", 0)),
        "CT":  g(base, bumps.get("CT", 0)),
        "SI":  g(base, bumps.get("SI", 0)),
        "SPL": g(base, bumps.get("SPL", 0)),
    }
    # Command grade from CommandTier: +/-5 grade per 0.10 (tunable)
    cmd_grade = int(round(max(30, min(80, 50 + ((cmd_tier - 1.00) / 0.10) * 5.0))))
    grades["CMD"] = cmd_grade

    # Stuff index: avg of best two pitch grades
    pitch_only = [grades[k] for k in ["FB","SL","CB","CH","CT","SI","SPL"]]
    pitch_only.sort(reverse=True)
    stuff_index = sum(pitch_only[:2]) / 2.0 if pitch_only else 50.0

    # Role weighting
    alpha, beta = (0.45, 0.55) if role == "SP" else (0.35, 0.65)
    fv = _round5(alpha * cmd_grade + beta * stuff_index)
    fv = int(max(40, min(80, fv)))

    grades["_FV"] = fv
    grades["_Role"] = _fv_to_role(fv)
    return grades

def _sample_hitter_grades(tier: str, archetype: str, rng: random.Random) -> dict:
    base = _tier_grade_center(tier)
    sd   = 5.0
    bumps = {
        "Balanced": {"Hit": +1, "GamePower": +1, "RawPower": +1, "Speed": 0,  "Field": 0,  "Arm": 0},
        "Contact":  {"Hit": +6, "GamePower": -2, "RawPower": -2, "Speed": +1, "Field": +2, "Arm": 0},
        "Power":    {"Hit": -2, "GamePower": +7, "RawPower": +7, "Speed": -1, "Field": 0,  "Arm": 0},
        "Speed":    {"Hit": +1, "GamePower": -1, "RawPower": -2, "Speed": +8, "Field": +2, "Arm": 0},
        "OBP":      {"Hit": +4, "GamePower": +1, "RawPower":  0, "Speed": +2, "Field": 0,  "Arm": 0},
    }.get(archetype or "Balanced", {"Hit":0,"GamePower":0,"RawPower":0,"Speed":0,"Field":0,"Arm":0})

    def g(mean, bump=0):
        return int(round(max(30, min(80, rng.normalvariate(mean + bump, sd)))))

    grades = {
        "Hit":       g(base, bumps["Hit"]),
        "GamePower": g(base, bumps["GamePower"]),
        "RawPower":  g(base, bumps["RawPower"]),
        "Speed":     g(base, bumps["Speed"]),
        "Field":     g(base, bumps["Field"]),
        "Arm":       g(base, bumps["Arm"]),
    }
    fv = _round5(0.55*grades["Hit"] + 0.30*grades["GamePower"] + 0.10*grades["Field"] + 0.05*grades["Speed"])
    fv = int(max(40, min(80, fv)))
    grades["_FV"] = fv
    grades["_Role"] = _fv_to_role(fv)
    return grades

def _bias_pitch_usage_by_grades(means: dict, grades: dict) -> dict:
    keymap = {"FB":"Fastball","SL":"Slider","CB":"Curveball","CH":"Changeup","CT":"Cutter","SI":"Sinker","SPL":"Splitter"}
    weights = {}
    for pitch_name, mean_p in means.items():
        gkey = None
        for gk, pname in keymap.items():
            if pname == pitch_name:
                gkey = gk; break
        bump = 1.0
        if gkey and gkey in grades:
            bump = 1.0 + 0.04 * ((grades[gkey] - 50.0) / 10.0)  # +4% per +10 grade
        weights[pitch_name] = max(0.0, float(mean_p)) * bump
    z = sum(weights.values()) or 1.0
    return {k: v/z for k, v in weights.items()}

def _compute_pitching_weight(gr: dict, role: str) -> float:
    stuff = sorted([gr.get(k,50) for k in ["FB","SL","CB","CH","CT","SI","SPL"]], reverse=True)[:2]
    stuff = sum(stuff)/2.0 if stuff else 50.0
    cmd   = gr.get("CMD", 50)
    if role == "SP":
        return max(0.85, min(1.25, _gmult(cmd, 0.06) * _gmult(stuff, 0.06)))
    else:
        return max(0.85, min(1.25, _gmult(stuff, 0.08) * _gmult(cmd, 0.04)))

def _compute_hitting_weight(gr: dict) -> float:
    return max(0.85, min(1.25,
           _gmult(gr.get("Hit",50), 0.07) *
           _gmult(gr.get("GamePower",50), 0.05) *
           _gmult(gr.get("Speed",50), 0.02)))

# --- Excel-safe role label helper (prevents Excel "May-55") ---
def _format_scout_role(present, future, excel_safe: bool = True) -> str:
    # coerce; ignore non-numeric present
    try:
        p = float(present) if present is not None else None
    except (TypeError, ValueError):
        p = None
    f = int(future) if future is not None else None

    if f is None and p is None:
        s = ""
    elif p is None:
        s = f"{f}"
    else:
        s = f"{p:g}/{f}"

    return ("\u200B" + s) if (excel_safe and s) else s



# --- Lightweight samplers for present role & future FV ---
# You can tune these weights later; they match the 4, 4.5, 5, 6 + 45–65 ranges you’ve seen.

_ROLE_PRESENT_PRIORS = {
    # SP skews a bit higher than RP; Rookie skews lower than Majors
    ("Rookie", "SP"): {4: 0.40, 4.5: 0.40, 5: 0.18, 6: 0.02},
    ("Rookie", "RP"): {4: 0.50, 4.5: 0.35, 5: 0.14, 6: 0.01},
    ("AAA",    "SP"): {4: 0.15, 4.5: 0.40, 5: 0.38, 6: 0.07},
    ("AAA",    "RP"): {4: 0.20, 4.5: 0.45, 5: 0.30, 6: 0.05},
    ("Majors", "SP"): {4: 0.05, 4.5: 0.20, 5: 0.55, 6: 0.20},
    ("Majors", "RP"): {4: 0.08, 4.5: 0.32, 5: 0.48, 6: 0.12},
}

# Future FV by tier (independent of role unless you want to split)
_FV_PRIORS = {
    "Rookie": {40: 0.15, 45: 0.45, 50: 0.30, 55: 0.08, 60: 0.02, 65: 0.00},
    "AAA":    {40: 0.05, 45: 0.25, 50: 0.40, 55: 0.22, 60: 0.07, 65: 0.01},
    "Majors": {40: 0.02, 45: 0.12, 50: 0.36, 55: 0.36, 60: 0.12, 65: 0.02},
}

def _wchoice(d: dict, rng) -> float | int:
    keys, wts = list(d.keys()), list(d.values())
    total = sum(wts)
    r = rng.random() * total
    acc = 0.0
    for k, w in zip(keys, wts):
        acc += w
        if r <= acc:
            return k
    return keys[-1]

def sample_present_role(tier: str, role: str, rng) -> float:
    # role is "SP" or "RP"
    return float(_wchoice(_ROLE_PRESENT_PRIORS[(tier, role)], rng))

def sample_future_fv(tier: str, rng) -> int:
    return int(_wchoice(_FV_PRIORS[tier], rng))



# -----------------------------
# Ages by tier + growth curve
# -----------------------------
AGE_RANGES = {
    "Majors": (18, 21),
    "AAA":    (16, 18),
    "Rookie": (13, 15),
}

GROWTH_TABLE = {
    13: 0.92, 14: 0.95, 15: 0.96, 16: 0.98,
    17: 0.99, 18: 1.00, 19: 1.00, 20: 1.00, 21: 1.00
}

def _growth_fraction(age: int, rng: random.Random) -> float:
    base = GROWTH_TABLE.get(max(13, min(21, age))),
    frac = base[0] if isinstance(base, tuple) else base
    return max(0.88, min(1.02, frac + rng.normalvariate(0.0, 0.005)))

def _sample_age_for_tier(tier: str, rng: random.Random) -> int:
    lo, hi = AGE_RANGES[tier]
    a = min(hi, max(lo, int(round(lo + (hi - lo) * max(rng.random(), rng.random())))))
    return a

def age_factor_from_age(age):
    try:
        a = float(age)
    except Exception:
        return 1.0
    if a <= 13:
        f = 0.92
    elif a >= 18:
        f = 1.00
    else:
        f = 0.92 + (a - 13) * (0.08 / 5.0)
    if a > 18:
        f = min(1.03, f + 0.01 * min(a - 18, 3))
    return max(0.85, min(1.05, f))

# -----------------------------
# Body/Release geometry helpers
# -----------------------------

ARM_BUCKETS = [
    ("Submarine", 0, 25),
    ("Sidearm",   25, 45),
    ("Low34",     45, 60),
    ("High34",    60, 75),
    ("Overhand",  75, 91),
]

ARM_PRIORS = {
    "PowerFB":      {"Overhand":0.18, "High34":0.48, "Low34":0.26, "Sidearm":0.06, "Submarine":0.02},
    "SinkerSlider": {"Overhand":0.06, "High34":0.30, "Low34":0.40, "Sidearm":0.20, "Submarine":0.04},
    "CutterMix":    {"Overhand":0.10, "High34":0.40, "Low34":0.34, "Sidearm":0.12, "Submarine":0.04},
    "ChangeCmd":    {"Overhand":0.12, "High34":0.46, "Low34":0.30, "Sidearm":0.10, "Submarine":0.02},
    "BreakHeavy":   {"Overhand":0.22, "High34":0.46, "Low34":0.24, "Sidearm":0.06, "Submarine":0.02},
}

def _choose_arm_bucket(cluster: str, rng: random.Random) -> str:
    p = ARM_PRIORS.get(cluster, {"Overhand":0.2,"High34":0.4,"Low34":0.3,"Sidearm":0.08,"Submarine":0.02})
    return choice_weighted(p, rng)

def _sample_angle_in_bucket(bucket: str, rng: random.Random) -> float:
    for name, lo, hi in ARM_BUCKETS:
        if name == bucket:
            mid = 0.5*(lo+hi)
            u = (rng.random()+rng.random())/2.0
            ang = lo + (hi-lo)*u
            ang = 0.7*ang + 0.3*mid
            return round(ang, 1)
    return round(rng.uniform(62, 73), 1)

def _bucket_from_angle(angle_deg: float) -> str:
    for name, lo, hi in ARM_BUCKETS:
        if lo <= angle_deg < hi:
            return name
    return "High34"

def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

def _apply_arm_slot_to_geometry(angle_deg: float,
                                throws: str,
                                rel_z_ft: float,
                                rel_x_ft: float,
                                ext_ft: float,
                                rng: random.Random) -> Tuple[float,float,float]:
    a = max(0.0, min(90.0, angle_deg))
    overhand_factor = a / 90.0
    sidearm_factor  = 1.0 - overhand_factor

    rel_z_ft += (0.35 * (overhand_factor - 0.5)) + rng.normalvariate(0.0, 0.03)

    mag = abs(rel_x_ft)
    mag += (0.50 * (sidearm_factor - 0.5)) + rng.normalvariate(0.0, 0.05)
    mag = _clip(mag, 0.5, 3.8)
    rel_x_ft = mag if throws == "Right" else -mag

    ext_ft += 0.10 * (overhand_factor - 0.5) + rng.normalvariate(0.0, 0.03)

    rel_z_ft = _clip(rel_z_ft, 4.6, 7.1)
    ext_ft   = _clip(ext_ft, 4.6, 7.8)
    return (round(rel_z_ft,3), round(rel_x_ft,3), round(ext_ft,3))

def _height_inches_for(role: str, tier: str, age: int, rng: random.Random) -> int:
    adult_base = {"Rookie": 72.5, "AAA": 73.5, "Majors": 74.2}[tier]
    role_bump = 0.8 if role in ("SP", "RP") else 0.0
    adult_target = adult_base + role_bump

    frac = _growth_fraction(age, rng)
    h = rng.normalvariate(adult_target * frac, 1.4 if age < 18 else 1.75)

    lo = 66.0 if age <= 15 else 68.0
    hi = 80.0 if age >= 18 else 78.0
    return int(round(_clip(h, lo, hi)))

def _wingspan_inches_from_height(h_in: int, rng: random.Random) -> int:
    w = rng.normalvariate(h_in + 2.0, 2.5)
    return int(round(_clip(w, h_in - 2.0, h_in + 6.0)))

def _geom_from_body(throws: str, height_in: int, wingspan_in: int, rng: random.Random) -> Tuple[float,float,float]:
    sign = 1.0 if throws == "Right" else -1.0
    h_ft = height_in / 12.0
    w_ft = wingspan_in / 12.0
    ape_ft = (wingspan_in - height_in) / 12.0

    rel_z = 0.36 * h_ft + rng.normalvariate(0.0, 0.15)
    rel_z = _clip(rel_z, 4.6, 7.1)

    ext = 0.27 * h_ft + 0.12 * ape_ft + rng.normalvariate(0.0, 0.20)
    ext = _clip(ext, 4.8, 7.6)

    rel_x_mag = 0.12 * w_ft + rng.normalvariate(0.0, 0.15)
    rel_x_mag = _clip(abs(rel_x_mag), 0.6, 3.6)
    rel_x = sign * rel_x_mag
    return (round(rel_z, 3), round(rel_x, 3), round(ext, 3))

# -----------------------------
# CommandTier by tier / age
# -----------------------------
_CMD_TIER_PARAMS = {
    "Rookie": {"mu": 0.94, "sd": 0.17, "clip": (0.60, 1.25)},
    "AAA":    {"mu": 0.97, "sd": 0.13, "clip": (0.70, 1.22)},
    "Majors": {"mu": 1.00, "sd": 0.09, "clip": (0.80, 1.18)},
}

_CMD_MIX = {
    "Rookie": {"high": 1.05, "low": 0.92, "min_high": 1, "min_low": 1},
    "AAA":    {"high": 1.05, "low": 0.95, "min_high": 2, "min_low": 2},
    "Majors": {"high": 1.06, "low": 0.96, "min_high": 2, "min_low": 2},
}

def _sample_command_tier(tier: str, age: int, role: str, rng: random.Random) -> float:
    p = _CMD_TIER_PARAMS[tier]
    mu = p["mu"]
    sd = p["sd"]

    mu += 0.008 * (age - 18)
    sd *= 1.0 - 0.03 * (age - 16)
    sd = max(0.05, sd)
    if role == "SP":
        sd *= 0.95
    else:
        sd *= 1.08

    val = rng.normalvariate(mu, sd)
    lo, hi = p["clip"]
    return round(max(lo, min(hi, val)), 3)

def _enforce_team_command_mix(cmds: List[float], tier: str, rng: random.Random) -> List[float]:
    th = _CMD_MIX[tier]
    hi_th, lo_th = th["high"], th["low"]
    need_hi, need_lo = th["min_high"], th["min_low"]

    count_hi = sum(c >= hi_th for c in cmds)
    count_lo = sum(c <= lo_th for c in cmds)

    lo_clip, hi_clip = _CMD_TIER_PARAMS[tier]["clip"]

    if count_hi < need_hi:
        cands = sorted([(hi_th - c, idx) for idx, c in enumerate(cmds) if c < hi_th])
        for _, idx in cands[: max(0, need_hi - count_hi)]:
            cmds[idx] = min(hi_clip, hi_th + rng.uniform(0.00, 0.04))

    if count_lo < need_lo:
        cands = sorted([(c - lo_th, idx) for idx, c in enumerate(cmds) if c > lo_th])
        for _, idx in cands[: max(0, need_lo - count_lo)]:
            cmds[idx] = max(lo_clip, lo_th - rng.uniform(0.00, 0.04))

    return [round(c, 3) for c in cmds]

# -----------------------------
# Utility sampling
# -----------------------------
def choice_weighted(d: Dict[str, float], rng: random.Random) -> str:
    keys, wts = list(d.keys()), list(d.values())
    total = sum(wts)
    if total <= 0:
        return rng.choice(keys)
    r = rng.random() * total
    acc = 0.0
    for k, w in zip(keys, wts):
        acc += max(0.0, w)
        if r <= acc:
            return k
    return keys[-1]

def dirichlet_from_means(means: Dict[str, float], alpha_total: float, rng: random.Random) -> Dict[str, float]:
    keys = list(means.keys())
    vals = [max(0.0, float(means[k])) for k in keys]
    s = sum(vals)
    vals = [v/s for v in vals] if s > 0 else [1.0/len(keys)] * len(keys)
    draws = [random.gammavariate(max(1e-6, v * alpha_total), 1.0) for v in vals]
    total = sum(draws)
    return {k: d/total for k, d in zip(keys, draws)}

def generate_name(rng: random.Random) -> Tuple[str, str, str]:
    f = rng.choice(FIRSTS); l = rng.choice(LASTS)
    return f, l, f + " " + l

# -----------------------------
# Two-way / secondary position knobs
# -----------------------------
TWO_WAY_RATE = {
    "Rookie": 0.22,
    "AAA":    0.10,
    "Majors": 0.04,
}
TWO_WAY_PITCH_ROLE = {"RP": 0.75, "SP": 0.25}
SECONDARY_DEF_POOL = ["C","1B","2B","3B","SS","LF","CF","RF","UT","IF","OF"]

def _pick_secondary_position(primary: str, rng: random.Random) -> str:
    choices = [p for p in SECONDARY_DEF_POOL if p != primary]
    if primary in ("SS","2B"):
        bias = ["3B","OF","UT"]
    elif primary in ("LF","RF","CF"):
        bias = ["1B","3B","UT"]
    elif primary == "C":
        bias = ["1B","UT","OF"]
    else:
        bias = ["OF","UT","2B"]
    pool = choices + bias
    return rng.choice(pool)

def _pick_two_way_role(rng: random.Random) -> str:
    return choice_weighted(TWO_WAY_PITCH_ROLE, rng)

# -----------------------------
# Durability / workload priors
# -----------------------------
DURABILITY_PRIORS = {
    "Rookie": {
        "SP": {"stam_mu": 58, "stam_sd": 10, "pc_mu": 55, "pc_sd": 8, "ip_mu": 32, "ip_sd": 10},
        "RP": {"stam_mu": 48, "stam_sd": 10, "pc_mu": 35, "pc_sd": 6, "ip_mu": 22, "ip_sd": 8},
    },
    "AAA": {
        "SP": {"stam_mu": 66, "stam_sd": 9,  "pc_mu": 80, "pc_sd": 10, "ip_mu": 62, "ip_sd": 15},
        "RP": {"stam_mu": 54, "stam_sd": 9,  "pc_mu": 45, "pc_sd": 8,  "ip_mu": 40, "ip_sd": 12},
    },
    "Majors": {
        "SP": {"stam_mu": 74, "stam_sd": 8,  "pc_mu": 95, "pc_sd": 12, "ip_mu": 85, "ip_sd": 20},
        "RP": {"stam_mu": 60, "stam_sd": 8,  "pc_mu": 55, "pc_sd": 8,  "ip_mu": 55, "ip_sd": 15},
    },
}

_VELO_PRIORS = {
    "Majors":   {"PowerFB": (90, 2.5), "SinkerSlider": (88, 2.2), "CutterMix": (86, 2.0), "ChangeCmd": (86, 2.8), "BreakHeavy": (83, 1.8)},
    "AAA":      {"PowerFB": (88, 2.5), "SinkerSlider": (86, 2.2), "CutterMix": (83, 2.0), "ChangeCmd": (84, 2.8), "BreakHeavy": (80, 1.8)},
    "Rookie":   {"PowerFB": (84, 2.5), "SinkerSlider": (83, 2.2), "CutterMix": (78, 2.0), "ChangeCmd": (76, 2.8), "BreakHeavy": (74, 1.8)},
}

def _sample_avg_velo_by_cluster(tier: str, cluster: str, age: int, rng: random.Random) -> float:
    mu, sd = _VELO_PRIORS.get(tier, _VELO_PRIORS["AAA"]).get(cluster, (87, 2.0))
    age_adj = 0.15 * max(0, min(20, age) - 16)
    v = rng.normalvariate(mu + age_adj, sd)
    return round(max(72.0, min(99.5, v)), 1)

def _bounded_int(mu: float, sd: float, lo: int, hi: int, rng: random.Random) -> int:
    return int(round(max(lo, min(hi, rng.normalvariate(mu, sd)))))

def _clipf(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _sample_prev_ip(tier: str, role: str, age: int, rng: random.Random) -> int:
    p = DURABILITY_PRIORS[tier][role]
    age_scale = 1.0 + 0.03 * (age - {"Rookie":14, "AAA":17, "Majors":19}[tier])
    mu = p["ip_mu"] * _clipf(age_scale, 0.75, 1.25)
    sd = p["ip_sd"]
    lo, hi = (10, 120) if role == "SP" else (5, 90)
    return _bounded_int(mu, sd, lo, hi, rng)

def _sample_stamina_and_limits(tier: str, role: str, age: int, cmd: float, avg_velo: float, rng: random.Random):
    p = DURABILITY_PRIORS[tier][role]
    stam = rng.normalvariate(p["stam_mu"], p["stam_sd"])
    stam += 0.6 * (age - {"Rookie":15, "AAA":17, "Majors":19}[tier])
    stam += 6.0 * (cmd - 1.00)
    stam -= 0.35 * max(0.0, avg_velo - {"Rookie":83, "AAA":87, "Majors":91}[tier])
    stam = int(round(_clipf(stam, 30, 95)))

    pc = rng.normalvariate(p["pc_mu"], p["pc_sd"])
    pc *= (0.80 + 0.004 * (stam - 50))
    pc = int(round(_clipf(pc, 25 if role == "RP" else 50, 120 if role == "SP" else 70)))

    if role == "SP":
        ppi = 15.5 - 2.0 * (cmd - 1.0)
        exp_ip = _clipf(stam / 16.0, 2.5, 6.8)
        avg_pitches = int(round(_clipf(ppi * exp_ip, 40, pc)))
        exp_bf = int(round(_clipf(3.8 * exp_ip, 9, 28)))
    else:
        avg_pitches = int(round(_clipf(12 + 20 * (stam / 100.0), 8, pc)))
        exp_bf = int(round(_clipf(0.8 * avg_pitches, 3, 12)))

    rec = 1 if role == "RP" else 4
    rec += 1 if avg_pitches > (55 if role == "SP" else 25) else 0
    rec += 1 if avg_velo > (92 if tier == "Majors" else 88) else 0
    rec = int(_clipf(rec, 1, 6))
    return stam, pc, avg_pitches, exp_bf, rec

def _sample_injury_flag(tier: str, role: str, age: int, avg_velo: float, rng: random.Random) -> int:
    base = {"Rookie": 0.03, "AAA": 0.05, "Majors": 0.07}[tier]
    velo_tax = 0.03 if avg_velo > {"Rookie":86, "AAA":90, "Majors":94}[tier] else 0.0
    age_tax = 0.02 if (tier != "Rookie" and age >= 20) else 0.0
    p = _clipf(base + velo_tax + age_tax, 0.01, 0.15)
    return 1 if random.random() < p else 0

# -----------------------------
# Orgs / teams / rosters
# -----------------------------
def build_organizations(num_orgs: int, rng: random.Random) -> List[Dict]:
    orgs = []
    used = set()
    for i in range(num_orgs):
        city = rng.choice(CITIES)
        mascot = rng.choice(MASCOTS)
        name = f"{city} {mascot}"
        while name in used:
            city = rng.choice(CITIES); mascot = rng.choice(MASCOTS)
            name = f"{city} {mascot}"
        used.add(name)
        orgs.append({"OrgID": i+1, "OrgName": name})
    return orgs

def build_teams(orgs: List[Dict]) -> List[Dict]:
    teams = []
    tid = 1
    for org in orgs:
        for tier in ["Majors", "AAA", "Rookie"]:
            team_name = f"{org['OrgName']} {tier}"
            teams.append({
                "TeamID": tid,
                "TeamName": team_name,
                "OrgID": org["OrgID"],
                "OrgName": org["OrgName"],
                "Tier": tier,
            })
            tid += 1
    return teams

def _pick_bats(rng: random.Random) -> str:
    return choice_weighted(BAT_SIDE_P, rng)

def _pick_throws(rng: random.Random) -> str:
    return choice_weighted(PITCH_HAND_P, rng)

def _pick_pitcher_cluster(throws: str, rng: random.Random) -> str:
    pool = CLUSTER_WEIGHTS_L if throws == "Left" else CLUSTER_WEIGHTS_R
    return choice_weighted(pool, rng)

def build_roster_for_team(team: Dict, rng: random.Random) -> List[Dict]:
    n = SPLIT_PER_TIER[team["Tier"]]
    players: List[Dict] = []

    if team["Tier"] in ("Majors", "AAA"):
        n_pitch = 13  # 5 SP + 8 RP
        n_hit = n - n_pitch  # 13
        n_sp = 5
    else:
        n_pitch = 6   # 2 SP + 4 RP
        n_hit = n - n_pitch  # 7
        n_sp = 2

    # -----------------
    # Pitchers
    # -----------------
    role_list = ["SP"] * n_sp + ["RP"] * (n_pitch - n_sp)
    ages = [_sample_age_for_tier(team["Tier"], rng) for _ in range(n_pitch)]

    cmd_list = [
        _sample_command_tier(team["Tier"], ages[i], role_list[i], rng)
        for i in range(n_pitch)
    ]
    cmd_list = _enforce_team_command_mix(cmd_list, team["Tier"], rng)

    for i in range(n_pitch):
        age = ages[i]
        role = role_list[i]

        throws = _pick_throws(rng)
        bats = _pick_bats(rng)
        cluster = _pick_pitcher_cluster(throws, rng)

        # Baseline command
        cmd = cmd_list[i]

        # per-pitch command factors (pre-scouting)
        pitch_types = list(PITCH_CLUSTERS[cluster].keys())
        sorted_pitches = sorted(pitch_types, key=lambda k: PITCH_CLUSTERS[cluster][k], reverse=True)
        per_pitch_factors: Dict[str, float] = {}
        for idx, pt in enumerate(sorted_pitches):
            if idx == 0:       mean, sd = 1.10, 0.06
            elif idx == 1:     mean, sd = 1.00, 0.06
            else:              mean, sd = 0.94, 0.07
            if role == "RP":
                if idx == 0:   mean *= 1.04
                elif idx >= 2: mean *= 0.97
            factor = rng.normalvariate(mean, sd)
            per_pitch_factors[pt] = round(max(0.75, min(1.35, factor)), 3)

        # ---- Scouting grades (NEW) ----
        p_gr = _sample_pitcher_grades(team["Tier"], cluster, role, cmd, rng)

        # Pitch usage biased by grades
        usage = dirichlet_from_means(PITCH_CLUSTERS[cluster], alpha_total=35.0, rng=rng)
        usage = _bias_pitch_usage_by_grades(usage, p_gr)

        # Scale per-pitch command by pitch+command grades
        for pt in per_pitch_factors:
            gk = {"Fastball":"FB","Slider":"SL","Curveball":"CB","Changeup":"CH",
                  "Cutter":"CT","Sinker":"SI","Splitter":"SPL"}.get(pt, None)
            g_here = (p_gr.get(gk, 50) + p_gr.get("CMD", 50)) / 2.0
            per_pitch_factors[pt] = round(per_pitch_factors[pt] * _gmult(g_here, 0.06), 3)

        cmd_by_pitch = {pt: round(cmd * per_pitch_factors[pt], 3) for pt in pitch_types}
        command_by_pitch_json = json.dumps(cmd_by_pitch)

        f, l, full = generate_name(rng)

        # Body + geometry
        h_in = _height_inches_for("SP" if role=="SP" else "RP", team["Tier"], age, rng)
        w_in = _wingspan_inches_from_height(h_in, rng)
        rel_z, rel_x, ext = _geom_from_body(throws, h_in, w_in, rng)
        arm_bucket = _choose_arm_bucket(cluster, rng)
        arm_deg = _sample_angle_in_bucket(arm_bucket, rng)
        arm_bucket = _bucket_from_angle(arm_deg)
        rel_z, rel_x, ext = _apply_arm_slot_to_geometry(arm_deg, throws, rel_z, rel_x, ext, rng)
        if role in ("SP", "RP"):
            af = age_factor_from_age(age)
            rel_z *= af
            ext   *= af

        # Durability / velo with FB grade bump
        fb_grade = p_gr.get("FB", 50)
        avg_velo = _sample_avg_velo_by_cluster(team["Tier"], cluster, age, rng)
        avg_velo += 0.18 * ((fb_grade - 50.0) / 10.0)
        avg_velo = round(max(72.0, min(99.5, avg_velo)), 1)

        prev_ip  = _sample_prev_ip(team["Tier"], role, age, rng)
        stam, pitch_cap, avg_pitches, exp_bf, rec_days = _sample_stamina_and_limits(
            team["Tier"], role, age, cmd, avg_velo, rng
        )
        inj_flag = _sample_injury_flag(team["Tier"], role, age, avg_velo, rng)

        pitching_weight = _compute_pitching_weight(p_gr, role)
        future_fv = sample_future_fv(team["Tier"], rng)
        present_role = sample_present_role(team["Tier"], role, rng)
        players.append({
            "TeamID": team["TeamID"], "TeamName": team["TeamName"], "OrgID": team["OrgID"],
            "OrgName": team["OrgName"], "Tier": team["Tier"], "PlayerID": f"P{team['TeamID']:02d}{i+1:03d}",
            "FirstName": f, "LastName": l, "FullName": full, "Role": role, "Position": "",
            "Throws": throws, "Bats": bats, "Cluster": cluster,
            "CommandTier": round(cmd, 3),
            "UsageJSON": json.dumps(usage),
            "CommandByPitchJSON": command_by_pitch_json,
            "HitterArchetype": "", "Notes": "",
            "AgeYears": age,
            "HeightIn": h_in, "WingspanIn": w_in,
            "ArmSlotDeg": arm_deg, "ArmSlotBucket": arm_bucket,
            "RelHeight_ft": round(rel_z,3), "RelSide_ft": round(rel_x,3), "Extension_ft": round(ext,3),
            "AvgFBVelo": avg_velo,
            "PrevSeasonIP": prev_ip,
            "StaminaScore": stam,
            "PitchCountLimit": pitch_cap,
            "AvgPitchesPerOuting": avg_pitches,
            "ExpectedBattersFaced": exp_bf,
            "RecoveryDaysNeeded": rec_days,
            "InjuryFlag": inj_flag,

            # NEW scouting/weights
            "ScoutFV": p_gr["_FV"],
            "ScoutGradesJSON": json.dumps(p_gr),
            "PitchingWeight": round(pitching_weight, 3),
            "HittingWeight": "",
            "ScoutRolePresent": present_role,                   # NEW
            "ScoutRoleFuture": int(future_fv),                  # NEW
            "ScoutRole": _format_scout_role(present_role, int(future_fv)),  # NEW (Excel-safe)

        })

    # -----------------
    # Hitters
    # -----------------
    pos_cycle = POSITIONS.copy()
    rng.shuffle(pos_cycle)
    starter_pos = pos_cycle[:9]

    bench_pool = ["C", "IF", "OF", "UT", "BatFirst"]
    while len(bench_pool) < (n_hit - 9):
        bench_pool.append(rng.choice(["IF", "OF", "UT"]))

    for i in range(n_hit):
        age = _sample_age_for_tier(team["Tier"], rng)
        bats = _pick_bats(rng)
        throws = _pick_throws(rng)
        f, l, full = generate_name(rng)
        pos = starter_pos[i] if i < 9 else bench_pool[i - 9]
        archetype = rng.choices(
            ["Balanced","Contact","Power","Speed","OBP"],
            weights=[0.40, 0.22, 0.18, 0.12, 0.08], k=1
        )[0]

        # Hit grades + weight
        h_gr = _sample_hitter_grades(team["Tier"], archetype, rng)
        hitting_weight = _compute_hitting_weight(h_gr)

        h_in = _height_inches_for("BAT", team["Tier"], age, rng)
        w_in = _wingspan_inches_from_height(h_in, rng)
        secondary_pos = _pick_secondary_position(pos, rng)
        is_two_way = (rng.random() < TWO_WAY_RATE[team["Tier"]])

        if is_two_way:
            p_role = _pick_two_way_role(rng)
            cluster = _pick_pitcher_cluster(throws, rng)
            cmd = _sample_command_tier(team["Tier"], age, p_role, rng)
            usage = dirichlet_from_means(PITCH_CLUSTERS[cluster], alpha_total=35.0, rng=rng)

            # Two-way also gets pitcher grades and usage/command shaped by grades
            p2_gr = _sample_pitcher_grades(team["Tier"], cluster, p_role, cmd, rng)
            usage = _bias_pitch_usage_by_grades(usage, p2_gr)

            pitch_types = list(PITCH_CLUSTERS[cluster].keys())
            sorted_pitches = sorted(pitch_types, key=lambda k: PITCH_CLUSTERS[cluster][k], reverse=True)
            per_pitch_factors: Dict[str, float] = {}
            for idx_pt, pt in enumerate(sorted_pitches):
                if idx_pt == 0:       mean, sd = 1.10, 0.06
                elif idx_pt == 1:     mean, sd = 1.00, 0.06
                else:                 mean, sd = 0.94, 0.07
                if p_role == "RP":
                    if idx_pt == 0:   mean *= 1.04
                    elif idx_pt >= 2: mean *= 0.97
                factor = rng.normalvariate(mean, sd)
                factor = max(0.75, min(1.35, factor))
                # grade scaling
                gk = {"Fastball":"FB","Slider":"SL","Curveball":"CB","Changeup":"CH",
                      "Cutter":"CT","Sinker":"SI","Splitter":"SPL"}.get(pt, None)
                g_here = (p2_gr.get(gk, 50) + p2_gr.get("CMD", 50)) / 2.0
                factor *= _gmult(g_here, 0.06)
                per_pitch_factors[pt] = round(factor, 3)

            cmd_by_pitch = {pt: round(cmd * per_pitch_factors[pt], 3) for pt in pitch_types}
            command_by_pitch_json = json.dumps(cmd_by_pitch)

            # Geometry for pitcher persona
            rel_z, rel_x, ext = _geom_from_body(throws, h_in, w_in, rng)
            arm_bucket = _choose_arm_bucket(cluster, rng)
            arm_deg = _sample_angle_in_bucket(arm_bucket, rng)
            arm_bucket = _bucket_from_angle(arm_deg)
            rel_z, rel_x, ext = _apply_arm_slot_to_geometry(arm_deg, throws, rel_z, rel_x, ext, rng)

            pitching_weight = _compute_pitching_weight(p2_gr, p_role)
             # --- Scouting grades for two-ways (based on their pitching role) ---
            future_fv = sample_future_fv(team["Tier"], rng)
            present_role = "Hitter"

            players.append({
                "TeamID": team["TeamID"], "TeamName": team["TeamName"], "OrgID": team["OrgID"],
                "OrgName": team["OrgName"], "Tier": team["Tier"], "PlayerID": f"H{team['TeamID']:02d}{i+1:03d}",
                "FirstName": f, "LastName": l, "FullName": full, "Role": "BAT", "Position": pos,
                "Throws": throws, "Bats": bats,

                "Cluster": cluster,
                "CommandTier": round(cmd, 3),
                "UsageJSON": json.dumps(usage),
                "CommandByPitchJSON": command_by_pitch_json,

                "HitterArchetype": archetype, "Notes": "",
                "AgeYears": age,
                "HeightIn": h_in, "WingspanIn": w_in,
                "ArmSlotDeg": arm_deg, "ArmSlotBucket": arm_bucket,
                "RelHeight_ft": rel_z, "RelSide_ft": rel_x, "Extension_ft": ext,

                "SecondaryPosition": secondary_pos,
                "IsTwoWay": 1,
                "PitcherSecondaryRole": p_role,

                # NEW scouting/weights (both hitter & pitcher personas)
                "ScoutFV": h_gr["_FV"],
                "ScoutGradesJSON": json.dumps({"HIT": h_gr, "PIT": p2_gr}),
                "PitchingWeight": round(pitching_weight, 3),
                "HittingWeight": round(hitting_weight, 3),
                "ScoutRolePresent": "",                              # NEW
                "ScoutRoleFuture": int(future_fv),                   # NEW (use your hitter FV variable)
                "ScoutRole": _format_scout_role(None, int(future_fv)), 
            })

        else:
           # --- Non–two-way hitter ---
           future_fv = sample_future_fv(team["Tier"], rng)
           present_role = None  # hitters don't have a numeric pitcher-present role

           players.append({
                   "TeamID": team["TeamID"], "TeamName": team["TeamName"], "OrgID": team["OrgID"],
                    "OrgName": team["OrgName"], "Tier": team["Tier"], "PlayerID": f"H{team['TeamID']:02d}{i+1:03d}",
                    "FirstName": f, "LastName": l, "FullName": full, "Role": "BAT", "Position": pos,
                     "Throws": throws, "Bats": bats,

                     "Cluster": "",
                     "CommandTier": "",
                     "UsageJSON": "",
                     "CommandByPitchJSON": "",

                     "HitterArchetype": archetype, "Notes": "",
                      "AgeYears": age,
                      "HeightIn": h_in, "WingspanIn": w_in,
                      "ArmSlotDeg": "", "ArmSlotBucket": "",
                      "RelHeight_ft": "", "RelSide_ft": "", "Extension_ft": "",

                      "SecondaryPosition": secondary_pos,
                       "IsTwoWay": 0,
                       "PitcherSecondaryRole": "",

                       # Scouting / weights (hitters)
                       "ScoutFV": h_gr["_FV"],
                      "ScoutGradesJSON": json.dumps(h_gr),
                      "PitchingWeight": "",
                       "HittingWeight": round(hitting_weight, 3),

                     # Role label: just future for hitters (e.g., "55")
                       "ScoutRolePresent": "",                        # leave blank
                       "ScoutRoleFuture": int(future_fv),
                     "ScoutRole": _format_scout_role(None, int(future_fv)),
                     })


    return players

def build_rosters(teams: List[Dict], rng: random.Random) -> List[Dict]:
    roster = []
    for t in teams:
        roster.extend(build_roster_for_team(t, rng))
    return roster

# -----------------------------
# Scheduling utilities
# -----------------------------
def round_robin_pairs(team_ids: List[int]) -> List[List[Tuple[int, int]]]:
    teams = team_ids[:]
    if len(teams) % 2 == 1:
        teams.append(None)
    n = len(teams)
    half = n // 2
    arr = teams[:]
    rounds: List[List[Tuple[int, int]]] = []
    for r in range(n - 1):
        left = arr[:half]
        right = arr[half:]
        right.reverse()
        pairs: List[Tuple[int, int]] = []
        for i in range(half):
            t1, t2 = left[i], right[i]
            if t1 is None or t2 is None:
                continue
            if r % 2 == 0:
                pairs.append((t1, t2))
            else:
                pairs.append((t2, t1))
        rounds.append(pairs)
        arr = [arr[0]] + [arr[-1]] + arr[1:-1]
    return rounds

def double_rr(rounds: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    mirrored: List[List[Tuple[int, int]]] = []
    for rnd in rounds:
        mirrored.append([(h, a) for (h, a) in rnd])
    for rnd in rounds:
        mirrored.append([(a, h) for (h, a) in rnd])
    return mirrored

def _round_dates_cadence(
    start_monday: date,
    num_rounds: int,
    two_game_offsets: Sequence[int] = (1, 4),
    one_game_offsets: Sequence[int] = (3,),
) -> List[date]:
    dates: List[date] = []
    week = 0
    while len(dates) < num_rounds:
        anchor = start_monday + timedelta(weeks=week)
        for off in two_game_offsets:
            if len(dates) >= num_rounds: break
            dates.append(anchor + timedelta(days=off))
        if len(dates) >= num_rounds: break
        week += 1
        anchor = start_monday + timedelta(weeks=week)
        for off in one_game_offsets:
            if len(dates) >= num_rounds: break
            dates.append(anchor + timedelta(days=off))
        week += 1
    return dates

def _last_date(rows: List[Dict]) -> date:
    return max(date.fromisoformat(r["Date"]) for r in rows) if rows else date.today()

def _max_gid(rows: List[Dict], prefix: str) -> int:
    mx = 0
    for r in rows:
        gid = r.get("GameID")
        if gid and gid.startswith(prefix):
            try:
                mx = max(mx, int(gid[len(prefix):]))
            except ValueError:
                pass
    return mx

def _seed_top6(team_ids: List[int]) -> List[int]:
    out = sorted(team_ids)
    assert len(out) >= 6, "Need at least 6 teams to seed playoffs"
    return out[:6]

def _series_dates(start_anchor: date, offsets: Sequence[int]) -> List[date]:
    return [start_anchor + timedelta(days=o) for o in offsets]

def _emit_series_rows(
    tier_name: str,
    game_prefix: str,
    round_label: str,
    home_team: str,
    away_team: str,
    dates: List[date],
    start_num: int
) -> Tuple[List[Dict], int]:
    rows: List[Dict] = []
    gid = start_num
    for d in dates:
        rows.append({
            "GameID": f"{game_prefix}{gid:04d}",
            "Date": d.isoformat(),
            "Tier": tier_name,
            "Round": round_label,
            "HomeTeamID": home_team,
            "AwayTeamID": away_team
        })
        gid += 1
    return rows, gid

def build_postseason_for_tier(
    tier_name: str,
    game_prefix: str,
    seeded_team_ids: List[int],
    reg_season_last_date: date,
    start_gid_num: int,
    wc_day_gap: int = 1,
    sf_offsets: Sequence[int] = (3, 4, 6),
    fin_offsets: Sequence[int] = (10, 11, 13)
) -> List[Dict]:
    assert len(seeded_team_ids) >= 6, "Need at least 6 playoff seeds"
    s1, s2, s3, s4, s5, s6 = seeded_team_ids[:6]
    out: List[Dict] = []
    gid = start_gid_num

    wc_date = reg_season_last_date + timedelta(days=wc_day_gap)
    rows, gid = _emit_series_rows(tier_name, game_prefix, "WC (3v6)", str(s3), str(s6), [wc_date], gid)
    out += rows
    rows, gid = _emit_series_rows(tier_name, game_prefix, "WC (4v5)", str(s4), str(s5), [wc_date], gid)
    out += rows

    wc_low  = f"WIN({game_prefix}-3v6)"
    wc_high = f"WIN({game_prefix}-4v5)"

    sf_dates = _series_dates(reg_season_last_date, sf_offsets)
    rows, gid = _emit_series_rows(tier_name, game_prefix, "SF (1 vs WC-L)", str(s1), wc_low, sf_dates, gid)
    out += rows
    rows, gid = _emit_series_rows(tier_name, game_prefix, "SF (2 vs WC-H)", str(s2), wc_high, sf_dates, gid)
    out += rows

    fin_dates = _series_dates(reg_season_last_date, fin_offsets)
    rows, gid = _emit_series_rows(tier_name, game_prefix, "FINAL", f"WIN({game_prefix}-SF-HI)", f"WIN({game_prefix}-SF-LO)", fin_dates, gid)
    out += rows
    return out

# ---------- Extra games filler ----------
def add_extra_games(team_ids: List[int],
                    base_rounds: List[List[Tuple[int,int]]],
                    games_per_team: int,
                    start_round_index: int,
                    max_vs_opponent: int,
                    rng: random.Random) -> Tuple[List[List[Tuple[int,int]]], int]:
    per_team = defaultdict(int)
    home_away = {t: {"home": 0, "away": 0} for t in team_ids}
    pair_count = defaultdict(int)

    for rnd in base_rounds:
        for h, a in rnd:
            per_team[h] += 1; home_away[h]["home"] += 1
            per_team[a] += 1; home_away[a]["away"] += 1
            key = (h, a) if h < a else (a, h)
            pair_count[key] += 1

    needed = {t: max(0, games_per_team - per_team[t]) for t in team_ids}
    extra_rounds: List[List[Tuple[int,int]]] = []
    next_round = start_round_index
    safety = 0

    while sum(needed.values()) > 0:
        safety += 1
        if safety > 10000:
            raise RuntimeError("Extra-game scheduler failed to converge; check constraints.")

        used_this_round = set()
        this_round: List[Tuple[int,int]] = []

        teams_by_need = sorted(team_ids, key=lambda t: (-needed[t], home_away[t]["home"] - home_away[t]["away"], t))
        for t in teams_by_need:
            if needed[t] <= 0 or t in used_this_round:
                continue

            opps = []
            for o in teams_by_need:
                if o == t or needed[o] <= 0 or o in used_this_round:
                    continue
                key = (t, o) if t < o else (o, t)
                if pair_count[key] >= max_vs_opponent:
                    continue
                opps.append(o)

            if not opps:
                continue

            def score(o):
                need_gap = abs(needed[t] - needed[o])
                ha_bias_t = home_away[t]["home"] - home_away[t]["away"]
                ha_bias_o = home_away[o]["home"] - home_away[o]["away"]
                return (need_gap, abs(ha_bias_t + ha_bias_o), rng.random())

            opps.sort(key=score)
            o = opps[0]

            if home_away[t]["home"] > home_away[t]["away"] and home_away[o]["home"] < home_away[o]["away"]:
                h, a = (o, t)
            elif home_away[o]["home"] > home_away[o]["away"] and home_away[t]["home"] < home_away[t]["away"]:
                h, a = (t, o)
            else:
                h, a = (t, o) if rng.random() < 0.5 else (o, t)

            this_round.append((h, a))
            used_this_round.update([h, a])
            needed[h] -= 1; needed[a] -= 1
            home_away[h]["home"] += 1; home_away[a]["away"] += 1
            key = (h, a) if h < a else (a, h)
            pair_count[key] += 1

            if sum(needed.values()) == 0:
                break

        if this_round:
            extra_rounds.append(this_round)
            next_round += 1
        else:
            max_vs_opponent += 1

    return extra_rounds, next_round

def _balance_home_away(rows: List[Dict], team_ids: List[int], max_diff: int = 1) -> None:
    team_keys = {str(t) for t in team_ids}
    home_games: Dict[str, List[int]] = {k: [] for k in team_keys}
    away_games: Dict[str, List[int]] = {k: [] for k in team_keys}
    ha = {k: {"H": 0, "A": 0} for k in team_keys}
    for idx, r in enumerate(rows):
        h = str(r["HomeTeamID"]); a = str(r["AwayTeamID"])
        if h in team_keys:
            ha[h]["H"] += 1; home_games[h].append(idx)
        if a in team_keys:
            ha[a]["A"] += 1; away_games[a].append(idx)

    def flip_game(i: int):
        r = rows[i]
        h = str(r["HomeTeamID"]); a = str(r["AwayTeamID"])
        if h in team_keys:
            ha[h]["H"] -= 1; ha[h]["A"] += 1
            if i in home_games[h]: home_games[h].remove(i)
            away_games[h].append(i)
        if a in team_keys:
            ha[a]["A"] -= 1; ha[a]["H"] += 1
            if i in away_games[a]: away_games[a].remove(i)
            home_games[a].append(i)
        r["HomeTeamID"], r["AwayTeamID"] = r["AwayTeamID"], r["HomeTeamID"]

    for _ in range(8):
        too_home = [t for t in team_keys if ha[t]["H"] - ha[t]["A"] > max_diff]
        too_away = [t for t in team_keys if ha[t]["A"] - ha[t]["H"] > max_diff]
        if not too_home and not too_away:
            break
        for t in list(too_home):
            fixed = False
            for i in list(home_games[t]):
                opp = str(rows[i]["AwayTeamID"])
                if opp in team_keys and (ha[opp]["A"] - ha[opp]["H"] > max_diff):
                    flip_game(i); fixed = True; break
            if not fixed and home_games[t]:
                flip_game(home_games[t][0])
        for t in list(too_away):
            fixed = False
            for i in list(away_games[t]):
                opp = str(rows[i]["HomeTeamID"])
                if opp in team_keys and (ha[opp]["H"] - ha[opp]["A"] > max_diff):
                    flip_game(i); fixed = True; break
            if not fixed and away_games[t]:
                flip_game(away_games[t][0])

    def offenders():
        th = [t for t in team_keys if ha[t]["H"] - ha[t]["A"] > max_diff]
        ta = [t for t in team_keys if ha[t]["A"] - ha[t]["H"] > max_diff]
        return th, ta

    safety = 0
    while True:
        safety += 1
        if safety > 2000:
            break
        th, ta = offenders()
        if not th and not ta:
            break
        progressed = False
        for hteam in th:
            if progressed: break
            for ateam in ta:
                if progressed: break
                cand = None
                if len(home_games[hteam]) <= len(away_games[ateam]):
                    for i in home_games[hteam]:
                        if str(rows[i]["AwayTeamID"]) == ateam:
                            cand = i; break
                else:
                    for i in away_games[ateam]:
                        if str(rows[i]["HomeTeamID"]) == hteam:
                            cand = i; break
                if cand is not None:
                    flip_game(cand)
                    progressed = True
        if not progressed:
            any_team = None
            max_gap = 0
            for t in team_keys:
                gap = abs(ha[t]["H"] - ha[t]["A"])
                if gap > max_gap:
                    max_gap = gap; any_team = t
            if any_team is None:
                break
            if ha[any_team]["H"] > ha[any_team]["A"] and home_games[any_team]:
                flip_game(home_games[any_team][0]); progressed = True
            elif away_games[any_team]:
                flip_game(away_games[any_team][0]); progressed = True
            if not progressed:
                break

# -----------------------------
# Public API
# -----------------------------
def build_tier_schedule(team_ids: List[int],
                        games_per_team: int,
                        start_date: date,
                        rng: random.Random) -> List[Dict]:
    assert len(team_ids) >= 2, "Need at least 2 teams"
    n = len(team_ids)
    single = round_robin_pairs(team_ids)
    double = double_rr(single)

    single_cap = n - 1
    double_cap = 2 * (n - 1)

    def _safe_prefix_len(G: int, N: int, max_rounds: int) -> int:
        if N % 2 == 0:
            return min(G, max_rounds)
        import math
        return min(max_rounds, int(math.floor(G * N / (N - 1))))

    if games_per_team <= single_cap:
        base_len = _safe_prefix_len(games_per_team, n, len(single))
        base = single[:base_len]
        extra_rounds, _ = add_extra_games(
            team_ids=team_ids,
            base_rounds=base,
            games_per_team=games_per_team,
            start_round_index=len(base) + 1,
            max_vs_opponent=1,
            rng=rng
        )
        schedule_rounds = base + extra_rounds

    elif games_per_team <= double_cap:
        base_len = _safe_prefix_len(games_per_team, n, len(double))
        base = double[:base_len]
        extra_rounds, _ = add_extra_games(
            team_ids=team_ids,
            base_rounds=base,
            games_per_team=games_per_team,
            start_round_index=len(base) + 1,
            max_vs_opponent=2,
            rng=rng
        )
        schedule_rounds = base + extra_rounds

    else:
        base = double[:]
        extra_rounds, _ = add_extra_games(
            team_ids=team_ids,
            base_rounds=base,
            games_per_team=games_per_team,
            start_round_index=len(base) + 1,
            max_vs_opponent=3,
            rng=rng
        )
        schedule_rounds = base + extra_rounds

    round_dates = _round_dates_cadence(
        start_monday=start_date,
        num_rounds=len(schedule_rounds),
        two_game_offsets=(1, 4),
        one_game_offsets=(3,),
    )
    assert len(round_dates) == len(schedule_rounds)

    rows: List[Dict] = []
    for rnd_idx, (rnd, d) in enumerate(zip(schedule_rounds, round_dates), start=1):
        rnd_local = rnd[:]
        rng.shuffle(rnd_local)
        for h, a in rnd_local:
            rows.append({
                "GameID": None,
                "Date": d.isoformat(),
                "Round": rnd_idx,
                "HomeTeamID": h,
                "AwayTeamID": a
            })

    _balance_home_away(rows, team_ids, max_diff=1)

    team_keys = {str(t) for t in team_ids}
    per_team = {k: 0 for k in team_keys}
    home_away = {k: {"H": 0, "A": 0} for k in team_keys}

    for r in rows:
        h = str(r["HomeTeamID"]); a = str(r["AwayTeamID"])
        if h in team_keys:
            per_team[h] += 1; home_away[h]["H"] += 1
        if a in team_keys:
            per_team[a] += 1; home_away[a]["A"] += 1

    bad = [t for t in team_keys if per_team[t] != games_per_team]
    assert not bad, (
        f"Schedule error: these teams miss target {games_per_team}: "
        + ", ".join(f"{t}={per_team[t]}" for t in bad)
    )

    if games_per_team <= double_cap:
        ha_bad = [t for t, ha in home_away.items() if abs(ha["H"] - ha["A"]) > 1]
        assert not ha_bad, (
            "Home/Away imbalance >1 for teams: "
            + ", ".join(f"{t} (H={home_away[t]['H']}, A={home_away[t]['A']})" for t in ha_bad)
        )

    return rows

# -----------------------------
# I/O helpers
# -----------------------------
def write_csv(path: Path, fieldnames: List[str], rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            safe = {}
            for k in fieldnames:
                v = r.get(k, "")
                safe[k] = "" if v is None else v
            w.writerow(safe)


def _dbg_schedule(tag: str, rows: List[Dict]):
    if not rows:
        print(f"[{tag}] ❌ No schedule rows")
        return
    print(f"[{tag}] ✅ {len(rows)} games; first 3 rows:")
    for r in rows[:3]:
        print("   ", r)
    per = defaultdict(int)
    for r in rows:
        per[r["HomeTeamID"]] += 1
        per[r["AwayTeamID"]] += 1
    print(f"[{tag}] sample per-team counts:", list(per.items())[:5])

# -----------------------------
# CLI + Orchestration
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Generate league orgs/teams/rosters/schedule")
    ap.add_argument("--num_orgs", type=int, default=DEFAULT_NUM_ORGS)
    ap.add_argument("--out_dir", type=str, default="league_out")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--start_date", type=str, default=START_DATE.isoformat(),
                    help="YYYY-MM-DD (Majors start; AAA +1d; Rookie +2d)")
    ap.add_argument("--maj_games", type=int, default=GAMES_PER_TEAM["Majors"])
    ap.add_argument("--aaa_games", type=int, default=GAMES_PER_TEAM["AAA"])
    ap.add_argument("--rookie_games", type=int, default=GAMES_PER_TEAM["Rookie"])
    ap.add_argument("--with_postseason", action="store_true",
                    help="Append 6-team bracket: seeds 1-2 byes; WC single; SF/Final best-of-3")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start_dt = date.fromisoformat(args.start_date)

    # 1) Organizations
    orgs = build_organizations(args.num_orgs, rng)

    # 2) Teams (3 per org)
    teams = build_teams(orgs)

    # 3) Rosters
    rosters = build_rosters(teams, rng)

    # 4) Regular-season schedules per tier
    tier_team_ids: Dict[str, List[int]] = {"Majors": [], "AAA": [], "Rookie": []}
    for t in teams:
        tier_team_ids[t["Tier"]].append(t["TeamID"])
    for tier in tier_team_ids:
        tier_team_ids[tier].sort()

    sched_M = build_tier_schedule(tier_team_ids["Majors"], args.maj_games, start_dt, rng)
    sched_A = build_tier_schedule(tier_team_ids["AAA"],    args.aaa_games, start_dt + timedelta(days=1), rng)
    sched_R = build_tier_schedule(tier_team_ids["Rookie"], args.rookie_games, start_dt + timedelta(days=2), rng)

    # Tag tiers + assign GameIDs (prefix M/A/R)
    gid = 1
    for r in sched_M:
        r["Tier"] = "Majors"; r["GameID"] = f"M{gid:04d}"; gid += 1
    gid = 1
    for r in sched_A:
        r["Tier"] = "AAA";    r["GameID"] = f"A{gid:04d}"; gid += 1
    gid = 1
    for r in sched_R:
        r["Tier"] = "Rookie"; r["GameID"] = f"R{gid:04d}"; gid += 1

    # Optional: append postseason for each tier
    if args.with_postseason:
        seeds_M = _seed_top6(tier_team_ids["Majors"])
        last_M  = _last_date(sched_M)
        next_M  = _max_gid(sched_M, "M") + 1
        po_M    = build_postseason_for_tier("Majors", "M", seeds_M, last_M, next_M)

        seeds_A = _seed_top6(tier_team_ids["AAA"])
        last_A  = _last_date(sched_A)
        next_A  = _max_gid(sched_A, "A") + 1
        po_A    = build_postseason_for_tier("AAA", "A", seeds_A, last_A, next_A)

        seeds_R = _seed_top6(tier_team_ids["Rookie"])
        last_R  = _last_date(sched_R)
        next_R  = _max_gid(sched_R, "R") + 1
        po_R    = build_postseason_for_tier("Rookie", "R", seeds_R, last_R, next_R)
    else:
        po_M = po_A = po_R = []

    _dbg_schedule("Majors RS", sched_M)
    _dbg_schedule("AAA RS",    sched_A)
    _dbg_schedule("Rookie RS", sched_R)
    if args.with_postseason:
        _dbg_schedule("Majors PO", po_M)
        _dbg_schedule("AAA PO",    po_A)
        _dbg_schedule("Rookie PO", po_R)

    schedules_all = []
    schedules_all.extend(sched_M + po_M)
    schedules_all.extend(sched_A + po_A)
    schedules_all.extend(sched_R + po_R)

    # 5) Write CSVs
    out_dir = Path(args.out_dir)
    write_csv(out_dir / "organizations.csv",
              ["OrgID", "OrgName"],
              orgs)

    write_csv(out_dir / "teams.csv",
              ["TeamID","TeamName","OrgID","OrgName","Tier"],
              teams)

    write_csv(
        out_dir / "rosters.csv",
        [
            # --- Team / org identity ---
            "TeamID", "TeamName", "OrgID", "OrgName", "Tier", "PlayerID",

            # --- Player identity ---
            "FirstName", "LastName", "FullName", "Role", "Position",
            "SecondaryPosition", "IsTwoWay", "PitcherSecondaryRole",
            "Throws", "Bats",

            # --- Pitching skillset ---
            "Cluster", "CommandTier", "UsageJSON", "CommandByPitchJSON",

            # --- Hitting skillset ---
            "HitterArchetype", "Notes",

            # --- Bio / body metrics ---
            "AgeYears", "HeightIn", "WingspanIn",
            "ArmSlotDeg", "ArmSlotBucket",
            "RelHeight_ft", "RelSide_ft", "Extension_ft",

            # --- Durability / workload ---
            "AvgFBVelo", "PrevSeasonIP", "StaminaScore", "PitchCountLimit",
            "AvgPitchesPerOuting", "ExpectedBattersFaced",
            "RecoveryDaysNeeded", "InjuryFlag",

            # --- Scouting / weights (NEW) ---
            "ScoutFV", "ScoutRolePresent", "ScoutRoleFuture", "ScoutRole", "ScoutGradesJSON",
        ],
        rosters
    )

    write_csv(out_dir / "schedule.csv",
              ["GameID","Date","Tier","Round","HomeTeamID","AwayTeamID"],
              schedules_all)

    print(f"\nOutput folder: {(out_dir.resolve())}")
    print(f"✅ Wrote organizations.csv ({len(orgs)} rows)")
    print(f"✅ Wrote teams.csv        ({len(teams)} rows)")
    print(f"✅ Wrote rosters.csv      ({len(rosters)} rows)")
    print(f"✅ Wrote schedule.csv     ({len(schedules_all)} games)")

    totals = {"Majors":0,"AAA":0,"Rookie":0}
    per_team_counts = defaultdict(int)
    for r in sched_M + sched_A + sched_R:
        totals[r["Tier"]] += 1
        per_team_counts[r["HomeTeamID"]] += 1
        per_team_counts[r["AwayTeamID"]] += 1
    for tier, tot in totals.items():
        nteams = len(tier_team_ids[tier]) or 1
        print(f"   {tier}: {tot} games total; ~{tot*2//nteams} per team (RS only)")
    if args.with_postseason:
        print("   (+ postseason games appended)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback, sys
        print("\n=== CRASH ===")
        print(type(e).__name__, e)
        traceback.print_exc()
        sys.exit(1)
