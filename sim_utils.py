#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, json, random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from faker import Faker

fake = Faker()

# ----------------------------- Season knobs -----------------------------
DEFAULT_SP_RECOVERY = 4     # days
DEFAULT_RP_RECOVERY = 1     # days baseline for relief work
INJURY_CHANCE_HEAVY_OVER = 0.05  # 5% if > limit + 25 pitches
INJURY_DUR_RANGE = (10, 30)      # days out
EXTRA_INNING_RECOVERY_BONUS_DAYS = 1
# Mid-game pull thresholds (can override via CLI)
PULL_RUNS = 4
PULL_STRESS_PITCHES = 35

# Fatigue penalties (can override via CLI)
FATIGUE_PER_PITCH_OVER = 0.015  # command tax per pitch beyond limit
FATIGUE_PER_BF_OVER     = 0.03  # command tax per BF beyond expected
VELO_LOSS_PER_OVER10    = 0.15  # mph per 10 pitches over limit
SPIN_LOSS_PER_OVER10    = 20.0  # rpm per 10 pitches over limit
TTO_PENALTY             = 0.10  # 3rd time through order command penalty

# Extra-innings knobs
EXTRA_INNING_FATIGUE_SCALE       = 0.50  # multiplies fatigue per pitch/BF in extras (per extra inning)
EXTRA_INNING_CMD_FLAT_PENALTY    = 0.03  # extra command tax per extra inning (3% each inning past 9)

# ------------- Shared helpers for game simulation (compact) -------------
INPLAY_RESULTS = ["Out","Single","Double","Triple","HomeRun"]
VALID_HITTYPE_BY_RESULT = {
    "Out":        ["GroundBall", "FlyBall", "LineDrive", "Popup"],
    "Single":     ["GroundBall", "LineDrive", "FlyBall"],
    "Double":     ["LineDrive", "FlyBall"],
    "Triple":     ["LineDrive", "FlyBall"],
    "HomeRun":    ["FlyBall", "LineDrive"]
}
ANGLE_BINS = {"GroundBall": (-20.0, 10.0), "LineDrive": (10.0, 25.0), "FlyBall": (25.0, 50.0), "Popup": (50.0, 90.0)}
EV_BOUNDS  = {"GroundBall": (60,105), "LineDrive": (80,112), "FlyBall": (75,110), "Popup": (50,90)}
HITTYPE_DISTANCE_BOUNDS = {"GroundBall": (30.0,120.0), "LineDrive": (120.0,320.0), "FlyBall": (150.0,430.0), "Popup": (40.0,200.0)}
HAND_CANON = {"R":"R","RIGHT":"R","Right":"R","L":"L","LEFT":"L","Left":"L"}



def canon_hand(x: str) -> str:
    return HAND_CANON.get(str(x), "R")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def sample_categorical(d: Dict[str, float], rng: random.Random) -> str:
    keys = list(d.keys())
    wts  = [max(0.0, float(d[k] or 0.0)) for k in keys]
    s = sum(wts) or 1.0
    wts = [w/s for w in wts]
    return rng.choices(keys, weights=wts, k=1)[0]

def sample_statpack(pack, rng: random.Random, round_to: Optional[int] = None):
    if not isinstance(pack, (list, tuple)) or len(pack) < 4 or pack[0] is None:
        return None
    mu, sd, lo, hi = pack
    sd = float(sd or 0.0)
    x = float(mu) if sd == 0.0 else rng.normalvariate(mu, sd)
    if lo is not None: x = max(x, lo)
    if hi is not None: x = min(x, hi)
    if round_to is not None: x = round(x, round_to)
    return x

def sample_feature_value(spec: Any, rng: random.Random, round_to: Optional[int] = None):
    if spec is None: return None
    if isinstance(spec, dict): return sample_categorical(spec, rng)
    return sample_statpack(spec, rng, round_to=round_to)

def mvn_sample(mean: List[float], cov: List[List[float]]) -> Tuple[float, float]:
    m = np.array(mean, dtype=float).reshape(2)
    C = np.array(cov, dtype=float).reshape(2,2)
    eigvals = np.linalg.eigvalsh(C)
    if np.any(eigvals < 1e-9):
        C = C + np.eye(2) * (1e-6 - min(0.0, eigvals.min()))
    samp = np.random.multivariate_normal(m, C, size=1)[0]
    return float(samp[0]), float(samp[1])

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return cur

def infer_hittype_from_angle(angle: float) -> Optional[str]:
    if angle is None: return None
    if angle < ANGLE_BINS["GroundBall"][1]: return "GroundBall"
    if angle < ANGLE_BINS["LineDrive"][1]:  return "LineDrive"
    if angle < ANGLE_BINS["FlyBall"][1]:    return "FlyBall"
    return "Popup"

def repair_batted_ball_fields(row: dict, play_result: str, hit_type: Optional[str], rng) -> Optional[str]:
    def ffloat(x):
        try:
            if x is None: return None
            if isinstance(x, str):
                s = x.strip()
                if s == "" or s.lower() == "nan": return None
                return float(s)
            return float(x)
        except Exception:
            return None

    ev   = ffloat(row.get("ExitSpeed"))
    ang  = ffloat(row.get("Angle"))
    dist = ffloat(row.get("Distance"))

    if not hit_type and ang is not None:
        if ang < ANGLE_BINS["GroundBall"][1]: hit_type = "GroundBall"
        elif ang < ANGLE_BINS["LineDrive"][1]: hit_type = "LineDrive"
        elif ang < ANGLE_BINS["FlyBall"][1]: hit_type = "FlyBall"
        else: hit_type = "Popup"

    if hit_type in ANGLE_BINS:
        lo, hi = ANGLE_BINS[hit_type]
        if ang is None or not (lo <= ang <= hi):
            m = 0.5
            tlo, thi = lo + m, hi - m
            if tlo > thi: tlo, thi = lo, hi
            ang = rng.uniform(tlo, thi)
            row["Angle"] = round(ang, 2)

    if hit_type in EV_BOUNDS:
        lo_ev, hi_ev = EV_BOUNDS[hit_type]
        if ev is None:
            ev = rng.uniform(lo_ev, (lo_ev + hi_ev*0.6)/1.6)
        ev = clamp(ev, lo_ev, hi_ev)
        row["ExitSpeed"] = round(ev, 1)

    if hit_type in HITTYPE_DISTANCE_BOUNDS:
        dlo, dhi = HITTYPE_DISTANCE_BOUNDS[hit_type]
        if hit_type == "FlyBall":
            base = 260.0
            if ev is not None and ang is not None:
                base = 1.5 * ev + 1.8 * ang - 120.0
                base = clamp(base, 200.0, 420.0)
            if dist is None or dist < 150.0:
                dist = rng.uniform(base - 20.0, base + 20.0)
            dist = clamp(dist, dlo, dhi)
            row["Distance"] = round(dist, 1)
        elif hit_type == "GroundBall":
            if dist is None or dist > dhi or dist < dlo:
                dist = rng.uniform(dlo, min(dhi, 95.0))
            dist = clamp(dist, dlo, dhi)
            row["Distance"] = round(dist, 1)
        else:
            if dist is None or not (dlo <= dist <= dhi):
                dist = rng.uniform(dlo, dhi)
            dist = clamp(dist, dlo, dhi)
            row["Distance"] = round(dist, 1)

    if hit_type == "GroundBall":
        if ang is None or ang > 10.0:
            ang = rng.uniform(-15.0, 10.0)
            row["Angle"] = round(ang, 2)
        dcur = ffloat(row.get("Distance"))
        if dcur is None or dcur > 120.0:
            row["Distance"] = round(rng.uniform(30.0, 90.0), 1)

    return hit_type

# ----------------------------- Scouting/weights helpers -----------------------------
def _safe_float(x, default=None):
    try:
        if x is None: return default
        s = str(x).strip()
        if s == "" or s.lower() == "nan": return default
        return float(s)
    except Exception:
        return default

def _grade_to_weight(g: Optional[float]) -> float:
    try:
        g = float(g)
    except Exception:
        return 1.0
    return float(round(0.04 * ((g - 50.0) / 5.0) + 1.0, 3))

def _extract_present_from_role(text: Optional[str]) -> Optional[float]:
    if not text: return None
    s = str(text).strip()
    import re
    m = re.search(r'(\d+(?:\.\d+)?)$', s)
    if not m: return None
    val = float(m.group(1))
    if 2.0 <= val <= 8.0: return val * 10.0
    return val

def _platoon_batter_bonus(bats: str, p_throws_canon: str) -> float:
    b = (bats or "").upper()
    p = (p_throws_canon or "R").upper()
    if b == "SWITCH": return 0.02
    if b.startswith("L") and p == "R": return 0.04
    if b.startswith("R") and p == "L": return 0.04
    return -0.02

def _adjust_pitchcall(base: Dict[str,float], batter_q: float, pitch_cmd: float) -> Dict[str,float]:
    d = {k: max(0.0, float(base.get(k, 0.0))) for k in ("BallCalled","StrikeCalled","StrikeSwinging","Foul","InPlay")}
    edge = (batter_q / max(1e-6, pitch_cmd)) - 1.0
    move_to_inplay = max(-0.08, min(0.08, 0.06 * edge))  # cap ±8%

    def pull_from(k, amt):
        take = min(d[k], amt); d[k] -= take; return take

    if move_to_inplay > 0:
        gain = 0.0
        gain += pull_from("StrikeSwinging", move_to_inplay * 0.60)
        gain += pull_from("StrikeCalled",  move_to_inplay * 0.36)
        gain += pull_from("Foul",          move_to_inplay * 0.25)
        d["InPlay"] += gain
    else:
        give = pull_from("InPlay", abs(move_to_inplay))
        d["StrikeSwinging"] += give * 0.55
        d["Foul"]           += give * 0.30
        d["StrikeCalled"]   += give * 0.15

    z = sum(d.values()) or 1.0
    return {k: v/z for k, v in d.items()}

def _adjust_inplay_split(split: Dict[str,float], batter_q: float) -> Dict[str,float]:
    s = {k: max(0.0, float(split.get(k, 0.0))) for k in ("Out","Single","Double","Triple","HomeRun","Error","FielderChoice","Sacrifice")}
    tilt = max(-0.15, min(0.25, 0.20 * (batter_q - 1.0)))
    if tilt >= 0:
        take_out = min(s["Out"], tilt * 0.8)
        take_1b  = min(s["Single"], tilt * 0.2)
        gain = take_out + take_1b
        s["Out"]    -= take_out
        s["Single"] -= take_1b
        s["HomeRun"] += gain * 0.45
        s["Double"]  += gain * 0.40
        s["Triple"]  += gain * 0.15
    else:
        gain = min(sum([s["HomeRun"], s["Double"], s["Triple"]]), abs(tilt))
        give_hr = min(s["HomeRun"], gain * 0.45); s["HomeRun"] -= give_hr
        give_2b = min(s["Double"],  gain * 0.35); s["Double"]  -= give_2b
        give_3b = min(s["Triple"],  gain * 0.20); s["Triple"]  -= give_3b
        back = give_hr + give_2b + give_3b
        s["Out"]    += back * 0.60
        s["Single"] += back * 0.40
    keys = [k for k in ("Out","Single","Double","Triple","HomeRun","Error","FielderChoice","Sacrifice") if s.get(k,0)>0]
    z = sum(s[k] for k in keys) or 1.0
    return {k: (s[k]/z if k in keys else 0.0) for k in ("Out","Single","Double","Triple","HomeRun","Error","FielderChoice","Sacrifice")}

# ----------------------------- Season utilities -----------------------------
def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def parse_json_field(text: str) -> Dict[str, float]:
    if not text: return {}
    try: return json.loads(text)
    except Exception: return {}

def parse_date(s: str) -> datetime:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try: return datetime.strptime(s, fmt)
        except Exception: pass
    return datetime.now()

# ---------- Roster extraction (pitchers) ----------
def staff_from_roster_rows(rows: List[Dict[str, str]], team_key: str) -> Dict[str, Any]:
    pitchers = []
    for r in rows:
        if r.get("Role") not in ("SP","RP"): continue
        if (r.get("TeamID") or r.get("TeamName")) != team_key: continue
        injured = str(r.get("InjuryFlag","")).strip().lower() in ("1","true","yes","y")

        throws_raw = str(r.get("Throws","Right")).strip().upper()
        throws_text = "Right" if throws_raw.startswith("R") else "Left"
        throws_canon = "R" if throws_text == "Right" else "L"

        fv = _safe_float(r.get("ScoutFV"), 50.0)
        present = r.get("ScoutRolePresent")
        if present is None and r.get("ScoutRole"):
            present = _extract_present_from_role(r.get("ScoutRole"))
        present = _safe_float(present, 5.0)
        present_20_80 = (present*10.0) if (present is not None and present <= 8.0) else present
        pitching_weight = _safe_float(r.get("PitchingWeight"), None)
        if pitching_weight is None:
            pitching_weight = _grade_to_weight((present_20_80 or 50.0))

        p = {
            "PitcherId": r.get("PlayerID",""),
            "Pitcher": f"{r.get('FirstName','')} {r.get('LastName','')}".strip() or f"Pitcher {r.get('PlayerID','')}",
            "TeamKey": team_key,
            "Role": r.get("Role","RP"),
            "Throws": throws_text,
            "_ThrowsCanon": throws_canon,
            "_Usage": parse_json_field((r.get("UsageJSON") or "").strip()),
            "_CmdBase": float(r.get("CommandTier") or 1.0),
            "_CmdByPitch": parse_json_field((r.get("CommandByPitchJSON") or "").strip()),
            "_Stamina": float(r.get("StaminaScore") or 50.0),
            "_Limit": int(float(r.get("PitchCountLimit") or 85)),
            "_ExpBF": float(r.get("ExpectedBattersFaced") or 18.0),
            "_AvgOut": float(r.get("AvgPitchesPerOuting") or 80.0),
            "_RecDays": int(float(r.get("RecoveryDaysNeeded") or (DEFAULT_SP_RECOVERY if r.get("Role")=="SP" else DEFAULT_RP_RECOVERY))),
            "_Injured": injured,
            "_NextOK": None,
            "_PitchingWeight": float(pitching_weight),
            "_AvgFBVelo": _safe_float(r.get("AvgFBVelo"), None),
            "_RelHeight_ft": _safe_float(r.get("RelHeight_ft"), None),
            "_RelSide_ft":   _safe_float(r.get("RelSide_ft"), None),
            "_Extension_ft": _safe_float(r.get("Extension_ft"), None),
            "_ScoutFV": fv,
            "_ScoutPresent": present_20_80,
        }
        pitchers.append(p)

    rotation = [p for p in pitchers if p["Role"] == "SP" and not p["_Injured"]]
    pen      = [p for p in pitchers if p["Role"] == "RP" and not p["_Injured"]]
    return {"rotation": rotation, "pen": pen, "all": pitchers, "_rot_idx": 0}

# ---------- Roster extraction (batters/lineups) ----------
def lineup_from_roster_rows(rows: List[Dict[str,str]], team_key: str, rng: random.Random) -> List[Dict[str,str]]:
    bats = []
    for r in rows:
        key = r.get("TeamID") or r.get("TeamName")
        if str(key) != str(team_key): continue
        role = (r.get("Role") or "").upper()
        is_two_way = str(r.get("IsTwoWay","0")).strip().lower() in ("1","true","yes","y")
        if role == "BAT" or is_two_way:
            fv = _safe_float(r.get("ScoutFV"), 50.0)
            w  = _safe_float(r.get("HittingWeight"), None)
            if w is None: w = _grade_to_weight(fv)
            bats.append({
                "Batter": f"{r.get('FirstName','')} {r.get('LastName','')}".strip() or f"Batter {r.get('PlayerID','')}",
                "BatterId": r.get("PlayerID") or str(20000 + rng.randrange(10000)),
                "BatterTeam": team_key,
                "BatterSide": r.get("Bats","Right"),
                "_Quality": float(w),
                "_TimesFaced": 0,
            })
    while len(bats) < 9:
        bats.append({"Batter": fake.name(), "BatterId": str(20000 + rng.randrange(10000)),
                     "BatterTeam": team_key, "BatterSide": rng.choice(["Right","Left","Switch"]),
                     "_Quality": 1.0, "_TimesFaced": 0})
    bats = sorted(bats, key=lambda b: b["_Quality"], reverse=True)[:9]
    rng.shuffle(bats)
    return bats

# ---------- Availability / rotation ----------
def is_available(p: Dict[str,Any], when: datetime) -> bool:
    if p["_Injured"]: return False
    if p["_NextOK"] is None: return True
    return when >= p["_NextOK"]

def choose_sp_for_date(staff: Dict[str,Any], game_dt: datetime) -> Optional[Dict[str,Any]]:
    start_idx = staff["_rot_idx"]; n = len(staff["rotation"])
    for k in range(n or 1):
        i = (start_idx + k) % (n or 1)
        p = staff["rotation"][i] if n else None
        if p and is_available(p, game_dt):
            staff["_rot_idx"] = (i + 1) % (n or 1)
            return p
    avail_rp = [rp for rp in staff["pen"] if is_available(rp, game_dt)]
    if avail_rp:
        return sorted(avail_rp, key=lambda x: (x["_NextOK"] or datetime.min))[0]
    return None

def mark_recovery(p: Dict[str,Any], game_dt: datetime, pitches: int, role: str):
    base = p["_RecDays"] if role == "SP" else 0
    if role == "RP":
        if   pitches >= 61: base = 4
        elif pitches >= 46: base = 3
        elif pitches >= 31: base = 2
        elif pitches >= 1:  base = 1
        else:               base = 0
    extra = 0
    if pitches > p["_Limit"] + 10: extra += 1
    if pitches > p["_Limit"] + 25: extra += 1
    p["_NextOK"] = (game_dt + timedelta(days=base + extra)).replace(hour=0, minute=0, second=0, microsecond=0)

def maybe_injure(p: Dict[str,Any], rng: random.Random, pitches: int):
    if pitches > p["_Limit"] + 25 and rng.random() < INJURY_CHANCE_HEAVY_OVER:
        days = rng.randint(*INJURY_DUR_RANGE)
        p["_Injured"] = True
        p["_NextOK"] = (p["_NextOK"] or datetime.now()) + timedelta(days=days)
