#!/usr/bin/env python
"""
Build compact PRIOR YAML + Pitcher Archetype Clusters from YakkerTech-style CSVs.

Outputs ONE YAML with:
- meta.source_files
- distributions:
    PitcherHand, BatterSide, BatterSideGivenPitcherHand (if present),
    PitchesPerInning (pmf + stats), PitchesPerAB (pmf), outcomes_overall
- pitches: per pitch type -> per hand:
    core statpacks (velo/spin/ivb/hb)
    extra feature statpacks (includes PlateLoc*, Release*, SpinAxis, Extension, Tilt as categorical)
    outcomes / outcomes_mean
    inplay_split (Out, Single, Double, Triple, HomeRun, Error, FielderChoice, Sacrifice)
    hit_types
    outcomes_detail (batted-ball statpacks per in-play result, same categories as inplay_split)
    location_model  (2D Gaussian over [PlateLocSide, PlateLocHeight])
    pitch_call_by_zone (3x3 grid + outside edge/chase)
- clusters: learned pitcher archetypes (separately for R/L) with:
    prior_weight
    {Hand}:
      command_tier_prior
      global_zone_target
      repertoire_presence (prob a pitch is in repertoire)
      baseline_usage (mean usage shares)
      summary_fastball: (FB velo/IVB/HB/rel_height/rel_side + FB spinrate, spinaxis, extension, rel angles)
- counts:
    start_distribution
    transitions (single-pitch transitions + terminals)
    pitch_mix_by_count
    joint_count_pitch: legal (from_count,to_count,pitch) probabilities
"""

import argparse, sys, yaml, math, warnings, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

# ---------------- Config ----------------
ROUND = 3
PMF_TOPK = 25
SMOOTH = 1e-9
SPORT = "baseball"
CANON_PITCHES = ["Fastball","Sinker","Cutter","Slider","Curveball","Changeup","Splitter"]
LOC_X, LOC_Z = "PlateLocSide", "PlateLocHeight"
FB = "Fastball"
PLATE_HALF_WIDTH = 0.708  # ~17" / 2 in feet
ZONE_Z_LOW  = 1.50
ZONE_Z_HIGH = 3.50
ALL_COUNTS = [f"{b}-{s}" for b in range(4) for s in range(3)]  # 0-0 ... 3-2

# Global knobs / defaults
WILD_MULT_DEFAULT = 1.15      # >1 widens base location covariances (everyone a bit wilder)
ZONE_MARGIN_DEFAULT = 0.15    # feet beyond zone treated as “edge” for outside calls
DEFAULT_ZONE_GRID = {"x_edges": [-0.708, 0.0, 0.708], "z_edges": [1.5, 2.5, 3.5]}
DEFAULT_ZONE_CELLS_FB = {
    "r0c0": {"BallCalled":0.08,"StrikeCalled":0.46,"StrikeSwinging":0.15,"Foul":0.21,"InPlay":0.10},
    "r0c1": {"BallCalled":0.07,"StrikeCalled":0.50,"StrikeSwinging":0.18,"Foul":0.15,"InPlay":0.10},
    "r0c2": {"BallCalled":0.09,"StrikeCalled":0.44,"StrikeSwinging":0.16,"Foul":0.21,"InPlay":0.10},
    "r1c0": {"BallCalled":0.09,"StrikeCalled":0.43,"StrikeSwinging":0.17,"Foul":0.21,"InPlay":0.10},
    "r1c1": {"BallCalled":0.05,"StrikeCalled":0.55,"StrikeSwinging":0.20,"Foul":0.15,"InPlay":0.05},
    "r1c2": {"BallCalled":0.09,"StrikeCalled":0.43,"StrikeSwinging":0.17,"Foul":0.21,"InPlay":0.10},
    "r2c0": {"BallCalled":0.10,"StrikeCalled":0.42,"StrikeSwinging":0.16,"Foul":0.22,"InPlay":0.10},
    "r2c1": {"BallCalled":0.08,"StrikeCalled":0.46,"StrikeSwinging":0.16,"Foul":0.20,"InPlay":0.10},
    "r2c2": {"BallCalled":0.11,"StrikeCalled":0.41,"StrikeSwinging":0.16,"Foul":0.22,"InPlay":0.10},
}
DEFAULT_OUTSIDE = {
    "edge":  {"BallCalled":0.45,"StrikeCalled":0.18,"StrikeSwinging":0.12,"Foul":0.17,"InPlay":0.08},
    "chase": {"BallCalled":0.72,"StrikeCalled":0.03,"StrikeSwinging":0.09,"Foul":0.09,"InPlay":0.07},
}

# Canonical columns
PITCH_COL = {"baseball": "TaggedPitchType", "softball": "TaggedPitchtype"}
HAND_COL  = {"baseball": "PitcherThrows",   "softball": "PitcherThrows"}
BATTER_COL = "BatterSide"

# Selected features for pitch priors
CORE_FEATURES = dict(
    velo="RelSpeed",
    spin="SpinRate",
    ivb ="InducedVertBreak",
    hb  ="HorzBreak",
)

EXTRA_FEATURES = [
    "PlateLocHeight","PlateLocSide",
    "RelHeight","RelSide","Extension",
    "VertApprAngle","HorzApprAngle","ZoneSpeed","ZoneTime",
    "SpinAxis","Tilt","VertRelAngle","HorzRelAngle","VertBreak",
    "pfxx","pfxz","x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0",
]

BATTED_BALL_FEATURES = [
    "ExitSpeed","Angle","Direction","HitSpinRate",
    "PositionAt110X","PositionAt110Y","PositionAt110Z",
    "Distance","LastTrackedDistance","Bearing","HangTime",
]

# >>> Expanded in-play results <<<
INPLAY_RESULTS = [
    "Out","FielderChoice","Error","Sacrifice",
    "Single","Double","Triple","HomeRun"
]

PITCH_CALL_BUCKETS = ["BallCalled","StrikeCalled","StrikeSwinging","Foul","InPlay"]

# Prefer fast C dumper if available
DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)

# --------------- Small helpers ---------------
def r(x):
    if x is None: return None
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x): return None
        return round(float(x), ROUND)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x

def roundf(x): return r(x)

def statpack(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return None
    return [r(s.mean()), r(s.std(ddof=0)), r(s.min()), r(s.max())]

def prob_dict(series, keys=None):
    s = series.dropna().astype(str)
    if keys is None:
        vc = s.value_counts()
    else:
        vc = pd.Series({k: (s == k).sum() for k in keys})
    vc = vc + SMOOTH
    z = float(vc.sum()) if float(vc.sum()) > 0 else 1.0
    probs = (vc / z).to_dict()
    return {k: r(float(v)) for k, v in probs.items() if v > 0}

def compact_pmf_from_counts(series, topk=PMF_TOPK):
    s = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    if s.empty: return {}
    vc = s.value_counts()
    if len(vc) > topk:
        vc = vc.sort_values(ascending=False).head(topk)
    vc = vc.sort_index()
    vc = vc + SMOOTH
    z = float(vc.sum())
    return {int(k): r(float(v/z)) for k, v in vc.items()}

def normalize_hands(s, kind="pitcher"):
    m = {"Right":"R","Left":"L","R":"R","L":"L","RHP":"R","LHP":"L","S":"S","Switch":"S"}
    s = s.map(m).fillna(s)
    if kind == "pitcher":
        s = s.where(s.isin(["R","L"]), np.nan)
    else:
        s = s.where(s.isin(["R","L","S"]), np.nan)
    return s

def to_builtin(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(to_builtin(k)): to_builtin(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(v) for v in obj]
    return obj

# --------- Cluster helpers (archetypes) ----------
def safe_mean(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return None if x.empty else float(x.mean())

def safe_std(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return None if x.empty else float(x.std(ddof=0))

def zone_pct(df):
    if "PitchCall" in df.columns:
        if "InZone" in df.columns:
            z = pd.to_numeric(df["InZone"], errors="coerce")
            z = z[~z.isna()]
            return None if z.empty else float(z.mean())
    if LOC_X in df.columns and LOC_Z in df.columns:
        x = pd.to_numeric(df[LOC_X], errors="coerce")
        z = pd.to_numeric(df[LOC_Z], errors="coerce")
        m = ~(x.isna() | z.isna())
        if not m.any(): return None
        in_zone = (x[m].between(-PLATE_HALF_WIDTH, PLATE_HALF_WIDTH)) & (z[m].between(ZONE_Z_LOW, ZONE_Z_HIGH))
        return float(in_zone.mean())
    return None

def command_proxy(df):
    """Higher = better command. Inverse of location covariance area."""
    if LOC_X not in df.columns or LOC_Z not in df.columns:
        return None
    x = pd.to_numeric(df[LOC_X], errors="coerce")
    z = pd.to_numeric(df[LOC_Z], errors="coerce")
    m = ~(x.isna() | z.isna())
    if not m.any(): return None
    X = np.vstack([x[m].values, z[m].values]).T
    cov = np.cov(X, rowvar=False, ddof=0)
    try:
        area = float(np.linalg.det(cov))
    except Exception:
        return None
    if area <= 0 or np.isnan(area): return None
    return float(1.0 / (1.0 + area) * 2.0)

def compute_usage(df, pitch_col):
    counts = df[pitch_col].value_counts(dropna=True)
    total = float(counts.sum()) if counts.sum() else 1.0
    return {p: float(c/total) for p,c in counts.items()}

def boolean_presence(usg, threshold=0.03):
    return {p: float(usg.get(p,0.0) >= threshold) for p in CANON_PITCHES}

def fb_stats(df, pitch_col):
    fb = df[df[pitch_col] == FB]
    if fb.empty:
        return {"velo": None, "ivb": None, "hb": None, "rel_h": None, "rel_x": None,
                "spin": None, "spinaxis": None, "ext": None, "rel_ang_v": None, "rel_ang_h": None}
    return {
        "velo": safe_mean(fb.get("RelSpeed")),
        "ivb":  safe_mean(fb.get("InducedVertBreak")),
        "hb":   safe_mean(fb.get("HorzBreak")),
        "rel_h": safe_mean(fb.get("RelHeight")),
        "rel_x": safe_mean(fb.get("RelSide")),
        "spin": safe_mean(fb.get("SpinRate")),
        "spinaxis": safe_mean(fb.get("SpinAxis")),
        "ext": safe_mean(fb.get("Extension")),
        "rel_ang_v": safe_mean(fb.get("VertRelAngle")),
        "rel_ang_h": safe_mean(fb.get("HorzRelAngle")),
    }

def build_pitcher_table(df, hand_col, pitch_col):
    pid = None
    for c in ["PitcherId","PitcherID","Pitcher","playerID","PitcherName"]:
        if c in df.columns: pid = c; break
    if pid is None:
        raise SystemExit("❌ No pitcher identifier column found (e.g., PitcherId/Pitcher).")

    rows = []
    for (p_id, hand), g in df.groupby([pid, hand_col], dropna=False):
        if pd.isna(hand) or pd.isna(p_id): continue
        usage = compute_usage(g, pitch_col)
        presence = boolean_presence(usage, threshold=0.03)
        fbs = fb_stats(g, pitch_col)
        vel_sd = safe_std(g.get("RelSpeed"))
        cmd = command_proxy(g)
        zrate = zone_pct(g)
        feat = {
            "PitcherId": p_id,
            "Hand": hand,
            "FB_Velo": fbs["velo"],
            "FB_IVB":  fbs["ivb"],
            "FB_HB":   fbs["hb"],
            "FB_Spin": fbs["spin"],
            "FB_SpinAxis": fbs["spinaxis"],
            "FB_Ext":  fbs["ext"],
            "FB_RelAngV": fbs["rel_ang_v"],
            "FB_RelAngH": fbs["rel_ang_h"],
            "RelH":    fbs["rel_h"],
            "RelX":    fbs["rel_x"],
            "Vel_SD":  vel_sd,
            "CommandScore": cmd,
            "ZonePct": zrate,
            **{f"U_{p}": float(usage.get(p,0.0)) for p in CANON_PITCHES},
            **{f"P_{p}": float(presence[p]) for p in CANON_PITCHES},
        }
        rows.append(feat)
    return pd.DataFrame(rows)

def fit_gmm_by_hand(tbl, hand, k_min=4, k_max=10, random_state=7):
    sub = tbl[tbl["Hand"]==hand].copy()
    if sub.empty:
        return None, None, None

    feat_cols = [
        "FB_Velo","FB_IVB","FB_HB","FB_Spin","FB_SpinAxis","FB_Ext","FB_RelAngV","FB_RelAngH",
        "RelH","RelX","Vel_SD","CommandScore","ZonePct"
    ] + [f"U_{p}" for p in CANON_PITCHES] + [f"P_{p}" for p in CANON_PITCHES]

    X = sub[feat_cols].astype(float)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scl = StandardScaler()
    Xz = scl.fit_transform(X_imp)

    best = None
    best_bic = np.inf
    for k in range(k_min, k_max+1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state, n_init=5)
        gmm.fit(Xz)
        bic = gmm.bic(Xz)
        if bic < best_bic:
            best = gmm; best_bic = bic

    labels = best.predict(Xz)
    sub = sub.assign(_cluster=labels)
    return sub, best, (imp, scl, feat_cols)

def summarize_clusters(assigned, hand):
    clusters = {}
    total = len(assigned)
    for idx, grp in assigned.groupby("_cluster"):
        weight = float(len(grp) / total)

        presence_prob = {p: float(grp[f"P_{p}"].mean()) for p in CANON_PITCHES}
        usage_mean    = {p: float(grp[f"U_{p}"].mean()) for p in CANON_PITCHES}

        cmd_prior  = float(np.nanmean(grp["CommandScore"])) if "CommandScore" in grp else None
        zone_target = float(np.nanmean(grp["ZonePct"])) if "ZonePct" in grp else None

        fb_velo = float(np.nanmean(grp["FB_Velo"]))
        fb_ivb  = float(np.nanmean(grp["FB_IVB"]))
        fb_hb   = float(np.nanmean(grp["FB_HB"]))
        fb_spin = float(np.nanmean(grp["FB_Spin"]))
        fb_axis = float(np.nanmean(grp["FB_SpinAxis"]))
        fb_ext  = float(np.nanmean(grp["FB_Ext"]))
        fb_rv   = float(np.nanmean(grp["FB_RelAngV"]))
        fb_rh   = float(np.nanmean(grp["FB_RelAngH"]))
        relh    = float(np.nanmean(grp["RelH"]))
        relx    = float(np.nanmean(grp["RelX"]))

        label = f"{'power' if (fb_velo and fb_velo>=92) else 'soft'}_" \
                f"{'sink' if (fb_ivb and fb_ivb<10) else 'ride'}_{hand.lower()}"

        clusters[f"{label}_{idx}"] = {
            "prior_weight": r(weight),
            hand: {
                "command_tier_prior": r(cmd_prior) if cmd_prior is not None else 1.0,
                "global_zone_target": r(zone_target) if zone_target is not None else 0.5,
                "repertoire_presence": {p: r(presence_prob[p]) for p in CANON_PITCHES},
                "baseline_usage": {p: r(usage_mean[p]) for p in CANON_PITCHES},
                "summary_fastball": {
                    "velo": r(fb_velo), "ivb": r(fb_ivb), "hb": r(fb_hb),
                    "spin": r(fb_spin), "spinaxis": r(fb_axis), "extension": r(fb_ext),
                    "rel_ang_v": r(fb_rv), "rel_ang_h": r(fb_rh),
                    "rel_height": r(relh), "rel_side": r(relx),
                },
            }
        }
    return clusters

def zone_edges_3x3():
    xl, xr = -PLATE_HALF_WIDTH, PLATE_HALF_WIDTH
    zl, zr = ZONE_Z_LOW, ZONE_Z_HIGH
    xe = np.linspace(xl, xr, 4)
    ze = np.linspace(zl, zr, 4)
    return xe, ze

def in_zone_mask(x, z):
    return (x >= -PLATE_HALF_WIDTH) & (x <= PLATE_HALF_WIDTH) & \
           (z >= ZONE_Z_LOW) & (z <= ZONE_Z_HIGH)

def near_edge_mask(x, z, margin=None):
    if margin is None:
        margin = ZONE_MARGIN
    inside = in_zone_mask(x, z)
    xl, xr = -PLATE_HALF_WIDTH, PLATE_HALF_WIDTH
    zl, zr = ZONE_Z_LOW, ZONE_Z_HIGH
    dx = np.where(x < xl, xl - x, np.where(x > xr, x - xr, 0.0))
    dz = np.where(z < zl, zl - z, np.where(z > zr, z - zr, 0.0))
    dist = np.hypot(dx, dz)
    return (~inside) & (dist <= margin)

def build_pitch_call_by_zone(df_sub):
    need_cols = {"PlateLocSide","PlateLocHeight","PitchCall"}
    if not need_cols.issubset(set(df_sub.columns)):
        return {}

    s = pd.to_numeric(df_sub["PlateLocSide"], errors="coerce")
    h = pd.to_numeric(df_sub["PlateLocHeight"], errors="coerce")
    pc = df_sub["PitchCall"].astype(str)
    m = ~(s.isna() | h.isna() | pc.isna())
    if not m.any():
        return {}

    s, h, pc = s[m].values, h[m].values, pc[m].values

    buckets = ["BallCalled","StrikeCalled","StrikeSwinging","Foul","InPlay"]
    pc = np.where(np.isin(pc, buckets), pc, None)
    keep = pc != np.array(None)
    if not keep.any():
        return {}

    s, h, pc = s[keep], h[keep], pc[keep]

    xe, ze = zone_edges_3x3()
    inside = in_zone_mask(s, h)
    cells = {}
    if inside.any():
        xi = np.clip(np.digitize(s[inside], xe) - 1, 0, 2)
        zi = np.clip(np.digitize(h[inside], ze) - 1, 0, 2)
        pc_in = pc[inside]
        for ri in range(3):
            for ci in range(3):
                mask = (zi == ri) & (xi == ci)
                if not mask.any():
                    continue
                vc = pd.Series(pc_in[mask]).value_counts()
                probs = (vc + 1e-9) / float(vc.sum() + 1e-9*len(vc))
                cells[f"r{ri}c{ci}"] = {k: float(probs.get(k, 0.0)) for k in buckets if probs.get(k, 0.0) > 0.0}

    edge = near_edge_mask(s, h, margin=ZONE_MARGIN)
    outside = ~inside
    chase = outside & (~edge)

    def probs_for(mask):
        if not mask.any():
            return {}
        vc = pd.Series(pc[mask]).value_counts()
        probs = (vc + 1e-9) / float(vc.sum() + 1e-9*len(vc))
        return {k: float(probs.get(k, 0.0)) for k in buckets if probs.get(k, 0.0) > 0.0}

    outside_dict = {"edge": probs_for(edge), "chase": probs_for(chase)}
    grid = {
        "x_edges": [float(x) for x in xe],
        "z_edges": [float(z) for z in ze],
        "cells": cells
    }
    out = {"grid": grid, "outside": outside_dict}
    if not cells and not any(outside_dict.values()):
        return {}
    return out

# ---- Count transition helpers (SAFE) ----
def _parse_count(c):
    """
    Extract B-S from anything that looks like it (e.g., '0-1', '0 - 1', '0-1 (auto)').
    Returns (b, s) as ints, or None if not matched.
    """
    m = re.search(r'([0-3])\s*-\s*([0-2])', str(c))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def _next_count_from_pitch(balls, strikes, pitch_call):
    pc = (str(pitch_call or "").strip())
    if pc in {"HitByPitch"}:
        return (balls, strikes, "End:HBP")
    if pc in {"BallCalled", "Ball"}:
        nb = balls + 1
        if nb >= 4: return (3, strikes, "End:Walk")
        return (nb, strikes, None)
    if pc in {"StrikeCalled", "StrikeSwinging"}:
        ns = strikes + 1
        if ns >= 3: return (balls, 2, "End:K")
        return (balls, ns, None)
    if pc in {"FoulBall", "Foul"}:
        if strikes < 2: return (balls, strikes + 1, None)
        return (balls, 2, None)
    if pc in {"InPlay"}:
        return (balls, strikes, "End:InPlay")
    # default → treat as ball
    nb = balls + 1
    if nb >= 4: return (3, strikes, "End:Walk")
    return (nb, strikes, None)

def is_legal_transition(from_count: str, to_count: str) -> bool:
    """
    One-pitch legality:
      - Either +1 ball OR +1 strike
      - No decreases
      - Staying the same only allowed at 2 strikes (foul with 2 strikes).
    """
    pf = _parse_count(from_count)
    pt = _parse_count(to_count)
    if pf is None or pt is None:
        return False
    b, s = pf
    nb, ns = pt
    db, ds = nb - b, ns - s
    if db == 1 and ds == 0: return True
    if db == 0 and ds == 1: return True
    if db == 0 and ds == 0 and s == 2 and ns == 2: return True
    return False

def build_count_transitions(df, pa_cols=("GameID","Inning","ABNum","PlateApprID")):
    """
    df must include: Count ('B-S'), PitchCall.
    Returns:
      transitions: {from_count: {to_count or terminal: prob}}
      start_dist : {start_count: prob}
    """
    pa_keys = [c for c in pa_cols if c in df.columns] or \
              [c for c in ["GameID","Pitcher","Batter"] if c in df.columns]

    transitions = defaultdict(Counter)
    start_counts = Counter()

    sort_cols = [c for c in ["GameID","Inning","ABNum","PlateApprID","PitchofPA","PitchNo"] if c in df.columns]
    df_sorted = df.sort_values(sort_cols, kind="mergesort") if sort_cols else df

    for _, pa in df_sorted.groupby(pa_keys, dropna=False):
        if pa.empty: continue
        first = pa.iloc[0]
        parsed = _parse_count(first.get("Count"))
        if not parsed: continue
        cb, cs = parsed
        start_counts[f"{cb}-{cs}"] += 1

        for _, row in pa.iterrows():
            from_c = f"{cb}-{cs}"
            nb, ns, terminal = _next_count_from_pitch(cb, cs, row.get("PitchCall"))
            if terminal:
                transitions[from_c][terminal] += 1
                break
            else:
                to_c = f"{nb}-{ns}"
                transitions[from_c][to_c] += 1
                cb, cs = nb, ns

    trans_probs = {}
    for from_c, hist in transitions.items():
        total = sum(hist.values())
        if total:
            trans_probs[from_c] = {k: v/total for k, v in hist.items()}

    total_starts = sum(start_counts.values()) or 1
    start_dist = {k: v/total_starts for k, v in start_counts.items()}
    return trans_probs, start_dist

def tilt_prob(series):
    """
    Build probabilities for Tilt with an explicit 'Unknown' bucket.
    - Known values: normalized frequency
    - Unknown (NaN or empty): share of rows missing
    """
    s = series.astype("object")
    n = len(s)
    if n == 0:
        return {}

    known_mask = s.notna() & (s.astype(str).str.strip() != "")
    vals = s[known_mask].astype(str).str.strip()

    counts = vals.value_counts()
    probs = {str(k): round(float(v) / n, 3) for k, v in counts.items()}

    unknown_share = (n - int(known_mask.sum())) / n
    if unknown_share > 0:
        probs["Unknown"] = round(float(unknown_share), 3)

    return probs

# --- NEW: canonicalize PlayResult into our INPLAY_RESULTS set ---
def canonical_play_result(val: object):
    """
    Map raw PlayResult text to one of:
      Out, FielderChoice, Error, Sacrifice, Single, Double, Triple, HomeRun
    Returns the canonical string or None if not mappable.
    """
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    low = s.lower()

    # Singles / extra-base hits
    if low in {"single","1b"}:
        return "Single"
    if low in {"double","2b"}:
        return "Double"
    if low in {"triple","3b"}:
        return "Triple"
    if low in {"home run","homerun","hr","home-run"}:
        return "HomeRun"

    # Sacrifice (bunt or fly)
    if "sac" in low or low in {"sacrifice","sac fly","sacrifice fly","sacrifice bunt","sac bunt","sf","sh"}:
        return "Sacrifice"

    # Error (reached on error)
    if "error" in low or low in {"roe","reached on error"}:
        return "Error"

    # Fielder's choice
    if low in {"fielderschoice","fielderchoice","fielder's choice","fielder’s choice","fc"}:
        return "FielderChoice"

    # General outs
    if "out" in low or low in {
        "groundout","flyout","lineout","popup","popout","strikeout","buntout"
    }:
        return "Out"

    # Some feeds encode plays like "InPlay, Out(s)" – keep as Out
    if "inplay" in low and "out" in low:
        return "Out"

    return None

# ---------------- Main build ----------------
def main():
    par = argparse.ArgumentParser("Build Priors + Archetypes YAML from CSVs")
    par.add_argument("--csv_folder", required=True, help="Folder with tagged CSVs")
    par.add_argument("--out", required=True, help="Output YAML path")
    par.add_argument("--sport", default=SPORT, choices=["baseball","softball"])
    par.add_argument("--kmin", type=int, default=9)
    par.add_argument("--kmax", type=int, default=9)
    par.add_argument("--random_state", type=int, default=7)
    # NEW CLI knobs
    par.add_argument("--wild_mult", type=float, default=WILD_MULT_DEFAULT,
                     help="Inflate location covariances globally (>=1.0 makes everyone wilder).")
    par.add_argument("--zone_margin", type=float, default=ZONE_MARGIN_DEFAULT,
                     help="Feet beyond the zone considered 'edge' for outside-call buckets.")
    par.add_argument("--no_fallback_zone_defaults", action="store_true",
                     help="Disable inserting default pitch_call_by_zone when data is insufficient.")
    args = par.parse_args()

    pitch_col = PITCH_COL[args.sport]
    hand_col  = HAND_COL[args.sport]

    # Bind globals for helper functions
    global WILD_MULT, ZONE_MARGIN, FALLBACK_ZONE_DEFAULTS
    WILD_MULT = float(args.wild_mult)
    ZONE_MARGIN = float(args.zone_margin)
    FALLBACK_ZONE_DEFAULTS = not bool(args.no_fallback_zone_defaults)

    csv_dir = Path(args.csv_folder)
    files = list(csv_dir.glob("*.csv"))
    if not files:
        sys.exit(f"❌ No CSVs found in {csv_dir}")

    print(f"▶ Loading {len(files)} CSV file(s)…")
    df_list = []
    for f in files:
        try:
            d = pd.read_csv(f, low_memory=False)
            d["__source_file__"] = f.name
            df_list.append(d)
        except Exception as e:
            print(f"⚠️  Skipping {f.name}: {e}")
    if not df_list:
        sys.exit("❌ No readable CSVs.")

    df = pd.concat(df_list, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]

    # Normalize hands
    if hand_col in df.columns:
        df[hand_col] = normalize_hands(df[hand_col], kind="pitcher")
    if BATTER_COL in df.columns:
        df[BATTER_COL] = normalize_hands(df[BATTER_COL], kind="batter")

    # ---------- Ensure Count & PitchCall exist with expected names ----------
    if "PitchCall" not in df.columns and "TaggedPitchCall" in df.columns:
        df["PitchCall"] = df["TaggedPitchCall"]

    if "PitchCall" in df.columns:
        pc_map = {
            "CalledStrike": "StrikeCalled",
            "StrikeCalled": "StrikeCalled",
            "StrikeSwinging": "StrikeSwinging",
            "SwingingStrike": "StrikeSwinging",
            "BallCalled": "BallCalled",
            "Ball": "BallCalled",
            "FoulBall": "Foul",
            "FoulTip": "Foul",
            "InPlay": "InPlay",
            "HitByPitch": "HitByPitch",
        }
        df["PitchCall"] = df["PitchCall"].astype(str).str.strip().map(lambda x: pc_map.get(x, x))

    # Count present? clean; else build from balls/strikes or PitchCount
    if "Count" in df.columns:
        df["Count"] = df["Count"].astype(str).str.extract(r"^\s*([0-3]\s*-\s*[0-2])", expand=True)[0]
        df["Count"] = df["Count"].str.replace(r"\s*", "", regex=True)
    else:
        ball_cols    = [c for c in ["Balls","Ball","B","balls"] if c in df.columns]
        strike_cols  = [c for c in ["Strikes","Strike","S","strikes"] if c in df.columns]
        if ball_cols and strike_cols:
            b = pd.to_numeric(df[ball_cols[0]], errors="coerce").fillna(0).clip(0, 3).astype(int)
            s = pd.to_numeric(df[strike_cols[0]], errors="coerce").fillna(0).clip(0, 2).astype(int)
            df["Count"] = b.astype(str) + "-" + s.astype(str)
        elif "PitchCount" in df.columns:
            df["Count"] = df["PitchCount"].astype(str).str.extract(r"^\s*([0-3]\s*-\s*[0-2])", expand=True)[0]
            df["Count"] = df["Count"].str.replace(r"\s*", "", regex=True)
    # -----------------------------------------------------------------------

    # Basic checks
    for c in [pitch_col, hand_col]:
        if c not in df.columns:
            sys.exit(f"❌ Missing required column: {c}")

    # --- META ---
    source_files = sorted(df["__source_file__"].dropna().unique().tolist()) if "__source_file__" in df.columns else []

    # --- DISTRIBUTIONS: Hands ---
    dist_pitcher = prob_dict(df[hand_col], keys=["R","L"])
    dist_batter  = prob_dict(df[BATTER_COL], keys=["R","L","S"]) if BATTER_COL in df.columns else {"R":0.5,"L":0.5,"S":0.0}

    # BatterSide conditioned on PitcherHand
    bph = {}
    if BATTER_COL in df.columns:
        for ph in ["R","L"]:
            mask = df[hand_col] == ph
            if mask.any():
                bph[ph] = prob_dict(df.loc[mask, BATTER_COL], keys=["R","L","S"])

    # --- DISTRIBUTIONS: PitchesPerInning ---
    inning_keys = [c for c in ["GameID","Top/Bottom","Inning"] if c in df.columns]
    pitches_per_inning = {}
    if inning_keys:
        inn_counts = df.groupby(inning_keys, dropna=False).size()
        if len(inn_counts) > 0:
            pitches_per_inning = {
                "mean": r(inn_counts.mean()),
                "sd":   r(inn_counts.std(ddof=0) if len(inn_counts)>1 else 0.0),
                "min":  int(inn_counts.min()),
                "max":  int(inn_counts.max()),
                "pmf":  compact_pmf_from_counts(inn_counts),
            }

    # --- DISTRIBUTIONS: PitchesPerAB ---
    pa_keys = [c for c in ["GameID","Top/Bottom","Inning","PAofInning"] if c in df.columns]
    pitches_per_ab = {}
    if pa_keys:
        pa_counts = df.groupby(pa_keys, dropna=False).size()
        if len(pa_counts) > 0:
            pitches_per_ab = compact_pmf_from_counts(pa_counts)

    # --- GLOBAL outcomes_overall ---
    outcomes_overall = {}
    if "PitchCall" in df.columns:
        pc_all = df["PitchCall"].astype(str)
        pc_all = pc_all.where(pc_all.isin(PITCH_CALL_BUCKETS), other=np.nan)
        outcomes_overall = prob_dict(pc_all, keys=PITCH_CALL_BUCKETS)

    # --- PITCH ROOT MIX ---
    pitch_mix = df[pitch_col].value_counts(normalize=True)
    pitch_mix = (pitch_mix / pitch_mix.sum()).to_dict()

    # --- Build per-pitch-type blocks (incl. 2D location model) ---
    pitches_block = {}
    for pt, pt_df in df.groupby(pitch_col, dropna=False):
        if pd.isna(pt): continue

        mix_pct = float(pitch_mix.get(pt, 0.0))
        hands_block = {}

        for hand, sub in pt_df.groupby(hand_col, dropna=False):
            if pd.isna(hand): continue

            # core stats
            core = {"n": int(len(sub))}
            for k, colname in CORE_FEATURES.items():
                if colname in sub.columns:
                    sp = statpack(sub[colname])
                    if sp: core[k] = sp

            # extra features
            features = {}
            for col in EXTRA_FEATURES:
                if col not in sub.columns: continue
                if col == "Tilt":
                    tilt_probs = tilt_prob(sub[col])
                    if tilt_probs:
                        features[col] = tilt_probs
                else:
                    sp = statpack(sub[col])
                    if sp: features[col] = sp

            # outcomes (pitch-call buckets)
            if "PitchCall" in sub.columns:
                pc = sub["PitchCall"].astype(str)
                pc = pc.where(pc.isin(PITCH_CALL_BUCKETS), other=np.nan)
                outcomes = prob_dict(pc, keys=PITCH_CALL_BUCKETS)
            else:
                outcomes = {}
            outcomes_mean = dict(outcomes)

            # --- In-play split with expanded categories ---
            inplay_split = {}
            if "PitchCall" in sub.columns and "PlayResult" in sub.columns:
                mask = sub["PitchCall"].astype(str) == "InPlay"
                if mask.any():
                    plays_raw = sub.loc[mask, "PlayResult"]
                    plays = plays_raw.map(canonical_play_result)
                    inplay_split = prob_dict(plays.dropna(), keys=INPLAY_RESULTS)

            # hit_types (in-play only)
            hit_types = {}
            if "PitchCall" in sub.columns and "HitType" in sub.columns:
                im = sub["PitchCall"].astype(str) == "InPlay"
                if im.any():
                    hit_types = prob_dict(sub.loc[im, "HitType"])

            # outcomes_detail: batted-ball statpacks per in-play result (expanded)
            outcomes_detail = {}
            if "PitchCall" in sub.columns and "PlayResult" in sub.columns:
                inplay = sub[sub["PitchCall"].astype(str) == "InPlay"].copy()
                if not inplay.empty:
                    inplay["__CanonPR__"] = inplay["PlayResult"].map(canonical_play_result)
                    for res in INPLAY_RESULTS:
                        grp = inplay[inplay["__CanonPR__"] == res]
                        if grp.empty:
                            continue
                        pack = {}
                        for bb in BATTED_BALL_FEATURES:
                            if bb in grp.columns:
                                sp = statpack(grp[bb])
                                if sp: pack[bb] = sp
                        if pack:
                            outcomes_detail[res] = pack

            # 2D Location Model
            loc_model = {}
            if LOC_X in sub.columns and LOC_Z in sub.columns:
                xs = pd.to_numeric(sub[LOC_X], errors="coerce")
                zs = pd.to_numeric(sub[LOC_Z], errors="coerce")
                m = ~(xs.isna() | zs.isna())
                if m.any():
                    X = np.vstack([xs[m].values, zs[m].values]).T
                    mu = X.mean(axis=0)
                    cov = np.cov(X, rowvar=False, ddof=0)

                    # floor tiny variances and inflate with wild_mult
                    cov = np.array(cov, dtype=float)
                    cov[0,0] = max(cov[0,0], 1e-5)
                    cov[1,1] = max(cov[1,1], 1e-5)
                    cov *= WILD_MULT

                    loc_model = {
                        "units": "feet",
                        "n": int(m.sum()),
                        "mean": [r(mu[0]), r(mu[1])],
                        "cov": [[r(cov[0,0]), r(cov[0,1])],
                                [r(cov[1,0]), r(cov[1,1])]],
                    }

            hand_key = str(hand)
            hblock = {"core": core}
            if features:        hblock["features"] = features
            if outcomes:        hblock["outcomes"] = outcomes
            if outcomes_mean:   hblock["outcomes_mean"] = outcomes_mean
            if inplay_split:    hblock["inplay_split"] = inplay_split
            if hit_types:       hblock["hit_types"] = hit_types
            if outcomes_detail: hblock["outcomes_detail"] = outcomes_detail
            if loc_model:       hblock["location_model"] = loc_model
            pcbz = build_pitch_call_by_zone(sub)
            if pcbz:
                hblock["pitch_call_by_zone"] = pcbz
            elif FALLBACK_ZONE_DEFAULTS:
                hblock["pitch_call_by_zone"] = {
                    "grid": dict(DEFAULT_ZONE_GRID, cells=DEFAULT_ZONE_CELLS_FB),
                    "outside": DEFAULT_OUTSIDE
                }
            hands_block[hand_key] = hblock

        pitches_block[str(pt)] = { "mix_pct": r(mix_pct), "hands": hands_block }

    # ---------------- Pitcher archetypes ----------------
    pid_col = None
    for c in ["PitcherId","PitcherID","Pitcher","playerID","PitcherName"]:
        if c in df.columns: pid_col = c; break
    if pid_col is None:
        print("⚠️  No pitcher id column found; skipping clusters section.")
        clusters = {}
    else:
        ptbl = build_pitcher_table(df, hand_col, pitch_col)
        clusters = {}
        if not ptbl.empty:
            for h in ["R","L"]:
                assigned, gmm, xform = fit_gmm_by_hand(ptbl, h, args.kmin, args.kmax, args.random_state)
                if assigned is None: continue
                clusters.update(summarize_clusters(assigned, h))

    # --- Final YAML object ---
    cfg = {
        "version": 1,
        "schema_version": "1.1",
        "meta": {"source_files": source_files},
        "distributions": {
            "PitcherHand": dist_pitcher or {"R":0.5,"L":0.5},
            "BatterSide":  dist_batter  or {"R":0.5,"L":0.5,"S":0.0},
        },
        "pitches": pitches_block,
        "clusters": clusters
    }
    if bph:
        cfg["distributions"]["BatterSideGivenPitcherHand"] = bph
    if 'pitches_per_inning' in locals() and pitches_per_inning:
        cfg["distributions"]["PitchesPerInning"] = pitches_per_inning
    if 'pitches_per_ab' in locals() and pitches_per_ab:
        cfg["distributions"]["PitchesPerAB"] = pitches_per_ab
    if outcomes_overall:
        cfg["distributions"]["outcomes_overall"] = outcomes_overall

    # --- Count transitions + pitch mix by count + legal joint ---
    if "Count" in df.columns and df["Count"].notna().any() and "PitchCall" in df.columns:
        transitions, start_dist = build_count_transitions(df)
        cfg["counts"] = {
            "start_distribution": start_dist or {"0-0": 1.0},
            "transitions": transitions,
        }

        pmix = (
            df.dropna(subset=["Count", pitch_col])
              .groupby(["Count", pitch_col])
              .size()
              .groupby(level=0)
              .apply(lambda s: (s/s.sum()).to_dict())
              .to_dict()
        )
        if pmix:
            cleaned_pmix = {}
            for k, v in pmix.items():
                ck = re.sub(r"\s*", "", str(k))  # "0 - 1" -> "0-1"
                if isinstance(v, dict):
                    vv = {str(pk): float(pv) for pk, pv in v.items() if pv == pv}
                    z = sum(vv.values())
                    if z > 0:
                        vv = {pk: pv / z for pk, pv in vv.items()}
                    cleaned_pmix[ck] = vv
            cfg["pitch_mix_by_count"] = cleaned_pmix

        counts_trans  = cfg["counts"]["transitions"]
        pmix_by_count = cfg.get("pitch_mix_by_count", {})

        joint = {}
        for from_c, pitch_pmf in pmix_by_count.items():
            if _parse_count(from_c) is None:
                continue
            trans_pmf = counts_trans.get(from_c, {})
            for to_c in ALL_COUNTS:
                if not is_legal_transition(from_c, to_c):
                    continue
                p_trans = float(trans_pmf.get(to_c, 0.0))
                for pitch, p_pitch in pitch_pmf.items():
                    p = float(p_pitch) * p_trans
                    if p != p:
                        p = 0.0
                    if p > 0.0:
                        joint[(from_c, (to_c, pitch))] = p
        cfg["joint_count_pitch"] = {f"{fc}|{tc}|{pt}": v for (fc, (tc, pt)), v in joint.items()}

    # --- Save ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(to_builtin(cfg), f, sort_keys=False, allow_unicode=True, Dumper=DUMPER)

    print(f"✅ Combined Priors + Archetypes YAML written to {out_path}")

if __name__ == "__main__":
    main()
