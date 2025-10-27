# loc_params_builder.py
from __future__ import annotations
import math, csv
from typing import Dict, Any, Tuple
from aim_engine import AimParams, PLATE_HALF_WIDTH, ZONE_Z_LOW, ZONE_Z_HIGH

def _safe(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict): return default
        cur = cur.get(k, None)
        if cur is None: return default
    return cur

def build_params_from(priors: Dict[str,Any], rules_yaml: Dict[str,Any] | None, pitcher_profiles_path: str | None) -> AimParams:
    p = AimParams()

    # 1) edge pads (fallback 0.15/0.15)
    pad_x = float(_safe(priors, "PitchCallZones", "edge_pad_x", default=0.15) or 0.15)
    pad_z = float(_safe(priors, "PitchCallZones", "edge_pad_z", default=0.15) or 0.15)
    p.edge_pad_x, p.edge_pad_z = pad_x, pad_z

    # 2) intent & region weights from rules_yaml if present
    # expected structure (flexible): rules_yaml["intent"][f"{pt}|{count}"] = {"zone":..., "shadow":...}
    if isinstance(rules_yaml, dict):
        for k,v in (rules_yaml.get("intent") or {}).items():
            p.intent_weights[str(k)] = {kk: float(vv) for kk,vv in v.items()}
        for k,v in (rules_yaml.get("region") or {}).items():
            p.region_weights[str(k)] = {kk: float(vv) for kk,vv in v.items()}

    # if not provided, leave empty → engine uses built-ins

    # 3) dispersion defaults per pitch type (edit if you like)
    p.sigma_base = {
        # (ft) horizontal ~ wider than vertical for FB/CT; CB has taller vertical
        "Fastball": (0.22, 0.18),
        "FourSeamFastball": (0.22, 0.18),
        "TwoSeamFastball":  (0.24, 0.18),
        "Sinker":           (0.24, 0.20),
        "Cutter":           (0.20, 0.18),
        "Slider":           (0.21, 0.20),
        "Sweeper":          (0.24, 0.20),
        "Curveball":        (0.20, 0.24),
        "KnuckleCurve":     (0.20, 0.24),
        "Changeup":         (0.22, 0.21),
        "Splitter":         (0.22, 0.22),
        "*":                (0.22, 0.20),
    }
    p.rho = {
        "Fastball": 0.10, "FourSeamFastball": 0.10, "TwoSeamFastball": 0.12,
        "Sinker": 0.12, "Cutter": 0.08, "Slider": 0.10, "Sweeper": 0.10,
        "Curveball": 0.05, "KnuckleCurve": 0.05, "Changeup": 0.12, "Splitter": 0.12,
        "*": 0.10,
    }

    # 4) miss bias (arm-side, down), tiny defaults
    p.miss_bias = {
        "*": {
            "R_R": (+0.05, -0.03), "R_L": (+0.06, -0.04),
            "L_R": (-0.06, -0.04), "L_L": (-0.05, -0.03),
        }
    }

    # 5) command scaling + count pressure
    p.alpha_cmd = 1.15   # raise to tighten more per unit of command
    p.count_pressure = {"behind": 0.06, "even": 0.02, "ahead": 0.00}

    # 6) per-pitcher overrides from CSV — OPTIONAL
    if pitcher_profiles_path:
        try:
            with open(pitcher_profiles_path, newline='', encoding="utf-8") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    pid = str(r.get("PitcherId") or r.get("id") or "").strip()
                    pt  = str(r.get("PitchType") or r.get("pt") or "").strip()
                    if not pid or not pt: continue
                    o = p.per_pitcher.setdefault(pid, {}).setdefault(pt, {})
                    # read optional override columns if they exist
                    for col, key in [("SigmaX","SigmaX"),("SigmaZ","SigmaZ"),("Rho","Rho"),
                                     ("ArmSideBias","ArmSideBias"),("DownBias","DownBias")]:
                        if r.get(col) not in (None, "", "NA"):
                            try:
                                o[key] = float(r[col])
                            except: pass
                    # convert ArmSide/Down to bias vector if both present
                    if "ArmSideBias" in o or "DownBias" in o:
                        ax = float(o.get("ArmSideBias", 0.0))
                        dz = float(o.get("DownBias", 0.0))
                        # store as (dx, dz); we’ll apply in engine by handedness pair if needed
                        o["SigmaX"] = o.get("SigmaX")  # keep existing if set
                        o["SigmaZ"] = o.get("SigmaZ")
                        o["Rho"]    = o.get("Rho")
        except Exception:
            pass

    return p
