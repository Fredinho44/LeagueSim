# aim_engine.py
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

# geometry defaults (feet)
PLATE_HALF_WIDTH = 0.708
ZONE_Z_LOW = 1.55
ZONE_Z_HIGH = 3.45

@dataclass
class AimParams:
    # weights
    intent_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)   # key: (pt|*):count_bucket -> {intent: w}
    region_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)   # key: (pt|*):(intent, bats) -> {region: w}

    # dispersion & bias defaults by pitch type (falls back to "*")
    sigma_base: Dict[str, Tuple[float, float]] = field(default_factory=dict)    # pt -> (sig_x, sig_z)
    rho: Dict[str, float] = field(default_factory=dict)                         # pt -> correlation [-0.5, 0.5]
    miss_bias: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    # miss_bias[pt][("R","R") or "R_L" etc.] -> (dx, dz)

    # command scaling
    alpha_cmd: float = 1.0               # how strongly command tightens the cloud (higher = tighter)
    min_sigma: float = 0.03              # floor on dispersion (ft)

    # count pressure pushes aim outward a bit (0–0.2 feet typical)
    count_pressure: Dict[str, float] = field(default_factory=dict)              # count_bucket -> outward ft

    # edge pads when aiming "shadow"
    edge_pad_x: float = 0.15
    edge_pad_z: float = 0.15

    # per-pitcher overrides (optional)
    per_pitcher: Dict[str, Dict[str, Any]] = field(default_factory=dict)        # pid -> {pt: {...overrides...}}

def _safe(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict): return default
        cur = cur.get(k, None)
        if cur is None: return default
    return cur

def _choose(weights: Dict[str, float], rng: random.Random) -> Optional[str]:
    if not weights: return None
    s = sum(max(0.0, float(v)) for v in weights.values()) or 1.0
    r = rng.random() * s
    acc = 0.0
    for k, v in weights.items():
        acc += max(0.0, float(v))
        if r <= acc: return k
    return next(iter(weights.keys()))

def _count_bucket(b: int, s: int) -> str:
    # match your existing count_bucket semantics
    if b - s >= 2: return "behind"   # pitcher behind
    if s - b >= 1: return "ahead"
    return "even"

def _clip(v, lo, hi): return max(lo, min(hi, v))

class PitchAimEngine:
    def __init__(self, params: AimParams, rng: Optional[random.Random] = None):
        self.p = params
        self.rng = rng or random.Random(17)

    def _get_pair_key(self, pthrows: str, bats: str) -> str:
        return f"{(pthrows or 'R')[0].upper()}_{(bats or 'R')[0].upper()}"

    def _target_for_region(self, region: str) -> Tuple[float, float]:
        # canonical target points by label (feet)
        cx = 0.0
        cz = 0.5 * (ZONE_Z_LOW + ZONE_Z_HIGH)
        # margins for shadow
        mx, mz = self.p.edge_pad_x, self.p.edge_pad_z

        # “zone_*” are inside plate; “shadow_*” are on the border; “chase_*” are just outside
        mapping = {
            "zone_center":        (cx, cz),
            "zone_up":            (cx, ZONE_Z_HIGH - mz),
            "zone_down":          (cx, ZONE_Z_LOW + mz),
            "zone_in":            (-PLATE_HALF_WIDTH + mx, cz),
            "zone_away":          ( PLATE_HALF_WIDTH - mx, cz),

            "shadow_up":          (cx, ZONE_Z_HIGH + 0.00),
            "shadow_down":        (cx, ZONE_Z_LOW  - 0.00),
            "shadow_in":          (-PLATE_HALF_WIDTH - 0.00, cz),
            "shadow_away":        ( PLATE_HALF_WIDTH + 0.00, cz),

            "chase_up":           (cx, ZONE_Z_HIGH + 0.25),
            "chase_down":         (cx, ZONE_Z_LOW  - 0.25),
            "chase_in":           (-PLATE_HALF_WIDTH - 0.25, cz),
            "chase_away":         ( PLATE_HALF_WIDTH + 0.25, cz),

            "waste_up":           (cx, ZONE_Z_HIGH + 0.6),
            "waste_down":         (cx, ZONE_Z_LOW  - 0.6),
            "waste_in":           (-PLATE_HALF_WIDTH - 0.6, cz),
            "waste_away":         ( PLATE_HALF_WIDTH + 0.6, cz),
        }
        return mapping.get(region, (cx, cz))

    def _anisotropic_sample(self, mu_x, mu_z, sig_x, sig_z, rho) -> Tuple[float,float]:
        # sample correlated normals via Cholesky
        u = self.rng.normalvariate(0.0, 1.0)
        v = self.rng.normalvariate(0.0, 1.0)
        x = mu_x + sig_x * u
        z = mu_z + sig_z * (rho * u + math.sqrt(max(1e-6, 1 - rho*rho)) * v)
        return (x, z)

    def sample(self, ctx: Dict[str, Any]) -> Optional[Tuple[float,float]]:
        """
        ctx: {
          pt, pthrows, bats, balls, strikes, command (float), pitcher_id
        }
        """
        pt = ctx.get("pt")
        pthrows = (ctx.get("pthrows") or "R")[0].upper()
        bats    = (ctx.get("bats") or "R")[0].upper()
        b,s     = int(ctx.get("balls",0)), int(ctx.get("strikes",0))
        cmd     = float(ctx.get("command", 1.0))
        pid     = str(ctx.get("pitcher_id") or "")

        cb = _count_bucket(b, s)

        # 1) choose intent (zone/shadow/chase/waste)
        iw = (_safe(self.p.intent_weights, f"{pt}|{cb}") or
              _safe(self.p.intent_weights, f"{pt}|*") or
              _safe(self.p.intent_weights, f"*|{cb}") or
              _safe(self.p.intent_weights, f"*|*") or
              {"zone": 0.55, "shadow": 0.30, "chase": 0.12, "waste": 0.03})

        intent = _choose(iw, self.rng) or "zone"

        # 2) choose region given intent, pitch, and platoon
        rw_key_options = [f"{pt}|{intent}|{bats}", f"{pt}|{intent}|*", f"*|{intent}|{bats}", f"*|{intent}|*"]
        rw = None
        for key in rw_key_options:
            rw = self.p.region_weights.get(key)
            if rw: break
        if not rw:
            base = {
                "zone":   {"zone_center":0.25,"zone_down":0.25,"zone_in":0.25,"zone_away":0.15,"zone_up":0.10},
                "shadow": {"shadow_away":0.35,"shadow_down":0.30,"shadow_in":0.20,"shadow_up":0.15},
                "chase":  {"chase_away":0.45,"chase_down":0.30,"chase_in":0.15,"chase_up":0.10},
                "waste":  {"waste_away":0.40,"waste_up":0.20,"waste_down":0.25,"waste_in":0.15},
            }
            rw = base.get(intent, base["zone"])
        region = _choose(rw, self.rng) or ("zone_center" if intent=="zone" else f"{intent}_away")

        # 3) base target point
        tx, tz = self._target_for_region(region)

        # 4) nudge outward for pressure counts (e.g., 3-0 → slightly more "waste")
        out_push = self.p.count_pressure.get(cb, 0.0)
        if "away" in region: tx += out_push if tx >= 0 else -out_push
        if "in" in region:   tx += -out_push if tx >= 0 else out_push
        if "up" in region:   tz += out_push
        if "down" in region: tz -= out_push

        # 5) per-pitcher overrides (optional)
        over = _safe(self.p.per_pitcher, pid, pt, default={}) or {}
        sig_x, sig_z = over.get("SigmaX"), over.get("SigmaZ")
        rho = over.get("Rho")
        if sig_x is None or sig_z is None:
            sx, sz = self.p.sigma_base.get(pt, self.p.sigma_base.get("*", (0.20, 0.18)))
            sig_x = sig_x if sig_x is not None else sx
            sig_z = sig_z if sig_z is not None else sz
        rho = rho if rho is not None else self.p.rho.get(pt, self.p.rho.get("*", 0.10))

        # 6) scale by command (higher cmd → tighter)
        # effective_sigma = base_sigma / (alpha_cmd * command)
        scale = max(0.35, self.p.alpha_cmd * max(0.25, cmd))
        sig_x = max(self.p.min_sigma, float(sig_x) / scale)
        sig_z = max(self.p.min_sigma, float(sig_z) / scale)
        rho = float(rho)

        # 7) systematic miss bias (arm-side, down) by pt & handedness pair
        pair_key = f"{pthrows}_{bats}"
        bx, bz = (0.0, 0.0)
        mb = _safe(self.p.miss_bias, pt, pair_key) or _safe(self.p.miss_bias, "*", pair_key)
        if mb: bx, bz = float(mb[0]), float(mb[1])

        # 8) Enhanced edge concentration - Simple approach
        # Apply subtle adjustments to increase edge concentration for certain pitch types
        # This helps create more realistic pitch distributions that concentrate at the edges
        if pt in ["Slider", "Curveball"]:
            # For breaking balls, apply a slight bias toward the edges
            if "away" in region:
                tx += 0.05 if tx > 0 else -0.05
            elif "in" in region:
                tx -= 0.05 if tx > 0 else 0.05
            elif "up" in region:
                tz += 0.03
            elif "down" in region:
                tz -= 0.03
        elif pt in ["Sinker", "TwoSeam"]:
            # For sinking pitches, apply a slight bias toward the bottom
            if "down" in region:
                tz -= 0.04

        # 9) finally sample around (target + bias)
        x, z = self._anisotropic_sample(tx + bx, tz + bz, sig_x, sig_z, rho)

        # very light soft-clamp to avoid absurd values
        x = _clip(x, -PLATE_HALF_WIDTH-1.5, PLATE_HALF_WIDTH+1.5)
        z = _clip(z, ZONE_Z_LOW-2.0, ZONE_Z_HIGH+2.0)
        return (x, z)
