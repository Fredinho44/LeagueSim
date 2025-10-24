# plate_loc_model.py
import json
import ast
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import math

# --- Attribute-driven plate location sampler (from rosters.csv) ---
import math, numpy as np

PLATE_X_MIN, PLATE_X_MAX = -0.95, 0.95
PLATE_Z_MIN, PLATE_Z_MAX =  0.00, 6.00

# Base “aim pockets” vs RHB; we flip x by platoon later
_BASES_ATTR = {
    "FourSeam":  (+0.30, 3.05, 0.16, 0.18),
    "Sinker":    (+0.20, 2.45, 0.17, 0.19),
    "TwoSeam":   (+0.20, 2.45, 0.17, 0.19),
    "Slider":    (+0.45, 2.10, 0.20, 0.20),
    "Curveball": (+0.10, 2.10, 0.18, 0.22),
    "Changeup":  (-0.10, 2.25, 0.18, 0.20),
    "Cutter":    (+0.35, 2.60, 0.18, 0.19),
}

def _truncate_mvn_attr(mean, cov, tries=12, rng=None):
    if rng is None: rng = np.random
    for _ in range(tries):
        x, z = rng.multivariate_normal(mean, cov)
        if PLATE_X_MIN <= x <= PLATE_X_MAX and PLATE_Z_MIN <= z <= PLATE_Z_MAX:
            return float(round(x, 3)), float(round(z, 3))
    # smooth fallback (avoid edge spikes)
    x, z = rng.multivariate_normal(mean, cov)
    x = 0.95 * math.tanh(x / 0.95)
    z = 2.5 + 1.5 * math.tanh((z - 2.5) / 1.5)
    return float(round(x, 3)), float(round(z, 3))

def sample_loc_from_roster_attrs(
    pitch_type: str,
    roster_row: dict,
    batter_side: str,           # "L" or "R"
    count_bucket: str = "even", # "behind"/"even"/"ahead"
    rng=None
):
    if rng is None: rng = np.random

    # 0) Base pocket (RHP vs RHB convention; platoon flip later)
    mu_x, mu_z, sx, sz = _BASES_ATTR.get(pitch_type, (+0.25, 2.60, 0.20, 0.20))

    # 1) Platoon handling: +x is “away to RHB”
    PTH = (roster_row.get("Throws") or "Right")[0].upper()
    BATS = (batter_side or "R")[0].upper()
    same_hand = (PTH == BATS)
    mu_x = (+abs(mu_x) if PTH == "R" else -abs(mu_x)) if same_hand else (-abs(mu_x) if PTH == "R" else +abs(mu_x))

    # 2) Arm slot offsets
    slot = (roster_row.get("ArmSlotBucket") or "").lower()
    if "over" in slot:
        mu_z += 0.12
    elif "high34" in slot or "3/4" in slot or "three" in slot:
        mu_z += 0.05; mu_x += (0.04 if PTH == "R" else -0.04)
    elif "side" in slot:
        mu_z -= 0.10; mu_x += (0.10 if PTH == "R" else -0.10); sx *= 1.08

    # 3) Release geometry nudges (small, safe)
    rel_h = float(roster_row.get("RelHeight_ft") or 5.5)
    rel_x = float(roster_row.get("RelSide_ft") or 0.0)
    mu_z += 0.06 * math.tanh((rel_h - 5.5) / 0.6)
    mu_x += 0.10 * math.tanh(rel_x / 0.7) * (1 if PTH == "R" else -1)

    # 4) Velo influence (mainly 4S)
    if "four" in pitch_type.lower() or "4" in pitch_type:
        v = float(roster_row.get("AvgFBVelo") or 90.0)
        mu_z += 0.02 * (v - 90.0)
        sx *= max(0.90, 1.0 - 0.01*(v - 90.0))

    # 5) Count bucket behavior
    if count_bucket == "ahead":
        mu_z -= 0.05; sx *= 1.05; sz *= 1.05
    elif count_bucket == "behind":
        mu_z += 0.03; sx *= 0.98; sz *= 0.98

    # 6) Command & extension → variance (gentle, floored)
    cmd = float(roster_row.get("CommandTier") or 1.0)
    eff = 0.5 + 0.5 * max(0.85, min(1.30, cmd))
    sx_eff, sz_eff = sx / eff, sz / eff

    ext = float(roster_row.get("Extension_ft") or 5.5)
    sz_eff *= max(0.85, min(1.10, 1.0 - 0.02*(ext - 5.5)))

    cov = np.diag([sx_eff**2, sz_eff**2])

    return _truncate_mvn_attr((mu_x, mu_z), cov, rng=rng)



def _safe_norm_map(d: dict[str, float]) -> dict[str, float]:
    if not d: return {}
    vals = {k: max(0.0, float(v)) for k, v in d.items()}
    s = sum(vals.values())
    return {k: (v / s if s > 0 else 0.0) for k, v in vals.items()}

def _count_bucket_aim(balls: int, strikes: int) -> str:
    # map to the names you'll use in YAML
    if strikes > balls: return "pitcher_ahead"
    if balls   > strikes: return "hitter_ahead"
    return "even"

def _pocket_center(key: str, grid: dict) -> tuple[float, float]:
    """Return (x,z) center of a grid pocket. Works with either 'row_col' or numeric '1..9' cell names."""
    rows = grid.get("row_labels") or []
    cols = grid.get("col_labels") or []
    xe, ze = grid["x_edges"], grid["z_edges"]
    if "_" in key and rows and cols:
        r, c = key.split("_", 1); ri, ci = rows.index(r), cols.index(c)
    elif key.startswith("r") and "c" in key:
        rc = key.replace("r","").split("c"); ri, ci = int(rc[0]), int(rc[1])
    else:
        # numeric 3×3 cells '1'..'9', row-major
        idx = int(key) - 1
        cols_n = len(xe) - 1
        ri, ci = divmod(idx, cols_n)
    x = 0.5 * (xe[ci] + xe[ci+1])
    z = 0.5 * (ze[ri] + ze[ri+1])
    return x, z

def _mvnorm_from_kernel(sx: float, sz: float, rho: float) -> np.ndarray:
    cov_xz = rho * sx * sz
    return np.array([[sx*sx, cov_xz],[cov_xz, sz*sz]], dtype=float)


def default_mixtures_demo():
    """
    Fallback mixtures that vary by pitch type, handedness matchups, and count.
    Keys are (pitch_type, pitcher_throws, batter_side, count_bucket).
    Units should match your PlateLocSide / PlateLocHeight conventions (usually feet).
    """
    mixtures = {}
    
    # Default fallback if nothing else matches
    sigx, sigy = 0.25, 0.30  # Standard variance parameters
    
    # --- Global default ---
    w = np.array([0.35, 0.30, 0.35], dtype=float)
    mu = np.array([[-0.35, 2.30],   # glove-side edge, belt-ish
                 [ 0.00, 2.55],   # middle up
                 [ 0.35, 2.30]])  # arm-side edge
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["__default__", "__", "__", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # --- RHP vs RHB ---
    # Fastballs: tend to work away and up
    w = np.array([0.40, 0.35, 0.25], dtype=float)
    mu = np.array([[ 0.40, 2.60],   # away, up
                   [ 0.20, 2.30],   # away, middle
                   [-0.10, 2.20]])  # inside edge
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["Fastball", "R", "R", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["FourSeamFastball", "R", "R", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # Breaking balls: low and away or backdoor
    w = np.array([0.45, 0.30, 0.25], dtype=float)
    mu = np.array([[ 0.40, 1.70],   # low and away
                   [ 0.10, 1.50],   # low middle
                   [-0.30, 2.10]])  # backdoor
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["Slider", "R", "R", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["Curveball", "R", "R", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # --- RHP vs LHB ---
    # Fastballs: tend to work inside and up
    w = np.array([0.40, 0.35, 0.25], dtype=float)
    mu = np.array([[-0.40, 2.60],   # inside, up
                   [-0.20, 2.30],   # inside, middle
                   [ 0.10, 2.20]])  # outside edge
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["Fastball", "R", "L", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["FourSeamFastball", "R", "L", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # Breaking balls: low and in or backdoor
    w = np.array([0.45, 0.30, 0.25], dtype=float)
    mu = np.array([[-0.40, 1.70],   # low and in
                   [-0.10, 1.50],   # low middle
                   [ 0.30, 2.10]])  # backdoor
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["Slider", "R", "L", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["Curveball", "R", "L", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # --- LHP vs RHB --- (mirror of RHP vs LHB)
    # Fastballs: tend to work inside and up
    w = np.array([0.40, 0.35, 0.25], dtype=float)
    mu = np.array([[ 0.40, 2.60],   # inside, up
                   [ 0.20, 2.30],   # inside, middle
                   [-0.10, 2.20]])  # outside edge
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["Fastball", "L", "R", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["FourSeamFastball", "L", "R", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # Breaking balls: low and in or backdoor
    w = np.array([0.45, 0.30, 0.25], dtype=float)
    mu = np.array([[ 0.40, 1.70],   # low and in
                   [ 0.10, 1.50],   # low middle
                   [-0.30, 2.10]])  # backdoor
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["Slider", "L", "R", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["Curveball", "L", "R", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # --- LHP vs LHB --- (mirror of RHP vs RHB)
    # Fastballs: tend to work away and up
    w = np.array([0.40, 0.35, 0.25], dtype=float)
    mu = np.array([[-0.40, 2.60],   # away, up
                   [-0.20, 2.30],   # away, middle
                   [ 0.10, 2.20]])  # inside edge
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["Fastball", "L", "L", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["FourSeamFastball", "L", "L", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # Breaking balls: low and away or backdoor
    w = np.array([0.45, 0.30, 0.25], dtype=float)
    mu = np.array([[-0.40, 1.70],   # low and away
                   [-0.10, 1.50],   # low middle
                   [ 0.30, 2.10]])  # backdoor
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["Slider", "L", "L", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["Curveball", "L", "L", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # --- Count-specific pitch locations ---
    
    # Ahead in count (0-2, 1-2): Expanded zone, more off-plate
    w = np.array([0.30, 0.30, 0.40], dtype=float)
    mu = np.array([[-0.70, 2.50],   # far outside
                   [ 0.00, 3.20],   # high
                   [ 0.70, 2.50]])  # far outside
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    mixtures["__default__", "__", "__", "ahead"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # Behind in count (3-0, 3-1): Tighter zone, more in middle
    w = np.array([0.20, 0.60, 0.20], dtype=float)
    mu = np.array([[-0.15, 2.30],   # slight inside
                   [ 0.00, 2.40],   # middle
                   [ 0.15, 2.30]])  # slight outside
    Sigma = np.array([[[0.15**2, 0.0], [0.0, 0.15**2]]] * 3)  # Tighter spread
    mixtures["__default__", "__", "__", "behind"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # --- Special pitch types ---
    
    # Changeups: typically low in the zone
    w = np.array([0.30, 0.50, 0.20], dtype=float)
    mu = np.array([[-0.30, 1.80],   # low inside
                   [ 0.00, 1.70],   # low middle
                   [ 0.30, 1.80]])  # low outside
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigx**2]]] * 3)
    mixtures["Changeup", "__", "__", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # Sinkers/Two-seam: typically lower and more to arm side
    w = np.array([0.20, 0.50, 0.30], dtype=float)
    mu = np.array([[ 0.10, 2.00],   # middle-in for RHP
                   [ 0.25, 1.90],   # down and in for RHP
                   [ 0.40, 2.10]])  # inside corner for RHP
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigx**2]]] * 3)
    mixtures["Sinker", "R", "__", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["TwoSeamFastball", "R", "__", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # Mirror for LHP sinkers
    mu = np.array([[-0.10, 2.00],   # middle-in for LHP
                   [-0.25, 1.90],   # down and in for LHP
                   [-0.40, 2.10]])  # inside corner for LHP
    mixtures["Sinker", "L", "__", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    mixtures["TwoSeamFastball", "L", "__", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # Cutters: Move toward glove side
    w = np.array([0.40, 0.35, 0.25], dtype=float)
    mu = np.array([[-0.40, 2.40],   # inside for RHP
                   [-0.20, 2.30],   # inside edge for RHP 
                   [ 0.10, 2.20]])  # middle for RHP
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigx**2]]] * 3)
    mixtures["Cutter", "R", "__", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    # Mirror for LHP cutters
    mu = np.array([[ 0.40, 2.40],   # inside for LHP
                   [ 0.20, 2.30],   # inside edge for LHP
                   [-0.10, 2.20]])  # middle for LHP
    mixtures["Cutter", "L", "__", "__"] = dict(w=w, mu=mu, Sigma=Sigma)
    
    return mixtures

def _coerce_mix(d):
    # convert python lists to numpy arrays with correct shapes
    w  = np.asarray(d["w"], dtype=float)
    mu = np.asarray(d["mu"], dtype=float)
    Sig = np.asarray(d["Sigma"], dtype=float)
    assert mu.shape[1] == 2 and Sig.shape[1:] == (2, 2), "mixture shapes must be [K,2] and [K,2,2]"
    w = w / w.sum()
    return dict(w=w, mu=mu, Sigma=Sig)

@dataclass
class PlateLocBundle:
    mixtures: dict
    region_usage: dict
    pitch_call_by_region: dict
    tilt_distribution: dict
    mixture_info: dict
    meta: dict

def _normalize_key(key) -> tuple:
    if isinstance(key, tuple):
        parts = list(key)
    elif isinstance(key, list):
        parts = list(key)
    elif isinstance(key, str):
        if "|" in key:
            parts = key.split("|")
        else:
            try:
                parsed = ast.literal_eval(key)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple)):
                parts = list(parsed)
            else:
                parts = [key]
    else:
        parts = [str(key)]
    if len(parts) < 4:
        parts += ["__"] * (4 - len(parts))
    elif len(parts) > 4:
        parts = parts[:4]
    return tuple(str(p) for p in parts)

def _decode_mixture_map(raw):
    mixtures = {}
    info = {}
    if not isinstance(raw, dict):
        return mixtures, info
    for k, d in raw.items():
        key = _normalize_key(k)
        mixtures[key] = _coerce_mix(d)
        extras = {ek: ev for ek, ev in d.items() if ek not in {"w", "mu", "Sigma"}}
        if extras:
            info[key] = extras
    return mixtures, info

def load_location_bundle(path: str | Path) -> PlateLocBundle:
    path = Path(path)
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "mixtures" in raw:
        mix_raw = raw.get("mixtures", {})
        region = raw.get("region_usage", {})
        region_calls = raw.get("pitch_call_by_region", {})
        tilt = raw.get("tilt_distribution", {})
        meta = raw.get("meta", {}) or {}
    else:
        mix_raw = raw
        region = {}
        region_calls = {}
        tilt = {}
        meta = {}
    mixtures, mix_info = _decode_mixture_map(mix_raw)
    reg = {_normalize_key(k): v for k, v in (region or {}).items()}
    reg_calls = {_normalize_key(k): v for k, v in (region_calls or {}).items()}
    tilts = {_normalize_key(k): v for k, v in (tilt or {}).items()}
    return PlateLocBundle(
        mixtures=mixtures,
        region_usage=reg,
        pitch_call_by_region=reg_calls,
        tilt_distribution=tilts,
        mixture_info=mix_info,
        meta=meta,
    )

def load_mixtures_json(path: str | Path):
    return load_location_bundle(path).mixtures

def save_mixtures_json(path: str | Path, mixtures: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ser = {
        "|".join(str(part) for part in k): dict(w=v["w"].tolist(), mu=v["mu"].tolist(), Sigma=v["Sigma"].tolist())
        for k, v in mixtures.items()
    }
    path.write_text(json.dumps(ser, indent=2))
def count_bucket(balls: int, strikes: int) -> str:
    # coarse bucket: behind/even/ahead (from PITCHER perspective)
    if balls - strikes >= 2: return "behind"
    if strikes - balls >= 1: return "ahead"
    return "even"

class PlateLocSampler:
    def __init__(self, rng: np.random.Generator, mixtures: dict,
                 rho: float = 0.3, noise_x: float = 0.03, noise_y: float = 0.04):
        self.rng = rng
        self.mix = mixtures
        self.rho = float(rho)
        self.nx  = float(noise_x)
        self.ny  = float(noise_y)
        self._prev = {}  # per-pitcher AR(1) memory

    def _pick_mix(self, ctx):
        """Pick the most specific mixture for the given context."""
        pitch_type = ctx.get("pitch_type", "__")
        pthrows = ctx.get("pthrows", "__")
        bats = ctx.get("bats", "__")
        count = ctx.get("count_bucket", "__")
        
        # Try most specific match first (pitch + handedness + count)
        key = (pitch_type, pthrows, bats, count)
        if key in self.mix: 
            return self.mix[key]
            
        # Try without count
        key = (pitch_type, pthrows, bats, "__")
        if key in self.mix: 
            return self.mix[key]
            
        # Try with just pitch type and pitcher handedness
        key = (pitch_type, pthrows, "__", "__")
        if key in self.mix: 
            return self.mix[key]
            
        # Try with just the pitch type
        key = (pitch_type, "__", "__", "__")
        if key in self.mix: 
            return self.mix[key]
            
        # Try with just the count
        key = ("__default__", "__", "__", count)
        if key in self.mix: 
            return self.mix[key]
            
        # Final fallback
        key = ("__default__", "__", "__", "__")
        if key in self.mix: 
            return self.mix[key]
            
        # If all else fails, return the first mixture
        return list(self.mix.values())[0]


def sample_with_strategy(self,
                             grid: dict,
                             cfg_pitch_hand: dict,
                             balls: int, strikes: int,
                             command: float,
                             fatigue: float,
                             rng: np.random.Generator,
                             pthrows: str = "R") -> tuple[float, float] | None:
        """
        grid: dict with x_edges, z_edges, row_labels, col_labels, cells (3×3 or your current grid)
        cfg_pitch_hand: YAML block for this pitch & hand:
            {
              "location_model": {
                "aim_map": {
                  "counts": {
                    "pitcher_ahead": {...pocket->prob...},
                    "even": {...},
                    "hitter_ahead": {...}
                  }
                },
                "error_kernel": {
                  "sigma_x_base": 0.22,
                  "sigma_z_base": 0.18,
                  "rho": -0.15,
                  "arm_side_bias": 0.05,   # +X for RHP, -X for LHP if you want
                  "up_down_bias":  0.00,
                  "command_scale": {"slope": -0.30, "intercept": 1.00},
                  "fatigue_scale": {"slope":  0.35, "intercept": 1.00},
                  "count_scale":   {"pitcher_ahead":0.95, "even":1.00, "hitter_ahead":1.10}
                }
              }
            }
        """
        loc = (cfg_pitch_hand or {}).get("location_model") or {}
        aim = (loc.get("aim_map") or {}).get("counts") or {}
        kernel = (loc.get("error_kernel") or {})

        # 1) choose an aim pocket (strategy)
        bucket = _count_bucket_aim(balls, strikes)
        probs = _safe_norm_map(aim.get(bucket) or aim.get("even") or {})
        if not probs:
            return None
        pockets = list(probs.keys())
        weights = np.array([probs[k] for k in pockets], dtype=float)
        target_key = rng.choice(pockets, p=weights)

        # 2) target center (+ light jitter so samples don't glue to centers)
        tx, tz = _pocket_center(target_key, grid)
        # jitter scaled by cell size
        xw = (grid["x_edges"][-1]-grid["x_edges"][0]) / max(1,(len(grid["x_edges"])-1))
        zw = (grid["z_edges"][-1]-grid["z_edges"][0]) / max(1,(len(grid["z_edges"])-1))
        tx += rng.normal(0, 0.10 * xw)
        tz += rng.normal(0, 0.10 * zw)

        # 3) build error kernel (execution)
        sx0 = float(kernel.get("sigma_x_base", 0.22))
        sz0 = float(kernel.get("sigma_z_base", 0.18))
        rho = float(kernel.get("rho", -0.15))

        cmd = max(1e-6, float(command))
        cs = kernel.get("command_scale") or {}
        fs = kernel.get("fatigue_scale") or {}
        c_fac = max(0.25, float(cs.get("intercept", 1.0)) + float(cs.get("slope", -0.30)) * (1.0/np.sqrt(cmd)))
        f_fac = max(0.50, float(fs.get("intercept", 1.0)) + float(fs.get("slope",  0.35)) * float(fatigue))
        cnt_fac = float((kernel.get("count_scale") or {}).get(bucket, 1.0))
        scale = np.clip(c_fac * f_fac * cnt_fac, 0.25, 2.5)
        sx = max(0.02, sx0 * scale)
        sz = max(0.02, sz0 * scale)
        cov = _mvnorm_from_kernel(sx, sz, rho)

        # 4) draw an error + systematic biases
        err = rng.multivariate_normal([0.0, 0.0], cov)
        arm_bias = float(kernel.get("arm_side_bias", 0.0))
        # flip sign for lefties if you want arm-side to be "negative X"
        if str(pthrows).upper().startswith("L"):
            arm_bias = -arm_bias
        updown_bias = float(kernel.get("up_down_bias", 0.0))

        x = float(np.clip(tx + err[0] + arm_bias, -1.15, 1.15))
        z = float(np.clip(tz + err[1] + updown_bias, 1.00, 4.60))
        return (x, z)

# ---------------------- Calibration helpers ----------------------
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _one_comp_inzone_prob(mux: float, sigx: float, muy: float, sigy: float,
                          x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> float:
    # Independent Gaussian assumption (Sigma diagonal in our mixtures)
    xl, xh = x_bounds; yl, yh = y_bounds
    if sigx <= 0 or sigy <= 0:
        return 0.0
    px = max(0.0, min(1.0, _norm_cdf((xh - mux) / sigx) - _norm_cdf((xl - mux) / sigx)))
    py = max(0.0, min(1.0, _norm_cdf((yh - muy) / sigy) - _norm_cdf((yl - muy) / sigy)))
    return px * py

def inzone_prob(mix: dict, x_bounds: tuple[float, float] = (-0.83, 0.83),
                y_bounds: tuple[float, float] = (1.5, 3.5)) -> float:
    """Approximate probability that a sample from the mixture lands in the strike zone.
    Assumes diagonal covariance (which our defaults use)."""
    w = np.asarray(mix["w"], dtype=float)
    mu = np.asarray(mix["mu"], dtype=float)
    Sig = np.asarray(mix["Sigma"], dtype=float)
    ps = []
    for k in range(len(w)):
        sigx = math.sqrt(max(1e-12, float(Sig[k, 0, 0])))
        sigy = math.sqrt(max(1e-12, float(Sig[k, 1, 1])))
        ps.append(_one_comp_inzone_prob(float(mu[k, 0]), sigx, float(mu[k, 1]), sigy, x_bounds, y_bounds))
    return float(np.dot(w / max(1e-12, w.sum()), np.asarray(ps)))

def _best_target_for_key(targets: dict, key: tuple[str, str, str, str]) -> float | None:
    pitch_type, pthrows, bats, count = key
    cands = [
        (pitch_type, pthrows, bats, count),
        (pitch_type, pthrows, bats, "__"),
        (pitch_type, pthrows, "__", "__"),
        (pitch_type, "__", "__", "__"),
        ("__default__", "__", "__", count),
        ("__default__", "__", "__", "__"),
    ]
    for cand in cands:
        skey = "|".join(cand)
        if skey in targets:
            try:
                t = float(targets[skey])
                if 0.0 <= t <= 1.0:
                    return t
            except Exception:
                continue
    return None

def calibrate_mixtures(mixtures: dict,
                       targets: dict,
                       x_bounds: tuple[float, float] = (-0.83, 0.83),
                       y_bounds: tuple[float, float] = (1.5, 3.5),
                       max_iter: int = 20,
                       step: float = 0.06,
                       tol: float = 0.01) -> dict:
    """Adjust mixture means to better match target in-zone probabilities.
    Simple heuristic: scale horizontal means away from center to reduce in-zone %, toward center to increase.
    Leaves covariances and weights unchanged.

    targets: mapping of "pitch|pthrows|bats|count" -> target_inzone_fraction (0..1)
    """
    out = {k: dict(w=v["w"].copy(), mu=v["mu"].copy(), Sigma=v["Sigma"].copy()) for k, v in mixtures.items()}
    for key, mix in list(out.items()):
        tgt = _best_target_for_key(targets, key)
        if tgt is None:
            continue
        for _ in range(max_iter):
            pin = inzone_prob(mix, x_bounds, y_bounds)
            if abs(pin - tgt) <= tol:
                break
            # Scale horizontal means away/toward center
            mu = mix["mu"]
            if not isinstance(mu, np.ndarray):
                mu = np.asarray(mu, dtype=float)
            center_pull = (1.0 - step) if pin < tgt else (1.0 + step)
            mu[:, 0] = mu[:, 0] * center_pull
            mix["mu"] = mu
    return out
