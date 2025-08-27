# plate_loc_model.py
import json
from pathlib import Path
import numpy as np

def default_mixtures_demo():
    """
    Fallback 3-component mixture that puts mass on the two edges and middle.
    Keys are (pitch_type, pitcher_throws, batter_side, count_bucket).
    We include a global fallback ('__default__', '__', '__', '__').
    Units should match your PlateLocSide / PlateLocHeight conventions (usually feet).
    """
    w  = np.array([0.40, 0.35, 0.25], dtype=float)
    mu = np.array([[-0.35, 2.30],   # glove-side edge, belt-ish
                   [ 0.00, 2.55],   # middle up
                   [ 0.35, 2.30]])  # arm-side edge
    # diagonal covariances (spread); tweak to taste (ft^2)
    sigx, sigy = 0.08, 0.12
    Sigma = np.array([[[sigx**2, 0.0], [0.0, sigy**2]]] * 3)
    return {("__default__", "__", "__", "__"): dict(w=w, mu=mu, Sigma=Sigma)}

def _coerce_mix(d):
    # convert python lists to numpy arrays with correct shapes
    w  = np.asarray(d["w"], dtype=float)
    mu = np.asarray(d["mu"], dtype=float)
    Sig = np.asarray(d["Sigma"], dtype=float)
    assert mu.shape[1] == 2 and Sig.shape[1:] == (2, 2), "mixture shapes must be [K,2] and [K,2,2]"
    # normalize weights
    w = w / w.sum()
    return dict(w=w, mu=mu, Sigma=Sig)

def load_mixtures_json(path: str | Path):
    path = Path(path)
    raw = json.loads(path.read_text())
    out = {}
    for k, d in raw.items():
        # keys stored as strings; convert back to tuple
        if isinstance(k, str):
            k = tuple(k.split("|")) if "|" in k else eval(k)
        out[k] = _coerce_mix(d)
    return out

def save_mixtures_json(path: str | Path, mixtures: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # make keys serializable
    ser = {"|".join(k): dict(w=v["w"].tolist(), mu=v["mu"].tolist(), Sigma=v["Sigma"].tolist())
           for k, v in mixtures.items()}
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
        key = (ctx.get("pitch_type","__"),
               ctx.get("pthrows","__"),
               ctx.get("bats","__"),
               ctx.get("count_bucket","__"))
        if key in self.mix: return self.mix[key]
        # progressively back off to less-specific keys
        keys = [
            ("__default__", "__", "__", "__"),
            (ctx.get("pitch_type","__"), "__", "__", "__")
        ]
        for k in keys:
            if k in self.mix:
                return self.mix[k]
        # final fallback
        return list(self.mix.values())[0]

    def sample(self, ctx):
        m = self._pick_mix(ctx)
        k = self.rng.choice(len(m["w"]), p=m["w"])
        base = self.rng.multivariate_normal(mean=m["mu"][k], cov=m["Sigma"][k])
        pid = ctx.get("pitcher_id", "global")
        prev = self._prev.get(pid)

        if prev is None:
            x = base[0] + self.rng.normal(0, self.nx)
            y = base[1] + self.rng.normal(0, self.ny)
        else:
            x = self.rho * prev[0] + (1 - self.rho) * base[0] + self.rng.normal(0, self.nx)
            y = self.rho * prev[1] + (1 - self.rho) * base[1] + self.rng.normal(0, self.ny)

        self._prev[pid] = (x, y)
        return float(x), float(y)
