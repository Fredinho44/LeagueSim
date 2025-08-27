
# streamlit_app.py
import streamlit as st
import sys, io, contextlib, importlib, tempfile, zipfile, json, copy, hashlib, types
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np

st.set_page_config(page_title="League Builder + Tuner", layout="wide")

# ---------- small helpers ----------
def _ensure_on_path(p: Path):
    p = p.resolve()
    if str(p.parent) not in sys.path:
        sys.path.insert(0, str(p.parent))

@st.cache_resource
def _load_module(mod_name: str, file_hint: str | None = None):
    try:
        return importlib.import_module(mod_name)
    except ModuleNotFoundError:
        if file_hint:
            _ensure_on_path(Path(file_hint))
            return importlib.import_module(mod_name)
        raise

def _fmt_pct(x): 
    try: return f"{float(x)*100:.0f}%"
    except: return "—"

def _cfg_cards(cfg):
    """Render read-only summary cards for what's selected on Tab 1."""
    bs = cfg.get("BAT_SIDE_P", {})
    ph = cfg.get("PITCH_HAND_P", {})
    tw = cfg.get("TWO_WAY_RATE", {})
    g  = cfg.get("GAMES_PER_TEAM", {})
    sk = cfg.get("SIM_KNOBS", {})

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Demographics")
        st.write(f"**Throws**: R {_fmt_pct(ph.get('Right',0))} | L {_fmt_pct(ph.get('Left',0))}")
        st.write(f"**Bats**: R {_fmt_pct(bs.get('Right',0))} | L {_fmt_pct(bs.get('Left',0))} | S {_fmt_pct(bs.get('Switch',0))}")
        st.write(f"**Two-way**: Rk {_fmt_pct(tw.get('Rookie',0))} | AAA {_fmt_pct(tw.get('AAA',0))} | MLB {_fmt_pct(tw.get('Majors',0))}")

    with c2:
        st.markdown("#### Command & Velo")
        p = cfg.get("_CMD_TIER_PARAMS", {})
        st.write(f"**Rookie** μ {p.get('Rookie',{}).get('mu','—')} σ {p.get('Rookie',{}).get('sd','—')}")
        st.write(f"**AAA** μ {p.get('AAA',{}).get('mu','—')} σ {p.get('AAA',{}).get('sd','—')}")
        st.write(f"**Majors** μ {p.get('Majors',{}).get('mu','—')} σ {p.get('Majors',{}).get('sd','—')}")
        # show velo shift if you track it; otherwise just say priors have been shifted
        if "tune_vshift" in st.session_state:
            st.write(f"**Global FB velo shift**: {st.session_state.tune_vshift:+d} mph")
        st.write(f"**Games**: MLB {g.get('Majors',30)} | AAA {g.get('AAA',30)} | Rk {g.get('Rookie',20)}")

    with c3:
        st.markdown("#### Game Model (sim_utils)")
        st.write(f"SP rest: {sk.get('DEFAULT_SP_RECOVERY',4)} d | RP rest: {sk.get('DEFAULT_RP_RECOVERY',1)} d")
        st.write(f"Pull: {sk.get('PULL_RUNS',4)} runs | Stress: {sk.get('PULL_STRESS_PITCHES',35)} pitches")
        st.write(f"Fatigue: pitch {_fmt_pct(sk.get('FATIGUE_PER_PITCH_OVER',0.015))}, BF {_fmt_pct(sk.get('FATIGUE_PER_BF_OVER',0.03))}")
        st.write(f"TTO penalty: {_fmt_pct(sk.get('TTO_PENALTY',0.10))}")
        st.write(f"Velo/10: {sk.get('VELO_LOSS_PER_OVER10',0.15)} mph | Spin/10: {int(sk.get('SPIN_LOSS_PER_OVER10',20))} rpm")
        st.write(f"Injury@over: {_fmt_pct(sk.get('INJURY_CHANCE_HEAVY_OVER',0.05))} | Days: {sk.get('INJURY_DUR_RANGE',(10,30))}")
        st.write(f"Extras: fatigue× {sk.get('EXTRA_INNING_FATIGUE_SCALE',0.5)} | cmd +{sk.get('EXTRA_INNING_CMD_FLAT_PENALTY',0.03)}")


def _reload_module(mod_or_name, file_hint=None):
    """Reload by name if the cached module object is detached."""
    import sys as _sys, importlib as _imp
    _imp.invalidate_caches()
    name = mod_or_name if isinstance(mod_or_name, str) else getattr(mod_or_name, "__name__", None)
    if not name:
        raise ImportError("Cannot determine module name to reload")
    if name in _sys.modules:
        return _imp.reload(_sys.modules[name])
    if file_hint:
        _ensure_on_path(Path(file_hint))
    return _imp.import_module(name)

def _apply_overrides_to_module(mod: types.ModuleType, cfg: dict):
    """Assign tuned dicts back to the imported make_league module so its helpers/main() use them."""
    for k in [
        "PITCH_HAND_P","BAT_SIDE_P","TWO_WAY_RATE",
        "_CMD_TIER_PARAMS","_CMD_MIX","_VELO_PRIORS",
        "DURABILITY_PRIORS","CLUSTER_WEIGHTS_R","CLUSTER_WEIGHTS_L",
        "PITCH_CLUSTERS","ARM_PRIORS","AGE_RANGES","GAMES_PER_TEAM","SPLIT_PER_TIER"
    ]:
        if k in cfg:
            setattr(mod, k, copy.deepcopy(cfg[k]))
    if "DEFAULT_NUM_ORGS" in cfg:
        setattr(mod, "DEFAULT_NUM_ORGS", int(cfg["DEFAULT_NUM_ORGS"]))

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

def _run_module_main(mod, argv) -> tuple[str, int | None]:
    out = io.StringIO()
    old_argv = sys.argv[:]
    exit_code = None
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                mod.main()
            except SystemExit as e:
                exit_code = int(e.code) if isinstance(e.code, int) else 0
    finally:
        sys.argv = old_argv
    return out.getvalue(), exit_code

def _robust_minmax_scale(series: pd.Series, new_min: float, new_max: float, q_lo=5, q_hi=95) -> pd.Series:
    """Rescale values to [new_min, new_max] using robust bounds (q_lo..q_hi), then clip.
    Preserves rank order; outliers are clipped to the new range."""
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return x  # nothing to do
    lo = np.nanpercentile(x, q_lo)
    hi = np.nanpercentile(x, q_hi)
    if hi <= lo:
        # degenerate; return constant at new_min
        return pd.Series(np.full(len(x), new_min), index=series.index)
    scaled = (x - lo) / (hi - lo)
    y = new_min + scaled * (new_max - new_min)
    mn, mx = (new_min, new_max) if new_min <= new_max else (new_max, new_min)
    return y.clip(lower=mn, upper=mx)

def _json_pretty(d): return json.dumps(d, indent=2, ensure_ascii=False)

def _coerce_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    return x

def _summary_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    for c in cols:
        x = _coerce_numeric(df[c]) if c in df.columns else pd.Series(dtype=float)
        x = x.dropna()
        if x.empty:
            continue
        qv = x.quantile(qs)
        rows.append({
            "Metric": c,
            "Count": int(x.count()),
            "Mean": round(x.mean(), 6),
            "Std": round(x.std(ddof=1), 6) if x.count() > 1 else 0.0,
            "Min": round(x.min(), 6),
            "P05": round(qv.iloc[0], 6),
            "P25": round(qv.iloc[1], 6),
            "P50": round(qv.iloc[2], 6),
            "P75": round(qv.iloc[3], 6),
            "P95": round(qv.iloc[4], 6),
            "Max": round(x.max(), 6),
        })
    out = pd.DataFrame(rows, columns=["Metric","Count","Mean","Std","Min","P05","P25","P50","P75","P95","Max"])
    return out

def _cfg_sig(cfg: dict) -> str:
    """Stable hash of current tuning to detect stale previews."""
    try:
        return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    except Exception:
        return "NA"

def _norm_probs(d: dict, keys: list[str]):
    """Clamp negatives to 0 and renormalize so sum(keys)=1 (if possible)."""
    vals = [max(0.0, float(d.get(k, 0.0))) for k in keys]
    s = sum(vals)
    if s <= 0:
        # fallback to uniform
        vals = [1.0/len(keys)]*len(keys)
    else:
        vals = [v/s for v in vals]
    for k, v in zip(keys, vals):
        d[k] = v

# ---------- sidebar: script locations ----------
st.sidebar.header("Script folder")
default_dir = Path(__file__).parent.resolve()
scripts_dir = Path(st.sidebar.text_input(
    "Folder containing make_league.py / simulate_seasons_roster_style.py",
    value=str(default_dir)
))

make_league_path = scripts_dir / "make_league.py"
simulate_path    = scripts_dir / "simulate_seasons_roster_style.py"
sim_utils_path   = scripts_dir / "sim_utils.py"

ok_scripts = (make_league_path.exists() and simulate_path.exists() and sim_utils_path.exists())
st.sidebar.write("Found make_league.py:", "✅" if make_league_path.exists() else "❌")
st.sidebar.write("Found simulate_seasons_roster_style.py:", "✅" if simulate_path.exists() else "❌")
st.sidebar.write("Found sim_utils.py:", "✅" if sim_utils_path.exists() else "❌")

# ---------- load defaults from make_league ----------
@st.cache_resource
def _load_make_and_defaults(make_path: Path):
    mod = _load_module("make_league", str(make_path))
    # Snapshot defaults from make_league
    def _get(name, default):
        return copy.deepcopy(getattr(mod, name, default))

    defaults = {
        "DEFAULT_NUM_ORGS": int(_get("DEFAULT_NUM_ORGS", 13)),
        "PITCH_HAND_P":     _get("PITCH_HAND_P", {"Right":0.7,"Left":0.3}),
        "BAT_SIDE_P":       _get("BAT_SIDE_P", {"Right":0.55,"Left":0.35,"Switch":0.10}),
        "TWO_WAY_RATE":     _get("TWO_WAY_RATE", {"Rookie":0.20,"AAA":0.10,"Majors":0.04}),
        "_CMD_TIER_PARAMS": _get("_CMD_TIER_PARAMS", {"Rookie":{"mu":1.0,"sd":0.10},"AAA":{"mu":1.02,"sd":0.10},"Majors":{"mu":1.04,"sd":0.10}}),
        "_CMD_MIX":         _get("_CMD_MIX", {}),
        "_VELO_PRIORS":     _get("_VELO_PRIORS", {}),
        "DURABILITY_PRIORS":_get("DURABILITY_PRIORS", {}),
        "CLUSTER_WEIGHTS_R":_get("CLUSTER_WEIGHTS_R", {}),
        "CLUSTER_WEIGHTS_L":_get("CLUSTER_WEIGHTS_L", {}),
        "PITCH_CLUSTERS":   _get("PITCH_CLUSTERS", {}),
        "ARM_PRIORS":       _get("ARM_PRIORS", {}),
        "AGE_RANGES":       _get("AGE_RANGES", {}),
        "GAMES_PER_TEAM":   _get("GAMES_PER_TEAM", {"Majors":30,"AAA":30,"Rookie":20}),
        "SPLIT_PER_TIER":   _get("SPLIT_PER_TIER", {}),
    }
    return mod, defaults

# ---------- load defaults from sim_utils & helpers to apply ----------
def _load_sim_defaults(sim_utils_path: Path):
    mod = _load_module("sim_utils", str(sim_utils_path))
    def _dget(name, default):
        return getattr(mod, name, default)
    defaults = {
        "SIM_KNOBS": {
            "DEFAULT_SP_RECOVERY":              int(_dget("DEFAULT_SP_RECOVERY", 4)),
            "DEFAULT_RP_RECOVERY":              int(_dget("DEFAULT_RP_RECOVERY", 1)),
            "INJURY_CHANCE_HEAVY_OVER":         float(_dget("INJURY_CHANCE_HEAVY_OVER", 0.05)),
            "INJURY_DUR_RANGE":                 tuple(_dget("INJURY_DUR_RANGE", (10, 30))),
            "EXTRA_INNING_RECOVERY_BONUS_DAYS": int(_dget("EXTRA_INNING_RECOVERY_BONUS_DAYS", 1)),

            "PULL_RUNS":                        int(_dget("PULL_RUNS", 4)),
            "PULL_STRESS_PITCHES":              int(_dget("PULL_STRESS_PITCHES", 35)),

            "FATIGUE_PER_PITCH_OVER":           float(_dget("FATIGUE_PER_PITCH_OVER", 0.015)),
            "FATIGUE_PER_BF_OVER":              float(_dget("FATIGUE_PER_BF_OVER", 0.03)),
            "VELO_LOSS_PER_OVER10":             float(_dget("VELO_LOSS_PER_OVER10", 0.15)),
            "SPIN_LOSS_PER_OVER10":             float(_dget("SPIN_LOSS_PER_OVER10", 20.0)),
            "TTO_PENALTY":                      float(_dget("TTO_PENALTY", 0.10)),

            "EXTRA_INNING_FATIGUE_SCALE":       float(_dget("EXTRA_INNING_FATIGUE_SCALE", 0.50)),
            "EXTRA_INNING_CMD_FLAT_PENALTY":    float(_dget("EXTRA_INNING_CMD_FLAT_PENALTY", 0.03)),
        }
    }
    return mod, defaults

def _apply_sim_knobs(sim_mod, sim_knobs: dict):
    """Assign tuned sim knobs back to sim_utils before a run."""
    if not sim_knobs: 
        return
    for k, v in sim_knobs.items():
        setattr(sim_mod, k, v)

@st.cache_resource
def _load_sim_and_defaults(sim_path: Path):
    return _load_sim_defaults(sim_path)

if ok_scripts:
    make_mod, defaults = _load_make_and_defaults(make_league_path)
    sim_mod_defaults, sim_defaults = _load_sim_and_defaults(sim_utils_path)

# Initialize session cfg + velo baseline
if ok_scripts and "cfg" not in st.session_state:
    st.session_state.cfg = copy.deepcopy(defaults)
    # store an immutable velo baseline to avoid cumulative shifts
    st.session_state._velo_base = copy.deepcopy(defaults.get("_VELO_PRIORS", {}))
    # merge in SIM_KNOBS defaults
    st.session_state.cfg.update(sim_defaults)  # brings in {"SIM_KNOBS": {...}}

# ---------- Tabs ----------
tab_tune, tab_league, tab_run, tab_help = st.tabs(
    ["1) League Tuning", "2) League Build & Preview", "3) Run Simulation", " Help / README"]
)

# ========== 1) LEAGUE TUNING ==========
with tab_tune:
    st.header("League Tuning")
    st.caption("Configure demographics, command/velo, schedule size, and in-game model weights.")

    if not ok_scripts:
        st.warning("Point the sidebar to your scripts folder first.")
        st.stop()

    cfg = st.session_state.cfg

    # Top bar: auto-apply + quick summary
    barL, barR = st.columns([3, 2])
    auto_apply = barL.toggle(
        "Auto-apply tuning on change",
        value=True, key="tune_auto_apply",
        help="If ON, changes immediately update the in-session config. If OFF, use the Apply button."
    )
    with barR:
        # tiny live summary
        bs = cfg.get("BAT_SIDE_P", {})
        ph = cfg.get("PITCH_HAND_P", {})
        g  = cfg.get("GAMES_PER_TEAM", {})
        st.markdown(
            f"""
            <div style="padding:10px;border:1px solid #EEE;border-radius:10px">
              <b>Live summary</b><br>
              Bats: R {bs.get('Right',0):.2f} | L {bs.get('Left',0):.2f} | S {bs.get('Switch',0):.2f}<br>
              Throws: R {ph.get('Right',0):.2f} | L {ph.get('Left',0):.2f}<br>
              Games: Maj {g.get('Majors',30)} | AAA {g.get('AAA',30)} | Rk {g.get('Rookie',20)}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Internal: apply function
    def _apply_quick_knobs():
        cfg = st.session_state.cfg
        # handedness
        cfg["PITCH_HAND_P"]["Right"] = float(st.session_state.tune_ph_r)/100.0
        cfg["PITCH_HAND_P"]["Left"]  = 1.0 - cfg["PITCH_HAND_P"]["Right"]
        _norm_probs(cfg["PITCH_HAND_P"], ["Right","Left"])
        # bats
        cfg["BAT_SIDE_P"]["Right"] = float(st.session_state.tune_bs_r)/100.0
        cfg["BAT_SIDE_P"]["Left"]  = float(st.session_state.tune_bs_l)/100.0
        cfg["BAT_SIDE_P"]["Switch"]= max(0.0, 1.0 - cfg["BAT_SIDE_P"]["Right"] - cfg["BAT_SIDE_P"]["Left"])
        _norm_probs(cfg["BAT_SIDE_P"], ["Right","Left","Switch"])
        # two-way
        cfg["TWO_WAY_RATE"]["Rookie"] = float(st.session_state.tune_tw_rk)/100.0
        cfg["TWO_WAY_RATE"]["AAA"]    = float(st.session_state.tune_tw_aa)/100.0
        cfg["TWO_WAY_RATE"]["Majors"] = float(st.session_state.tune_tw_mj)/100.0
        # games
        cfg["GAMES_PER_TEAM"]["Majors"] = int(st.session_state.tune_g_maj)
        cfg["GAMES_PER_TEAM"]["AAA"]    = int(st.session_state.tune_g_aaa)
        cfg["GAMES_PER_TEAM"]["Rookie"] = int(st.session_state.tune_g_rookie)
        # velo shift applied from immutable baseline
        base = st.session_state.get("_velo_base", {})
        if base:
            newp = {}
            for tier, tier_map in base.items():
                newp[tier] = {}
                for cluster, pair in tier_map.items():
                    try:
                        mu, sd = float(pair[0]), float(pair[1])
                    except Exception:
                        mu, sd = float(pair[0]), float(pair[-1])
                    newp[tier][cluster] = [mu + float(st.session_state.tune_vshift), sd]
            cfg["_VELO_PRIORS"] = newp

    # Sub-tabs inside Tuning
    sub_demo, sub_cmdvelo, sub_games, sub_sim, sub_adv = st.tabs(
        ["Demographics", "Command & Velo", "Games per Tier", "Game Model (sim_utils)", "Advanced JSON"]
    )

    # -------------------- Demographics --------------------
    with sub_demo:
        st.subheader("Roster Demographics")
        c1, c2, c3 = st.columns(3)
        ph_r = c1.slider(
            "Pitcher throws: Right (%)", 0, 100,
            int(cfg["PITCH_HAND_P"].get("Right",0.7)*100),
            key="tune_ph_r",
            help="Share of pitchers who throw right-handed. Left-handed share fills the remainder."
        )
        bs_r = c2.slider(
            "Batter side: Right (%)", 0, 100,
            int(cfg["BAT_SIDE_P"].get("Right",0.55)*100),
            key="tune_bs_r",
            help="Share of right-handed batters. Left and Switch are derived from these."
        )
        bs_l = c3.slider(
            "Batter side: Left (%)", 0, 100,
            int(cfg["BAT_SIDE_P"].get("Left",0.35)*100),
            key="tune_bs_l",
            help="Share of left-handed batters. Switchers are auto-computed."
        )
        st.caption(f"Switch bats auto = **{max(0, 100 - bs_r - bs_l)}%**")

        tw_rk = st.slider(
            "Two-way rate — Rookie (%)", 0, 50,
            int(cfg["TWO_WAY_RATE"].get("Rookie",0.22)*100),
            key="tune_tw_rk",
            help="Probability that a Rookie player is two-way."
        )
        cA, cB = st.columns(2)
        tw_aa = cA.slider(
            "Two-way rate — AAA (%)", 0, 50,
            int(cfg["TWO_WAY_RATE"].get("AAA",0.10)*100),
            key="tune_tw_aa",
            help="Probability that a AAA player is two-way."
        )
        tw_mj = cB.slider(
            "Two-way rate — Majors (%)", 0, 50,
            int(cfg["TWO_WAY_RATE"].get("Majors",0.04)*100),
            key="tune_tw_mj",
            help="Probability that a MLB player is two-way."
        )

    # -------------------- Command & Velo --------------------
    with sub_cmdvelo:
        st.subheader("CommandTier & Velo")
        st.markdown("Adjust the center (μ) and spread (σ) of command by tier, plus a global velo shift.")

        cmdc1, cmdc2, cmdc3 = st.columns(3)
        cfg["_CMD_TIER_PARAMS"].setdefault("Rookie", {"mu":1.0,"sd":0.10})
        cfg["_CMD_TIER_PARAMS"].setdefault("AAA",    {"mu":1.02,"sd":0.10})
        cfg["_CMD_TIER_PARAMS"].setdefault("Majors", {"mu":1.04,"sd":0.10})

        def _cmd_pair(tier, col):
            mu0 = float(cfg["_CMD_TIER_PARAMS"][tier]["mu"])
            sd0 = float(cfg["_CMD_TIER_PARAMS"][tier]["sd"])
            mu = col.number_input(f"{tier} μ", value=mu0, step=0.01, format="%.3f",
                                  key=f"tune_cmd_mu_{tier}",
                                  help="Center of the command multiplier for this tier.")
            sd = col.number_input(f"{tier} σ", value=sd0, step=0.01, format="%.3f",
                                  key=f"tune_cmd_sd_{tier}",
                                  help="Standard deviation of the command multiplier for this tier.")
            cfg["_CMD_TIER_PARAMS"][tier]["mu"] = float(mu)
            cfg["_CMD_TIER_PARAMS"][tier]["sd"] = float(sd)

        _cmd_pair("Rookie", cmdc1); _cmd_pair("AAA", cmdc2); _cmd_pair("Majors", cmdc3)

        vshift = st.slider(
            "Global fastball velo shift (mph)", -5, 5, 0,
            key="tune_vshift",
            help="Adds this many mph to every (tier, cluster) fastball mean; spreads unchanged."
        )

    # -------------------- Games per Tier --------------------
    with sub_games:
        st.subheader("Schedule Size")
        g1, g2, g3 = st.columns(3)
        g_m = g1.number_input(
            "Games per team — Majors", min_value=10, max_value=120,
            value=int(cfg["GAMES_PER_TEAM"].get("Majors", 30)),
            key="tune_g_maj",
            help="Number of regular season games per MLB team."
        )
        g_a = g2.number_input(
            "Games per team — AAA", min_value=10, max_value=120,
            value=int(cfg["GAMES_PER_TEAM"].get("AAA", 30)),
            key="tune_g_aaa",
            help="Number of regular season games per AAA team."
        )
        g_r = g3.number_input(
            "Games per team — Rookie", min_value=10, max_value=120,
            value=int(cfg["GAMES_PER_TEAM"].get("Rookie", 20)),
            key="tune_g_rookie",
            help="Number of regular season games per Rookie team."
        )

    # -------------------- Game Model (sim_utils) --------------------
    with sub_sim:
        st.subheader("Game Model (Fatigue · TTO · Injuries · Pulls · Extras)")
        sim_knobs = st.session_state.cfg.setdefault("SIM_KNOBS", {})

        c1, c2, c3 = st.columns(3)
        sim_knobs["DEFAULT_SP_RECOVERY"] = c1.number_input(
            "SP recovery days", min_value=0, max_value=10,
            value=int(sim_knobs.get("DEFAULT_SP_RECOVERY", 4)),
            help="Baseline rest days required for starters after an outing."
        )
        sim_knobs["DEFAULT_RP_RECOVERY"] = c2.number_input(
            "RP recovery days", min_value=0, max_value=10,
            value=int(sim_knobs.get("DEFAULT_RP_RECOVERY", 1)),
            help="Baseline rest days required for relievers."
        )
        sim_knobs["EXTRA_INNING_RECOVERY_BONUS_DAYS"] = c3.number_input(
            "Extra-inning recovery bonus (days)", min_value=0, max_value=3,
            value=int(sim_knobs.get("EXTRA_INNING_RECOVERY_BONUS_DAYS", 1)),
            help="Teams get this bonus rest when a game goes to extras."
        )

        c4, c5 = st.columns(2)
        sim_knobs["PULL_RUNS"] = c4.number_input(
            "Auto-pull if trailing by (runs)", min_value=0, max_value=12,
            value=int(sim_knobs.get("PULL_RUNS", 4)),
            help="Manager pulls pitcher if trailing by at least this many."
        )
        sim_knobs["PULL_STRESS_PITCHES"] = c5.number_input(
            "Stress-pitch threshold", min_value=0, max_value=120,
            value=int(sim_knobs.get("PULL_STRESS_PITCHES", 35)),
            help="High-stress pitch count that increases pull tendency."
        )

        st.markdown("**Fatigue penalties**")
        c6, c7, c8 = st.columns(3)
        sim_knobs["FATIGUE_PER_PITCH_OVER"] = c6.number_input(
            "Cmd tax per pitch over limit", min_value=0.0, max_value=0.2, step=0.001,
            value=float(sim_knobs.get("FATIGUE_PER_PITCH_OVER", 0.015)), format="%.3f",
            help="Command multiplier deducted per pitch beyond a pitch-count limit."
        )
        sim_knobs["FATIGUE_PER_BF_OVER"] = c7.number_input(
            "Cmd tax per BF over expected", min_value=0.0, max_value=0.2, step=0.001,
            value=float(sim_knobs.get("FATIGUE_PER_BF_OVER", 0.03)), format="%.3f",
            help="Command multiplier deducted per batter faced beyond expectation."
        )
        sim_knobs["TTO_PENALTY"] = c8.number_input(
            "TTO penalty (3rd time)", min_value=0.0, max_value=0.3, step=0.005,
            value=float(sim_knobs.get("TTO_PENALTY", 0.10)), format="%.3f",
            help="Extra command tax the 3rd time through the order."
        )

        c9, c10 = st.columns(2)
        sim_knobs["VELO_LOSS_PER_OVER10"] = c9.number_input(
            "Velo loss per +10 pitches", min_value=0.0, max_value=1.0, step=0.01,
            value=float(sim_knobs.get("VELO_LOSS_PER_OVER10", 0.15)), format="%.2f",
            help="Fastball mph lost per 10 pitches over limit (fatigue effect)."
        )
        sim_knobs["SPIN_LOSS_PER_OVER10"] = c10.number_input(
            "Spin loss per +10 pitches (rpm)", min_value=0.0, max_value=200.0, step=1.0,
            value=float(sim_knobs.get("SPIN_LOSS_PER_OVER10", 20.0)), format="%.0f",
            help="Spin lost per 10 pitches over limit (fatigue effect)."
        )

        st.markdown("**Injuries**")
        c11, c12 = st.columns(2)
        sim_knobs["INJURY_CHANCE_HEAVY_OVER"] = c11.number_input(
            "Injury chance (heavy over)", min_value=0.0, max_value=0.5, step=0.005,
            value=float(sim_knobs.get("INJURY_CHANCE_HEAVY_OVER", 0.05)), format="%.3f",
            help="Probability of injury when pushed far over fatigue thresholds."
        )
        dur_lo, dur_hi = sim_knobs.get("INJURY_DUR_RANGE", (10, 30))
        dur_lo = c12.number_input("Injury min days", min_value=1, max_value=180, value=int(dur_lo),
                                  help="Minimum injury downtime.")
        dur_hi = c12.number_input("Injury max days", min_value=dur_lo, max_value=365, value=int(dur_hi),
                                  help="Maximum injury downtime.")
        sim_knobs["INJURY_DUR_RANGE"] = (int(dur_lo), int(dur_hi))

        st.markdown("**Extra innings**")
        c13, c14 = st.columns(2)
        sim_knobs["EXTRA_INNING_FATIGUE_SCALE"] = c13.number_input(
            "Fatigue scale per extra inning", min_value=0.0, max_value=2.0, step=0.01,
            value=float(sim_knobs.get("EXTRA_INNING_FATIGUE_SCALE", 0.50)), format="%.2f",
            help="Multiplier on fatigue penalties for each extra inning."
        )
        sim_knobs["EXTRA_INNING_CMD_FLAT_PENALTY"] = c14.number_input(
            "Cmd tax per extra inning", min_value=0.0, max_value=0.2, step=0.001,
            value=float(sim_knobs.get("EXTRA_INNING_CMD_FLAT_PENALTY", 0.03)), format="%.3f",
            help="Flat command penalty layered per extra inning."
        )

    # -------------------- Advanced JSON --------------------
    with sub_adv:
        st.subheader("Advanced dictionaries (JSON)")
        with st.expander("Edit: _CMD_TIER_PARAMS", expanded=False):
            txt = st.text_area("_CMD_TIER_PARAMS", value=_json_pretty(cfg["_CMD_TIER_PARAMS"]), height=180, key="tune_cmd_json")
            if st.button("Save _CMD_TIER_PARAMS", key="tune_save_cmd"):
                cfg["_CMD_TIER_PARAMS"] = json.loads(txt); st.success("Saved.")
        with st.expander("Edit: _VELO_PRIORS", expanded=False):
            txt = st.text_area("_VELO_PRIORS", value=_json_pretty(cfg["_VELO_PRIORS"]), height=220, key="tune_velo_json")
            if st.button("Save _VELO_PRIORS", key="tune_save_velo"):
                cfg["_VELO_PRIORS"] = json.loads(txt); st.success("Saved.")
                st.session_state._velo_base = copy.deepcopy(cfg["_VELO_PRIORS"])
        with st.expander("Edit: DURABILITY_PRIORS", expanded=False):
            txt = st.text_area("DURABILITY_PRIORS", value=_json_pretty(cfg["DURABILITY_PRIORS"]), height=260, key="tune_dur_json")
            if st.button("Save DURABILITY_PRIORS", key="tune_save_dur"):
                cfg["DURABILITY_PRIORS"] = json.loads(txt); st.success("Saved.")
        with st.expander("Other dicts (cluster weights, pitch mixes, arm priors…)", expanded=False):
            cA, cB = st.columns(2)
            t1 = cA.text_area("CLUSTER_WEIGHTS_R", value=_json_pretty(cfg["CLUSTER_WEIGHTS_R"]), height=150, key="tune_cwr")
            t2 = cA.text_area("CLUSTER_WEIGHTS_L", value=_json_pretty(cfg["CLUSTER_WEIGHTS_L"]), height=150, key="tune_cwl")
            t3 = cB.text_area("PITCH_CLUSTERS",    value=_json_pretty(cfg["PITCH_CLUSTERS"]),    height=150, key="tune_pcl")
            t4 = cB.text_area("ARM_PRIORS",        value=_json_pretty(cfg["ARM_PRIORS"]),        height=150, key="tune_arm")
            if st.button("Save these four", key="tune_save_four"):
                cfg["CLUSTER_WEIGHTS_R"] = json.loads(t1)
                cfg["CLUSTER_WEIGHTS_L"] = json.loads(t2)
                cfg["PITCH_CLUSTERS"]    = json.loads(t3)
                cfg["ARM_PRIORS"]        = json.loads(t4)
                st.success("Saved.")

    # ----------- Apply / Save / Load / Reset -----------
    st.divider()
    actL, actR, actZ = st.columns([1.2, 1.2, 1])
    if not auto_apply:
        if actL.button("Apply changes now", key="tune_apply_btn"):
            _apply_quick_knobs()
            st.success("Applied to session config.")
    else:
        # keep in sync continuously
        _apply_quick_knobs()

    # Save/Load config (single JSON holds both league tuning + SIM_KNOBS)
    with actR.popover("Save / Load config"):
        st.caption("Export or import your entire tuning as JSON.")
        cA, cB = st.columns(2)
        if cA.button("Save JSON", key="tune_save_cfg_btn"):
            st.session_state["_last_cfg_json"] = json.dumps(cfg, indent=2)
            st.success("Config staged for download below.")
        if "_last_cfg_json" in st.session_state:
            st.download_button(
                "Download league_tuning_config.json",
                data=st.session_state["_last_cfg_json"],
                file_name="league_tuning_config.json",
                mime="application/json",
                key="tune_save_cfg_dl"
            )
        upload = st.file_uploader("Load config JSON", type=["json"], key="tune_upload_cfg")
        if upload:
            st.session_state.cfg = json.loads(upload.getvalue().decode("utf-8"))
            st.session_state._velo_base = copy.deepcopy(st.session_state.cfg.get("_VELO_PRIORS", {}))
            st.success("Config loaded into session state.")

    if actZ.button("Reset to defaults", key="tune_reset_btn"):
        _reload_module("make_league", make_league_path)
        make_mod, defaults_ml = _load_make_and_defaults(make_league_path)
        _, defaults_sim = _load_sim_and_defaults(sim_utils_path)
        merged = copy.deepcopy(defaults_ml); merged.update(defaults_sim)
        st.session_state.cfg = merged
        st.session_state._velo_base = copy.deepcopy(defaults_ml.get("_VELO_PRIORS", {}))
        st.success("Reset to defaults from make_league.py + sim_utils.py")


        


# ========== 2) LEAGUE BUILD & PREVIEW ==========
with tab_league:
    st.header("League Build & Preview")
    st.caption("Quickly preview rosters/schedule in memory or write official CSVs to a folder.")

    if not ok_scripts:
        st.warning("Point the sidebar to your scripts folder first.")
        st.stop()

    # two sub-tabs: Quick Preview (in-memory) and Write CSVs (official)
    sub_preview, sub_write = st.tabs(["Quick Preview (in-memory)", "Write CSVs to Folder"])

    # -------------------- Quick Preview (in-memory) --------------------
    with sub_preview:
        pc1, pc2, pc3, pc4 = st.columns([1,1,1,1])
        prev_num_orgs = pc1.number_input(
            "Preview orgs", min_value=2,
            value=int(st.session_state.cfg.get("DEFAULT_NUM_ORGS", 13)),
            key="prev_num_orgs",
            help="How many organizations to generate for this in-memory preview."
        )
        prev_seed = pc2.number_input("Preview seed", min_value=0, value=7, key="prev_seed")
        prev_date = pc3.date_input("Preview start date", value=date(2025, 4, 1), key="prev_start")
        prev_po   = pc4.checkbox("Include postseason in preview", value=False, key="prev_po")
        st.markdown("### Config snapshot (read-only)")
        _cfg_cards(st.session_state.cfg)

        # show a signature to make it obvious what config is live
        live_sig = _cfg_sig(st.session_state.cfg) 
        st.caption(f"Config signature: `{live_sig}`")
        build_preview = st.button("Build / Refresh Preview", key="prev_build_btn", use_container_width=True)

        if build_preview:
            import importlib as _imp, random
            from datetime import timedelta, date as _date
            try:
                _reload_module("make_league", make_league_path)
                ml = _imp.import_module("make_league")
                _apply_overrides_to_module(ml, st.session_state.cfg)

                rng = random.Random(int(prev_seed))
                orgs    = ml.build_organizations(int(prev_num_orgs), rng)
                teams   = ml.build_teams(orgs)
                rosters = ml.build_rosters(teams, rng)

                # build schedules by tier
                tier_ids = {"Majors": [], "AAA": [], "Rookie": []}
                for t in teams:
                    tier_ids[t["Tier"]].append(t["TeamID"])
                for k in tier_ids: tier_ids[k].sort()

                sched_M = ml.build_tier_schedule(tier_ids["Majors"], ml.GAMES_PER_TEAM["Majors"], prev_date, rng)
                sched_A = ml.build_tier_schedule(tier_ids["AAA"],    ml.GAMES_PER_TEAM["AAA"],    prev_date + timedelta(days=1), rng)
                sched_R = ml.build_tier_schedule(tier_ids["Rookie"], ml.GAMES_PER_TEAM["Rookie"], prev_date + timedelta(days=2), rng)

                # tag Tier + GameIDs
                gid = 1
                for r in sched_M: r["Tier"] = "Majors"; r["GameID"] = f"M{gid:04d}"; gid += 1
                gid = 1
                for r in sched_A: r["Tier"] = "AAA";    r["GameID"] = f"A{gid:04d}"; gid += 1
                gid = 1
                for r in sched_R: r["Tier"] = "Rookie"; r["GameID"] = f"R{gid:04d}"; gid += 1

                schedules_all = sched_M + sched_A + sched_R

                if prev_po:
                    try:
                        end_M = max(_date.fromisoformat(r["Date"]) for r in sched_M) if sched_M else prev_date
                        end_A = max(_date.fromisoformat(r["Date"]) for r in sched_A) if sched_A else prev_date
                        end_R = max(_date.fromisoformat(r["Date"]) for r in sched_R) if sched_R else prev_date
                        seeds_M = sorted(tier_ids["Majors"])[:6]
                        seeds_A = sorted(tier_ids["AAA"])[:6]
                        seeds_R = sorted(tier_ids["Rookie"])[:6]
                        next_M, next_A, next_R = len(sched_M)+1, len(sched_A)+1, len(sched_R)+1
                        po_M = ml.build_postseason_for_tier("Majors","M",seeds_M,end_M,next_M)
                        po_A = ml.build_postseason_for_tier("AAA","A",seeds_A,end_A,next_A)
                        po_R = ml.build_postseason_for_tier("Rookie","R",seeds_R,end_R,next_R)
                        schedules_all += po_M + po_A + po_R
                    except Exception:
                        st.warning("Postseason helper not available or failed; preview shown without PO.")

                st.session_state.preview = {
                    "organizations": pd.DataFrame(orgs),
                    "teams":         pd.DataFrame(teams),
                    "rosters":       pd.DataFrame(rosters),
                    "schedule":      pd.DataFrame(schedules_all),
                }
                st.session_state.preview_sig = _cfg_sig(st.session_state.cfg)
                st.success("Preview built.")
            except Exception as e:
                st.exception(e)

        # If we have a preview, show compact dashboard + export
        if "preview" in st.session_state:
            # warn if stale vs current tuning
            if st.session_state.get("preview_sig") != _cfg_sig(st.session_state.cfg):
                st.warning("Preview is stale relative to current tuning. Click **Build / Refresh Preview**.")

            df_orgs   = st.session_state.preview["organizations"]
            df_teams  = st.session_state.preview["teams"]
            df_roster = st.session_state.preview["rosters"]
            df_sched  = st.session_state.preview["schedule"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Organizations", len(df_orgs))
            m2.metric("Teams", len(df_teams))
            m3.metric("Players", len(df_roster))
            m4.metric("Games (RS+PO)", len(df_sched))

            st.dataframe(df_roster.head(20), use_container_width=True, height=300)

            st.subheader("Roster Metrics")
            scope = st.radio("Scope", ["All players", "Pitchers only", "Hitters only"], index=0,
                             horizontal=True, key="prev_scope")
            if scope == "Pitchers only":
                dfv = df_roster[df_roster["Role"].isin(["SP","RP"])] if "Role" in df_roster.columns else df_roster.iloc[0:0]
            elif scope == "Hitters only":
                dfv = df_roster[df_roster["Role"].isin(["BAT"])] if "Role" in df_roster.columns else df_roster.iloc[0:0]
            else:
                dfv = df_roster

            default_metrics = [
                "CommandTier","AgeYears","HeightIn","WingspanIn","ArmSlotDeg",
                "RelHeight_ft","RelSide_ft","Extension_ft","AvgFBVelo",
                "PrevSeasonIP","StaminaScore","PitchCountLimit",
                "AvgPitchesPerOuting","ExpectedBattersFaced","RecoveryDaysNeeded"
            ]
            metrics = st.multiselect(
                "Metrics", options=sorted(dfv.columns.tolist()),
                default=[m for m in default_metrics if m in dfv.columns],
                key="prev_metrics"
            )
            tbl = _summary_table(dfv, metrics) if not dfv.empty else pd.DataFrame(columns=[
                "Metric","Count","Mean","Std","Min","P05","P25","P50","P75","P95","Max"
            ])
            st.dataframe(tbl, use_container_width=True)
            st.download_button(
                "Download roster summary (CSV)",
                data=tbl.to_csv(index=False).encode("utf-8"),
                file_name="roster_summary_preview.csv",
                mime="text/csv",
                key="prev_summary_dl",
                use_container_width=True,
            )

            st.markdown("### Save this Preview as League CSVs")
            save_col1, save_col2 = st.columns([2,1])
            preview_out_dir = save_col1.text_input(
                "Folder for organizations/teams/rosters/schedule",
                value=str((scripts_dir / "league_out").resolve()),
                key="prev_save_dir",
            )
            do_save = save_col2.button("Write preview CSVs", key="prev_save_btn")
            if do_save:
                try:
                    outp = Path(preview_out_dir); outp.mkdir(parents=True, exist_ok=True)
                    df_orgs.to_csv(outp / "organizations.csv", index=False)
                    df_teams.to_csv(outp / "teams.csv", index=False)
                    df_roster.to_csv(outp / "rosters.csv", index=False)
                    df_sched.to_csv(outp / "schedule.csv", index=False)
                    st.session_state["league_out_dir"] = str(outp.resolve())
                    st.success(f"Wrote CSVs to: {outp}")
                except Exception as e:
                    st.exception(e)

    # -------------------- Write CSVs to Folder (official build) --------------------
    with sub_write:
        st.subheader("Generate League CSVs (official)")
        c1, c2, c3 = st.columns(3)
        num_orgs   = c1.number_input("Number of organizations", min_value=2,
                                     value=int(st.session_state.cfg.get("DEFAULT_NUM_ORGS", 13)), key="build_num_orgs")
        seed       = c2.number_input("Seed", min_value=0, value=7, key="build_seed")
        start_date_str = c3.text_input("Start date (YYYY-MM-DD)",
                                       value=date(2025, 4, 1).isoformat(), key="build_start")
        with_po    = c1.checkbox("Include postseason bracket (6-team)", value=False, key="build_with_po")
        out_dir    = st.text_input("Output folder for league CSVs",
                                   value=str((scripts_dir / "league_out").resolve()), key="build_out_dir")

        run_build = st.button("Generate League CSVs using current tuning", key="build_btn", use_container_width=True)

        if run_build:
            try:
                # fresh import + apply overrides
                _reload_module("make_league", make_league_path)
                make_mod = importlib.import_module("make_league")
                _apply_overrides_to_module(make_mod, st.session_state.cfg)

                argv = [
                    "make_league.py",
                    "--num_orgs", str(int(num_orgs)),
                    "--out_dir", str(out_dir),
                    "--seed", str(int(seed)),
                    "--start_date", start_date_str,
                    "--maj_games", str(int(st.session_state.cfg["GAMES_PER_TEAM"]["Majors"])),
                    "--aaa_games", str(int(st.session_state.cfg["GAMES_PER_TEAM"]["AAA"])),
                    "--rookie_games", str(int(st.session_state.cfg["GAMES_PER_TEAM"]["Rookie"])),
                ]
                if with_po:
                    argv.append("--with_postseason")

                with st.status("Generating league with overrides…", expanded=True) as status:
                    logs, code = _run_module_main(make_mod, argv)
                    st.code(logs or "(no stdout)")
                    status.update(label="League generated" if code in (None, 0) else "Exited with errors", state="complete")
                if code not in (None, 0):
                    st.error(f"make_league.py exited with code {code}. See message above.")
                    st.stop()

                out_dir_p = Path(out_dir)
                if (out_dir_p / "rosters.csv").exists():
                    st.success("Wrote rosters.csv + schedule.csv + organizations/teams.")
                    df_ro = pd.read_csv(out_dir_p / "rosters.csv")
                    st.dataframe(df_ro.head(12), use_container_width=True)

                    # ZIP download
                    zbuf = io.BytesIO()
                    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                        for fn in ["organizations.csv", "teams.csv", "rosters.csv", "schedule.csv"]:
                            fp = out_dir_p / fn
                            if fp.exists():
                                z.write(fp, fp.name)
                    zbuf.seek(0)
                    st.download_button(
                        "Download league_out (ZIP)",
                        data=zbuf,
                        file_name="league_out.zip",
                        mime="application/zip",
                        use_container_width=True,
                        key="build_zip_dl"
                    )
                    st.session_state["league_out_dir"] = str(out_dir_p.resolve())
                else:
                    st.warning("Finished, but rosters.csv not found. Check logs above.")

            except Exception as e:
                st.exception(e)


# ========== 4) RUN SIMULATOR ==========
with tab_run:
    st.subheader("Run simulate_seasons_roster_style.py")

    if not ok_scripts:
        st.warning("Point the sidebar to your scripts folder first.")
    else:
        # ---------- Inputs: League (can regenerate) ----------
        st.markdown("### League settings (optional: regenerate from current tuning before running)")
        rc1, rc2, rc3 = st.columns(3)
        run_num_orgs = rc1.number_input(
            "Number of organizations",
            min_value=2,
            value=int(st.session_state.cfg.get("DEFAULT_NUM_ORGS", 13)),
            key="run_num_orgs",
        )
        run_seed = rc2.number_input("Seed", min_value=0, value=7, key="run_seed")
        run_start_date = rc3.date_input("Start date", value=date(2025, 4, 1), key="run_start_date")
        run_with_po = rc1.checkbox("Include postseason bracket (6-team)", value=False, key="run_with_po")

        league_out_default = str((scripts_dir / "league_out").resolve())
        league_out_dir = st.text_input(
            "league_out folder (where organizations.csv / teams.csv / rosters.csv / schedule.csv will live)",
            value=st.session_state.get("league_out_dir", league_out_default),
            key="run_league_out_dir",
        )

        st.markdown("### Simulator paths")
        priors_yaml = st.text_input(
            "Path to rules_pitch_by_pitch.yaml",
            value=str((scripts_dir / "rules_pitch_by_pitch.yaml").resolve()),
            key="run_priors_yaml",
        )
        template_csv = st.text_input(
            "Path to a YakkerTech template CSV (RECOMMENDED)",
            value="",
            key="run_template_csv",
        )
        season_out = st.text_input(
            "Season output folder (simulator writes Season01/… here)",
            value=str((scripts_dir / "season_out").resolve()),
            key="run_season_out",
        )

        colA, colB = st.columns([1,1])
        do_regen = colA.checkbox("Regenerate league from current tuning before running", value=True, key="run_do_regen")
        run_btn = colB.button("Regenerate (if checked) + Run Simulator", key="run_run_btn")

        if run_btn:
            try:
                # ---------- Optional: regenerate league from current tuning ----------
                out_dir_p = Path(league_out_dir)
                if do_regen:
                    _reload_module("make_league", make_league_path)
                    make_mod = importlib.import_module("make_league")
                    _apply_overrides_to_module(make_mod, st.session_state.cfg)

                    argv_make = [
                        "make_league.py",
                        "--num_orgs", str(int(run_num_orgs)),
                        "--out_dir", str(out_dir_p),
                        "--seed", str(int(run_seed)),
                        "--start_date", run_start_date.isoformat(),
                        "--maj_games", str(int(st.session_state.cfg["GAMES_PER_TEAM"]["Majors"])),
                        "--aaa_games", str(int(st.session_state.cfg["GAMES_PER_TEAM"]["AAA"])),
                        "--rookie_games", str(int(st.session_state.cfg["GAMES_PER_TEAM"]["Rookie"])),
                    ]
                    if run_with_po:
                        argv_make.append("--with_postseason")

                    with st.status("Regenerating league from current tuning…", expanded=True) as status:
                        logs_make, code_make = _run_module_main(make_mod, argv_make)
                        st.code(logs_make or "(no stdout)")
                        status.update(label="League regenerated" if code_make in (None,0) else "Exited with errors", state="complete")

                    st.session_state["league_out_dir"] = str(out_dir_p.resolve())

                # ---------- Validate required league CSVs ----------
                roster_p   = out_dir_p / "rosters.csv"
                schedule_p = out_dir_p / "schedule.csv"
                if not roster_p.is_file() or not schedule_p.is_file():
                    st.error(f"Missing league CSVs in {out_dir_p}. Ensure rosters.csv and schedule.csv exist.")
                    st.stop()

                # ---------- Validate priors & optional template ----------
                priors_p = Path(priors_yaml.strip())
                if not priors_p.is_file():
                    st.error(f"Priors YAML not found: {priors_p}")
                    st.stop()

                tmpl_str = template_csv.strip()
                tmpl_ok = Path(tmpl_str).is_file() if tmpl_str else False
                if tmpl_str and not tmpl_ok:
                    st.warning(f"Template CSV not found; proceeding without it: {tmpl_str}")

                # ---------- Apply sim_utils knobs BEFORE run ----------
                _reload_module("sim_utils", sim_utils_path)
                sim_utils_mod = importlib.import_module("sim_utils")
                _apply_sim_knobs(sim_utils_mod, st.session_state.cfg.get("SIM_KNOBS", {}))

                # Ensure downstream modules will see updated sim_utils
                for m in ("game_sim", "season_runner"):
                    if m in sys.modules:
                        del sys.modules[m]
                importlib.invalidate_caches()

                # ---------- Run simulator ----------
                out_root = Path(season_out.strip())
                out_root.mkdir(parents=True, exist_ok=True)

                _reload_module("simulate_seasons_roster_style", simulate_path)
                sim_mod = importlib.import_module("simulate_seasons_roster_style")

                argv_sim = [
                    "simulate_seasons_roster_style.py",
                    "--roster_csv",   str(roster_p),
                    "--schedule_csv", str(schedule_p),
                    "--priors",       str(priors_p),
                    "--out_dir",      str(out_root),
                    "--seed",         str(int(run_seed)),
                ]
                if tmpl_ok:
                    argv_sim += ["--template_csv", tmpl_str]

                st.markdown("**Command:**")
                st.code(" ".join(argv_sim), language="bash")

                with st.status("Running simulator…", expanded=True) as status:
                    logs_sim, code = _run_module_main(sim_mod, argv_sim)
                    st.code(logs_sim or "(no stdout)")
                    status.update(label="Simulation finished" if code in (None, 0) else "Exited with errors", state="complete")
                if code not in (None, 0):
                    st.error(f"Simulator exited with code {code}. See message above for details.")
                    st.stop()

                # ---------- Show outputs ----------
                seasons = sorted([p for p in out_root.glob("Season*") if p.is_dir()])
                if not seasons:
                    st.warning(f"No Season* folders found in {out_root}. Check logs above for argparse or path issues.")
                    st.stop()

                total_csvs = 0
                for sdir in seasons:
                    csvs = sorted(sdir.glob("*.csv"))
                    total_csvs += len(csvs)
                    st.write(f"**{sdir.name}** — {len(csvs)} game CSVs")
                    if csvs:
                        st.caption(f"Sample: {csvs[0].name}")
                        try:
                            st.dataframe(pd.read_csv(csvs[0]).head(5), use_container_width=True)
                        except Exception:
                            st.info("(Sample CSV loaded, but could not preview due to size/encoding.)")

                st.success(f"✅ Produced {total_csvs} CSVs across {len(seasons)} season folder(s).")

                # ZIP everything for convenience
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
                    for sdir in seasons:
                        for f in sdir.glob("*.csv"):
                            z.write(f, f.relative_to(out_root).as_posix())
                zbuf.seek(0)
                st.download_button(
                    "Download Season ZIP",
                    data=zbuf,
                    file_name="season_out.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="run_season_zip_btn",
                )

            except SystemExit as e:
                st.error(f"Simulator exited (argparse). Double-check flags & paths. Code: {e.code}")
            except Exception as e:
                st.exception(e)

# ========== 5) README ==========
with tab_help:
    st.subheader("Documentation")
    readme_file = Path("README_ModelCA_League_Sim.md")
    if readme_file.exists():
        st.markdown(readme_file.read_text(encoding="utf-8"))
    else:
        st.info("Place README_ModelCA_League_Sim.md next to streamlit_app.py to render it here.")
