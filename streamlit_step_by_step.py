
"""
streamlit_step_by_step.py
-------------------------
A step-by-step interface for the League Simulator with expandable sections
that appear after completing each step, allowing users to generate organizations
and teams, and run simulations.

This improved version includes:
- Better error handling
- Enhanced documentation
- Improved UI/UX
- More robust file handling
"""

import streamlit as st
import sys
import subprocess
import time
import tempfile
import io
import contextlib
import importlib
import json
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta

st.set_page_config(page_title="League Simulator", layout="wide")

# Enhanced CSS styling for professional appearance
st.markdown("""
<style>
.step-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
}

.step-header {
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}

.step-subheader {
    font-size: 1.2rem;
    font-weight: bold;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

.config-section {
    border-left: 4px solid #1f77b4;
    padding-left: 1rem;
    margin: 1rem 0;
    background-color: #f8f9fa;
    border-radius: 0 8px 8px 0;
}

.metric-card {
    background: linear-gradient(135deg, #2196f3 0%, #21cbf3 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.preset-button {
    background: linear-gradient(45deg,#2196f3 0%, #21cbf3 100%);
    border: none;
    border-radius: 8px;
    color: white;
    padding: 0.5rem 1rem;
    margin: 0.25rem;
    transition: transform 0.2s;
}

.preset-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.preview-dashboard {
    background: linear-gradient(135deg, #2196f3 0%, #21cbf3 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    margin: 1rem 0;
}

.tour-highlight {
    border: 2px dashed #ff6b6b;
    border-radius: 8px;
    padding: 1rem;
    background-color: #fff5f5;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 8px;
}

.stButton>button:hover {
    background-color: #45a049;
}

.stButton>button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
}
</style>
""", unsafe_allow_html=True)

# ---------- sidebar: script locations ----------
st.sidebar.header("📁 Script Folder")
default_dir = Path(__file__).parent.resolve()
scripts_dir = Path(st.sidebar.text_input(
    "Folder containing make_league.py",
    value=str(default_dir)
))

make_league_path = scripts_dir / "make_league.py"
ok_scripts = make_league_path.exists()
st.sidebar.write("Found make_league.py:", "✅" if make_league_path.exists() else "❌")

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1  # Start at step 1

if "league_config" not in st.session_state:
    st.session_state.league_config = {}

if "preview_data" not in st.session_state:
    st.session_state.preview_data = None

if "league_generated" not in st.session_state:
    st.session_state.league_generated = False

# ---------- load defaults from make_league ----------
def _load_make_defaults(make_path: Path):
    """Load default configuration from make_league.py."""
    if not make_path.exists():
        st.error("make_league.py not found!")
        return {}
    
    try:
        # Add the directory to sys.path if needed
        script_dir = str(make_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        # Import the module
        make_mod = importlib.import_module("make_league")
        
        # Extract defaults
        defaults = {
            "DEFAULT_NUM_ORGS": int(getattr(make_mod, "DEFAULT_NUM_ORGS", 13)),
            "PITCH_HAND_P": getattr(make_mod, "PITCH_HAND_P", {"Right": 0.70, "Left": 0.30}),
            "BAT_SIDE_P": getattr(make_mod, "BAT_SIDE_P", {"Right": 0.55, "Left": 0.35, "Switch": 0.10}),
            "TWO_WAY_RATE": getattr(make_mod, "TWO_WAY_RATE", {"Rookie": 0.20, "AAA": 0.10, "Majors": 0.04}),
            "_CMD_TIER_PARAMS": getattr(make_mod, "_CMD_TIER_PARAMS", {
                "Rookie": {"mu": 1.0, "sd": 0.10, "clip": (0.60, 1.25)},
                "AAA": {"mu": 1.02, "sd": 0.10, "clip": (0.70, 1.22)},
                "Majors": {"mu": 1.04, "sd": 0.10, "clip": (0.80, 1.18)}
            }),
            "GAMES_PER_TEAM": getattr(make_mod, "GAMES_PER_TEAM", {"Majors": 30, "AAA": 30, "Rookie": 20}),
        }
        return defaults
    except Exception as e:
        st.error(f"Error loading make_league.py: {e}")
        return {}

def _fmt_pct(x): 
    try: return f"{float(x)*100:.0f}%"
    except: return "—"

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

def _ensure_on_path(p: Path):
    p = p.resolve()
    if str(p.parent) not in sys.path:
        sys.path.insert(0, str(p.parent))

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

def _apply_overrides_to_module(mod: object, cfg: dict):
    """Assign tuned dicts back to the imported make_league module so its helpers/main() use them."""
    for k in [
        "PITCH_HAND_P","BAT_SIDE_P","TWO_WAY_RATE",
        "_CMD_TIER_PARAMS","GAMES_PER_TEAM"
    ]:
        if k in cfg:
            # For _CMD_TIER_PARAMS, ensure clip key is present for each tier
            if k == "_CMD_TIER_PARAMS":
                cmd_tier_params = copy.deepcopy(cfg[k])
                for tier in ["Rookie", "AAA", "Majors"]:
                    if tier in cmd_tier_params and "clip" not in cmd_tier_params[tier]:
                        # Add default clip values if missing
                        if tier == "Rookie":
                            cmd_tier_params[tier]["clip"] = (0.60, 1.25)
                        elif tier == "AAA":
                            cmd_tier_params[tier]["clip"] = (0.70, 1.22)
                        elif tier == "Majors":
                            cmd_tier_params[tier]["clip"] = (0.80, 1.18)
                setattr(mod, k, cmd_tier_params)
            else:
                setattr(mod, k, copy.deepcopy(cfg[k]))
    if "DEFAULT_NUM_ORGS" in cfg:
        setattr(mod, "DEFAULT_NUM_ORGS", int(cfg["DEFAULT_NUM_ORGS"]))
    
    # Apply SIM_KNOBS to sim_utils if they exist
    if "SIM_KNOBS" in cfg:
        try:
            # Import sim_utils module
            sim_utils_mod = importlib.import_module("sim_utils")
            sim_knobs = cfg["SIM_KNOBS"]
            for knob, value in sim_knobs.items():
                if hasattr(sim_utils_mod, knob):
                    setattr(sim_utils_mod, knob, value)
        except Exception as e:
            st.warning(f"Could not apply simulation knobs: {e}")

def _apply_sim_knobs(sim_mod, sim_knobs: dict):
    """Assign tuned sim knobs back to sim_utils before a run."""
    if not sim_knobs: 
        return
    for k, v in sim_knobs.items():
        if hasattr(sim_mod, k):
            setattr(sim_mod, k, v)

# Load defaults if not already in session state
if ok_scripts and not st.session_state.league_config:
    defaults = _load_make_defaults(make_league_path)
    st.session_state.league_config = copy.deepcopy(defaults)

# Main title
st.title("League Simulator")
st.markdown("---")

# Step 1: League Configuration
if st.session_state.step >= 1:
    with st.expander(" Step 1: League Configuration", expanded=(st.session_state.step == 1)):
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-header">Step 1: Configure Your League</div>', unsafe_allow_html=True)
        
        if not ok_scripts:
            st.error("⚠️ Point to your scripts folder in the sidebar first.")
            st.stop()
        
        cfg = st.session_state.league_config
        
        # Demographics
        st.markdown('<div class="step-subheader">Player Demographics</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Pitcher Handedness**")
            ph_r = st.slider(
                "Right-handed Pitchers (%)", 
                0, 100, 
                int(cfg.get("PITCH_HAND_P", {}).get("Right", 0.70) * 100),
                help="Percentage of right-handed pitchers in the league"
            )
            ph_l = 100 - ph_r
            cfg["PITCH_HAND_P"] = {"Right": ph_r/100.0, "Left": ph_l/100.0}
            st.write(f"Left-handed Pitchers: {ph_l}%")
        
        with col2:
            st.markdown("**Batter Handedness**")
            bs_r = st.slider(
                "Right-handed Batters (%)", 
                0, 100, 
                int(cfg.get("BAT_SIDE_P", {}).get("Right", 0.55) * 100),
                help="Percentage of right-handed batters in the league"
            )
            bs_l = st.slider(
                "Left-handed Batters (%)", 
                0, 100-bs_r, 
                int(cfg.get("BAT_SIDE_P", {}).get("Left", 0.35) * 100),
                help="Percentage of left-handed batters in the league"
            )
            bs_s = 100 - bs_r - bs_l
            total = bs_r + bs_l + bs_s
            if total > 0:
                cfg["BAT_SIDE_P"] = {
                    "Right": bs_r/total, 
                    "Left": bs_l/total, 
                    "Switch": bs_s/total
                }
            st.write(f"Switch Hitters: {bs_s}%")
        
        # Two-way players
        st.markdown('<div class="step-subheader">Two-way Players Rate</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            tw_rk = st.slider(
                "Rookie Level (%)", 
                0, 100, 
                int(cfg.get("TWO_WAY_RATE", {}).get("Rookie", 0.20) * 100),
                help="Percentage of two-way players at Rookie level"
            )
            cfg["TWO_WAY_RATE"]["Rookie"] = tw_rk/100.0
        with col2:
            tw_aa = st.slider(
                "AAA Level (%)", 
                0, 100, 
                int(cfg.get("TWO_WAY_RATE", {}).get("AAA", 0.10) * 100),
                help="Percentage of two-way players at AAA level"
            )
            cfg["TWO_WAY_RATE"]["AAA"] = tw_aa/100.0
        with col3:
            tw_mj = st.slider(
                "Majors Level (%)", 
                0, 100, 
                int(cfg.get("TWO_WAY_RATE", {}).get("Majors", 0.04) * 100),
                help="Percentage of two-way players at Majors level"
            )
            cfg["TWO_WAY_RATE"]["Majors"] = tw_mj/100.0
        
        # Configuration summary
        st.markdown('<div class="step-subheader">Configuration Summary</div>', unsafe_allow_html=True)
        bs = cfg.get("BAT_SIDE_P", {})
        ph = cfg.get("PITCH_HAND_P", {})
        tw = cfg.get("TWO_WAY_RATE", {})
        g = cfg.get("GAMES_PER_TEAM", {})
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Demographics**")
            st.write(f"**Throws**: R {_fmt_pct(ph.get('Right',0))} | L {_fmt_pct(ph.get('Left',0))}")
            st.write(f"**Bats**: R {_fmt_pct(bs.get('Right',0))} | L {_fmt_pct(bs.get('Left',0))} | S {_fmt_pct(bs.get('Switch',0))}")
            st.write(f"**Two-way**: Rk {_fmt_pct(tw.get('Rookie',0))} | AAA {_fmt_pct(tw.get('AAA',0))} | MLB {_fmt_pct(tw.get('Majors',0))}")
        
        with c2:
            st.markdown("**Games**")
            st.write(f"**Majors**: {g.get('Majors',30)} games")
            st.write(f"**AAA**: {g.get('AAA',30)} games")
            st.write(f"**Rookie**: {g.get('Rookie',20)} games")
            st.write(f"**Organizations**: {cfg.get('DEFAULT_NUM_ORGS',13)}")
        
        with c3:
            st.markdown("**Command Tiers**")
            p = cfg.get("_CMD_TIER_PARAMS", {})
            st.write(f"**Rookie** μ {p.get('Rookie',{}).get('mu','—')} σ {p.get('Rookie',{}).get('sd','—')}")
            st.write(f"**AAA** μ {p.get('AAA',{}).get('mu','—')} σ {p.get('AAA',{}).get('sd','—')}")
            st.write(f"**Majors** μ {p.get('Majors',{}).get('mu','—')} σ {p.get('Majors',{}).get('sd','—')}")
        
        # Quick Presets System
        with st.expander("🎮 **Quick Presets**", expanded=False):
            st.markdown("**One-click configurations for common scenarios:**")
            preset_cols = st.columns(4)
            
            with preset_cols[0]:
                if st.button("🔵 Modern MLB", use_container_width=True, help="Realistic modern baseball parameters"):
                    # Modern MLB settings
                    cfg["PITCH_HAND_P"] = {"Right": 0.72, "Left": 0.28}
                    cfg["BAT_SIDE_P"] = {"Right": 0.55, "Left": 0.35, "Switch": 0.10}
                    cfg["GAMES_PER_TEAM"] = {"Majors": 162, "AAA": 144, "Rookie": 76}
                    if "SIM_KNOBS" not in cfg:
                        cfg["SIM_KNOBS"] = {}
                    cfg["SIM_KNOBS"]["BALL_BIAS"] = 1.00
                    cfg["SIM_KNOBS"]["TTO_PENALTY"] = 0.08
                    st.success("✅ Applied Modern MLB preset!")
                    st.rerun()
                    
            with preset_cols[1]:
                if st.button("⚾ Classic Baseball", use_container_width=True, help="1990s-style parameters"):
                    # 1990s-style baseball
                    cfg["PITCH_HAND_P"] = {"Right": 0.75, "Left": 0.25}
                    cfg["BAT_SIDE_P"] = {"Right": 0.65, "Left": 0.30, "Switch": 0.05}
                    cfg["GAMES_PER_TEAM"] = {"Majors": 162, "AAA": 144, "Rookie": 76}
                    if "SIM_KNOBS" not in cfg:
                        cfg["SIM_KNOBS"] = {}
                    cfg["SIM_KNOBS"]["BALL_BIAS"] = 0.85
                    cfg["SIM_KNOBS"]["TTO_PENALTY"] = 0.05
                    st.success("✅ Applied Classic Baseball preset!")
                    st.rerun()
                    
            with preset_cols[2]:
                if st.button("🚀 High-Offense", use_container_width=True, help="Hitter-friendly environment"):
                    # Hitter-friendly environment
                    cfg["PITCH_HAND_P"] = {"Right": 0.70, "Left": 0.30}
                    cfg["BAT_SIDE_P"] = {"Right": 0.50, "Left": 0.40, "Switch": 0.10}
                    cfg["GAMES_PER_TEAM"] = {"Majors": 162, "AAA": 144, "Rookie": 76}
                    if "SIM_KNOBS" not in cfg:
                        cfg["SIM_KNOBS"] = {}
                    cfg["SIM_KNOBS"]["BALL_BIAS"] = 1.25
                    cfg["SIM_KNOBS"]["TTO_PENALTY"] = 0.12
                    st.success("✅ Applied High-Offense preset!")
                    st.rerun()
                    
            with preset_cols[3]:
                if st.button("🛡️ Pitcher-Dominant", use_container_width=True, help="Pitcher-friendly environment"):
                    # Pitcher-dominant environment
                    cfg["PITCH_HAND_P"] = {"Right": 0.80, "Left": 0.20}
                    cfg["BAT_SIDE_P"] = {"Right": 0.60, "Left": 0.30, "Switch": 0.10}
                    cfg["GAMES_PER_TEAM"] = {"Majors": 162, "AAA": 144, "Rookie": 76}
                    if "SIM_KNOBS" not in cfg:
                        cfg["SIM_KNOBS"] = {}
                    cfg["SIM_KNOBS"]["BALL_BIAS"] = 0.75
                    cfg["SIM_KNOBS"]["TTO_PENALTY"] = 0.05
                    st.success("✅ Applied Pitcher-Dominant preset!")
                    st.rerun()
        
        # Real-time Preview Dashboard
        if "SIM_KNOBS" not in cfg:
            cfg["SIM_KNOBS"] = {}
        
        sim_knobs = cfg["SIM_KNOBS"]
        current_bias = float(sim_knobs.get("BALL_BIAS", 1.0))
        games_majors = int(cfg.get("GAMES_PER_TEAM", {}).get("Majors", 30))
        games_aaa = int(cfg.get("GAMES_PER_TEAM", {}).get("AAA", 30))
        games_rookie = int(cfg.get("GAMES_PER_TEAM", {}).get("Rookie", 20))
        total_games = (games_majors + games_aaa + games_rookie) * cfg.get("DEFAULT_NUM_ORGS", 13)
        expected_walks_per_game = 6.5 * current_bias
        expected_strikeouts = 16.2 / current_bias if current_bias > 0 else 16.2
        estimated_time_minutes = total_games * (2 + current_bias * 0.5)
        game_length_impact = 100 + (current_bias - 1) * 15
        
        # Live Preview removed per request





























        # Quick status overview removed









        # Smart Recommendations
        recommendations = []
        if current_bias > 1.5:
            recommendations.append("🔍 Consider reducing walk bias - current setting may produce unrealistic game lengths")
        elif current_bias < 0.7:
            recommendations.append("⚾ Very low walk rate - games will be dominated by strikeouts")
        
        if games_majors < 20:
            recommendations.append("📅 Consider increasing games per team for more realistic season statistics")
        elif games_majors > 100:
            recommendations.append("⏱️ High game count will significantly increase simulation time")
        
        if total_games > 2000:
            recommendations.append("🚀 Large league detected - consider reducing team count or games for faster testing")
        
        if recommendations:
            with st.expander("💡 **Smart Recommendations**", expanded=False):
                for rec in recommendations:
                    st.info(rec)
        
        # Advanced Configuration
        st.markdown('<div class="step-subheader">Advanced Configuration</div>', unsafe_allow_html=True)
        
        with st.expander("⚙️ **Game Simulation Settings**", expanded=False):
            st.markdown("**🏃‍♂️ Player Management**")
            c1, c2 = st.columns(2)
            sim_knobs["DEFAULT_SP_RECOVERY"] = c1.number_input(
                "SP rest (days)", 0.0, 10.0,
                float(sim_knobs.get("DEFAULT_SP_RECOVERY", 4)),
                key="sp_rest",
                help="Days off required for starting pitchers"
            )
            sim_knobs["DEFAULT_RP_RECOVERY"] = c2.number_input(
                "RP rest (days)", 0.0, 10.0, 
                float(sim_knobs.get("DEFAULT_RP_RECOVERY", 1)),
                key="rp_rest",
                help="Days off required for relief pitchers"
            )
            
            st.markdown("**🚫 Manager Decisions**")
            c1, c2 = st.columns(2)
            sim_knobs["PULL_RUNS"] = c1.number_input(
                "Auto-pull threshold (runs)", 0.0, 12.0,
                float(sim_knobs.get("PULL_RUNS", 4)),
                key="pull_runs",
                help="Pull pitcher when trailing by this many runs"
            )
            sim_knobs["PULL_STRESS_PITCHES"] = c2.number_input(
                "Stress pitch limit", 0.0, 120.0,
                float(sim_knobs.get("PULL_STRESS_PITCHES", 35)),
                key="pull_stress",
                help="High-stress pitch count threshold"
            )
            
            st.markdown("**⚾ Walk Generation Control**")
            sim_knobs["BALL_BIAS"] = st.slider(
                "Walk rate multiplier", 0.5, 3.0,
                float(sim_knobs.get("BALL_BIAS", 1.00)),
                step=0.1, format="%.1f",
                key="ball_bias",
                help="1.0 = realistic walks, >1.0 = more walks, <1.0 = fewer walks"
            )

            # Count and TTO mix scaling
            c1, c2 = st.columns(2)
            sim_knobs["COUNT_MIX_SCALE"] = c1.slider(
                "Count mix scale", 0.0, 2.0,
                float(sim_knobs.get("COUNT_MIX_SCALE", 1.0)),
                step=0.1, format="%.1f",
                help="0 = disable count-based mix; 1 = default; 2 = strong"
            )
            sim_knobs["TTO_MIX_SCALE"] = c2.slider(
                "TTO mix scale", 0.0, 2.0,
                float(sim_knobs.get("TTO_MIX_SCALE", 1.0)),
                step=0.1, format="%.1f",
                help="0 = disable TTO mix shifts; 1 = default; 2 = strong"
            )
            
            walk_guidance = {
                (0.5, 0.8): ("🟢 Very low walk rates", "Few walks, pitcher-friendly"),
                (0.8, 1.2): ("🔵 Normal walk rates", "Realistic baseball statistics"), 
                (1.2, 2.0): ("🟡 High walk rates", "Hitter-friendly, more baserunners"),
                (2.0, 3.1): ("🔴 Very high walk rates", "Extreme offensive environment")
            }
            
            current_bias = float(sim_knobs.get("BALL_BIAS", 1.0))
            for (low, high), (status, desc) in walk_guidance.items():
                if low <= current_bias < high:
                    st.caption(f"{status}: {desc}")
                    break
        
        # Advanced Simulation Settings
        with st.expander("⚙️ **Advanced Simulation Settings**", expanded=False):
            st.markdown("**💪 Fatigue & Performance**")
            c1, c2, c3 = st.columns(3)
            sim_knobs["FATIGUE_PER_PITCH_OVER"] = c1.number_input(
                "Fatigue per extra pitch", 0.000, 0.200, 
                float(sim_knobs.get("FATIGUE_PER_PITCH_OVER", 0.015)),
                step=0.001, format="%.3f",
                help="Command penalty per pitch over limit"
            )
            sim_knobs["FATIGUE_PER_BF_OVER"] = c2.number_input(
                "Fatigue per extra batter", 0.000, 0.200,
                float(sim_knobs.get("FATIGUE_PER_BF_OVER", 0.03)),
                step=0.001, format="%.3f",
                help="Command penalty per batter over expected"
            )
            sim_knobs["TTO_PENALTY"] = c3.number_input(
                "3rd time penalty", 0.000, 0.300,
                float(sim_knobs.get("TTO_PENALTY", 0.10)),
                step=0.005, format="%.3f",
                help="Extra penalty 3rd time through order"
            )

            # Save/Load SIM_KNOBS presets
            st.markdown("**Presets**")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                import json as _json
                knobs_json = _json.dumps(sim_knobs, indent=2)
                st.download_button(
                    label="Download Knobs JSON",
                    data=knobs_json.encode("utf-8"),
                    file_name="sim_knobs.json",
                    mime="application/json",
                    help="Save current simulation knobs to a JSON file"
                )
            with pcol2:
                up = st.file_uploader("Load Knobs JSON", type=["json"], key="knobs_upload")
                if up is not None:
                    try:
                        import json as _json
                        loaded = _json.loads(up.getvalue().decode("utf-8"))
                        if not isinstance(loaded, dict):
                            raise ValueError("Expected a JSON object with knob keys")
                        # Update SIM_KNOBS in config and refresh
                        cfg["SIM_KNOBS"].update(loaded)
                        st.success("Loaded knobs from JSON. Settings updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load knobs JSON: {e}")
            
            st.markdown("**🏃 Velocity & Spin Loss**")
            c1, c2 = st.columns(2)
            sim_knobs["VELO_LOSS_PER_OVER10"] = c1.number_input(
                "Velo loss per 10 pitches", 0.0, 1.0,
                float(sim_knobs.get("VELO_LOSS_PER_OVER10", 0.15)),
                step=0.01, format="%.2f",
                help="MPH lost per 10 pitches over limit"
            )
            sim_knobs["SPIN_LOSS_PER_OVER10"] = c2.number_input(
                "Spin loss per 10 pitches", 0.0, 200.0,
                float(sim_knobs.get("SPIN_LOSS_PER_OVER10", 20.0)),
                step=1.0, format="%.0f",
                help="RPM lost per 10 pitches over limit"
            )
            
            st.markdown("**🚑 Injury System**")
            c1, c2, c3 = st.columns(3)
            injury_chance_pct = c1.number_input(
                "Injury chance (%)", 0.0, 50.0,
                float(sim_knobs.get("INJURY_CHANCE_HEAVY_OVER", 0.05) * 100),
                step=0.1, format="%.1f",
                help="Injury probability when overused"
            )
            sim_knobs["INJURY_CHANCE_HEAVY_OVER"] = injury_chance_pct / 100.0
            
            dur_lo, dur_hi = sim_knobs.get("INJURY_DUR_RANGE", (10, 30))
            dur_lo = c2.number_input(
                "Min injury days", 1.0, 180.0,
                float(dur_lo),
                help="Minimum injury duration"
            )
            dur_hi = c3.number_input(
                "Max injury days", dur_lo, 365.0,
                float(dur_hi),
                help="Maximum injury duration"
            )
            sim_knobs["INJURY_DUR_RANGE"] = (int(dur_lo), int(dur_hi))
            
            st.markdown("**⏰ Extra Innings**")
            c1, c2 = st.columns(2)
            sim_knobs["EXTRA_INNING_FATIGUE_SCALE"] = c1.number_input(
                "Fatigue scale per inning", 0.0, 2.0,
                float(sim_knobs.get("EXTRA_INNING_FATIGUE_SCALE", 0.50)),
                step=0.01, format="%.2f",
                help="Fatigue multiplier for extra innings"
            )
            sim_knobs["EXTRA_INNING_CMD_FLAT_PENALTY"] = c2.number_input(
                "Command penalty per inning", 0.0, 0.200,
                float(sim_knobs.get("EXTRA_INNING_CMD_FLAT_PENALTY", 0.03)),
                step=0.001, format="%.3f",
                help="Flat command penalty per extra inning"
            )

        # Advanced Configuration (JSON) for Pitching Model
        with st.expander("?? Advanced Configuration (JSON)", expanded=False):
            st.caption("Provide raw JSON for: command data, pitch usage, pitcher types, and arm angle by type.")
            cfg = st.session_state.league_config
            default_model = {
                "command_data": {
                    "Rookie": {"mu": 1.00, "sd": 0.10, "clip": [0.60, 1.25]},
                    "AAA":    {"mu": 1.02, "sd": 0.10, "clip": [0.70, 1.22]},
                    "Majors": {"mu": 1.04, "sd": 0.10, "clip": [0.80, 1.18]},
                },
                "pitcher_types": {
                    "Power": {
                        "arm_angle": "Overhand",
                        "usage": {"FourSeam": 0.45, "Slider": 0.30, "Changeup": 0.10, "Curve": 0.10, "Cutter": 0.05}
                    },
                    "Sinkerballer": {
                        "arm_angle": "ThreeQuarters",
                        "usage": {"Sinker": 0.40, "Slider": 0.25, "Changeup": 0.20, "FourSeam": 0.10, "Curve": 0.05}
                    },
                    "Command": {
                        "arm_angle": "ThreeQuarters",
                        "usage": {"FourSeam": 0.30, "Changeup": 0.30, "Curve": 0.20, "Slider": 0.15, "Cutter": 0.05}
                    }
                }
            }
            # Use existing if present
            model_key = "PITCHING_MODEL"
            pitching_model = copy.deepcopy(cfg.get(model_key) or default_model)

            json_str = st.text_area(
                "Pitching Model JSON",
                value=json.dumps(pitching_model, indent=2),
                height=320,
            )
            up1, up2, up3 = st.columns([1,1,1])
            uploaded = up1.file_uploader("Upload JSON", type=["json"], accept_multiple_files=False)
            if uploaded is not None:
                try:
                    uploaded_json = json.loads(uploaded.getvalue().decode("utf-8"))
                    json_str = json.dumps(uploaded_json, indent=2)
                    st.success("Loaded JSON from file. Click Apply to save.")
                except Exception as e:
                    st.error(f"Upload parse error: {e}")

            def _validate_model(m: dict) -> list[str]:
                errs = []
                if not isinstance(m, dict):
                    errs.append("Top-level must be an object")
                    return errs
                if "command_data" not in m:
                    errs.append("Missing 'command_data'")
                if "pitcher_types" not in m:
                    errs.append("Missing 'pitcher_types'")
                # Validate command_data
                cd = m.get("command_data", {})
                for tier in ("Rookie","AAA","Majors"):
                    t = cd.get(tier)
                    if not isinstance(t, dict):
                        errs.append(f"command_data.{tier} must be an object")
                        continue
                    for k in ("mu","sd","clip"):
                        if k not in t:
                            errs.append(f"command_data.{tier} missing '{k}'")
                # Validate pitcher_types usage sums
                for tname, tinfo in (m.get("pitcher_types") or {}).items():
                    usage = (tinfo or {}).get("usage", {})
                    if usage:
                        s = sum(float(v) for v in usage.values())
                        if not (0.99 <= s <= 1.01):
                            errs.append(f"usage for '{tname}' sums to {s:.2f} (should be ~1.00)")
                return errs

            if up2.button("Validate JSON"):
                try:
                    parsed = json.loads(json_str)
                    errs = _validate_model(parsed)
                    if errs:
                        for e in errs:
                            st.error(e)
                    else:
                        st.success("Pitching model JSON looks valid.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

            if up3.button("Apply JSON"):
                try:
                    parsed = json.loads(json_str)
                    errs = _validate_model(parsed)
                    if errs:
                        for e in errs:
                            st.error(e)
                    else:
                        st.session_state.league_config[model_key] = parsed
                        st.success("Applied pitching model JSON to configuration.")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
        
        # Diagnostics & Troubleshooting
        with st.expander("🔧 **Diagnostics & Troubleshooting**", expanded=False):
            st.markdown("**Built-in diagnostic tools for validation and debugging:**")
            
            diag_col1, diag_col2 = st.columns(2)
            
            with diag_col1:
                if st.button("🧪 Test Configuration", use_container_width=True):
                    try:
                        # Run quick validation
                        errors = []
                        
                        # Validate walk bias
                        if current_bias < 0.3 or current_bias > 3.0:
                            errors.append(f"Walk bias {current_bias} is outside recommended range (0.3-3.0)")
                        
                        # Validate games
                        if games_majors < 10 or games_majors > 200:
                            errors.append(f"Majors games {games_majors} outside reasonable range (10-200)")
                        
                        # Check for missing configuration
                        if not cfg.get("PITCH_HAND_P") or not cfg.get("BAT_SIDE_P"):
                            errors.append("Missing handedness configuration")
                        
                        if errors:
                            for error in errors:
                                st.error(f"❌ {error}")
                        else:
                            st.success("✅ Configuration valid - ready for simulation")
                            
                    except Exception as e:
                        st.error(f"❌ Configuration test failed: {str(e)}")
            
            with diag_col2:
                if st.button("📊 Export Debug Info", use_container_width=True):
                    # Generate comprehensive debug report
                    debug_info = {
                        "config": cfg,
                        "simulation_knobs": sim_knobs,
                        "preview_metrics": {
                            "total_games": total_games,
                            "walks_per_game": expected_walks_per_game,
                            "strikeouts_per_game": expected_strikeouts,
                            "estimated_time": estimated_time_minutes,
                            "game_length_impact": game_length_impact
                        },
                        "recommendations": recommendations,
                        "timestamp": str(pd.Timestamp.now()),
                        "version": "2.0"
                    }
                    st.download_button(
                        "Download Debug Report",
                        data=json.dumps(debug_info, indent=2, default=str),
                        file_name=f"modelca_debug_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    st.success("✅ Debug report prepared for download")
        
        # Next step button
        if st.button("✅ Complete Step 1 - Configure League", key="step1_complete"):
            st.session_state.step = max(st.session_state.step, 2)
            st.success("Step 1 completed! You can now proceed to generate your league.")
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Summary between Step 1 and Step 2
if st.session_state.step >= 1:
    with st.container():
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### Current Configuration Summary")
        cfg = st.session_state.league_config or {}
        ph = cfg.get("PITCH_HAND_P", {})
        bs = cfg.get("BAT_SIDE_P", {})
        tw = cfg.get("TWO_WAY_RATE", {})
        g  = cfg.get("GAMES_PER_TEAM", {})
        sim_knobs = cfg.get("SIM_KNOBS", {})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Roster Mix**")
            st.write(f"Pitchers R/L: {int(ph.get('Right',0.7)*100)}% / {int(ph.get('Left',0.3)*100)}%")
            st.write(f"Batters R/L/S: {int(bs.get('Right',0.55)*100)}% / {int(bs.get('Left',0.35)*100)}% / {int(bs.get('Switch',0.10)*100)}%")
            st.write(f"Two-way Rk/AAA/MLB: {int(tw.get('Rookie',0)*100)}% / {int(tw.get('AAA',0)*100)}% / {int(tw.get('Majors',0)*100)}%")
        with c2:
            st.markdown("**Structure**")
            st.write(f"Organizations: {cfg.get('DEFAULT_NUM_ORGS', 13)}")
            st.write(f"Games Maj/AAA/Rk: {g.get('Majors',30)} / {g.get('AAA',30)} / {g.get('Rookie',20)}")
        with c3:
            st.markdown("**Simulation Knobs**")
            st.write(f"Walk bias: {float(sim_knobs.get('BALL_BIAS',1.0)):.2f}x")
            st.write(f"SP/RP rest: {sim_knobs.get('DEFAULT_SP_RECOVERY',4)}/{sim_knobs.get('DEFAULT_RP_RECOVERY',1)} days")
            st.write(f"Pull runs/stress: {sim_knobs.get('PULL_RUNS',4)}/{sim_knobs.get('PULL_STRESS_PITCHES',35)}")
        st.markdown('</div>', unsafe_allow_html=True)

# Step 2: Generate League
if st.session_state.step >= 2:
    with st.expander(" Step 2: Generate League", expanded=(st.session_state.step == 2)):
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-header">Step 2: Generate Your League</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            seed = st.number_input("Random Seed", min_value=0, value=7, help="Seed for reproducible league generation")
        with col2:
            start_date = st.date_input("Start Date", value=date(2025, 4, 1), help="First game date for Majors (AAA and Rookie start 1-2 days later)")

        # League structure controls (moved here to avoid duplication on Step 1)
        cfg = st.session_state.league_config
        lc1, lc2, lc3, lc4 = st.columns(4)
        cfg["DEFAULT_NUM_ORGS"] = lc1.number_input("Organizations", min_value=2, max_value=50, value=int(cfg.get("DEFAULT_NUM_ORGS", 13)))
        gpt = cfg.setdefault("GAMES_PER_TEAM", cfg.get("GAMES_PER_TEAM", {"Majors": 30, "AAA": 30, "Rookie": 20}))
        gpt["Majors"] = int(lc2.number_input("Majors games", 10, 200, int(gpt.get("Majors", 30))))
        gpt["AAA"]    = int(lc3.number_input("AAA games",    10, 200, int(gpt.get("AAA", 30))))
        gpt["Rookie"]  = int(lc4.number_input("Rookie games", 10, 200, int(gpt.get("Rookie", 20))))

        # Multi-season planning inputs (stored for future multi-season runner)
        ms1, ms2, ms3 = st.columns(3)
        cfg["NUM_SEASONS"] = int(ms1.number_input("Number of seasons", min_value=1, max_value=50, value=int(cfg.get("NUM_SEASONS", 1))))
        cfg["DEV_SPEED"] = float(ms2.slider("Player development speed", 0.5, 2.0, float(cfg.get("DEV_SPEED", 1.0)), 0.1))
        ms3.info("Higher development speed may increase variance and runtime slightly.")
        
        # Preview button
        if st.button("🔍 Preview League (In-Memory)", key="preview_league"):
            try:
                # Reload and import make_league module
                _reload_module("make_league", make_league_path)
                ml = importlib.import_module("make_league")
                _apply_overrides_to_module(ml, st.session_state.league_config)
                
                import random
                rng = random.Random(int(seed))
                orgs = ml.build_organizations(int(st.session_state.league_config.get("DEFAULT_NUM_ORGS", 13)), rng)
                teams = ml.build_teams(orgs)
                rosters = ml.build_rosters(teams, rng)
                
                # Build schedules by tier
                tier_ids = {"Majors": [], "AAA": [], "Rookie": []}
                for t in teams:
                    tier_ids[t["Tier"]].append(t["TeamID"])
                for k in tier_ids: 
                    tier_ids[k].sort()
                
                sched_M = ml.build_tier_schedule(tier_ids["Majors"], ml.GAMES_PER_TEAM["Majors"], start_date, rng)
                sched_A = ml.build_tier_schedule(tier_ids["AAA"], ml.GAMES_PER_TEAM["AAA"], start_date + timedelta(days=1), rng)
                sched_R = ml.build_tier_schedule(tier_ids["Rookie"], ml.GAMES_PER_TEAM["Rookie"], start_date + timedelta(days=2), rng)
                
                # Tag Tier + GameIDs
                gid = 1
                for r in sched_M: r["Tier"] = "Majors"; r["GameID"] = f"M{gid:04d}"; gid += 1
                gid = 1
                for r in sched_A: r["Tier"] = "AAA"; r["GameID"] = f"A{gid:04d}"; gid += 1
                gid = 1
                for r in sched_R: r["Tier"] = "Rookie"; r["GameID"] = f"R{gid:04d}"; gid += 1
                
                schedules_all = sched_M + sched_A + sched_R
                
                # Store preview data
                st.session_state.preview_data = {
                    "organizations": pd.DataFrame(orgs),
                    "teams": pd.DataFrame(teams),
                    "rosters": pd.DataFrame(rosters),
                    "schedule": pd.DataFrame(schedules_all),
                }
                
                st.success("✅ League preview generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating preview: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Display preview if available
        if st.session_state.preview_data:
            st.markdown('<div class="step-subheader">League Preview</div>', unsafe_allow_html=True)
            df_orgs = st.session_state.preview_data["organizations"]
            df_teams = st.session_state.preview_data["teams"]
            df_roster = st.session_state.preview_data["rosters"]
            df_sched = st.session_state.preview_data["schedule"]
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Organizations", len(df_orgs))
            m2.metric("Teams", len(df_teams))
            m3.metric("Players", len(df_roster))
            m4.metric("Games", len(df_sched))
            
            # Preview dataframes
            with st.expander("📋 Organizations Preview"):
                st.dataframe(df_orgs, use_container_width=True)
            
            with st.expander("🏆 Teams Preview"):
                st.dataframe(df_teams, use_container_width=True)
            
            with st.expander("👥 Rosters Preview (First 20 Players)"):
                st.dataframe(df_roster.head(20), use_container_width=True)
            
            with st.expander("📅 Schedule Preview (First 20 Games)"):
                st.dataframe(df_sched.head(20), use_container_width=True)
        
        # Schedule Generation
        st.markdown("**Schedule Generation**")
        st.caption("Choose cadence used when generating schedule.csv")
        st.selectbox(
            "Schedule cadence",
            options=["weekend","balanced"],
            index=0,
            key="make_sched_cadence_step",
            help="Weekend: Fri/Sat/Sun with occasional Tue/Wed; Balanced: spread midweek/weekend"
        )
        st.slider(
            "Midweek game probability (per week)", 0.0, 1.0, 0.35, 0.05,
            key="make_midweek_prob_step",
            help="For weekend cadence, chance to add an extra Tue/Wed game"
        )

        # Generate league button
        if st.button("🏗️ Generate League CSVs", key="generate_league"):
            try:
                # Fresh import + apply overrides
                _reload_module("make_league", make_league_path)
                make_mod = importlib.import_module("make_league")
                _apply_overrides_to_module(make_mod, st.session_state.league_config)
                
                # Prepare output directory
                out_dir = scripts_dir / "league_out"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                argv = [
                    "make_league.py",
                    "--num_orgs", str(int(st.session_state.league_config.get("DEFAULT_NUM_ORGS", 13))),
                    "--out_dir", str(out_dir),
                    "--seed", str(int(seed)),
                    "--start_date", start_date.isoformat(),
                    "--maj_games", str(int(st.session_state.league_config["GAMES_PER_TEAM"]["Majors"])),
                    "--aaa_games", str(int(st.session_state.league_config["GAMES_PER_TEAM"]["AAA"])),
                    "--rookie_games", str(int(st.session_state.league_config["GAMES_PER_TEAM"]["Rookie"])),
                    "--schedule_cadence", str(st.session_state.get("make_sched_cadence_step", "weekend")),
                    "--midweek_prob", str(float(st.session_state.get("make_midweek_prob_step", 0.35))),
                ]
                
                with st.status("Generating league...", expanded=True) as status:
                    logs, code = _run_module_main(make_mod, argv)
                    st.code(logs or "(no output)")
                    status.update(label="League generated" if code in (None, 0) else "Exited with errors", state="complete")
                
                if code not in (None, 0):
                    st.error(f"make_league.py exited with code {code}. See message above.")
                else:
                    st.session_state.league_generated = True
                    st.success("✅ League generated successfully!")
                    st.session_state.step = max(st.session_state.step, 3)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error generating league: {e}")
                import traceback
                st.code(traceback.format_exc())
        # Roster tuning panel (Pitch usage, Type, Arm Angle)
        try:
            out_dir = scripts_dir / "league_out"
            roster_path = out_dir / "rosters.csv"
            if roster_path.exists():
                with st.expander("? Roster Tuning (Pitch Usage, Type, Arm Angle)", expanded=False):
                    try:
                        df_roster = pd.read_csv(roster_path)
                    except Exception as e:
                        st.error(f"Unable to read roster CSV: {e}")
                        df_roster = None

                    if df_roster is not None:
                        # Ensure editable columns exist
                        if "PitcherType" not in df_roster.columns:
                            df_roster["PitcherType"] = ""
                        if "ArmAngle" not in df_roster.columns:
                            df_roster["ArmAngle"] = ""
                        if "UsageJSON" not in df_roster.columns:
                            df_roster["UsageJSON"] = ""
                        if "CommandTier" not in df_roster.columns:
                            df_roster["CommandTier"] = 1.00

                        st.markdown("Adjust pitcher profiles by role, team, or in bulk. Saves to rosters_custom.csv.")
                        # Filters
                        colf1, colf2, colf3 = st.columns(3)
                        roles = sorted(set([str(x).upper() for x in df_roster.get("Role", pd.Series([])).fillna("").unique()]))
                        role_sel = colf1.selectbox("Role", ["ALL"] + roles, index=0)
                        teams = sorted(set([str(x) for x in df_roster.get("TeamID", pd.Series([])).fillna("").unique()]))
                        team_sel = colf2.selectbox("Team", ["ALL"] + teams, index=0)
                        tiers = sorted(set([str(x) for x in df_roster.get("Tier", pd.Series([])).fillna("").unique()]))
                        tier_sel = colf3.selectbox("Tier", ["ALL"] + tiers, index=0)

                        mask = pd.Series([True]*len(df_roster))
                        if role_sel != "ALL":
                            mask &= df_roster["Role"].astype(str).str.upper().eq(role_sel)
                        if team_sel != "ALL":
                            mask &= df_roster["TeamID"].astype(str).eq(team_sel)
                        if tier_sel != "ALL":
                            mask &= df_roster["Tier"].astype(str).eq(tier_sel)

                        model = st.session_state.league_config.get("PITCHING_MODEL", {})
                        type_defs = model.get("pitcher_types", {})
                        type_names = sorted(type_defs.keys())

                        cc1, cc2, cc3 = st.columns(3)
                        sel_type = cc1.selectbox("Pitcher Type", ["(no change)"] + type_names, index=0)
                        new_cmd = cc2.number_input("CommandTier (optional)", min_value=0.5, max_value=2.0, value=1.00, step=0.05)
                        apply_cmd = cc3.checkbox("Apply CommandTier", value=False)

                        if st.button("Apply to filtered pitchers"):
                            if sel_type != "(no change)" and sel_type in type_defs:
                                usage = type_defs[sel_type].get("usage", {})
                                arm = type_defs[sel_type].get("arm_angle", "")
                                df_roster.loc[mask, "PitcherType"] = sel_type
                                df_roster.loc[mask, "ArmAngle"] = arm
                                df_roster.loc[mask, "UsageJSON"] = json.dumps(usage)
                                # Optional: map arm angle to simple release metrics if present
                                if "RelHeight_ft" in df_roster.columns or "RelSide_ft" in df_roster.columns or "Extension_ft" in df_roster.columns:
                                    def _arm_defaults(a):
                                        a = (a or "").lower()
                                        if a == "overhand":
                                            return 6.5, 0.5, 6.5
                                        if a == "threequarters" or a == "threequarters":
                                            return 6.0, 1.5, 6.0
                                        if a == "sidearm":
                                            return 5.0, 2.5, 6.0
                                        if a == "submarine":
                                            return 4.5, 3.0, 5.8
                                        return None
                                    vals = _arm_defaults(arm)
                                    if vals is not None:
                                        rh, rs, ex = vals
                                        if "RelHeight_ft" in df_roster.columns:
                                            df_roster.loc[mask, "RelHeight_ft"] = rh
                                        if "RelSide_ft" in df_roster.columns:
                                            df_roster.loc[mask, "RelSide_ft"] = rs
                                        if "Extension_ft" in df_roster.columns:
                                            df_roster.loc[mask, "Extension_ft"] = ex
                            if apply_cmd:
                                df_roster.loc[mask, "CommandTier"] = float(new_cmd)
                            st.success(f"Applied changes to {int(mask.sum())} pitchers")

                        st.dataframe(
                            df_roster.loc[mask, [c for c in ["FirstName","LastName","TeamID","Role","Tier","PitcherType","ArmAngle","CommandTier","UsageJSON"] if c in df_roster.columns]].head(50),
                            use_container_width=True,
                        )

                        if st.button("Save as rosters_custom.csv"):
                            try:
                                df_roster.to_csv(out_dir / "rosters_custom.csv", index=False)
                                st.success("Saved league_out/rosters_custom.csv. Simulation will use it if present.")
                            except Exception as e:
                                st.error(f"Save failed: {e}")
        except Exception as e:
            st.warning(f"Roster tuning panel error: {e}")

        
        # Next step button (only if league is generated)
        if st.session_state.league_generated:
            if st.button("✅ Complete Step 2 - Generate League", key="step2_complete"):
                st.session_state.step = max(st.session_state.step, 3)
                st.success("Step 2 completed! You can now run simulations.")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

# Step 3: Required Files (Multi-Season Ready)
if st.session_state.step >= 3:
    with st.expander(" Step 3: Required Files", expanded=(st.session_state.step == 3)):
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-header">Step 3: Validate Required Files</div>', unsafe_allow_html=True)

        if not st.session_state.league_generated:
            st.warning("?? Please generate a league first (Step 2).")
            st.stop()

        out_dir = scripts_dir / "league_out"
        required_files = ["organizations.csv", "schedule.csv"]
        roster_ok = (out_dir / "rosters.csv").exists() or (out_dir / "rosters_custom.csv").exists()
        team_ok = (out_dir / "teams.csv").exists()
        missing_files = [f for f in required_files if not (out_dir / f).exists()]

        sim_script = scripts_dir / "simulate_seasons_roster_style.py"
        priors_path = scripts_dir / "rules_pitch_by_pitch.yaml"
        extras_missing = []
        if not sim_script.exists(): extras_missing.append("simulate_seasons_roster_style.py")
        if not priors_path.exists(): extras_missing.append("rules_pitch_by_pitch.yaml")

        st.markdown("- Required league CSVs: organizations.csv, teams.csv, rosters.csv, schedule.csv")
        st.markdown("- Simulation script: simulate_seasons_roster_style.py")
        st.markdown("- Priors file: rules_pitch_by_pitch.yaml")
        
        # Optional: Upload template CSVs
        st.markdown("- Optional templates: YakkerTech and TrackMan header templates")
        up_c1, up_c2 = st.columns(2)
        yakker_file = up_c1.file_uploader("Upload YakkerTech Template CSV", type=["csv"], key="yakker_tpl")
        trackman_file = up_c2.file_uploader("Upload TrackMan Template CSV", type=["csv"], key="trackman_tpl")
        templates_dir = scripts_dir / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        if yakker_file is not None:
            try:
                (templates_dir / "yakker_template.csv").write_bytes(yakker_file.getvalue())
                st.session_state["yakker_template_path"] = str(templates_dir / "yakker_template.csv")
                st.success("Saved YakkerTech template.")
            except Exception as e:
                st.error(f"Failed to save YakkerTech template: {e}")
        if trackman_file is not None:
            try:
                (templates_dir / "trackman_template.csv").write_bytes(trackman_file.getvalue())
                st.session_state["trackman_template_path"] = str(templates_dir / "trackman_template.csv")
                st.success("Saved TrackMan template.")
            except Exception as e:
                st.error(f"Failed to save TrackMan template: {e}")

        if missing_files or extras_missing or (not roster_ok) or (not team_ok):
            if missing_files:
                st.error(f"Missing league files: {', '.join(missing_files)}")
            if not team_ok:
                st.error("Missing teams.csv")
            if not roster_ok:
                st.error("Missing roster file (rosters.csv or rosters_custom.csv)")
            if extras_missing:
                st.error(f"Missing extras: {', '.join(extras_missing)}")
        else:
            st.success("All required files found. Ready to run simulation.")

        if st.button("? Complete Step 3 - Files Validated", key="step3_files_complete"):
            st.session_state.step = max(st.session_state.step, 4)
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# Step 4: Run Simulation
if st.session_state.step >= 4:
    with st.expander(" Step 4: Run Simulation", expanded=(st.session_state.step == 4)):
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-header">Step 4: Run Your Simulation</div>', unsafe_allow_html=True)
        
        if not st.session_state.league_generated:
            st.warning("⚠️ Please generate a league first (Step 2)")
            st.stop()
        
        # Check if required files exist
        out_dir = scripts_dir / "league_out"
        required_files = ["organizations.csv", "teams.csv", "rosters.csv", "schedule.csv"]
        missing_files = [f for f in required_files if not (out_dir / f).exists()]
        
        if missing_files:
            st.error(f"❌ Missing required files: {', '.join(missing_files)}. Please generate the league first.")
            st.stop()
        
        # Simulation parameters
        st.markdown('<div class="step-subheader">Simulation Parameters</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_seed = st.number_input("Simulation Seed", min_value=0, value=42, help="Seed for reproducible simulation")
        with col2:
            # Get ball bias from league config if available, otherwise default to 1.0
            default_ball_bias = 1.0
            if "SIM_KNOBS" in st.session_state.league_config:
                default_ball_bias = st.session_state.league_config["SIM_KNOBS"].get("BALL_BIAS", 1.0)
            ball_bias = st.slider("Ball Bias (Walk Rate)", 0.5, 2.0, float(default_ball_bias), 0.05, 
                                help="Controls walk rate - lower values = more strikeouts, higher values = more walks")
        with col3:
            # Number of seasons to simulate this run
            num_seasons_input = st.number_input(
                "Number of seasons",
                min_value=1,
                max_value=50,
                value=int(st.session_state.league_config.get("NUM_SEASONS", 1)),
                key="num_seasons_run"
            )
            st.session_state.league_config["NUM_SEASONS"] = int(num_seasons_input)

        # Timeout control (per season run)
        timeout_min = st.number_input(
            "Timeout per season (minutes)",
            min_value=1,
            max_value=240,
            value=30,
            help="Abort the season run if it exceeds this duration",
            key="season_timeout_min"
        )

        # Output formats and optional template
        fmt_opts = st.multiselect("Output formats", ["yakkertech", "trackman"], default=["yakkertech"], help="Choose one or both")
        tpl_yakker = st.session_state.get("yakker_template_path", "")
        if tpl_yakker:
            st.caption(f"Using YakkerTech template: {tpl_yakker}")
        tpl_tm = st.session_state.get("trackman_template_path", "")
        if tpl_tm:
            st.caption(f"Using TrackMan template: {tpl_tm}")
        verbose_logs = st.checkbox("Verbose logs (conversion/sim)", value=False)
        overwrite_from_1 = st.checkbox("Start at Season1 (overwrite)", value=False, help="If checked, writes Season1..N and removes any existing Season folders for this run.")

        # Optional plate-zone targets JSON for plate location calibration
        plate_targets_path = st.text_input(
            "Plate zone targets JSON (optional)",
            value="",
            key="plate_zone_targets_json_step",
            help="Path to a JSON mapping of context keys (pitch|pthrows|bats|count) to target in-zone fractions"
        ).strip()
        if plate_targets_path and not Path(plate_targets_path).exists():
            st.warning(f"Plate zone targets file not found; continuing without it: {plate_targets_path}")

        # Optional label for this run (writes to season_out/<label>/SeasonN)
        run_label = st.text_input("Run label (optional)", value="", help="If set, outputs to season_out/<label>/Season1..N")

        # Plan preview: which Season folders will be written
        try:
            season_root_preview = (scripts_dir / "season_out" / run_label) if run_label.strip() else (scripts_dir / "season_out")
            season_root_preview.mkdir(parents=True, exist_ok=True)
            import re
            existing_prev = [p.name for p in season_root_preview.iterdir() if p.is_dir() and p.name.lower().startswith("season")]
            def _idx_prev(name: str) -> int:
                m = re.search(r"(\d+)$", name)
                return int(m.group(1)) if m else 0
            if overwrite_from_1:
                base_idx_prev = 1
            else:
                base_idx_prev = max([_idx_prev(n) for n in existing_prev] or [0]) + 1
            planned = [f"Season{base_idx_prev + i}" for i in range(int(st.session_state.league_config.get("NUM_SEASONS", 1)))]
            st.caption("Will write into: " + ", ".join(planned))
        except Exception:
            pass

        if "sim_running" not in st.session_state:
            st.session_state.sim_running = False

        # Run simulation button
        if st.button("🏃 Run Simulation", key="run_simulation", use_container_width=True):
            try:
                # Check if simulate_seasons_roster_style.py exists
                simulate_path = scripts_dir / "simulate_seasons_roster_style.py"
                if not simulate_path.exists():
                    st.error("❌ simulate_seasons_roster_style.py not found in scripts folder")
                    st.stop()
                
                # Check if rules_pitch_by_pitch.yaml exists
                priors_path = scripts_dir / "rules_pitch_by_pitch.yaml"
                if not priors_path.exists():
                    st.error("❌ rules_pitch_by_pitch.yaml not found in scripts folder")
                    st.stop()
                
                # Add directory to path if needed
                script_dir = str(scripts_dir)
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                
                # Import and run simulation
                sim_mod = importlib.import_module("simulate_seasons_roster_style")
                
                # Apply simulation knobs if they exist
                if "SIM_KNOBS" in st.session_state.league_config:
                    # Import sim_utils and apply knobs
                    try:
                        sim_utils_mod = importlib.import_module("sim_utils")
                        _apply_sim_knobs(sim_utils_mod, st.session_state.league_config["SIM_KNOBS"])
                    except Exception as e:
                        st.warning(f"Could not apply simulation knobs: {e}")
                
                # Prefer custom roster if present
                custom_roster = out_dir / "rosters_custom.csv"
                roster_csv_path = custom_roster if custom_roster.exists() else (out_dir / "rosters.csv")

                # Multi-season run
                num_seasons = int(st.session_state.league_config.get("NUM_SEASONS", 1))
                last_code = 0
                # Determine next Season index based on existing folders (Season1, Season2, ...)
                season_root = scripts_dir / "season_out"
                season_root.mkdir(parents=True, exist_ok=True)
                import re, shutil
                existing = [p.name for p in season_root.iterdir() if p.is_dir() and p.name.lower().startswith("season")]
                def _idx(name: str) -> int:
                    m = re.search(r"(\d+)$", name)
                    return int(m.group(1)) if m else 0
                if overwrite_from_1:
                    base_idx = 1
                    # Remove Season1..SeasonN if present
                    for s in range(num_seasons):
                        tgt = season_root / f"Season{s+1}"
                        if tgt.exists():
                            try:
                                shutil.rmtree(tgt, ignore_errors=True)
                            except Exception:
                                pass
                else:
                    base_idx = max([_idx(n) for n in existing] or [0]) + 1

                # Stop-on-error option
                stop_on_error = st.checkbox("Stop on first season error", value=False, help="If unchecked, the app will attempt remaining seasons even if one fails.")

                for s in range(num_seasons):
                    season_target = season_root / f"Season{base_idx + s}"
                    season_target.mkdir(parents=True, exist_ok=True)

                for s in range(num_seasons):
                    season_target = season_root / f"Season{base_idx + s}"
                    season_target.mkdir(parents=True, exist_ok=True)

                    argv = [
                        "simulate_seasons_roster_style.py",
                        "--roster_csv", str(roster_csv_path),
                        "--schedule_csv", str(out_dir / "schedule.csv"),
                        "--priors", str(priors_path),
                        "--seed", str(int(sim_seed) + s),
                        "--ball_bias", str(ball_bias),
                        "--out_dir", str(season_target),
                        "--output_formats", *fmt_opts
                    ]
                    if tpl_yakker:
                        argv.extend(["--template_csv", tpl_yakker])
                    if tpl_tm:
                        argv.extend(["--trackman_template_csv", tpl_tm])
                    if plate_targets_path and Path(plate_targets_path).exists():
                        argv.extend(["--plate_zone_targets_json", plate_targets_path])
                    if verbose_logs:
                        argv.append("--verbose")

                    with st.status(f"Running Season{base_idx + s} ({s+1}/{num_seasons})...", expanded=True) as status:
                        # Polished, lightweight progress for this season (streams stdout)
                        try:
                            schedule_path = out_dir / "schedule.csv"
                            total_games = int(pd.read_csv(schedule_path).shape[0]) if schedule_path.exists() else 0
                        except Exception:
                            total_games = 0

                        prog = st.progress(0.0)
                        info = st.empty()
                        log_area = st.empty()

                        # Build subprocess command to stream output live
                        py = sys.executable
                        sim_script = scripts_dir / "simulate_seasons_roster_style.py"
                        cmd = [py, str(sim_script)] + argv[1:]  # drop the module name we used for in-proc

                        done = 0
                        captured_lines = []
                        code = None
                        last_update = time.time()
                        UPDATE_INTERVAL = 0.25  # seconds
                        try:
                            proc = subprocess.Popen(
                                cmd,
                                cwd=str(scripts_dir),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                bufsize=1,
                                universal_newlines=True,
                                encoding="utf-8",
                                errors="replace",
                            )
                            # Stream lines; bump progress when we see a game completed
                            start_time = time.time()
                            timeout_sec = max(60, int(timeout_min * 60))
                            timed_out = False
                            while True:
                                line = proc.stdout.readline() if proc.stdout else ""
                                if not line and (proc.poll() is not None):
                                    break
                                # Timeout check
                                if (time.time() - start_time) > timeout_sec:
                                    timed_out = True
                                    captured_lines.append(f"TIMEOUT: Season exceeded {timeout_min} minutes. Terminating...")
                                    try:
                                        proc.terminate()
                                    except Exception:
                                        pass
                                    try:
                                        proc.wait(timeout=5)
                                    except Exception:
                                        try:
                                            proc.kill()
                                        except Exception:
                                            pass
                                    break
                                if line:
                                    captured_lines.append(line.rstrip("\n"))
                                    if (" Final:" in line) or (" -> " in line):
                                        done += 1
                                    now = time.time()
                                    if (now - last_update) >= UPDATE_INTERVAL:
                                        if total_games > 0:
                                            prog.progress(min(1.0, done / max(1, total_games)))
                                            info.write(f"Season {s+1}: {done}/{total_games} games simulated")
                                        # Keep the last ~200 lines visible
                                        log_area.code("\n".join(captured_lines[-200:]) or "(no output)")
                                        last_update = now
                            code = proc.returncode
                            if timed_out and (code in (None, 0)):
                                code = 124  # common timeout code
                        except FileNotFoundError as e:
                            captured_lines.append(f"ERROR: {e}")
                            code = 1
                        except Exception as e:
                            captured_lines.append(f"ERROR: {e}")
                            code = 1

                        # Finalize UI
                        if total_games > 0:
                            prog.progress(1.0)
                            info.write(f"Season {s+1}: {max(done,total_games)}/{total_games} games simulated")
                        log_area.code("\n".join(captured_lines) or "(no output)")

                        # Count per-game CSVs written for this season (simple, robust patterns)
                        try:
                            import re
                            yt_pat = re.compile(r"^\d{8}-.+-G.+\.csv$", re.IGNORECASE)
                            tm_pat = re.compile(r"^\d{8}-.+-G.+_trackman\.csv$", re.IGNORECASE)
                            all_csv = list(season_target.glob("*.csv"))
                            yt_count = sum(1 for p in all_csv if yt_pat.match(p.name) and not tm_pat.match(p.name))
                            tm_count = sum(1 for p in all_csv if tm_pat.match(p.name))
                        except Exception:
                            yt_count, tm_count = 0, 0

                        if code in (None, 0):
                            status.update(label=f"Season{base_idx + s} complete - {yt_count} YakkerTech, {tm_count} TrackMan CSVs", state="complete")
                        else:
                            status.update(label=f"Season{base_idx + s} exited with errors (code {code}) - {yt_count} YakkerTech, {tm_count} TrackMan CSVs", state="error")
                            st.error(f"Season {s+1} exited with code {code}. See log above.")

                    last_code = code if code is not None else 0
                    if last_code not in (0, None):
                        st.error(f"Season {s+1} exited with code {last_code}. See message above.")
                        if stop_on_error:
                            st.stop()
                        # else: continue to next season
                
                if last_code not in (0, None):
                    st.error(f"Simulation exited with code {last_code}. See message above.")
                else:
                    st.success("✅ Simulation completed successfully!")
                    st.session_state.step = max(st.session_state.step, 5)
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error running simulation: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)

# Step 5: View Results
if st.session_state.step >= 5:
    with st.expander(" Step 5: View Results", expanded=(st.session_state.step == 5)):
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-header">Step 5: View Your Results</div>', unsafe_allow_html=True)
        
        # Check if season output exists
            # Check if season output exists
    season_out_dir = scripts_dir / "season_out"
    if not season_out_dir.exists():
        st.warning("No simulation results found. Please run a simulation first (Step 4).")
    else:
        # Seasons present
        season_dirs = [p for p in season_out_dir.iterdir() if p.is_dir() and p.name.lower().startswith("season")]
        st.caption(f"Found {len(season_dirs)} season folder(s). Use the buttons below to build and download team and player reports.")
        if season_dirs:
            sel = st.selectbox("Select a season to download per-season reports:", [p.name for p in season_dirs])
            sel_dir = season_out_dir / sel
            c1, c2, c3 = st.columns(3)
            with c1:
                tsr = sel_dir / "team_season_report.csv"
                if tsr.exists():
                    st.download_button("Download Team Season Report", tsr.read_bytes(), file_name=tsr.name, mime="text/csv")
            with c2:
                pbat = sel_dir / "player_batting.csv"
                if pbat.exists():
                    st.download_button("Download Player Batting (Season)", pbat.read_bytes(), file_name=pbat.name, mime="text/csv")
            with c3:
                ppit = sel_dir / "player_pitching.csv"
                if ppit.exists():
                    st.download_button("Download Player Pitching (Season)", ppit.read_bytes(), file_name=ppit.name, mime="text/csv")
        # Team reports builder
        st.markdown('<div class="step-subheader">Team Season Reports</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Build Team Reports (per season + combined)"):
                try:
                    import importlib
                    tr = importlib.import_module("build_team_reports")
                    # Build all season reports and combined outputs under season_out
                    seasons = sorted([p for p in season_out_dir.iterdir() if p.is_dir() and p.name.lower().startswith("season")])
                    built_any = False
                    for sd in seasons:
                        outp = tr.build_for_season(sd)
                        if outp:
                            st.success(f"Built {outp}")
                            built_any = True
                    cb, imp = tr.build_combined(season_out_dir)
                    if cb:
                        st.success(f"Combined report: {cb}")
                    if imp:
                        st.success(f"Improvements report: {imp}")
                    if not (built_any or cb):
                        st.info("No season data found.")
                except Exception as e:
                    st.error(f"Failed to build team reports: {e}")
        with c2:
            # Offer downloads if files exist
            try:
                import io
                comb = season_out_dir / "team_season_combined.csv"
                impf = season_out_dir / "team_season_improvements.csv"
                if comb.exists():
                    b = (comb.read_text(encoding="utf-8")).encode("utf-8")
                    st.download_button("Download Combined", b, file_name=comb.name, mime="text/csv")
                if impf.exists():
                    b2 = (impf.read_text(encoding="utf-8")).encode("utf-8")
                    st.download_button("Download Improvements", b2, file_name=impf.name, mime="text/csv")
            except Exception:
                pass

        # Player reports builder
        st.markdown('<div class="step-subheader">Player Season Reports</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Build Player Reports (batting + pitching)"):
                try:
                    import importlib
                    pr = importlib.import_module("build_player_reports")
                    seasons = sorted([p for p in season_out_dir.iterdir() if p.is_dir() and p.name.lower().startswith("season")])
                    built_any = False
                    for sd in seasons:
                        b = pr.build_batting_for_season(sd)
                        if b:
                            st.success(f"Built batting: {b}")
                            built_any = True
                        p = pr.build_pitching_for_season(sd)
                        if p:
                            st.info(f"Found pitching: {p}")
                    cb_b, imp_b = pr.build_combined(season_out_dir, "batting")
                    cb_p, imp_p = pr.build_combined(season_out_dir, "pitching")
                    if cb_b: st.success(f"Combined batting: {cb_b}")
                    if imp_b: st.success(f"Improvements batting: {imp_b}")
                    if cb_p: st.success(f"Combined pitching: {cb_p}")
                    if imp_p: st.success(f"Improvements pitching: {imp_p}")
                    if not (built_any or cb_b or cb_p):
                        st.info("No season game CSVs found to build player reports.")
                except Exception as e:
                    st.error(f"Failed to build player reports: {e}")
        with c2:
            try:
                comb_b = season_out_dir / "player_batting_combined.csv"
                comb_p = season_out_dir / "player_pitching_combined.csv"
                imp_b = season_out_dir / "player_batting_improvements.csv"
                imp_p = season_out_dir / "player_pitching_improvements.csv"
                if comb_b.exists():
                    st.download_button("Download Batting Combined", comb_b.read_bytes(), file_name=comb_b.name, mime="text/csv")
                if comb_p.exists():
                    st.download_button("Download Pitching Combined", comb_p.read_bytes(), file_name=comb_p.name, mime="text/csv")
                if imp_b.exists():
                    st.download_button("Download Batting Improvements", imp_b.read_bytes(), file_name=imp_b.name, mime="text/csv")
                if imp_p.exists():
                    st.download_button("Download Pitching Improvements", imp_p.read_bytes(), file_name=imp_p.name, mime="text/csv")
            except Exception:
                pass

        # One-click: build & download all reports (ZIP)
        st.markdown('<div class="step-subheader">One-Click Download</div>', unsafe_allow_html=True)
        if st.button("Build & Download All Reports (ZIP)"):
            try:
                import importlib, zipfile
                from io import BytesIO
                # Build both team and player reports
                tr = importlib.import_module("build_team_reports")
                pr = importlib.import_module("build_player_reports")
                seasons = sorted([p for p in season_out_dir.iterdir() if p.is_dir() and p.name.lower().startswith("season")])
                for sd in seasons:
                    try: tr.build_for_season(sd)
                    except Exception: pass
                    try: pr.build_batting_for_season(sd)
                    except Exception: pass
                    try: pr.build_pitching_for_season(sd)
                    except Exception: pass
                # Combined
                try: tr.build_combined(season_out_dir)
                except Exception: pass
                try: pr.build_combined(season_out_dir, "batting"); pr.build_combined(season_out_dir, "pitching")
                except Exception: pass

                # Collect files
                buf = BytesIO()
                with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    # Per-season
                    for sd in seasons:
                        for fname in ("team_season_report.csv","player_batting.csv","player_pitching.csv"):
                            fp = sd / fname
                            if fp.exists():
                                zf.write(fp, arcname=f"{sd.name}/{fname}")
                    # Combined
                    for fname in (
                        "team_season_combined.csv","team_season_improvements.csv",
                        "player_batting_combined.csv","player_pitching_combined.csv",
                        "player_batting_improvements.csv","player_pitching_improvements.csv",
                    ):
                        fp = season_out_dir / fname
                        if fp.exists():
                            zf.write(fp, arcname=fname)
                zf_bytes = buf.getvalue()
                st.download_button("Download reports_all.zip", zf_bytes, file_name="reports_all.zip", mime="application/zip")
            except Exception as e:
                st.error(f"Failed to build or package reports: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# Reset button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset All Steps"):
    st.session_state.step = 1
    st.session_state.league_config = {}
    st.session_state.preview_data = None
    st.session_state.league_generated = False
    st.success("🔄 All steps reset!")
    st.rerun()

st.sidebar.markdown("---")
_steps_display = ["", "League Configuration", "Generate League", "Run Simulation", "View Results"]
try:
    idx = int(st.session_state.step)
except Exception:
    idx = 1
idx = max(1, min(idx, 4))
st.sidebar.markdown("**Current Step:** " + _steps_display[idx])
st.sidebar.markdown("**Progress:** " + "✅" * (st.session_state.step - 1) + "⭕" + "⚪" * (4 - st.session_state.step))



