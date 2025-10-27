# attr_location_sampler.py
# Attribute-driven plate location sampler (drop-in for ModelCA)
# Produces (PlateLocSide, PlateLocHeight) in feet using roster.csv traits.
import math
import numpy as np
from typing import Optional, Dict, Any, Tuple

# Add a global variable to store season-specific randomization
_SEASON_RANDOMIZATION = {}

def set_season_randomization(season_id: str, seed: Optional[int] = None):
    """Set season-specific randomization parameters to ensure each season has unique patterns."""
    global _SEASON_RANDOMIZATION
    if seed is not None:
        np.random.seed(seed)
    
    # Generate season-specific randomization factors
    _SEASON_RANDOMIZATION[season_id] = {
        'horizontal_bias': np.random.normal(0, 0.03),  # Small horizontal shift
        'vertical_bias': np.random.normal(0, 0.02),    # Small vertical shift
        'variance_multiplier': np.random.uniform(0.95, 1.05),  # Variance adjustment
        'edge_concentration': np.random.uniform(0.9, 1.1),     # Edge concentration variation
        'random_seed': seed if seed is not None else np.random.randint(0, 10000)
    }
    
    # Reset random seed to ensure consistency
    np.random.seed(None)

def get_season_randomization(season_id: str) -> dict:
    """Get season-specific randomization parameters."""
    global _SEASON_RANDOMIZATION
    if season_id not in _SEASON_RANDOMIZATION:
        # Generate default randomization if not set
        set_season_randomization(season_id, hash(season_id) % 10000)
    return _SEASON_RANDOMIZATION[season_id]

PLATE_X_MIN, PLATE_X_MAX = -0.95, 0.95
PLATE_Z_MIN, PLATE_Z_MAX =  1.00, 4.00

# Base "aim pockets" vs RHB; we flip x by platoon later
# Enhanced values for better edge concentration and realism
_BASES_ATTR = {
    "FourSeam":  (+0.26, 2.85, 0.22, 0.22),  # Slightly higher challenge, keeps arm-side option
    "Sinker":    (+0.18, 2.25, 0.25, 0.30),  # Still low but not buried
    "TwoSeam":   (+0.18, 2.28, 0.25, 0.30),  # Similar to sinker with tiny elevation
    "Slider":    (+0.40, 2.25, 0.30, 0.30),  # Allow more elevated glove-side chase
    "Curveball": (+0.12, 2.25, 0.30, 0.35),  # Higher break point for backdoor mixes
    "Changeup":  (-0.05, 2.35, 0.25, 0.25),  # More mid-zone feel
    "Cutter":    (+0.30, 2.70, 0.24, 0.24),  # Elevated glove-side cutter
    "Splitter":  (-0.04, 2.05, 0.28, 0.32),  # Remains low but with some mid reach
    "Forkball":  (-0.04, 2.05, 0.28, 0.32),  # Similar to splitter
    "Knuckleball": (0.00, 2.55, 0.40, 0.40), # Unpredictable movement
}


_EDGE_BIAS_FALLBACK_CFG: Dict[str, Any] = {
    'default': {
        'R_vs_R': 0.0,
        'R_vs_L': 0.0,
        'L_vs_R': 0.0,
        'L_vs_L': 0.0,
        '*': 0.0,
    },
    'Slider': {
        'R_vs_R': 0.05,
        'R_vs_L': 0.05,
        'L_vs_R': 0.05,
        'L_vs_L': 0.05,
        '*': 0.05,
    },
    'Curveball': {
        'R_vs_R': 0.05,
        'R_vs_L': 0.05,
        'L_vs_R': 0.05,
        'L_vs_L': 0.05,
        '*': 0.05,
    },
    'Sinker': {
        'R_vs_R': 0.03,
        'R_vs_L': 0.03,
        'L_vs_R': 0.03,
        'L_vs_L': 0.03,
        '*': 0.03,
    },
    'TwoSeam': {
        'R_vs_R': 0.03,
        'R_vs_L': 0.03,
        'L_vs_R': 0.03,
        'L_vs_L': 0.03,
        '*': 0.03,
    },
    'Cutter': {
        'R_vs_R': 0.04,
        'R_vs_L': 0.04,
        'L_vs_R': 0.04,
        'L_vs_L': 0.04,
        '*': 0.04,
    },
    'Splitter': {
        'R_vs_R': {'magnitude': 0.06, 'vertical_shift': -0.05},
        'R_vs_L': {'magnitude': 0.06, 'vertical_shift': -0.05},
        'L_vs_R': {'magnitude': 0.06, 'vertical_shift': -0.05},
        'L_vs_L': {'magnitude': 0.06, 'vertical_shift': -0.05},
        '*': {'magnitude': 0.06, 'vertical_shift': -0.05},
    },
    'Forkball': {
        'R_vs_R': {'magnitude': 0.06, 'vertical_shift': -0.05},
        'R_vs_L': {'magnitude': 0.06, 'vertical_shift': -0.05},
        'L_vs_R': {'magnitude': 0.06, 'vertical_shift': -0.05},
        'L_vs_L': {'magnitude': 0.06, 'vertical_shift': -0.05},
        '*': {'magnitude': 0.06, 'vertical_shift': -0.05},
    },
}

def _coerce_edge_bias_value(value: Any) -> Tuple[float, float] | None:
    try:
        if isinstance(value, dict):
            mag = value.get('magnitude', value.get('mag', value.get('value')))
            if mag is None:
                return None
            vertical = float(value.get('vertical_shift', value.get('vz_shift', value.get('z_shift', 0.0))))
            return abs(float(mag)), vertical
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            mag = float(s)
            return abs(mag), 0.0
        if isinstance(value, (int, float)):
            return abs(float(value)), 0.0
    except (TypeError, ValueError):
        return None
    return None

def _lookup_edge_bias_from_mapping(mapping: Any, pair_key: str) -> Tuple[float, float] | None:
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        return _coerce_edge_bias_value(mapping)
    variants = {
        pair_key,
        pair_key.replace('_', '-'),
        pair_key.replace('_', '/'),
        pair_key.replace('_', '|'),
        pair_key.lower(),
        pair_key.upper(),
    }
    for key in variants:
        if key in mapping:
            result = _coerce_edge_bias_value(mapping[key])
            if result:
                return result
    left, _, right = pair_key.partition('_vs_')
    if left and right:
        left = left.strip()
        right = right.strip()
        same_side = left[:1].upper() == right[:1].upper()
        if same_side:
            alias_candidates = ('same', 'same_hand', 'samehand', 'same-handed', 'same_side', 'sameSide')
        else:
            alias_candidates = ('opposite', 'opp', 'opposite_hand', 'opposite-hand', 'opp_hand', 'oppSide', 'platoon')
        for alias in alias_candidates:
            if alias in mapping:
                result = _coerce_edge_bias_value(mapping[alias])
                if result:
                    return result
    for fallback_key in ('*', 'default', 'DEFAULT'):
        if fallback_key in mapping:
            result = _coerce_edge_bias_value(mapping[fallback_key])
            if result:
                return result
    return None

def _resolve_edge_bias(edge_bias_cfg: Dict[str, Any] | None, pitch_type: str, pair_key: str) -> Tuple[float, float] | None:
    if not isinstance(edge_bias_cfg, dict):
        return None
    if 'layers' in edge_bias_cfg and isinstance(edge_bias_cfg['layers'], (list, tuple)):
        for layer in edge_bias_cfg['layers']:
            result = _resolve_edge_bias(layer, pitch_type, pair_key)
            if result:
                return result
    layered_keys = ('overrides', 'priors', 'fallback')
    for layered_key in layered_keys:
        sub_cfg = edge_bias_cfg.get(layered_key)
        if isinstance(sub_cfg, dict) and sub_cfg is not edge_bias_cfg:
            result = _resolve_edge_bias(sub_cfg, pitch_type, pair_key)
            if result:
                return result
    candidates = [pitch_type, pitch_type.lower(), pitch_type.upper()]
    for key in candidates:
        if key in edge_bias_cfg:
            result = _lookup_edge_bias_from_mapping(edge_bias_cfg[key], pair_key)
            if result:
                return result
    groups = edge_bias_cfg.get('groups') if isinstance(edge_bias_cfg.get('groups'), dict) else None
    if groups:
        for group_cfg in groups.values():
            try:
                pitches = group_cfg.get('pitches')
            except AttributeError:
                pitches = None
            if isinstance(pitches, (list, tuple, set)) and pitch_type in pitches:
                result = _lookup_edge_bias_from_mapping(group_cfg.get('values'), pair_key)
                if result:
                    return result
    for key in ('*', 'default', 'DEFAULT'):
        if key in edge_bias_cfg:
            result = _lookup_edge_bias_from_mapping(edge_bias_cfg[key], pair_key)
            if result:
                return result
    return None


def _truncate_mvn_attr(mean, cov, tries=12, rng=None):
    if rng is None: 
        rng = np.random
    for _ in range(tries):
        x, z = rng.multivariate_normal(mean, cov)
        if PLATE_X_MIN <= x <= PLATE_X_MAX and PLATE_Z_MIN <= z <= PLATE_Z_MAX:
            return float(round(x, 3)), float(round(z, 3))
    # smooth fallback (avoid edge spikes) - use proper plate boundaries
    x, z = rng.multivariate_normal(mean, cov)
    x_center = (PLATE_X_MAX + PLATE_X_MIN) / 2
    x_range = (PLATE_X_MAX - PLATE_X_MIN) / 2
    z_center = (PLATE_Z_MAX + PLATE_Z_MIN) / 2
    z_range = (PLATE_Z_MAX - PLATE_Z_MIN) / 2
    x = x_center + x_range * math.tanh((x - x_center) / x_range)
    z = z_center + z_range * math.tanh((z - z_center) / z_range)
    return float(round(x, 3)), float(round(z, 3))

def sample_loc_from_roster_attrs(
    pitch_type: str,
    roster_row: dict,
    batter_side: str,           # "L" or "R"
    count_bucket: str = "even", # "behind"/"even"/"ahead"
    game_context: Optional[dict] = None,  # New parameter for game context
    pitcher_fatigue: float = 0.0,  # New parameter: 0.0 (fresh) to 1.0 (exhausted)
    season_id: str = "default",   # New parameter for season-specific variation
    rng=None,
    *, edge_bias_cfg: Dict[str, Any] | None = None
):
    """Return (PlateLocSide, PlateLocHeight) in feet based on roster traits.
    Expected roster_row keys:
      - Throws: 'Right'/'Left'
      - ArmSlotBucket: e.g., 'Overhand','High34','Sidearm'...
      - CommandTier: float ~ 0.8..1.3 (1.0 baseline)
      - AvgFBVelo: mph
      - Extension_ft: feet
      - RelHeight_ft: feet
      - RelSide_ft: feet (release side offset from rubber center)
    Optional config:
      - edge_bias_cfg can supply priors/overrides keyed by pitch type and handedness pairs (R_vs_R, R_vs_L, L_vs_R, L_vs_L) with magnitudes and optional vertical_shift values.
    """
    if rng is None: 
        rng = np.random
    
    if game_context is None:
        game_context = {}

    # Get season-specific randomization
    season_randomization = get_season_randomization(season_id)
    
    # 0) Base pocket (RHP vs RHB convention; platoon flip later)
    mu_x, mu_z, sx, sz = _BASES_ATTR.get(pitch_type, (+0.25, 2.60, 0.25, 0.25))

    # 1) Platoon handling: +x is "away to RHB"
    PTH = (roster_row.get("Throws") or "Right")[0].upper()
    BATS = (batter_side or "R")[0].upper()
    same_hand = (PTH == BATS)
    
    # Simplified and corrected platoon handling logic
    if same_hand:
        # Same-handed: pitch away from batter
        # For RHP vs RHB: away is left (-x)
        # For LHP vs LHB: away is right (+x)
        mu_x = -abs(mu_x) if PTH == "R" else +abs(mu_x)
    else:
        # Opposite-handed: pitch into batter
        # For RHP vs LHB: into batter is right (+x)
        # For LHP vs RHB: into batter is left (-x)
        mu_x = +abs(mu_x) if PTH == "R" else -abs(mu_x)

    # 2) Arm slot offsets
    slot = (roster_row.get("ArmSlotBucket") or "").lower()
    if "over" in slot:
        mu_z += 0.12
    elif "high34" in slot or "3/4" in slot or "three" in slot:
        mu_z += 0.05; mu_x += (0.04 if PTH == "R" else -0.04)
    elif "side" in slot:
        mu_z -= 0.10; mu_x += (0.10 if PTH == "R" else -0.10); sx *= 1.12

    # 3) Release geometry nudges (small, safe)
    try:
        rel_h = float(roster_row.get("RelHeight_ft") or 5.5)
    except:
        rel_h = 5.5
    try:
        rel_x = float(roster_row.get("RelSide_ft") or 0.0)
    except:
        rel_x = 0.0
    mu_z += 0.06 * math.tanh((rel_h - 5.5) / 0.6)
    mu_x += 0.10 * math.tanh(rel_x / 0.7) * (1 if PTH == "R" else -1)

    # 4) Velo influence (mainly 4S)
    if "four" in pitch_type.lower() or "4" in pitch_type:
        try:
            v = float(roster_row.get("AvgFBVelo") or 90.0)
        except:
            v = 90.0
        mu_z += 0.02 * (v - 90.0)
        sx *= max(0.90, 1.0 - 0.01*(v - 90.0))

    # 5) Count bucket behavior with enhanced edge concentration
    if count_bucket == "ahead":
        # When ahead in count, pitchers aim more carefully but with expanded zone awareness
        mu_z -= 0.03  # Slight drop
        sx *= 1.08    # Slight expansion horizontally
        sz *= 1.08    # Slight expansion vertically
    elif count_bucket == "behind":
        # When behind in count, pitchers try to be more precise but may miss more
        mu_z += 0.02  # Slight rise
        sx *= 0.96    # Tighter horizontally
        sz *= 0.96    # Tighter vertically

    # 6) Edge concentration driven by configurable priors
    pair_key = f"{PTH}_vs_{BATS}"
    edge_bias_factor = 0.0
    edge_bias_vshift = 0.0

    resolved_edge_bias = None
    if isinstance(edge_bias_cfg, dict):
        resolved_edge_bias = _resolve_edge_bias(edge_bias_cfg, pitch_type, pair_key)
    if resolved_edge_bias is None:
        resolved_edge_bias = _resolve_edge_bias(_EDGE_BIAS_FALLBACK_CFG, pitch_type, pair_key)

    if resolved_edge_bias:
        edge_bias_factor, edge_bias_vshift = resolved_edge_bias
        if edge_bias_vshift:
            mu_z += edge_bias_vshift

        if edge_bias_factor > 0:
            if PTH == "R":
                if BATS == "R":
                    mu_x += edge_bias_factor
                else:
                    mu_x -= edge_bias_factor
            else:
                if BATS == "L":
                    mu_x -= edge_bias_factor
                else:
                    mu_x += edge_bias_factor


    # 7) Game Context Effects
    score_diff = game_context.get("score_diff", 0)  # Positive = pitcher's team ahead
    inning = game_context.get("inning", 1)
    is_late_game = game_context.get("is_late_game", False)
    pressure_level = game_context.get("pressure_level", 0.0)  # 0.0 to 1.0
    
    # Pressure effects (moderate impact on control)
    if pressure_level > 0:
        # More subtle control changes
        pressure_factor = 1.0 + (0.20 * pressure_level)  # Up to 20% increase in variance
        sx *= pressure_factor
        sz *= pressure_factor
        
        # In high pressure, pitchers may aim more carefully to extremes
        if pressure_level > 0.7:
            # Subtle adjustment toward edges rather than drastic push
            if mu_x > 0.1:  # Already aiming right
                mu_x += 0.04  # Slight push to right edge
            elif mu_x < -0.1:  # Already aiming left
                mu_x -= 0.04  # Slight push to left edge
    
    # Score differential effects (gentle adjustments)
    if abs(score_diff) >= 3:  # Significant lead or deficit
        if score_diff > 0:  # Pitcher's team ahead (protect lead)
            # Slightly tighter locations, more conservative
            control_factor = 0.95
            sx *= control_factor
            sz *= control_factor
        else:  # Pitcher's team behind (need to speed up)
            # Slightly looser locations, more aggressive
            control_factor = 1.05
            sx *= control_factor
            sz *= control_factor
            # Try to get more swings - slight vertical adjustment
            mu_z += 0.03

    # Late game effects (subtle increased focus)
    if is_late_game and inning >= 7:
        # Very gentle tightening
        late_game_factor = 0.96
        sx *= late_game_factor
        sz *= late_game_factor

    # 8) Fatigue Effects (more realistic progression)
    if pitcher_fatigue > 0:
        # Fatigue increases variance, but in a more realistic way
        # Using a quadratic relationship for more natural progression
        fatigue_variance_factor = 1.0 + (0.30 * pitcher_fatigue**1.5)  # Increased effect
        sx *= fatigue_variance_factor
        sz *= fatigue_variance_factor
        
        # Fatigue affects vertical location (arm tiredness)
        mu_z -= 0.05 * pitcher_fatigue  # Increased drop effect
        
        # Extreme fatigue effects (more realistic threshold)
        if pitcher_fatigue > 0.7:
            # Additional small effects for very tired pitchers
            sx *= 1.15  # Increased effect
            sz *= 1.15

    # 9) Season-specific randomization - NEW FEATURE
    # Apply season-specific variations to make each season unique
    mu_x += season_randomization['horizontal_bias']
    mu_z += season_randomization['vertical_bias']
    sx *= season_randomization['variance_multiplier']
    sz *= season_randomization['variance_multiplier']
    edge_bias_factor *= season_randomization['edge_concentration']

    # 10) Command & extension → variance (gentle, floored)
    try:
        cmd = float(roster_row.get("CommandTier") or 1.0)
    except:
        cmd = 1.0
    eff = 0.5 + 0.5 * max(0.85, min(1.30, cmd))
    sx_eff, sz_eff = sx / eff, sz / eff

    try:
        ext = float(roster_row.get("Extension_ft") or 5.5)
    except:
        ext = 5.5
    sz_eff *= max(0.85, min(1.10, 1.0 - 0.02*(ext - 5.5)))

    cov = np.diag([sx_eff**2, sz_eff**2])
    return _truncate_mvn_attr((mu_x, mu_z), cov, rng=rng)