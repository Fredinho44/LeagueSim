# simulate_seasons_roster_style.py
import sys, importlib
from pathlib import Path

def _ensure_on_path(p: Path):
    p = p.resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

def main():
    root = Path(__file__).parent
    _ensure_on_path(root)                  # make sure our folder is first
    importlib.invalidate_caches()

    # Drop stale copies so fresh code is loaded
    for m in ("sim_utils", "game_sim", "season_runner"):
        if m in sys.modules:
            del sys.modules[m]

    # Re-import in dependency order, then run
    import sim_utils   # noqa: F401  (ensures constants are defined)
    import game_sim    # noqa: F401
    import season_runner as sr
    sr.main()

def build_team_tier_map(roster_rows):
    """Return {TeamID: Tier} using the roster. Defaults to 'Unknown'."""
    m = {}
    for r in roster_rows:
        tid = str(r.get("TeamID") or "").strip()
        if not tid:
            continue
        tier = (r.get("Tier") or "").strip() or "Unknown"
        # first seen wins; all players on a team should share Tier
        m.setdefault(tid, tier)
    return m



if __name__ == "__main__":
    main()
