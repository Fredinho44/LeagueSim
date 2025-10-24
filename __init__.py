"""
ModelCA League Simulator
========================

A baseball/softball league simulator that generates synthetic player data
and runs detailed pitch-by-pitch simulations.
"""

__version__ = "1.0.0"
__author__ = "Alfredo Caraballo"
__email__ = "alfredo.caraballo@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 Alfredo Caraballo"

# Import key modules for easier access
from .make_league import build_organizations, build_teams, build_rosters
from .game_sim import simulate_one_game
from .simulate_seasons_roster_style import main as simulate_seasons

__all__ = [
    "build_organizations",
    "build_teams", 
    "build_rosters",
    "simulate_one_game",
    "simulate_seasons"
]