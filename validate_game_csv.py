#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lightweight validator for one game CSV.

Checks:
- Column presence/order vs a template (YakkerTech or TrackMan)
- Core rate targets: BB%, K%, HR/FB, BABIP

Usage examples:
  python validate_game_csv.py --game_csv path/to/game.csv --format yakkertech \
      --template_csv templates/yt_template.csv

  python validate_game_csv.py --game_csv path/to/game_trackman.csv --format trackman \
      --bb_min 0.07 --bb_max 0.11 --k_min 0.18 --k_max 0.26
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd

# Local import (same directory)
try:
    from csv_format_manager import CSVFormatManager
except Exception:
    CSVFormatManager = None  # Fallback if not available


def _load_template_columns(fmt: str, template_csv: str | None) -> List[str]:
    fmt = (fmt or "yakkertech").lower()
    if CSVFormatManager is not None:
        mgr = CSVFormatManager()
        try:
            return mgr.get_template_columns(fmt, template_csv)
        except Exception:
            pass
    # Fallback: if a template file is provided, read columns from it
    if template_csv and Path(template_csv).exists():
        try:
            return list(pd.read_csv(template_csv, nrows=1, low_memory=False).columns)
        except Exception:
            pass
    # Last resort minimal defaults
    if fmt == "trackman":
        # Minimal subset that should exist
        return [
            "PitchNo","Date","Time","Pitcher","PitcherId","PitcherThrows","PitcherTeam",
            "Batter","BatterId","BatterSide","BatterTeam","Inning","Top/Bottom","Balls","Strikes",
            "PitchCall","PlayResult","TaggedHitType","OutsOnPlay","RunsScored"
        ]
    # YakkerTech minimal
    return [
        "GameID","Date","Inning","Top/Bottom","Outs","Balls","Strikes",
        "Pitcher","PitcherId","PitcherThrows","PitcherTeam",
        "Batter","BatterId","BatterSide","BatterTeam",
        "PitchCall","PlayResult","HitType","OutsOnPlay","RunsScored"
    ]


def _compare_columns(df_cols: List[str], template_cols: List[str]) -> Dict[str, Any]:
    df_set = set(df_cols)
    tmpl_set = set(template_cols)
    missing = [c for c in template_cols if c not in df_set]
    extra = [c for c in df_cols if c not in tmpl_set]

    # For columns in intersection, check relative order (first occurrence per template order)
    common = [c for c in template_cols if c in df_set]
    order_ok = True
    first_mismatch = None
    for i, c in enumerate(common):
        # expected index: position among common equals template order; actual index in df is increasing
        if i == 0:
            last_idx = -1
        idx = df_cols.index(c)
        if idx <= last_idx:
            order_ok = False
            first_mismatch = dict(column=c, prev_index=last_idx, index=idx)
            break
        last_idx = idx

    return {
        "missing": missing,
        "extra": extra,
        "order_ok": order_ok,
        "first_order_mismatch": first_mismatch,
    }


def _derive_counts(df: pd.DataFrame) -> Dict[str, int]:
    # Normalize key columns
    play = df.get("PlayResult")
    korbb = df.get("KorBB")
    htype = df.get("TaggedHitType") if "TaggedHitType" in df.columns else df.get("HitType")

    def has_val(series, val: str) -> pd.Series:
        if series is None:
            return pd.Series([False] * len(df))
        return series.astype(str).str.strip().str.lower().eq(val.lower())

    def contains_val(series, vals: List[str]) -> pd.Series:
        if series is None:
            return pd.Series([False] * len(df))
        s = series.astype(str).str.strip().str.lower()
        mask = pd.Series([False] * len(df))
        for v in vals:
            mask = mask | s.eq(v.lower())
        return mask

    # Terminal PAs: use presence of PlayResult OR KorBB in {Walk,Strikeout}
    is_walk = has_val(play, "Walk") | has_val(korbb, "Walk")
    is_k = contains_val(play, ["strikeoutswinging", "strikeoutlooking"]) | has_val(korbb, "Strikeout")

    hit_labels = ["single", "double", "triple", "homerun"]
    is_hit = contains_val(play, hit_labels)
    is_hr = has_val(play, "HomeRun")

    is_sf = has_val(play, "Sacrifice")
    is_hbp = has_val(play, "HitByPitch")

    # Outs on balls in play
    is_inplay_out = has_val(play, "Out") | has_val(play, "FielderChoice") | has_val(play, "Error")

    # AB definition: outs (incl ROE, FC) + hits, excludes walks, HBP, sacrifices
    is_ab = (is_inplay_out | is_hit) & (~is_walk) & (~is_hbp) & (~is_sf)

    # Plate appearances: AB + BB + HBP + SF
    pa = int(is_ab.sum() + is_walk.sum() + is_hbp.sum() + is_sf.sum())

    # Fly balls for HR/FB denominator: use TaggedHitType/HitType == FlyBall
    fb = 0
    if htype is not None:
        fb = int(htype.astype(str).str.strip().str.lower().eq("flyball").sum())

    return dict(
        PA=pa,
        BB=int(is_walk.sum()),
        K=int(is_k.sum()),
        AB=int(is_ab.sum()),
        H=int(is_hit.sum()),
        HR=int(is_hr.sum()),
        SF=int(is_sf.sum()),
        FB=fb,
    )


def _compute_rates(cnt: Dict[str, int]) -> Dict[str, float]:
    pa = max(1, cnt.get("PA", 0))
    ab = cnt.get("AB", 0)
    bb = cnt.get("BB", 0)
    k = cnt.get("K", 0)
    h = cnt.get("H", 0)
    hr = cnt.get("HR", 0)
    sf = cnt.get("SF", 0)
    fb = cnt.get("FB", 0)

    bb_rate = bb / pa
    k_rate = k / pa
    hr_fb = (hr / fb) if fb > 0 else float("nan")
    # BABIP = (H - HR) / (AB - K - HR + SF)
    denom = (ab - k - hr + sf)
    babip = ((h - hr) / denom) if denom > 0 else float("nan")

    return dict(BB_pct=bb_rate, K_pct=k_rate, HR_per_FB=hr_fb, BABIP=babip)


def _check_targets(rates: Dict[str, float], bb_min: float, bb_max: float,
                   k_min: float, k_max: float, hrfb_min: float, hrfb_max: float,
                   babip_min: float, babip_max: float) -> Dict[str, Any]:
    out = {}
    def within(val, lo, hi):
        try:
            return (val >= lo) and (val <= hi)
        except Exception:
            return False
    out["BB_pct_ok"] = within(rates.get("BB_pct", float("nan")), bb_min, bb_max)
    out["K_pct_ok"] = within(rates.get("K_pct", float("nan")), k_min, k_max)
    out["HR_per_FB_ok"] = within(rates.get("HR_per_FB", float("nan")), hrfb_min, hrfb_max)
    out["BABIP_ok"] = within(rates.get("BABIP", float("nan")), babip_min, babip_max)
    return out


def main():
    ap = argparse.ArgumentParser("Validate one game CSV vs template and target rates")
    ap.add_argument("--game_csv", required=True, help="Path to the per-game CSV to validate")
    ap.add_argument("--format", choices=["yakkertech", "trackman"], default="yakkertech")
    ap.add_argument("--template_csv", default="", help="Optional template CSV to enforce column order")
    ap.add_argument("--report_json", default="", help="Optional path to write a JSON report")

    # Target windows (defaults are reasonable for MiLB/MLB-like environments)
    ap.add_argument("--bb_min", type=float, default=0.070)
    ap.add_argument("--bb_max", type=float, default=0.110)
    ap.add_argument("--k_min", type=float, default=0.180)
    ap.add_argument("--k_max", type=float, default=0.260)
    ap.add_argument("--hrfb_min", type=float, default=0.08)
    ap.add_argument("--hrfb_max", type=float, default=0.18)
    ap.add_argument("--babip_min", type=float, default=0.270)
    ap.add_argument("--babip_max", type=float, default=0.320)

    args = ap.parse_args()

    game_path = Path(args.game_csv)
    if not game_path.exists():
        raise SystemExit(f"Game CSV not found: {game_path}")

    df = pd.read_csv(game_path)

    # Columns check
    template_cols = _load_template_columns(args.format, args.template_csv or None)
    col_report = _compare_columns(list(df.columns), template_cols)

    # Rate checks
    counts = _derive_counts(df)
    rates = _compute_rates(counts)
    targets_ok = _check_targets(
        rates,
        args.bb_min, args.bb_max,
        args.k_min, args.k_max,
        args.hrfb_min, args.hrfb_max,
        args.babip_min, args.babip_max,
    )

    report = {
        "file": str(game_path),
        "format": args.format,
        "columns": col_report,
        "counts": counts,
        "rates": rates,
        "targets_ok": targets_ok,
    }

    # Console summary
    print("Validation Summary:\n")
    print(f"- File: {report['file']}")
    print(f"- Format: {report['format']}")

    missing = report["columns"]["missing"]
    extra = report["columns"]["extra"]
    order_ok = report["columns"]["order_ok"]
    print("- Columns: ")
    print(f"  - Missing ({len(missing)}): {missing}")
    print(f"  - Extra   ({len(extra)}): {extra}")
    print(f"  - Order OK: {order_ok}")

    print("- Counts:")
    print(f"  PA={counts['PA']} AB={counts['AB']} BB={counts['BB']} K={counts['K']} H={counts['H']} HR={counts['HR']} SF={counts['SF']} FB={counts['FB']}")

    print("- Rates:")
    bbp = rates['BB_pct']
    kp = rates['K_pct']
    hrfb = rates['HR_per_FB']
    babip = rates['BABIP']
    print(f"  BB%={bbp:.3f}  K%={kp:.3f}  HR/FB={(hrfb if pd.notna(hrfb) else 'nan')}  BABIP={(babip if pd.notna(babip) else 'nan')}")

    print("- Target checks:")
    for k, ok in report["targets_ok"].items():
        print(f"  {k}: {'OK' if ok else 'OUT'}")

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON report: {args.report_json}")


if __name__ == "__main__":
    main()

