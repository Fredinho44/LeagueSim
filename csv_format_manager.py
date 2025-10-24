#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV Format Manager for ModelCA League Simulator
==============================================

Handles dual-format CSV output:
- YakkerTech format (existing)
- Trackman format (new)

Provides column mapping, data transformation, and format-specific output generation.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np


class CSVFormatManager:
    """Manages CSV output in multiple formats (YakkerTech and Trackman)"""
    
    # Define the Trackman column structure based on the template
    TRACKMAN_COLUMNS = [
        "PitchNo", "Date", "Time", "PAofInning", "PitchofPA", "Pitcher", "PitcherId", 
        "PitcherThrows", "PitcherTeam", "Batter", "BatterId", "BatterSide", "BatterTeam",
        "PitcherSet", "Inning", "Top/Bottom", "Outs", "Balls", "Strikes", "TaggedPitchType",
        "AutoPitchType", "PitchCall", "KorBB", "TaggedHitType", "PlayResult", "OutsOnPlay", 
        "RunsScored", "Notes", "RelSpeed", "VertRelAngle", "HorzRelAngle", "SpinRate", 
        "SpinAxis", "Tilt", "RelHeight", "RelSide", "Extension", "VertBreak", "InducedVertBreak", 
        "HorzBreak", "PlateLocHeight", "PlateLocSide", "ZoneSpeed", "VertApprAngle", "HorzApprAngle", 
        "ZoneTime", "ExitSpeed", "Angle", "Direction", "HitSpinRate", "PositionAt110X", 
        "PositionAt110Y", "PositionAt110Z", "Distance", "LastTrackedDistance", "Bearing", 
        "HangTime", "pfxx", "pfxz", "x0", "y0", "z0", "vx0", "vy0", "vz0", "ax0", "ay0", "az0",
        "HomeTeam", "AwayTeam", "Stadium", "Level", "League", "GameID", "PitchUID", 
        "EffectiveVelo", "MaxHeight", "MeasuredDuration", "SpeedDrop", "PitchLastMeasuredX", 
        "PitchLastMeasuredY", "PitchLastMeasuredZ", "ContactPositionX", "ContactPositionY", 
        "ContactPositionZ", "GameUID", "UTCDate", "UTCTime", "LocalDateTime", "UTCDateTime", 
        "AutoHitType", "System", "HomeTeamForeignID", "AwayTeamForeignID", "GameForeignID", 
        "Catcher", "CatcherId", "CatcherThrows", "CatcherTeam", "PlayID"
        # Note: Truncated for brevity - the template has many more trajectory columns
    ]
    
    def __init__(self, verbose: bool = False):
        """Initialize format manager
        verbose: if True, emit debug prints; default False to keep logs quiet.
        """
        self.supported_formats = ["yakkertech", "trackman"]
        self.verbose = bool(verbose)

        def _dbg(msg: str):
            if self.verbose:
                try:
                    print(msg)
                except Exception:
                    pass
        self._dbg = _dbg
        
        # Column mappings from YakkerTech (source) to Trackman (target)
        self.column_mappings = {
            # Basic game info
            "GameID": "GameID",
            "Date": "Date", 
            "HomeTeam": "HomeTeam",
            "AwayTeam": "AwayTeam",
            "Inning": "Inning",
            "Top/Bottom": "Top/Bottom",
            "Outs": "Outs",
            "Balls": "Balls", 
            "Strikes": "Strikes",
            "PAofInning": "PAofInning",
            "PitchofPA": "PitchofPA",
            
            # Pitcher info
            "PitcherName": "Pitcher",
            "Pitcher": "Pitcher",  # For game simulation output
            "PitcherId": "PitcherId",
            "Throws": "PitcherThrows",
            "PitcherThrows": "PitcherThrows",  # For game simulation output
            "PitcherTeam": "PitcherTeam",
            
            # Batter info  
            "BatterName": "Batter",
            "Batter": "Batter",  # For game simulation output
            "BatterId": "BatterId",
            "BatterSide": "BatterSide",
            "BatterTeam": "BatterTeam",
            
            # Pitch details
            "PitchType": "TaggedPitchType",
            "TaggedPitchType": "TaggedPitchType",  # For game simulation output
            "PitchCall": "PitchCall",
            "PlayResult": "PlayResult",
            "KorBB": "KorBB",
            "HitType": "TaggedHitType",
            # Note: TaggedHitType -> TaggedHitType mapping removed to prevent overwriting
            "OutsOnPlay": "OutsOnPlay",
            "RunsScored": "RunsScored",
            
            # Physics data
            "RelSpeed": "RelSpeed",
            "SpinRate": "SpinRate", 
            "SpinAxis": "SpinAxis",
            "RelHeight": "RelHeight",
            "RelSide": "RelSide",
            "Extension": "Extension",
            "VertBreak": "VertBreak",
            "HorzBreak": "HorzBreak",
            "PlateLocHeight": "PlateLocHeight",
            "PlateLocSide": "PlateLocSide",
            "ExitSpeed": "ExitSpeed",
            "Angle": "Angle",
            "Direction": "Direction",
            "Distance": "Distance",
            
            # Pass-through mappings for fields our simulator already outputs
            "VertRelAngle": "VertRelAngle",
            "HorzRelAngle": "HorzRelAngle",
            "VertApprAngle": "VertApprAngle",
            "HorzApprAngle": "HorzApprAngle",
            "ZoneSpeed": "ZoneSpeed",
            "ZoneTime": "ZoneTime",
            "InducedVertBreak": "InducedVertBreak",
            "VertBreak": "VertBreak",
            "HorzBreak": "HorzBreak",
            "Bearing": "Bearing",
            "HangTime": "HangTime",
            "EffectiveVelo": "EffectiveVelo",
            "MaxHeight": "MaxHeight",
            "MeasuredDuration": "MeasuredDuration",
            "SpeedDrop": "SpeedDrop",
            "PitchLastMeasuredX": "PitchLastMeasuredX",
            "PitchLastMeasuredY": "PitchLastMeasuredY",
            "PitchLastMeasuredZ": "PitchLastMeasuredZ",
            "ContactPositionX": "ContactPositionX",
            "ContactPositionY": "ContactPositionY",
            "ContactPositionZ": "ContactPositionZ",
            # Catcher info (from game_sim via roster)
            "Catcher": "Catcher",
            "CatcherId": "CatcherId",
            "CatcherThrows": "CatcherThrows",
            "CatcherTeam": "CatcherTeam",
            "UTCDate": "UTCDate",
            "UTCTime": "UTCTime",
            "LocalDateTime": "LocalDateTime",
            "UTCDateTime": "UTCDateTime",
            "System": "System",
            "GameUID": "GameUID",
            "PitchUID": "PitchUID",
        }
    
    def get_template_columns(self, format_type: str, template_path: Optional[str] = None) -> List[str]:
        """Get column structure for specified format"""
        format_type = format_type.lower()
        
        if format_type == "trackman":
            if template_path and Path(template_path).exists():
                try:
                    template_df = pd.read_csv(template_path, nrows=1, low_memory=False)
                    return list(template_df.columns)
                except Exception:
                    return self.TRACKMAN_COLUMNS.copy()
            return self.TRACKMAN_COLUMNS.copy()
        elif format_type == "yakkertech":
            if template_path and Path(template_path).exists():
                # Use provided YakkerTech template
                template_df = pd.read_csv(template_path, nrows=1, low_memory=False)
                columns = list(template_df.columns)
            else:
                # Default YakkerTech columns (minimal set)
                columns = [
                    "GameID", "Date", "Inning", "Top/Bottom", "Outs", "Balls", "Strikes",
                    "PitcherName", "PitcherId", "Throws", "PitcherTeam", 
                    "BatterName", "BatterId", "BatterSide", "BatterTeam",
                    "PitchType", "PitchCall", "PlayResult", "KorBB", "HitType", 
                    "OutsOnPlay", "RunsScored"
                ]
            
            # Ensure critical columns are present
            required_cols = ["Throws", "Top/Bottom", "BatterSide", "PlayResult", 
                           "KorBB", "RunsScored", "OutsOnPlay", "HitType"]
            for col in required_cols:
                if col not in columns:
                    columns.append(col)
            
            return columns
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def convert_yakkertech_to_trackman(self, yakkertech_df: pd.DataFrame) -> pd.DataFrame:
        """Convert YakkerTech format DataFrame to Trackman format"""
        
        # Handle special conversions and calculated fields
        num_rows = len(yakkertech_df)
        
        # Start with empty Trackman structure
        trackman_data = {}
        
        # Initialize all Trackman columns with appropriate defaults for the number of rows
        for col in self.TRACKMAN_COLUMNS:
            if col in ["Date", "Time", "LocalDateTime", "UTCDateTime", "UTCDate", "UTCTime"]:
                trackman_data[col] = [""] * num_rows
            elif col in ["PitchNo", "Inning", "Outs", "Balls", "Strikes", 
                        "OutsOnPlay", "RunsScored"]:
                trackman_data[col] = [0] * num_rows
            elif col in ["RelSpeed", "VertRelAngle", "HorzRelAngle", "SpinRate", "SpinAxis", 
                        "RelHeight", "RelSide", "Extension", "VertBreak", "InducedVertBreak",
                        "HorzBreak", "PlateLocHeight", "PlateLocSide", "ExitSpeed", "Angle",
                        "Direction", "Distance"]:
                trackman_data[col] = [np.nan] * num_rows
            elif col in ["PitchCall", "KorBB", "TaggedHitType", "PlayResult"]:
                trackman_data[col] = ["Undefined"] * num_rows
            else:
                # Create a new list for each column to avoid shared references
                trackman_data[col] = [""] * num_rows
        
        # Map available YakkerTech columns to Trackman
        for yak_col, track_col in self.column_mappings.items():
            if yak_col in yakkertech_df.columns and track_col in trackman_data:
                # Convert values properly: preserve actual values, convert NaN/None to empty string
                source_values = yakkertech_df[yak_col].copy()
                converted_values = []
                for val in source_values:
                    if pd.isna(val) or val is None:
                        converted_values.append("")
                    else:
                        converted_values.append(str(val))
                
                # Special handling for PitchCall: Convert Foul -> FoulBallNotFieldable
                if track_col == "PitchCall":
                    trackman_pitch_calls = []
                    for val in converted_values:
                        if val == "Foul":
                            trackman_pitch_calls.append("FoulBallNotFieldable")
                        elif val == "":
                            trackman_pitch_calls.append("Undefined")
                        else:
                            trackman_pitch_calls.append(val)
                    trackman_data[track_col] = trackman_pitch_calls
                # Special handling for columns that should have "Undefined" instead of empty string
                elif track_col in ["KorBB", "TaggedHitType", "PlayResult"]:
                    defined_values = []
                    for val in converted_values:
                        if val == "":
                            defined_values.append("Undefined")
                        else:
                            defined_values.append(val)
                    trackman_data[track_col] = defined_values
                else:
                    trackman_data[track_col] = converted_values

        # Sanitize Notes: remove SB2/SB3 tokens and clean separators
        try:
            notes = trackman_data.get("Notes")
            if isinstance(notes, list):
                cleaned = []
                for n in notes:
                    s = "" if n is None or (isinstance(n, float) and pd.isna(n)) else str(n)
                    parts = [t.strip() for t in s.split(";")]
                    parts = [t for t in parts if t and t not in ("SB2","SB3")]
                    cleaned.append(";".join(parts))
                trackman_data["Notes"] = cleaned
        except Exception:
            pass
        
        # CRITICAL FIX: Store TaggedHitType separately to prevent corruption
        tagged_hit_type_backup = None
        if "TaggedHitType" in trackman_data:
            tagged_hit_type_backup = trackman_data["TaggedHitType"].copy()
        
        # Generate sequential pitch numbers
        self._dbg(f"DEBUG: About to assign PitchNo. TaggedHitType before: {trackman_data.get('TaggedHitType', 'NOT_FOUND')[:5] if 'TaggedHitType' in trackman_data else 'NOT_FOUND'}")
        
        # Check what we're about to assign
        pitch_no_values = list(range(1, num_rows + 1))
        self._dbg(f"DEBUG: PitchNo values to assign: {pitch_no_values[:5]}")
        
        trackman_data["PitchNo"] = pitch_no_values
        
        self._dbg(f"DEBUG: PitchNo assigned. TaggedHitType after: {trackman_data.get('TaggedHitType', 'NOT_FOUND')[:5] if 'TaggedHitType' in trackman_data else 'NOT_FOUND'}")
        
        # Debug checkpoint - IMMEDIATELY after PitchNo
        if "TaggedHitType" in trackman_data:
            self._dbg(f"DEBUG: TaggedHitType IMMEDIATELY after PitchNo assignment: {trackman_data['TaggedHitType'][:5]}")
            # Also check if it's the same object reference as another column
            for col_name, col_data in trackman_data.items():
                if col_name != "TaggedHitType" and col_data is trackman_data["TaggedHitType"]:
                    self._dbg(f"DEBUG: SHARED REFERENCE FOUND! TaggedHitType shares reference with {col_name}")
        
        # Debug checkpoint
        if "TaggedHitType" in trackman_data:
            self._dbg(f"DEBUG: TaggedHitType after PitchNo generation: {trackman_data['TaggedHitType'][:5]}")
        
        # Handle date/time formatting
        if "Date" in yakkertech_df.columns:
            dates = pd.to_datetime(yakkertech_df["Date"], errors='coerce')
            trackman_data["Date"] = dates.dt.strftime('%m/%d/%Y').fillna("").tolist()
            trackman_data["Time"] = dates.dt.strftime('%I:%M:%S %p').fillna("").tolist()
            trackman_data["UTCDate"] = dates.dt.strftime('%m/%d/%Y').fillna("").tolist()  
            trackman_data["UTCTime"] = dates.dt.strftime('%H:%M.%f').fillna("").tolist()
            trackman_data["LocalDateTime"] = dates.dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z').fillna("").tolist()
            trackman_data["UTCDateTime"] = dates.dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ').fillna("").tolist()
        
        # Debug checkpoint
        if "TaggedHitType" in trackman_data:
            self._dbg(f"DEBUG: TaggedHitType after date formatting: {trackman_data['TaggedHitType'][:5]}")
        
        # Set system and metadata fields
        trackman_data["System"] = ["v3"] * num_rows
        trackman_data["Level"] = ["Simulated League"] * num_rows
        trackman_data["League"] = ["ModelCA"] * num_rows
        trackman_data["Stadium"] = ["SimField"] * num_rows
        
        # Generate unique IDs
        trackman_data["PitchUID"] = [f"sim-pitch-{i:08d}" for i in range(num_rows)]
        if "GameID" in yakkertech_df.columns:
            trackman_data["GameUID"] = yakkertech_df["GameID"].fillna("sim-game").tolist()
        else:
            trackman_data["GameUID"] = ["sim-game"] * num_rows
        trackman_data["PlayID"] = [f"play-{i:08d}" for i in range(num_rows)]
        
        # Debug checkpoint
        if "TaggedHitType" in trackman_data:
            self._dbg(f"DEBUG: TaggedHitType after ID generation: {trackman_data['TaggedHitType'][:5]}")
        
        # Format pitcher, batter, catcher names as "LastName, FirstName"
        for name_col in ["Pitcher", "Batter", "Catcher"]:
            if name_col in trackman_data and isinstance(trackman_data[name_col], list):
                formatted_names = []
                for name in trackman_data[name_col]:
                    if isinstance(name, str) and name.strip():
                        # Split "FirstName LastName" and reformat as "LastName, FirstName"
                        name_parts = name.strip().split()
                        if len(name_parts) >= 2:
                            first_name = name_parts[0]
                            last_name = " ".join(name_parts[1:])  # Handle multiple last names
                            formatted_names.append(f"{last_name}, {first_name}")
                        else:
                            formatted_names.append(name)  # Keep as-is if only one part
                    else:
                        formatted_names.append("")
                trackman_data[name_col] = formatted_names
        
        # Handle TaggedPitchType fallback and AutoPitchType mapping
        tagged = trackman_data.get("TaggedPitchType", [""] * num_rows)
        # Fallback heuristic for TaggedPitchType if missing/blank
        if tagged is None or all((not str(v).strip()) for v in tagged):
            rel = pd.to_numeric(pd.Series(trackman_data.get("RelSpeed", [None] * num_rows)), errors='coerce')
            hb  = pd.to_numeric(pd.Series(trackman_data.get("HorzBreak", [None] * num_rows)), errors='coerce')
            vb  = pd.to_numeric(pd.Series(trackman_data.get("VertBreak", [None] * num_rows)), errors='coerce')
            inferred = []
            for i in range(num_rows):
                v = rel.iat[i] if i < len(rel) else None
                h = hb.iat[i] if i < len(hb) else None
                z = vb.iat[i] if i < len(vb) else None
                if pd.notna(v):
                    if v >= 93:
                        inferred.append("FourSeamFastball")
                    elif v >= 88:
                        # Sinker vs TwoSeam based on arm-side run
                        if pd.notna(h) and h > 6:
                            inferred.append("Sinker")
                        else:
                            inferred.append("TwoSeamFastball")
                    elif v >= 84:
                        inferred.append("Changeup")
                    elif v >= 78:
                        inferred.append("Slider")
                    else:
                        # Curve if strong downward break
                        if pd.notna(z) and z < -6:
                            inferred.append("Curveball")
                        else:
                            inferred.append("Slider")
                else:
                    inferred.append("")
            trackman_data["TaggedPitchType"] = inferred
        # AutoPitchType from TaggedPitchType
        trackman_data["AutoPitchType"] = self._map_to_trackman_autopitch(trackman_data.get("TaggedPitchType", [""] * num_rows))
        
        # CRITICAL FIX: Restore TaggedHitType if it got corrupted
        if tagged_hit_type_backup is not None:
            trackman_data["TaggedHitType"] = tagged_hit_type_backup
            self._dbg(f"DEBUG: TaggedHitType restored from backup: {trackman_data['TaggedHitType'][:5]}")
        
        # Create DataFrame
        self._dbg(f"DEBUG: Creating DataFrame with {len(trackman_data)} columns")
        if "TaggedHitType" in trackman_data:
            self._dbg(f"DEBUG: TaggedHitType data before DataFrame creation: {trackman_data['TaggedHitType'][:10]}")
            self._dbg(f"DEBUG: TaggedHitType type: {type(trackman_data['TaggedHitType'])}")
            self._dbg(f"DEBUG: TaggedHitType length: {len(trackman_data['TaggedHitType'])}")
        
        # Final check to ensure empty strings are replaced with "Undefined" in critical columns
        for col in ["PitchCall", "KorBB", "TaggedHitType", "PlayResult"]:
            if col in trackman_data:
                trackman_data[col] = [val if val else "Undefined" for val in trackman_data[col]]
                
        trackman_df = pd.DataFrame(trackman_data)

        # ---- Derive commonly expected TrackMan fields when possible ----
        try:
            mph2fps = 1.46667
            # Safe extractors
            def fcol(name, default=np.nan):
                return pd.to_numeric(trackman_df.get(name, pd.Series([default]*num_rows)), errors='coerce')

            # Release to plate flight distance (approx): 60.5 ft minus extension
            ext = fcol("Extension", 6.0)
            flight = 60.5 - ext
            rel_speed = fcol("RelSpeed", np.nan)
            rel_h = fcol("RelHeight", np.nan)
            rel_s = fcol("RelSide", np.nan)
            plate_h = fcol("PlateLocHeight", np.nan)
            plate_s = fcol("PlateLocSide", np.nan)
            horz_break = fcol("HorzBreak", np.nan)
            vert_break = fcol("VertBreak", np.nan)

            # ZoneTime (s) ~ distance / speed (fps)
            zonetime = flight / (rel_speed * mph2fps)
            trackman_df["ZoneTime"] = zonetime

            # ZoneSpeed (mph) – rough estimate of plate velocity
            trackman_df["ZoneSpeed"] = rel_speed * 0.94

            # Approach angles using simple geometry (degrees)
            with np.errstate(invalid='ignore'):
                v_ang = np.degrees(np.arctan2(plate_h - rel_h, flight))
                h_ang = np.degrees(np.arctan2(plate_s - rel_s, flight))
            trackman_df["VertApprAngle"] = v_ang
            trackman_df["HorzApprAngle"] = h_ang

            # pfxx/pfxz: use HorzBreak/VertBreak as proxies
            trackman_df["pfxx"] = horz_break
            trackman_df["pfxz"] = vert_break

            # InducedVertBreak – if missing, mirror VertBreak
            if "InducedVertBreak" not in trackman_df.columns or trackman_df["InducedVertBreak"].isna().all():
                trackman_df["InducedVertBreak"] = vert_break

            # x0,y0,z0 – approximate from release metrics (ft)
            trackman_df["x0"] = rel_s
            trackman_df["y0"] = flight
            trackman_df["z0"] = rel_h

            # Initial velocities/accelerations – not simulated here; set zeros to satisfy schema
            for c in ("vx0","vy0","vz0","ax0","ay0","az0"):
                trackman_df[c] = 0.0

            # Bearing from batted ball Direction if present
            if "Direction" in trackman_df.columns:
                trackman_df["Bearing"] = pd.to_numeric(trackman_df["Direction"], errors='coerce')

            # HangTime – if MeasuredDuration exists, use it, else approximate
            if "MeasuredDuration" in trackman_df.columns and not trackman_df["MeasuredDuration"].isna().all():
                trackman_df["HangTime"] = pd.to_numeric(trackman_df["MeasuredDuration"], errors='coerce')
            elif "Distance" in trackman_df.columns and "ExitSpeed" in trackman_df.columns:
                es = pd.to_numeric(trackman_df["ExitSpeed"], errors='coerce')
                dist = pd.to_numeric(trackman_df["Distance"], errors='coerce')
                trackman_df["HangTime"] = dist / (es * mph2fps)

            # LastTrackedDistance – use Distance if not populated
            if "LastTrackedDistance" not in trackman_df.columns or trackman_df["LastTrackedDistance"].isna().all():
                trackman_df["LastTrackedDistance"] = pd.to_numeric(trackman_df.get("Distance", np.nan), errors='coerce')

            # HitSpinRate – if not provided, leave NaN
            if "HitSpinRate" not in trackman_df.columns:
                trackman_df["HitSpinRate"] = np.nan

            # Tilt – derive simple clock-face tilt from SpinAxis if missing
            if "Tilt" not in trackman_df.columns or trackman_df["Tilt"].isna().all() or (trackman_df["Tilt"] == "").all():
                sa = pd.to_numeric(trackman_df.get("SpinAxis", np.nan), errors='coerce')
                def axis_to_tilt(val):
                    if pd.isna(val):
                        return ""
                    deg = float(val) % 360.0
                    hour = int(round(deg / 30.0)) % 12
                    minute = int(round(((deg % 30.0) / 30.0) * 60.0))
                    if minute == 60:
                        minute = 0; hour = (hour + 1) % 12
                    if hour == 0:
                        hour = 12
                    return f"{hour}:{minute:02d}"
                trackman_df["Tilt"] = sa.apply(axis_to_tilt)
        except Exception:
            # Keep conversion robust even if some derivations fail
            pass
        
        if "TaggedHitType" in trackman_df.columns:
            self._dbg(f"DEBUG: TaggedHitType after DataFrame creation: {trackman_df['TaggedHitType'].head(10).tolist()}")
            self._dbg(f"DEBUG: TaggedHitType value counts after DataFrame: {trackman_df['TaggedHitType'].value_counts()}")
        
        # As a final safety, infer TaggedPitchType and AutoPitchType from numeric fields if still blank
        try:
            if "TaggedPitchType" in trackman_df.columns:
                mask_blank = trackman_df["TaggedPitchType"].astype(str).str.strip().eq("")
                if mask_blank.any():
                    rel = pd.to_numeric(trackman_df.get("RelSpeed", np.nan), errors='coerce')
                    hb  = pd.to_numeric(trackman_df.get("HorzBreak", np.nan), errors='coerce')
                    vb  = pd.to_numeric(trackman_df.get("VertBreak", np.nan), errors='coerce')
                    inferred = []
                    for i in range(len(trackman_df)):
                        if not mask_blank.iat[i]:
                            inferred.append(trackman_df["TaggedPitchType"].iat[i])
                            continue
                        v = rel.iat[i] if i < len(rel) else np.nan
                        h = hb.iat[i] if i < len(hb) else np.nan
                        z = vb.iat[i] if i < len(vb) else np.nan
                        if pd.notna(v):
                            if v >= 93:
                                inferred.append("FourSeamFastball")
                            elif v >= 88:
                                inferred.append("Sinker" if (pd.notna(h) and h > 6) else "TwoSeamFastball")
                            elif v >= 84:
                                inferred.append("Changeup")
                            elif v >= 78:
                                inferred.append("Slider")
                            else:
                                inferred.append("Curveball" if (pd.notna(z) and z < -6) else "Slider")
                        else:
                            inferred.append("")
                    trackman_df.loc[mask_blank, "TaggedPitchType"] = inferred
            # AutoPitchType mapping from TaggedPitchType
            if "TaggedPitchType" in trackman_df.columns:
                trackman_df["AutoPitchType"] = self._map_to_trackman_autopitch(trackman_df["TaggedPitchType"].tolist())
        except Exception:
            pass

        # Ensure all expected columns exist and proper order
        for col in self.TRACKMAN_COLUMNS:
            if col not in trackman_df.columns:
                # create with sensible default
                if col in ["PitchNo","Inning","Outs","Balls","Strikes","OutsOnPlay","RunsScored"]:
                    trackman_df[col] = 0
                else:
                    trackman_df[col] = "" if col not in (
                        "RelSpeed","VertRelAngle","HorzRelAngle","SpinRate","SpinAxis","RelHeight","RelSide","Extension",
                        "VertBreak","InducedVertBreak","HorzBreak","PlateLocHeight","PlateLocSide","ZoneSpeed","VertApprAngle",
                        "HorzApprAngle","ZoneTime","ExitSpeed","Angle","Direction","HitSpinRate","PositionAt110X","PositionAt110Y",
                        "PositionAt110Z","Distance","LastTrackedDistance","Bearing","HangTime","pfxx","pfxz","x0","y0","z0",
                        "vx0","vy0","vz0","ax0","ay0","az0"
                    ) else np.nan

        result_df = trackman_df[self.TRACKMAN_COLUMNS].copy()
        # Ensure we always return a DataFrame, not a Series
        if isinstance(result_df, pd.Series):
            result_df = result_df.to_frame().T
        return result_df
    
    def _map_to_trackman_autopitch(self, pitch_types):
        """
        Convert ModelCA pitch types to TrackMan AutoPitchType format.
        TrackMan's AutoPitchType provides more granular fastball classification.
        """
        trackman_mapping = {
            "Fastball": "Four-Seam",  # Generic fastball -> Four-Seam in TrackMan
            "FourSeamFastball": "Four-Seam",
            "TwoSeamFastball": "Two-Seam", 
            "Sinker": "Sinker",
            "Cutter": "Cutter",
            "Slider": "Slider",
            "Curveball": "Curveball", 
            "Changeup": "Changeup",
            "Splitter": "Splitter",
            "Knuckleball": "Knuckleball"
        }
        
        if isinstance(pitch_types, (list, pd.Series)):
            return [trackman_mapping.get(pt, pt) for pt in pitch_types]
        else:
            return trackman_mapping.get(pitch_types, pitch_types)
    
    def save_game_in_format(self, 
                           game_df: pd.DataFrame, 
                           output_path: Path, 
                           format_type: str,
                           yakkertech_template_path: Optional[str] = None,
                           trackman_template_path: Optional[str] = None) -> Path:
        """Save game DataFrame in specified format
        
        Returns:
            Path: The actual path where the file was saved
        """
        
        format_type = format_type.lower()
        
        if format_type == "yakkertech":
            # Save in YakkerTech format (existing functionality)
            game_df.to_csv(output_path, index=False)
            return output_path
            
        elif format_type == "trackman":
            # Convert to Trackman format and save
            trackman_df = self.convert_yakkertech_to_trackman(game_df)
            # If a TrackMan template was provided, enforce its column set and order
            tmpl_cols = self.get_template_columns("trackman", trackman_template_path) if trackman_template_path else None
            if tmpl_cols:
                # ensure all template columns exist
                for col in tmpl_cols:
                    if col not in trackman_df.columns:
                        trackman_df[col] = ""
                # also keep any existing extra columns by appending
                ordered = [c for c in tmpl_cols]
                extras = [c for c in trackman_df.columns if c not in ordered]
                trackman_df = trackman_df[ordered + extras]
            
            # Final check to ensure empty strings are replaced with "Undefined" in critical columns
            for col in ["PitchCall", "KorBB", "TaggedHitType", "PlayResult"]:
                if col in trackman_df.columns:
                    trackman_df[col] = trackman_df[col].apply(lambda x: "Undefined" if pd.isna(x) or x == "" else x)
            
            # Save directly to the provided output path (no additional suffix)
            trackman_df.to_csv(output_path, index=False)
            return output_path
            
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def save_game_dual_format(self, 
                             game_df: pd.DataFrame, 
                             base_output_path: Path,
                             yakkertech_template_path: Optional[str] = None) -> Dict[str, Path]:
        """Save game in both YakkerTech and Trackman formats"""
        
        output_paths = {}
        
        # YakkerTech format (original)
        yakkertech_path = base_output_path
        game_df.to_csv(yakkertech_path, index=False)
        output_paths["yakkertech"] = yakkertech_path
        
        # Trackman format
        trackman_df = self.convert_yakkertech_to_trackman(game_df)
        
        # Final check to ensure empty strings are replaced with "Undefined" in critical columns
        for col in ["PitchCall", "KorBB", "TaggedHitType", "PlayResult"]:
            if col in trackman_df.columns:
                trackman_df[col] = trackman_df[col].apply(lambda x: "Undefined" if pd.isna(x) or x == "" else x)
                
        trackman_path = base_output_path.with_name(
            base_output_path.stem + "_trackman" + base_output_path.suffix
        )
        trackman_df.to_csv(trackman_path, index=False)
        output_paths["trackman"] = trackman_path
        
        return output_paths
    
    def get_format_info(self, format_type: str) -> Dict[str, Any]:
        """Get information about a specific format"""
        
        format_type = format_type.lower()
        
        if format_type == "yakkertech":
            return {
                "name": "YakkerTech",
                "description": "Standard pitch-by-pitch format used by YakkerTech and similar platforms",
                "columns": "Variable (based on template)",
                "typical_size": "~50-100 columns",
                "use_case": "General baseball analytics, existing workflows"
            }
        elif format_type == "trackman":
            return {
                "name": "Trackman",
                "description": "Comprehensive format used by Trackman baseball systems",
                "columns": len(self.TRACKMAN_COLUMNS),
                "typical_size": "150+ columns",
                "use_case": "Advanced pitch tracking, biomechanics, detailed physics data"
            }
        else:
            return {"error": f"Unknown format: {format_type}"}


def convert_existing_csv_to_trackman(input_path: str, output_path: Optional[str] = None) -> str:
    """Utility function to convert existing YakkerTech CSV to Trackman format"""
    
    manager = CSVFormatManager()
    
    # Load YakkerTech CSV
    yakkertech_df = pd.read_csv(input_path)
    
    # Convert to Trackman
    trackman_df = manager.convert_yakkertech_to_trackman(yakkertech_df)
    
    # Final check to ensure empty strings are replaced with "Undefined" in critical columns
    for col in ["PitchCall", "KorBB", "TaggedHitType", "PlayResult"]:
        if col in trackman_df.columns:
            trackman_df[col] = trackman_df[col].apply(lambda x: "Undefined" if pd.isna(x) or x == "" else x)
    
    # Determine output path
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.with_name(
            input_path_obj.stem + "_trackman" + input_path_obj.suffix
        ))
    
    # Save
    trackman_df.to_csv(output_path, index=False)
    
    return output_path


if __name__ == "__main__":
    # Example usage and testing
    manager = CSVFormatManager()
    
    print("CSV Format Manager - Supported Formats:")
    for fmt in manager.supported_formats:
        info = manager.get_format_info(fmt)
        print(f"\n{info['name']}:")
        print(f"  Description: {info['description']}")
        print(f"  Columns: {info.get('columns', 'N/A')}")
        print(f"  Use case: {info['use_case']}")
    
    # Test column mapping
    print(f"\nTrackman column count: {len(manager.TRACKMAN_COLUMNS)}")
    print(f"Column mappings defined: {len(manager.column_mappings)}")
