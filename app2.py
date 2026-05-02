import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ---------------- CONFIG ----------------
MODEL_DIRS = {
    "ResStock": Path(r"A:\College\Thesis\SHD\models\resstock_rev6_zone_and_global_tsv_driven"),
    "ComStock": Path(r"A:\College\Thesis\SHD\models\comstock_rev2_monotonic"),
}
FT2_TO_M2 = 0.09290304
KWH_TO_KBTU = 1.0 / 0.29307107

# ── Vintage options pulled from actual ResStock / ComStock data dictionaries ──
RESSTOCK_VINTAGES = [
    "<1940", "1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s",
    "2010s",
]
COMSTOCK_VINTAGES = [
    "Before 1946", "1946 to 1959", "1960 to 1969", "1970 to 1979",
    "1980 to 1989", "1990 to 1999", "2000 to 2012", "2013 to 2018",
]

# ResStock heating setpoint values from data dictionary
RESSTOCK_HEATING_SETPOINTS = [
    "55F", "60F", "62F", "65F", "67F", "68F", "70F", "72F", "75F",
]
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Heat Demand Predictor", layout="centered")
st.title("🏠🏢 Heat Demand Predictor")
st.caption("Predict specific heat demand (kWh/m²/year) using trained ResStock or ComStock models.")


# ------------ Helpers ------------
def safe_zone_name(zone: str):
    return str(zone).replace(" ", "_").replace("-", "_").replace("/", "_")


def parse_year_from_text(v):
    if pd.isna(v):
        return np.nan
    import re
    years = re.findall(r"(19\d{2}|20\d{2})", str(v))
    return float(max(int(y) for y in years)) if years else np.nan


@st.cache_data(show_spinner=False)
def list_available_zones(stock_type: str):
    model_dir = MODEL_DIRS[stock_type]
    zones = set()
    for p in model_dir.glob("features_*.joblib"):
        zones.add(p.stem.replace("features_", ""))
    return sorted(zones)


@st.cache_resource(show_spinner=False)
def load_zone_artifacts(stock_type: str, climate_zone: str):
    import json as _json
    model_dir = MODEL_DIRS[stock_type]
    sz = safe_zone_name(climate_zone)

    # ── ResStock saves summary as JSON; ComStock saves as joblib ──────────────
    summary = None
    best_model_name = None

    if stock_type == "ResStock":
        # Training script: save_bundle writes summary_{prefix}.json
        # prefix for zone models is f"zone_{safe_zone}", for global it's "global"
        for json_stem in [f"zone_{sz}", sz, "global"]:
            json_path = model_dir / f"summary_{json_stem}.json"
            if json_path.exists():
                with open(json_path, encoding="utf-8") as f:
                    summary = _json.load(f)
                best_model_name = summary.get("selected_model")
                break
    else:
        # ComStock saves tuning_summary_{safe_zone}.joblib
        joblib_path = model_dir / f"tuning_summary_{sz}.joblib"
        if joblib_path.exists():
            summary = joblib.load(joblib_path)
            best_model_name = summary.get("selected_model")

    name_to_file = {
        "Stacking": f"stacking_{sz}.joblib",
        "CatBoost": f"catboost_{sz}.joblib",
        "XGBoost":  f"xgb_{sz}.joblib",
        "LightGBM": f"lightgbm_{sz}.joblib",
    }
    candidates = []
    if best_model_name in name_to_file:
        candidates.append(model_dir / name_to_file[best_model_name])
    candidates += [
        model_dir / f"stacking_{sz}.joblib",
        model_dir / f"lightgbm_{sz}.joblib",
        model_dir / f"catboost_{sz}.joblib",
        model_dir / f"xgb_{sz}.joblib",
    ]

    model, used_model_file = None, None
    for p in candidates:
        if p.exists():
            model = joblib.load(p)
            used_model_file = p.name
            break

    if model is None:
        raise FileNotFoundError(f"No model found for zone '{climate_zone}' in {model_dir}")

    for path in [model_dir / f"encoders_{sz}.joblib", model_dir / f"features_{sz}.joblib"]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}")

    encoders = joblib.load(model_dir / f"encoders_{sz}.joblib")
    features = joblib.load(model_dir / f"features_{sz}.joblib")
    if not isinstance(features, list):
        features = list(features)

    return model, encoders, features, used_model_file, summary


def infer_hdd_from_zone(zone: str) -> float:
    hdd_map = {
        "Hot-Humid": 1000.0, "Hot-Dry": 1500.0, "Marine": 3000.0,
        "Mixed-Dry": 3500.0, "Mixed-Humid": 4000.0, "Cold": 5500.0,
        "Very Cold": 7000.0, "Subarctic": 9000.0,
    }
    return hdd_map.get(zone, 4000.0)


def encode_row_with_saved_encoders(row_df: pd.DataFrame, features, encoders):
    X_df = row_df.copy()
    for col in features:
        if col not in X_df.columns:
            X_df[col] = "__MISSING__"
    X_df = X_df[features].copy()

    for col in features:
        if col in encoders:
            le = encoders[col]
            raw_val = str(X_df.at[0, col])
            if raw_val in le.classes_:
                X_df.at[0, col] = le.transform([raw_val])[0]
            elif "__MISSING__" in le.classes_:
                X_df.at[0, col] = le.transform(["__MISSING__"])[0]
            else:
                X_df.at[0, col] = le.transform([le.classes_[0]])[0]
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)

    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_df


# ── ResStock row builder ──────────────────────────────────────────────────────
# KEY FINDINGS from training script cross-exam:
#   • Features come from data_dictionary.tsv metadata columns (in.* fields)
#   • floor_area_m2 is BANNED (leakage guard in is_bad_feature_name)
#   • The raw area column seen by the model is in.sqft (numeric, no encoder)
#   • in.heating_setpoint is in must_have list → must be included
#   • Engineered: floor_area_m2 (banned), weekday/weekend_operating_hours_num,
#     operating_hours_week, is_missing__* flags
#   • All values are cast to string then label-encoded or parsed as numeric
def build_resstock_row(user):
    sqft = float(user.get("floor_area_ft2", np.nan))
    occ = float(user.get("occupants", np.nan))

    d = {
        # ── Core in.* features (from data dictionary metadata) ──
        "in.sqft": sqft,                                          # numeric, no encoder
        "in.geometry_building_type_recs": user.get("building_type", "__MISSING__"),
        "in.geometry_stories":            str(user.get("stories", "__MISSING__")),
        "in.geometry_foundation_type":    user.get("foundation_type", "Heated Basement"),
        "in.geometry_attic_type":         user.get("attic_type", "Vented Attic"),
        "in.vintage":                     user.get("vintage", "__MISSING__"),
        "in.occupants":                   str(int(occ)) if np.isfinite(occ) else "__MISSING__",
        "in.bedrooms":                    str(user.get("bedrooms", "__MISSING__")),
        "in.heating_fuel":                user.get("heating_fuel", "__MISSING__"),
        "in.hvac_heating_type":           user.get("hvac_heating_type", "Fuel Furnace"),
        "in.hvac_heating_efficiency":     user.get("hvac_heating_efficiency", "Fuel Furnace, 80% AFUE"),
        "in.hvac_has_ducts":              user.get("hvac_has_ducts", "Yes"),
        "in.duct_leakage_and_insulation": user.get("duct_leakage_and_insulation", "20% Leakage, R-4"),
        "in.infiltration":                user.get("infiltration", "7 ACH50"),
        "in.insulation_wall":             user.get("insulation_wall", "Wood Stud, R-13"),
        "in.insulation_ceiling":          user.get("insulation_ceiling", "R-30"),
        "in.insulation_roof":             user.get("insulation_roof", "R-19"),
        "in.insulation_slab":             user.get("insulation_slab", "Uninsulated"),
        "in.windows":                     user.get("windows", "Double, Clear, Metal"),
        "in.doors":                       user.get("doors", "__MISSING__"),
        # in.heating_setpoint IS in must_have list in training
        "in.heating_setpoint":            user.get("heating_setpoint", "68F"),
        "in.building_america_climate_zone": user.get("climate_zone", "__MISSING__"),
        "in.ashrae_iecc_climate_zone_2004": user.get("iecc_zone", "__MISSING__"),
        "in.census_region":               user.get("census_region", "__MISSING__"),
        "in.state":                       user.get("state", "__MISSING__"),
        "in.county_name":                 user.get("county_name", "__MISSING__"),
        # ── Engineered features created in prepare_dataset() ──
        # floor_area_m2 is BANNED (leakage guard) — do NOT include
        "in.weekday_operating_hours_num": np.nan,   # ResStock doesn't have this; stays NaN→0
        "in.weekend_operating_hours_num": np.nan,
        "operating_hours_week":           np.nan,
        # is_missing__ flags for the must_have features
        "is_missing__in.sqft":            0,
        "is_missing__in.heating_fuel":    0 if user.get("heating_fuel") else 1,
        "is_missing__in.hvac_heating_type": 0,
        "is_missing__in.insulation_wall": 0,
        "is_missing__in.insulation_roof": 0,
        "is_missing__in.windows":         0,
        "is_missing__in.infiltration":    0,
        "is_missing__in.geometry_building_type_recs": 0,
        "is_missing__in.geometry_stories": 0,
    }
    return pd.DataFrame([d])


# ── ComStock row builder ──────────────────────────────────────────────────────
# KEY FINDINGS from training script:
#   • FEATURE_MODE = "max_accuracy" → includes out.params.* U-value columns
#   • U-values are numeric in parquet → no LabelEncoder saved → app must pass
#     actual float values (not "__MISSING__") for them to have any effect
#   • Interaction terms hdd_x_wall_u etc. have monotonic constraint +1
#   • operating_hours_week is built from numeric casts of the operating hours
#   • in.weekday_operating_hours is cast numeric in training (safe_numeric)
#   • is_missing__ flags are created for U-value columns too
def build_comstock_row(user):
    area_ft2 = float(user.get("floor_area_ft2", np.nan))
    area_m2 = area_ft2 * FT2_TO_M2 if np.isfinite(area_ft2) else np.nan

    hdd65f = user.get("hdd65f", np.nan)
    if pd.isna(hdd65f) or hdd65f == 0.0:
        hdd65f = infer_hdd_from_zone(user.get("climate_zone", ""))

    wday = user.get("weekday_operating_hours", np.nan)
    wend = user.get("weekend_operating_hours", np.nan)
    wall_u = user.get("wall_u", np.nan)
    roof_u = user.get("roof_u", np.nan)
    win_u  = user.get("window_u", np.nan)

    vintage = user.get("vintage", "__MISSING__")
    year_built_num = parse_year_from_text(vintage) if vintage != "__MISSING__" else np.nan
    in_year_built = str(int(year_built_num)) if not pd.isna(year_built_num) else "__MISSING__"

    # Interaction terms — only compute if both operands are real numbers
    def prod(a, b):
        return float(a) * float(b) if pd.notna(a) and pd.notna(b) else np.nan

    d = {
        # ── deployable_features block ──
        "in.comstock_building_type":       user.get("building_type", "__MISSING__"),
        "in.comstock_building_type_group": user.get("building_type_group", "__MISSING__"),
        "in.building_subtype":             user.get("building_subtype", "__MISSING__"),
        "in.floor_area_category":          user.get("floor_area_category", "__MISSING__"),
        "in.number_of_stories":            user.get("stories", np.nan),   # numeric in training
        "in.aspect_ratio":                 user.get("aspect_ratio", np.nan),
        "in.rotation":                     user.get("rotation", np.nan),
        "in.wall_construction_type":       user.get("wall_construction_type", "__MISSING__"),
        "in.window_to_wall_ratio_category": user.get("wwr_category", "__MISSING__"),
        "in.window_type":                  user.get("window_type", "__MISSING__"),
        "in.vintage":                      vintage,
        "in.year_built":                   in_year_built,
        "in.building_america_climate_zone": user.get("climate_zone", "__MISSING__"),
        "in.ashrae_iecc_climate_zone_2006": user.get("iecc_zone", "__MISSING__"),
        "in.state":                        user.get("state", "__MISSING__"),
        "in.census_region_name":           user.get("census_region", "__MISSING__"),
        "in.census_division_name":         user.get("census_division", "__MISSING__"),
        "in.hvac_system_type":             user.get("hvac_system_type", "__MISSING__"),
        "in.hvac_category":                user.get("hvac_category", "__MISSING__"),
        "in.hvac_heat_type":               user.get("hvac_heat_type", "__MISSING__"),
        "in.hvac_cool_type":               user.get("hvac_cool_type", "__MISSING__"),
        "in.hvac_vent_type":               user.get("hvac_vent_type", "__MISSING__"),
        "in.hvac_combined_type":           user.get("hvac_combined_type", "__MISSING__"),
        "in.heating_fuel":                 user.get("heating_fuel", "__MISSING__"),
        "in.hvac_night_variability":       user.get("hvac_night_variability", "__MISSING__"),
        "in.weekday_opening_time":         user.get("weekday_opening_time", np.nan),   # numeric
        "in.weekday_operating_hours":      wday,   # numeric (cast in training)
        "in.weekend_opening_time":         user.get("weekend_opening_time", np.nan),
        "in.weekend_operating_hours":      wend,
        # ── max_accuracy_additional block ──
        "out.params.hdd65f":                      hdd65f,
        "out.params.average_wall_u_value":        wall_u,    # numeric → no encoder → float must be real
        "out.params.average_roof_u_value":        roof_u,
        "out.params.average_window_u_value":      win_u,
        "out.params.average_window_shgc":         user.get("window_shgc", np.nan),
        "out.params.window_to_wall_ratio":        user.get("wwr", np.nan),
        "out.params.ext_wall_area":               user.get("ext_wall_area_m2", np.nan),
        "out.params.ext_roof_area":               user.get("ext_roof_area_m2", np.nan),
        "out.params.ext_window_area":             user.get("ext_window_area_m2", np.nan),
        "out.params.building_fraction_heated":    user.get("building_fraction_heated", np.nan),
        "out.params.occupant_density_ppl_per_m_2": user.get("occupant_density_ppl_m2", np.nan),
        "out.params.interior_equipment_power_density": user.get("equip_pd_w_ft2", np.nan),
        "out.params.interior_lighting_power_density":  user.get("light_pd_w_ft2", np.nan),
        # ── engineered features ──
        "floor_area_ft2":     area_ft2,
        "floor_area_m2":      area_m2,
        "year_built_num":     year_built_num,
        "operating_hours_week": prod(wday, 5) + prod(wend, 2) if pd.notna(wday) and pd.notna(wend) else np.nan,
        "hdd_x_wall_u":       prod(hdd65f, wall_u),
        "hdd_x_roof_u":       prod(hdd65f, roof_u),
        "hdd_x_win_u":        prod(hdd65f, win_u),
        "hdd_x_wall_area":    prod(hdd65f, user.get("ext_wall_area_m2", np.nan)),
        "hdd_x_window_area":  prod(hdd65f, user.get("ext_window_area_m2", np.nan)),
        # ── is_missing__ flags (created in add_missing_flags during training) ──
        "is_missing__out.params.average_wall_u_value":   int(pd.isna(wall_u)),
        "is_missing__out.params.average_roof_u_value":   int(pd.isna(roof_u)),
        "is_missing__out.params.average_window_u_value": int(pd.isna(win_u)),
        "is_missing__out.params.hdd65f":                 int(pd.isna(hdd65f)),
        "is_missing__in.hvac_system_type":               int(not user.get("hvac_system_type")),
        "is_missing__in.window_type":                    int(not user.get("window_type")),
        "is_missing__in.weekday_operating_hours":        int(pd.isna(wday)),
        "is_missing__in.weekend_operating_hours":        int(pd.isna(wend)),
    }
    return pd.DataFrame([d])


# ------------ UI ------------
stock_type = st.radio("Dataset / Model Family", ["ResStock", "ComStock"], horizontal=True)

available_safe_zones = list_available_zones(stock_type)
display_zones = [z.replace("_", " ") for z in available_safe_zones]
if not display_zones:
    st.error(f"No zone artifacts found in {MODEL_DIRS[stock_type]}")
    st.stop()

st.subheader("Inputs")
col1, col2 = st.columns(2)
with col1:
    floor_area_ft2 = st.number_input("Floor area (ft²)", min_value=300.0, max_value=2_000_000.0, value=2000.0, step=50.0)
    climate_zone   = st.selectbox("Climate zone", sorted(display_zones))
    stories        = st.number_input("Stories", min_value=1, max_value=80, value=2, step=1)
with col2:
    heating_fuel   = st.selectbox("Heating fuel", ["Natural Gas", "Electricity", "Fuel Oil", "Propane", "DistrictHeating", "Other"])
    vintage_opts   = RESSTOCK_VINTAGES if stock_type == "ResStock" else COMSTOCK_VINTAGES
    vintage        = st.selectbox("Construction era / Vintage", vintage_opts)
    iecc_zone      = st.text_input("IECC/ASHRAE zone (optional)", "")

# ── ResStock extra inputs ──────────────────────────────────────────────────────
if stock_type == "ResStock":
    st.markdown("#### ResStock details")
    c1, c2, c3 = st.columns(3)
    with c1:
        building_type = st.selectbox(
            "Building type",
            ["Single-Family Detached", "Single-Family Attached",
             "Apartment", "Manufactured Home", "Multi-Family"]
        )
        occupants = st.number_input("Occupants", min_value=1, max_value=20, value=4, step=1)
        bedrooms  = st.number_input("Bedrooms",  min_value=0, max_value=10, value=3, step=1)
    with c2:
        infiltration    = st.selectbox("Infiltration", ["3 ACH50", "5 ACH50", "7 ACH50", "10 ACH50"], index=2)
        heating_setpoint = st.selectbox("Heating setpoint", RESSTOCK_HEATING_SETPOINTS,
                                        index=RESSTOCK_HEATING_SETPOINTS.index("68F"))
        census_region   = st.text_input("Census region (optional)", "")
    with c3:
        state       = st.text_input("State abbreviation (optional, e.g. TX)", "")
        insulation_wall = st.selectbox(
            "Wall insulation",
            ["Wood Stud, Uninsulated", "Wood Stud, R-7", "Wood Stud, R-13",
             "Wood Stud, R-15", "Wood Stud, R-19", "CMU, Uninsulated", "Brick, Uninsulated"]
        )
        insulation_ceiling = st.selectbox(
            "Ceiling insulation",
            ["Uninsulated", "R-7", "R-13", "R-19", "R-30", "R-38", "R-49"]
        )

# ── ComStock extra inputs ──────────────────────────────────────────────────────
else:
    st.markdown("#### ComStock details")
    c1, c2, c3 = st.columns(3)
    with c1:
        building_type = st.selectbox(
            "ComStock building type",
            ["SmallOffice", "MediumOffice", "LargeOffice", "RetailStandalone", "RetailStripmall",
             "PrimarySchool", "SecondarySchool", "Warehouse", "Hospital", "SmallHotel",
             "LargeHotel", "QuickServiceRestaurant", "FullServiceRestaurant", "Outpatient"]
        )
        building_type_group = st.text_input("Building type group (optional)", "")
        hvac_system_type    = st.text_input("HVAC system type (optional)", "")
    with c2:
        weekday_operating_hours = st.number_input("Weekday operating hours", 0.0, 24.0, 12.0, 0.5)
        weekend_operating_hours = st.number_input("Weekend operating hours", 0.0, 24.0, 6.0, 0.5)
        weekday_opening_time    = st.number_input("Weekday opening hour", 0.0, 24.0, 8.0, 0.5)
        weekend_opening_time    = st.number_input("Weekend opening hour", 0.0, 24.0, 9.0, 0.5)
    with c3:
        st.caption("U-values: 0 = unknown (model uses its own imputation)")
        hdd65f   = st.number_input("HDD65F (0 = auto by climate zone)",      0.0, 15000.0, 0.0, 50.0)
        wall_u   = st.number_input("Wall U-value   (W/m²K, 0 = unknown)",   0.0, 5.0,     0.0, 0.01)
        roof_u   = st.number_input("Roof U-value   (W/m²K, 0 = unknown)",   0.0, 5.0,     0.0, 0.01)
        window_u = st.number_input("Window U-value (W/m²K, 0 = unknown)",   0.0, 10.0,    0.0, 0.05)

debug_mode = st.checkbox("Debug mode (show feature diagnostics)", value=False)

# ------------ Predict ------------
if st.button("Predict Heat Demand"):
    try:
        model, encoders, features, model_file, summary = load_zone_artifacts(stock_type, climate_zone)

        if stock_type == "ResStock":
            user = {
                "floor_area_ft2":   floor_area_ft2,
                "building_type":    building_type,
                "stories":          stories,
                "vintage":          vintage,
                "occupants":        occupants,
                "bedrooms":         bedrooms,
                "heating_fuel":     heating_fuel,
                "heating_setpoint": heating_setpoint,
                "climate_zone":     climate_zone,
                "iecc_zone":        iecc_zone or "__MISSING__",
                "census_region":    census_region or "__MISSING__",
                "state":            state or "__MISSING__",
                "infiltration":     infiltration,
                "insulation_wall":  insulation_wall,
                "insulation_ceiling": insulation_ceiling,
            }
            row_df = build_resstock_row(user)
        else:
            user = {
                "floor_area_ft2":           floor_area_ft2,
                "building_type":            building_type,
                "building_type_group":      building_type_group or "__MISSING__",
                "stories":                  stories,
                "vintage":                  vintage,
                "climate_zone":             climate_zone,
                "iecc_zone":                iecc_zone or "__MISSING__",
                "heating_fuel":             heating_fuel,
                "hvac_system_type":         hvac_system_type or "",
                "weekday_operating_hours":  weekday_operating_hours,
                "weekend_operating_hours":  weekend_operating_hours,
                "weekday_opening_time":     weekday_opening_time,
                "weekend_opening_time":     weekend_opening_time,
                "hdd65f":                   np.nan if hdd65f == 0.0 else hdd65f,
                "wall_u":                   np.nan if wall_u == 0.0 else wall_u,
                "roof_u":                   np.nan if roof_u == 0.0 else roof_u,
                "window_u":                 np.nan if window_u == 0.0 else window_u,
            }
            row_df = build_comstock_row(user)

        X_df    = encode_row_with_saved_encoders(row_df, features, encoders)
        pred_raw = float(model.predict(X_df)[0])

        # Both ResStock (USE_LOG_TARGET=True) and ComStock use log1p target
        pred_specific = np.expm1(pred_raw)

        area_m2           = floor_area_ft2 * FT2_TO_M2
        pred_total_kwh    = pred_specific * area_m2
        pred_total_kbtu   = pred_total_kwh * KWH_TO_KBTU

        st.success("Prediction complete ✅")
        st.caption(f"Model family: {stock_type} | Model file: {model_file}")
        st.metric("Specific heat demand",   f"{pred_specific:,.2f} kWh/m²/year")
        st.metric("Total annual heat demand", f"{pred_total_kwh:,.0f} kWh/year")
        st.caption(f"≈ {pred_total_kbtu:,.0f} kBtu/year")

        # ── Debug ──────────────────────────────────────────────────────────────
        if debug_mode:
            st.markdown("### Debug diagnostics")
            st.write(f"Raw prediction (log scale): `{pred_raw:.4f}` → after expm1: `{pred_specific:.4f}` kWh/m²/year")
            st.write(f"Floor area: {area_m2:.1f} m²")

            # Feature presence audit
            st.markdown("#### Feature presence audit")
            audit_cols = {
                "ResStock": [
                    "in.sqft", "in.heating_setpoint", "in.infiltration",
                    "in.insulation_wall", "in.insulation_ceiling",
                    "in.hvac_heating_type", "in.heating_fuel",
                    "in.geometry_building_type_recs", "in.geometry_stories", "in.vintage",
                ],
                "ComStock": [
                    "out.params.hdd65f",
                    "out.params.average_wall_u_value",
                    "out.params.average_roof_u_value",
                    "out.params.average_window_u_value",
                    "hdd_x_wall_u", "hdd_x_roof_u", "hdd_x_win_u",
                    "operating_hours_week",
                    "in.weekday_operating_hours", "in.weekend_operating_hours",
                    "is_missing__out.params.average_wall_u_value",
                ],
            }[stock_type]

            rows = []
            for c in audit_cols:
                rows.append({
                    "feature": c,
                    "in model features": c in features,
                    "has LabelEncoder": c in encoders,
                    "raw value sent": str(row_df[c].iloc[0]) if c in row_df.columns else "NOT IN ROW",
                    "encoded value": f"{X_df[c].iloc[0]:.4f}" if c in X_df.columns else "NOT IN X",
                })
            st.dataframe(pd.DataFrame(rows).set_index("feature"))

            if stock_type == "ComStock":
                # U-value sensitivity
                st.markdown("#### U-value sensitivity (wall)")
                results = {}
                for label, wu in [("High U=1.5 (poor)", 1.5), ("Low U=0.25 (good)", 0.25)]:
                    r = row_df.copy()
                    r["out.params.average_wall_u_value"] = wu
                    r["hdd_x_wall_u"] = float(r["out.params.hdd65f"].iloc[0]) * wu
                    r["is_missing__out.params.average_wall_u_value"] = 0
                    Xe = encode_row_with_saved_encoders(r, features, encoders)
                    pred = np.expm1(float(model.predict(Xe)[0]))
                    results[label] = round(pred, 3)
                delta = results["High U=1.5 (poor)"] - results["Low U=0.25 (good)"]
                results["Δ (high−low)"] = round(delta, 3)
                results["verdict"] = "✅ U-values affect model" if abs(delta) > 0.5 else "⚠️ Little/no effect — check feature presence above"
                st.json(results)

                # Hours sensitivity
                st.markdown("#### Operating hours sensitivity")
                results_h = {}
                for label, wd, we in [("Long hours (16/10)", 16.0, 10.0), ("Short hours (6/0)", 6.0, 0.0)]:
                    r = row_df.copy()
                    r["in.weekday_operating_hours"] = wd
                    r["in.weekend_operating_hours"] = we
                    r["operating_hours_week"] = wd * 5 + we * 2
                    Xe = encode_row_with_saved_encoders(r, features, encoders)
                    pred = np.expm1(float(model.predict(Xe)[0]))
                    results_h[label] = round(pred, 3)
                results_h["Δ"] = round(results_h["Long hours (16/10)"] - results_h["Short hours (6/0)"], 3)
                st.json(results_h)

            st.write("**First 40 features in this zone model:**", features[:40])
            if summary:
                sel  = summary.get("selected_model", "N/A")
                rmse = summary.get("model_stats", {}).get(sel, {}).get("rmse", "N/A")
                st.write(f"Selected model: **{sel}** | Test RMSE: {rmse}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        if debug_mode:
            import traceback
            st.code(traceback.format_exc())