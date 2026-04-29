import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone

import optuna
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


# ========================= CONFIG =========================
DATA_PATH = r"A:\College\Thesis\SHD\USstock\comstock\baseline.parquet"
MODEL_SAVE_DIR = Path(r"A:\College\Thesis\Models\comstock_rev2_monotonic")
PLOTS_DIR = MODEL_SAVE_DIR / "plots"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5
MIN_SAMPLES_PER_ZONE = 250

N_TRIALS_XGB = 30
N_TRIALS_CAT = 30
N_TRIALS_LGB = 30

FEATURE_MODE = "max_accuracy"   # "deployable" or "max_accuracy"
USE_LOG_TARGET = True
GEN_GAP_WEIGHT = 0.5
CLEAR_OLD_PLOTS = False

# monotonic constraints toggle
USE_MONOTONIC_CONSTRAINTS = True
# =========================================================


# ---------------- Utility ----------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_actual_vs_pred_plot(y_true, y_pred, zone_name, model_name, out_path):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual specific heating demand (kWh/m²/yr)")
    plt.ylabel("Predicted specific heating demand (kWh/m²/yr)")
    plt.title(f"{zone_name} - {model_name}\nActual vs Predicted (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_year_from_text(v):
    if pd.isna(v):
        return np.nan
    import re
    s = str(v)
    years = re.findall(r"(19\d{2}|20\d{2})", s)
    if len(years) == 0:
        return np.nan
    return float(max(int(y) for y in years))


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def add_missing_flags(df, cols):
    for c in cols:
        if c in df.columns:
            df[f"is_missing__{c}"] = df[c].isna().astype(int)
    return df


def fit_label_encoders(X_df: pd.DataFrame):
    X_df = X_df.copy()
    encoders = {}
    for col in X_df.columns:
        s = X_df[col].astype("string").fillna("__MISSING__").astype(str)
        n = pd.to_numeric(s, errors="coerce")
        if n.notna().all():
            X_df[col] = n.astype(np.float64)
        else:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(s).astype(np.float64)
            encoders[col] = le
    return X_df, encoders


def transform_with_encoders(X_df: pd.DataFrame, encoders: dict):
    X_df = X_df.copy()
    for col in X_df.columns:
        s = X_df[col].astype("string").fillna("__MISSING__").astype(str)
        if col in encoders:
            le = encoders[col]
            cls = set(le.classes_)
            vals = []
            for v in s.values:
                if v in cls:
                    vals.append(le.transform([v])[0])
                elif "__MISSING__" in cls:
                    vals.append(le.transform(["__MISSING__"])[0])
                else:
                    vals.append(0)
            X_df[col] = np.array(vals, dtype=np.float64)
        else:
            X_df[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(np.float64)
    return X_df


def map_first_match(df, canonical, exact_names=None, contains_any=None):
    if canonical in df.columns:
        return canonical

    exact_names = exact_names or []
    contains_any = contains_any or []

    for c in exact_names:
        if c in df.columns:
            df[canonical] = df[c]
            return c

    for c in df.columns:
        cl = c.lower()
        if all(tok in cl for tok in contains_any):
            df[canonical] = df[c]
            return c
    return None


def crossval_rmse(model_builder, X, y, n_splits=5, random_state=42, log_target=False):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        y_fit = np.log1p(y_tr) if log_target else y_tr

        m = model_builder()
        m.fit(X_tr, y_fit)
        p = m.predict(X_va)

        if log_target:
            p = np.expm1(p)
            p = np.clip(p, 0, None)

        scores.append(rmse(y_va, p))
    return float(np.mean(scores)), float(np.std(scores))


def build_monotone_vectors(features):
    """
    Conservative monotonic constraints for heating demand.
    +1 only where physical direction is high-confidence.
    """
    monotone_map = {f: 0 for f in features}

    high_conf_must_increase = {
        # climate severity
        "out.params.hdd65f",
        "out.params.hdd50f",
        "out.params.hours_below_50f",
        "out.params.hours_below_17f",
        "out.params.hours_below_0f",

        # operating time
        "in.weekday_operating_hours",
        "in.weekend_operating_hours",
        "operating_hours_week",

        # interaction terms tied to climate/envelope load
        "hdd_x_wall_u",
        "hdd_x_roof_u",
        "hdd_x_win_u",
        "hdd_x_wall_area",
        "hdd_x_window_area",
    }

    for f in high_conf_must_increase:
        if f in monotone_map:
            monotone_map[f] = 1

    # XGBoost expects tuple aligned with feature order
    xgb_tuple = tuple(monotone_map[f] for f in features)
    # LightGBM expects list aligned with feature order
    lgb_list = [monotone_map[f] for f in features]

    return monotone_map, xgb_tuple, lgb_list

# ---------------- Load ----------------
if CLEAR_OLD_PLOTS:
    for p in PLOTS_DIR.glob("actual_vs_pred_*.png"):
        p.unlink()

print("=== Loading ComStock parquet ===")
df = pd.read_parquet(DATA_PATH).convert_dtypes()
print(f"Raw shape: {df.shape}")

# ---------------- Canonical mapping ----------------
alias_hits = {}
alias_hits["in.weekday_operating_hours"] = map_first_match(
    df, "in.weekday_operating_hours",
    exact_names=["in.weekday_operating_hours", "in.weekday_operating_hours..hr"],
    contains_any=["weekday", "operating", "hour"]
)
alias_hits["in.weekend_operating_hours"] = map_first_match(
    df, "in.weekend_operating_hours",
    exact_names=["in.weekend_operating_hours", "in.weekend_operating_hours..hr", "in.weekened_operating_hours..hr"],
    contains_any=["weeke", "operating", "hour"]
)
alias_hits["in.weekday_opening_time"] = map_first_match(
    df, "in.weekday_opening_time",
    exact_names=["in.weekday_opening_time", "in.weekday_opening_time..hr", "in.weekday_start_time"],
    contains_any=["weekday", "open", "time"]
)
alias_hits["in.weekend_opening_time"] = map_first_match(
    df, "in.weekend_opening_time",
    exact_names=["in.weekend_opening_time", "in.weekend_opening_time..hr", "in.weekened_opening_time..hr", "in.weekend_start_time"],
    contains_any=["weeke", "open", "time"]
)
alias_hits["out.params.average_heating_setpoint_max"] = map_first_match(
    df, "out.params.average_heating_setpoint_max",
    exact_names=["out.params.average_heating_setpoint_max", "out.params.average_heating_setpoint_max..c"],
    contains_any=["average", "heating", "setpoint", "max"]
)
alias_hits["out.params.average_heating_setpoint_min"] = map_first_match(
    df, "out.params.average_heating_setpoint_min",
    exact_names=["out.params.average_heating_setpoint_min", "out.params.average_heating_setpoint_min..c"],
    contains_any=["average", "heating", "setpoint", "min"]
)
alias_hits["out.params.hours_heating_setpoint_not_met"] = map_first_match(
    df, "out.params.hours_heating_setpoint_not_met",
    exact_names=[
        "out.params.hours_heating_setpoint_not_met",
        "out.params.hours_heating_setpoint_not_met..hr",
        "out.params.hours_heating_setpoint_not_met..c",
    ],
    contains_any=["hours", "heating", "setpoint", "not", "met"]
)

print("\nAlias mapping hits:")
for k, v in alias_hits.items():
    print(f"  {k} <- {v}")

# ---------------- Target + Area ----------------
if "calc.enduse_group.site_energy.heating.energy_consumption" in df.columns:
    df["heating_energy_kwh"] = safe_numeric(df["calc.enduse_group.site_energy.heating.energy_consumption"])
else:
    parts = [
        "out.electricity.heating.energy_consumption",
        "out.natural_gas.heating.energy_consumption",
        "out.other_fuel.heating.energy_consumption",
        "out.district_heating.heating.energy_consumption",
    ]
    missing = [c for c in parts if c not in df.columns]
    if missing:
        raise ValueError(f"Missing heating target columns: {missing}")
    s = 0
    for c in parts:
        s = s + safe_numeric(df[c]).fillna(0.0)
    df["heating_energy_kwh"] = s

if "in.sqft" in df.columns:
    df["floor_area_ft2"] = safe_numeric(df["in.sqft"])
elif "calc.weighted.sqft" in df.columns:
    df["floor_area_ft2"] = safe_numeric(df["calc.weighted.sqft"])
else:
    raise ValueError("No area column found (in.sqft or calc.weighted.sqft).")

df = df.dropna(subset=["heating_energy_kwh", "floor_area_ft2"]).copy()
df = df[(df["floor_area_ft2"] > 0) & (df["heating_energy_kwh"] >= 0)].copy()

df["floor_area_m2"] = df["floor_area_ft2"] * 0.09290304
df["specific_heat_demand_kwh_m2"] = df["heating_energy_kwh"] / df["floor_area_m2"]

df = df[np.isfinite(df["specific_heat_demand_kwh_m2"])].copy()
df = df[df["specific_heat_demand_kwh_m2"] > 0].copy()

q1 = df["specific_heat_demand_kwh_m2"].quantile(0.005)
q2 = df["specific_heat_demand_kwh_m2"].quantile(0.995)
df = df[(df["specific_heat_demand_kwh_m2"] >= q1) & (df["specific_heat_demand_kwh_m2"] <= q2)].copy()

if "upgrade" in df.columns:
    u = df["upgrade"].astype("string").str.strip().str.lower()
    is_baseline = (u.isin(["0", "00", "baseline", "base", "upgrade00"]) | u.str.fullmatch(r"0+"))
    if is_baseline.sum() > 1000:
        df = df[is_baseline].copy()
    else:
        print("Warning: baseline filter skipped (no clear baseline coding found).")

# ---------------- Feature blocks ----------------
CLIMATE_COL = "in.building_america_climate_zone"

deployable_features = [
    "in.comstock_building_type", "in.comstock_building_type_group", "in.building_subtype",
    "in.floor_area_category", "in.number_of_stories", "in.aspect_ratio", "in.rotation",
    "in.wall_construction_type", "in.window_to_wall_ratio_category", "in.window_type",
    "in.vintage", "in.year_built",
    CLIMATE_COL, "in.ashrae_iecc_climate_zone_2006", "in.state",
    "in.census_region_name", "in.census_division_name",
    "in.cambium_grid_region", "in.iso_rto_region",
    "in.cluster_id", "in.cluster_name", "in.county_name",
    "in.weather_file_2018", "in.weather_file_tmy3",
    "in.hvac_system_type", "in.hvac_category", "in.hvac_heat_type",
    "in.hvac_cool_type", "in.hvac_vent_type", "in.hvac_combined_type",
    "in.heating_fuel", "in.hvac_night_variability",
    "in.weekday_opening_time", "in.weekday_operating_hours",
    "in.weekend_opening_time", "in.weekend_operating_hours",
    "in.energy_code_followed_during_original_building_construction",
    "in.energy_code_followed_during_last_hvac_replacement",
    "in.energy_code_followed_during_last_roof_replacement",
    "in.energy_code_followed_during_last_walls_replacement",
    "in.ownership_type", "in.party_responsible_for_operation", "in.purchase_input_responsibility",
]

max_accuracy_additional = [
    "out.params.hdd65f", "out.params.hdd50f", "out.params.cdd65f",
    "out.params.hours_below_50f", "out.params.hours_below_17f", "out.params.hours_below_0f",
    "out.params.average_wall_u_value", "out.params.average_roof_u_value", "out.params.average_window_u_value",
    "out.params.average_window_shgc", "out.params.window_to_wall_ratio",
    "out.params.ext_wall_area", "out.params.ext_roof_area", "out.params.ext_window_area",
    "out.params.building_fraction_heated", "out.params.heating_equipment",
    "out.params.boiler_average_efficiency", "out.params.boiler_capacity",
    "out.params.dx_heating_average_cop", "out.params.dx_heating_design_cop",
    "out.params.heat_pump_heating_average_cop", "out.params.heat_pump_heating_capacity",
    "out.params.average_heating_setpoint_max",
    "out.params.average_heating_setpoint_min",
    "out.params.hours_heating_setpoint_not_met",
    "out.params.average_outdoor_air_fraction",
    "out.params.occupant_density_ppl_per_m_2", "out.params.occupant_eflh",
    "out.params.interior_equipment_power_density", "out.params.interior_electric_equipment_eflh",
    "out.params.interior_lighting_power_density", "out.params.interior_lighting_eflh",
]

candidate_features = list(deployable_features)
if FEATURE_MODE == "max_accuracy":
    candidate_features += max_accuracy_additional
candidate_features = [c for c in candidate_features if c in df.columns]

# ---------------- Engineered ----------------
for c in [
    "in.number_of_stories", "in.aspect_ratio", "in.rotation",
    "in.weekday_opening_time", "in.weekday_operating_hours",
    "in.weekend_opening_time", "in.weekend_operating_hours",
    "out.params.average_heating_setpoint_max", "out.params.average_heating_setpoint_min",
]:
    if c in df.columns:
        df[c] = safe_numeric(df[c])

if "in.year_built" in df.columns:
    df["year_built_num"] = df["in.year_built"].apply(parse_year_from_text)

if {"in.weekday_operating_hours", "in.weekend_operating_hours"}.issubset(df.columns):
    wkd = safe_numeric(df["in.weekday_operating_hours"])
    wke = safe_numeric(df["in.weekend_operating_hours"])
    df["operating_hours_week"] = 5.0 * wkd + 2.0 * wke
    df["is_missing__in.weekday_operating_hours"] = wkd.isna().astype(int)
    df["is_missing__in.weekend_operating_hours"] = wke.isna().astype(int)

if {"out.params.average_heating_setpoint_max", "out.params.average_heating_setpoint_min"}.issubset(df.columns):
    tmax = safe_numeric(df["out.params.average_heating_setpoint_max"])
    tmin = safe_numeric(df["out.params.average_heating_setpoint_min"])
    df["heating_setpoint_span_c"] = tmax - tmin
    df["is_missing__out.params.average_heating_setpoint_max"] = tmax.isna().astype(int)
    df["is_missing__out.params.average_heating_setpoint_min"] = tmin.isna().astype(int)

if "out.params.hdd65f" in df.columns:
    hdd = safe_numeric(df["out.params.hdd65f"])
    if "out.params.average_wall_u_value" in df.columns:
        df["hdd_x_wall_u"] = hdd * safe_numeric(df["out.params.average_wall_u_value"])
    if "out.params.average_roof_u_value" in df.columns:
        df["hdd_x_roof_u"] = hdd * safe_numeric(df["out.params.average_roof_u_value"])
    if "out.params.average_window_u_value" in df.columns:
        df["hdd_x_win_u"] = hdd * safe_numeric(df["out.params.average_window_u_value"])
    if "out.params.ext_wall_area" in df.columns:
        df["hdd_x_wall_area"] = hdd * safe_numeric(df["out.params.ext_wall_area"])
    if "out.params.ext_window_area" in df.columns:
        df["hdd_x_window_area"] = hdd * safe_numeric(df["out.params.ext_window_area"])

engineered = [
    "floor_area_ft2", "floor_area_m2", "year_built_num",
    "operating_hours_week",
    "heating_setpoint_span_c",
    "hdd_x_wall_u", "hdd_x_roof_u", "hdd_x_win_u",
    "hdd_x_wall_area", "hdd_x_window_area",
    "is_missing__in.weekday_operating_hours",
    "is_missing__in.weekend_operating_hours",
    "is_missing__out.params.average_heating_setpoint_max",
    "is_missing__out.params.average_heating_setpoint_min",
]

missing_flag_candidates = [
    "out.params.average_wall_u_value",
    "out.params.average_roof_u_value",
    "out.params.average_window_u_value",
    "out.params.hdd65f",
    "in.hvac_system_type",
    "in.window_type",
    "in.weekday_operating_hours",
    "in.weekend_operating_hours",
    "out.params.average_heating_setpoint_max",
    "out.params.average_heating_setpoint_min",
    "out.params.hours_heating_setpoint_not_met",
]
df = add_missing_flags(df, missing_flag_candidates)
engineered += [c for c in df.columns if c.startswith("is_missing__")]

features = candidate_features + [c for c in engineered if c in df.columns]
features = list(dict.fromkeys(features))

df = df.dropna(subset=["specific_heat_demand_kwh_m2", CLIMATE_COL]).copy()
df[features] = df[features].astype("string").fillna("__MISSING__")

print(f"\nRows ready: {len(df)}")
print(f"Feature mode: {FEATURE_MODE}")
print(f"Use log target: {USE_LOG_TARGET}")
print(f"Use monotonic constraints: {USE_MONOTONIC_CONSTRAINTS}")
print(f"Total features: {len(features)}")

# ---------------- Train per zone ----------------
zones = sorted(df[CLIMATE_COL].astype("string").dropna().unique().tolist())
saved = {}

for zone in zones:
    d = df[df[CLIMATE_COL].astype("string") == zone].copy()
    if len(d) < MIN_SAMPLES_PER_ZONE:
        print(f"Skipping zone '{zone}' ({len(d)} samples)")
        continue

    safe_zone = str(zone).replace(" ", "_").replace("-", "_").replace("/", "_")
    print(f"\n=== Zone: {zone} | samples={len(d)} ===")

    X_raw = d[features].copy()
    y = d["specific_heat_demand_kwh_m2"].to_numpy(dtype=np.float64)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_STATE
    )

    X_train_df, encoders = fit_label_encoders(X_train_raw)
    X_test_df = transform_with_encoders(X_test_raw, encoders)

    X_train = X_train_df.to_numpy(dtype=np.float64)
    X_test = X_test_df.to_numpy(dtype=np.float64)

    # monotonic vectors after final feature order is known
    monotone_map, xgb_mono, lgb_mono = build_monotone_vectors(features)

    # ----- tuning -----
    def tune_xgb():
        def objective(trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 250, 1200),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 40.0),
                "gamma": trial.suggest_float("gamma", 0.0, 15.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 60.0, log=True),
                "random_state": RANDOM_STATE,
                "tree_method": "hist",
                "n_jobs": -1,
            }
            if USE_MONOTONIC_CONSTRAINTS:
                p["monotone_constraints"] = xgb_mono

            def builder():
                return xgb.XGBRegressor(**p)

            m, _ = crossval_rmse(builder, X_train, y_train, n_splits=CV_FOLDS, random_state=RANDOM_STATE, log_target=USE_LOG_TARGET)
            return m

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS_XGB, show_progress_bar=False)
        return study.best_params, study.best_value

    def tune_cat():
        def objective(trial):
            p = {
                "iterations": trial.suggest_int("iterations", 250, 1200),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 60.0),
                "random_strength": trial.suggest_float("random_strength", 1.0, 12.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 6.0),
                "loss_function": "RMSE",
                "random_seed": RANDOM_STATE,
                "verbose": 0,
            }

            def builder():
                return CatBoostRegressor(**p)

            m, _ = crossval_rmse(builder, X_train, y_train, n_splits=CV_FOLDS, random_state=RANDOM_STATE, log_target=USE_LOG_TARGET)
            return m

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS_CAT, show_progress_bar=False)
        return study.best_params, study.best_value

    def tune_lgb():
        def objective(trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 250, 1200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 80),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 40, 300),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 60.0, log=True),
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            }
            if USE_MONOTONIC_CONSTRAINTS:
                p["monotone_constraints"] = lgb_mono

            def builder():
                return LGBMRegressor(**p)

            m, _ = crossval_rmse(builder, X_train, y_train, n_splits=CV_FOLDS, random_state=RANDOM_STATE, log_target=USE_LOG_TARGET)
            return m

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS_LGB, show_progress_bar=False)
        return study.best_params, study.best_value

    xgb_best, xgb_cv = tune_xgb()
    cat_best, cat_cv = tune_cat()
    lgb_best, lgb_cv = tune_lgb()

    print(f"Best CV RMSE -> XGB:{xgb_cv:.3f} | CAT:{cat_cv:.3f} | LGB:{lgb_cv:.3f}")

    # ensure constraints are carried into final models
    xgb_final_params = dict(xgb_best)
    lgb_final_params = dict(lgb_best)
    if USE_MONOTONIC_CONSTRAINTS:
        xgb_final_params["monotone_constraints"] = xgb_mono
        lgb_final_params["monotone_constraints"] = lgb_mono

    xgb_model = xgb.XGBRegressor(**xgb_final_params, random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1)
    cat_model = CatBoostRegressor(**cat_best, random_seed=RANDOM_STATE, verbose=0)
    lgb_model = LGBMRegressor(**lgb_final_params, random_state=RANDOM_STATE, n_jobs=-1)

    y_train_fit = np.log1p(y_train) if USE_LOG_TARGET else y_train

    xgb_model.fit(X_train, y_train_fit)
    cat_model.fit(X_train, y_train_fit, verbose=False)
    lgb_model.fit(X_train, y_train_fit)

    stack_model = StackingRegressor(
        estimators=[("xgb", clone(xgb_model)), ("cat", clone(cat_model)), ("lgb", clone(lgb_model))],
        final_estimator=Ridge(alpha=2.0),
        cv=5,
        n_jobs=-1,
        passthrough=False
    )
    stack_model.fit(X_train, y_train_fit)

    models = [("XGBoost", xgb_model), ("CatBoost", cat_model), ("LightGBM", lgb_model), ("Stacking", stack_model)]

    def predict_back(m, X):
        p = m.predict(X)
        if USE_LOG_TARGET:
            p = np.expm1(p)
            p = np.clip(p, 0, None)
        return p

    stats = {}
    for n, m in models:
        p_tr = predict_back(m, X_train)
        p_te = predict_back(m, X_test)

        rmse_tr = rmse(y_train, p_tr)
        rmse_te = rmse(y_test, p_te)
        gap = rmse_te - rmse_tr
        mae = mean_absolute_error(y_test, p_te)
        r2 = r2_score(y_test, p_te)
        gen_score = rmse_te + GEN_GAP_WEIGHT * max(gap, 0.0)

        print(f"{n:9s} | MAE:{mae:8.3f} | RMSE:{rmse_te:8.3f} | R²:{r2:7.4f} | Gap:{gap:7.3f} | GenScore:{gen_score:8.3f}")
        stats[n] = {"rmse": rmse_te, "gap": gap, "gen_score": gen_score}

    best_name = min(stats.items(), key=lambda kv: kv[1]["gen_score"])[0]
    pred_map = {n: predict_back(m, X_test) for n, m in models}
    best_pred = pred_map[best_name]

    baseline_pred = np.full_like(y_test, np.median(y_train), dtype=np.float64)
    baseline_rmse = rmse(y_test, baseline_pred)
    print(f"Baseline(zone median) RMSE: {baseline_rmse:.3f}")

    plot_path = PLOTS_DIR / f"actual_vs_pred_{safe_zone}_{best_name}.png"
    make_actual_vs_pred_plot(y_test, best_pred, zone, best_name, plot_path)

    joblib.dump(xgb_model, MODEL_SAVE_DIR / f"xgb_{safe_zone}.joblib")
    joblib.dump(cat_model, MODEL_SAVE_DIR / f"catboost_{safe_zone}.joblib")
    joblib.dump(lgb_model, MODEL_SAVE_DIR / f"lightgbm_{safe_zone}.joblib")
    joblib.dump(stack_model, MODEL_SAVE_DIR / f"stacking_{safe_zone}.joblib")
    joblib.dump(encoders, MODEL_SAVE_DIR / f"encoders_{safe_zone}.joblib")
    joblib.dump(features, MODEL_SAVE_DIR / f"features_{safe_zone}.joblib")
    joblib.dump(
        {
            "feature_mode": FEATURE_MODE,
            "use_log_target": USE_LOG_TARGET,
            "use_monotonic_constraints": USE_MONOTONIC_CONSTRAINTS,
            "monotone_map": monotone_map,
            "xgb_monotone_constraints": xgb_mono if USE_MONOTONIC_CONSTRAINTS else None,
            "lgb_monotone_constraints": lgb_mono if USE_MONOTONIC_CONSTRAINTS else None,
            "xgb_best_params": xgb_best,
            "cat_best_params": cat_best,
            "lgb_best_params": lgb_best,
            "xgb_cv_rmse": xgb_cv,
            "cat_cv_rmse": cat_cv,
            "lgb_cv_rmse": lgb_cv,
            "selected_model": best_name,
            "baseline_rmse": baseline_rmse,
            "model_stats": stats,
        },
        MODEL_SAVE_DIR / f"tuning_summary_{safe_zone}.joblib",
    )

    saved[zone] = len(d)
    print(f"Saved artifacts for zone '{zone}' -> best: {best_name}")

print("\n=== COMSTOCK TRAINING COMPLETE ===")
print(f"Output folder: {MODEL_SAVE_DIR}")
print(f"Plots folder : {PLOTS_DIR}")
print(f"Climate zones trained: {len(saved)}")
print("Target: specific_heat_demand_kwh_m2 = heating_energy_kwh / floor_area_m2")
print(f"Feature mode used: {FEATURE_MODE}")
print(f"USE_LOG_TARGET: {USE_LOG_TARGET}")
print(f"USE_MONOTONIC_CONSTRAINTS: {USE_MONOTONIC_CONSTRAINTS}")