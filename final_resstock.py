import json
import re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.base import clone

import optuna
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor


# ========================= CONFIG =========================
DATA_PATH = r"A:\College\Thesis\SHD\USstock\baseline.parquet"
DATA_DICTIONARY_PATH = r"A:\College\Thesis\SHD\USstock\data_dictionary.tsv"
MODEL_DIR = Path(r"A:\College\Thesis\Models\resstock_rev6_zone_and_global_tsv_driven")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5
MIN_SAMPLES_PER_ZONE = 500

N_TRIALS_XGB = 20
N_TRIALS_CAT = 20
N_TRIALS_LGB = 20

USE_LOG_TARGET = True
GEN_GAP_WEIGHT = 0.5
USE_MONOTONIC_CONSTRAINTS = True

# feature controls
MAX_MISSING_RATIO = 0.98     # drop columns with >98% missing
MAX_CARDINALITY = 300        # drop very high-cardinality categorical columns
INCLUDE_ID_LIKE = False      # usually False (prevents leakage-ish IDs)
# ==========================================================


# ---------------- Utility ----------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def parse_num_from_text(v):
    if pd.isna(v):
        return np.nan
    nums = re.findall(r"[-+]?\d*\.?\d+", str(v))
    return float(nums[0]) if nums else np.nan


def add_missing_flags(df, cols):
    for c in cols:
        if c in df.columns:
            df[f"is_missing__{c}"] = df[c].isna().astype(int)
    return df


def fit_label_encoders(X_df: pd.DataFrame):
    X_df = X_df.copy()
    enc = {}
    for c in X_df.columns:
        s = X_df[c].astype("string").fillna("__MISSING__").astype(str)
        n = pd.to_numeric(s, errors="coerce")
        if n.notna().all():
            X_df[c] = n.astype(np.float64)
        else:
            le = LabelEncoder()
            X_df[c] = le.fit_transform(s).astype(np.float64)
            enc[c] = le
    return X_df, enc


def transform_with_encoders(X_df: pd.DataFrame, encoders: dict):
    X_df = X_df.copy()
    for c in X_df.columns:
        s = X_df[c].astype("string").fillna("__MISSING__").astype(str)
        if c in encoders:
            le = encoders[c]
            cls = set(le.classes_)
            vals = []
            for v in s.values:
                if v in cls:
                    vals.append(le.transform([v])[0])
                elif "__MISSING__" in cls:
                    vals.append(le.transform(["__MISSING__"])[0])
                else:
                    vals.append(0)
            X_df[c] = np.array(vals, dtype=np.float64)
        else:
            X_df[c] = pd.to_numeric(s, errors="coerce").fillna(0).astype(np.float64)
    return X_df


def crossval_rmse(builder, X, y):
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, va in kf.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        yfit = np.log1p(ytr) if USE_LOG_TARGET else ytr
        m = builder()
        m.fit(Xtr, yfit)
        p = m.predict(Xva)
        if USE_LOG_TARGET:
            p = np.expm1(p)
            p = np.clip(p, 0, None)
        scores.append(rmse(yva, p))
    return float(np.mean(scores))


def predict_back(model, X):
    p = model.predict(X)
    if USE_LOG_TARGET:
        p = np.expm1(p)
        p = np.clip(p, 0, None)
    return p


def evaluate(models, Xtr, ytr, Xte, yte):
    stats = {}
    for n, m in models.items():
        p_tr = predict_back(m, Xtr)
        p_te = predict_back(m, Xte)

        rmse_tr = rmse(ytr, p_tr)
        rmse_te = rmse(yte, p_te)
        gap = rmse_te - rmse_tr

        stats[n] = {
            "mae": float(mean_absolute_error(yte, p_te)),
            "rmse": float(rmse_te),
            "r2": float(r2_score(yte, p_te)),
            "gap": float(gap),
            "gen_score": float(rmse_te + GEN_GAP_WEIGHT * max(gap, 0.0)),
        }
    return stats


def best_by_gen(stats):
    return min(stats.items(), key=lambda kv: kv[1]["gen_score"])[0]


def build_monotone_vectors(features):
    mono = {f: 0 for f in features}
    for f in [
        "in.sqft",
        "floor_area_m2",
        "in.weekday_operating_hours_num",
        "in.weekend_operating_hours_num",
        "operating_hours_week",
    ]:
        if f in mono:
            mono[f] = 1
    return mono, tuple(mono[f] for f in features), [mono[f] for f in features]


# ---------------- Schema helpers ----------------
def load_dictionary(tsv_path):
    ddf = pd.read_csv(tsv_path, sep="\t", dtype=str)
    required = {"field_name", "field_location"}
    miss = [c for c in required if c not in ddf.columns]
    if miss:
        raise ValueError(f"data_dictionary.tsv missing required column(s): {miss}")
    ddf["field_name"] = ddf["field_name"].astype(str)
    ddf["field_location"] = ddf["field_location"].astype(str)
    return ddf


def detect_heating_columns(df_cols):
    preferred = [
        "out.electricity.heating.energy_consumption.kwh",
        "out.natural_gas.heating.energy_consumption.kwh",
        "out.fuel_oil.heating.energy_consumption.kwh",
        "out.propane.heating.energy_consumption.kwh",
        "out.other_fuel.heating.energy_consumption.kwh",
    ]
    found = [c for c in preferred if c in df_cols]
    if found:
        return found

    discovered = [
        c for c in df_cols
        if c.startswith("out.")
        and ("heating" in c.lower())
        and ("energy_consumption" in c.lower())
    ]
    return discovered


def is_bad_feature_name(col):
    c = col.lower()
    banned_substrings = [
        "bldg_id", "building_id", "job_id",
        "applicability",
        "upgrade",  # target leakage via scenario id
        "weight", "sample_weight"
    ]
    if not INCLUDE_ID_LIKE and any(x in c for x in banned_substrings):
        return True

    # prevent direct target leakage
    leakage = [
        "heating_energy_kwh",
        "specific_heat_demand_kwh_m2",
        "floor_area_m2",
    ]
    if c in leakage:
        return True
    return False


def build_tsv_driven_feature_list(df, ddf, include_climate=True):
    CLIMATE_COL = "in.building_america_climate_zone"

    dict_fields = set(ddf["field_name"].tolist())
    meta_like = ddf["field_location"].str.contains("metadata", case=False, na=False)
    candidates = set(ddf.loc[meta_like, "field_name"].tolist())

    # Keep only fields present in parquet
    candidates = [c for c in candidates if c in df.columns and c in dict_fields]

    # keep "in." primarily; allow selected known "out.params" if present
    keep = []
    for c in candidates:
        if is_bad_feature_name(c):
            continue
        if c.startswith("in."):
            keep.append(c)
        elif c.startswith("out.params."):
            keep.append(c)

    # optional climate inclusion
    if not include_climate and CLIMATE_COL in keep:
        keep.remove(CLIMATE_COL)

    # heuristic pruning: very-missing / huge-cardinality identifiers
    final = []
    n = len(df)
    for c in keep:
        s = df[c]
        miss_ratio = float(s.isna().mean()) if n > 0 else 1.0
        nunq = int(s.astype("string").nunique(dropna=True))

        if miss_ratio > MAX_MISSING_RATIO:
            continue
        if nunq > MAX_CARDINALITY and not c.startswith("in.weather_"):
            # keep some high-card weather strings if needed; otherwise skip
            continue
        final.append(c)

    # must include some key features if present
    must_have = [
        CLIMATE_COL,
        "in.sqft",
        "in.heating_fuel",
        "in.hvac_heating_type",
        "in.hvac_heating_efficiency",
        "in.infiltration",
        "in.insulation_wall",
        "in.insulation_roof",
        "in.windows",
        "in.heating_setpoint",
        "in.geometry_building_type_recs",
        "in.geometry_stories",
        "in.state",
        "in.county_name",
    ]
    for c in must_have:
        if c in df.columns and c not in final:
            if include_climate or c != CLIMATE_COL:
                final.append(c)

    # dedupe preserve order
    final = list(dict.fromkeys(final))
    return final


# ---------------- Model training ----------------
def train_models(Xtr, ytr, features):
    mono_map, xgb_mono, lgb_mono = build_monotone_vectors(features)

    def tune_xgb():
        def obj(t):
            p = dict(
                n_estimators=t.suggest_int("n_estimators", 300, 1100),
                max_depth=t.suggest_int("max_depth", 3, 7),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
                subsample=t.suggest_float("subsample", 0.70, 0.95),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.60, 0.90),
                min_child_weight=t.suggest_float("min_child_weight", 3.0, 35.0),
                gamma=t.suggest_float("gamma", 0.0, 12.0),
                reg_alpha=t.suggest_float("reg_alpha", 1e-3, 25.0, log=True),
                reg_lambda=t.suggest_float("reg_lambda", 1e-2, 55.0, log=True),
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=-1,
            )
            if USE_MONOTONIC_CONSTRAINTS:
                p["monotone_constraints"] = xgb_mono
            return crossval_rmse(lambda: xgb.XGBRegressor(**p), Xtr, ytr)

        s = optuna.create_study(direction="minimize")
        s.optimize(obj, n_trials=N_TRIALS_XGB, show_progress_bar=False)
        return s.best_params

    def tune_cat():
        def obj(t):
            p = dict(
                iterations=t.suggest_int("iterations", 300, 1100),
                depth=t.suggest_int("depth", 4, 8),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
                l2_leaf_reg=t.suggest_float("l2_leaf_reg", 3.0, 55.0),
                random_strength=t.suggest_float("random_strength", 0.5, 10.0),
                bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 6.0),
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=0,
            )
            return crossval_rmse(lambda: CatBoostRegressor(**p), Xtr, ytr)

        s = optuna.create_study(direction="minimize")
        s.optimize(obj, n_trials=N_TRIALS_CAT, show_progress_bar=False)
        return s.best_params

    def tune_lgb():
        def obj(t):
            p = dict(
                n_estimators=t.suggest_int("n_estimators", 300, 1100),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
                num_leaves=t.suggest_int("num_leaves", 16, 90),
                max_depth=t.suggest_int("max_depth", 3, 8),
                min_child_samples=t.suggest_int("min_child_samples", 30, 260),
                subsample=t.suggest_float("subsample", 0.70, 0.95),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.60, 0.90),
                reg_alpha=t.suggest_float("reg_alpha", 1e-3, 25.0, log=True),
                reg_lambda=t.suggest_float("reg_lambda", 1e-2, 55.0, log=True),
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            if USE_MONOTONIC_CONSTRAINTS:
                p["monotone_constraints"] = lgb_mono
            return crossval_rmse(lambda: LGBMRegressor(**p), Xtr, ytr)

        s = optuna.create_study(direction="minimize")
        s.optimize(obj, n_trials=N_TRIALS_LGB, show_progress_bar=False)
        return s.best_params

    xgb_best = tune_xgb()
    cat_best = tune_cat()
    lgb_best = tune_lgb()

    if USE_MONOTONIC_CONSTRAINTS:
        xgb_best["monotone_constraints"] = xgb_mono
        lgb_best["monotone_constraints"] = lgb_mono

    xgb_m = xgb.XGBRegressor(**xgb_best, random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1)
    cat_m = CatBoostRegressor(**cat_best, random_seed=RANDOM_STATE, verbose=0)
    lgb_m = LGBMRegressor(**lgb_best, random_state=RANDOM_STATE, n_jobs=-1)

    yfit = np.log1p(ytr) if USE_LOG_TARGET else ytr
    xgb_m.fit(Xtr, yfit)
    cat_m.fit(Xtr, yfit, verbose=False)
    lgb_m.fit(Xtr, yfit)

    stack_m = StackingRegressor(
        estimators=[("xgb", clone(xgb_m)), ("cat", clone(cat_m)), ("lgb", clone(lgb_m))],
        final_estimator=Ridge(alpha=2.0),
        cv=5,
        n_jobs=-1
    )
    stack_m.fit(Xtr, yfit)

    return {
        "XGBoost": xgb_m,
        "CatBoost": cat_m,
        "LightGBM": lgb_m,
        "Stacking": stack_m
    }


def prepare_dataset():
    df = pd.read_parquet(DATA_PATH).convert_dtypes()
    ddf = load_dictionary(DATA_DICTIONARY_PATH)

    CLIMATE_COL = "in.building_america_climate_zone"
    if CLIMATE_COL not in df.columns:
        raise ValueError(f"Missing required climate column: {CLIMATE_COL}")

    # ----- target -----
    heating_cols = detect_heating_columns(set(df.columns))
    print("Detected heating columns:", heating_cols)
    if len(heating_cols) == 0:
        raise ValueError("No heating energy consumption columns detected in parquet.")

    df["heating_energy_kwh"] = 0.0
    for c in heating_cols:
        df["heating_energy_kwh"] += safe_numeric(df[c]).fillna(0.0)

    if "in.sqft" not in df.columns:
        raise ValueError("Missing in.sqft in parquet.")
    df["in.sqft"] = safe_numeric(df["in.sqft"])

    # ----- filtering -----
    print("Rows raw:", len(df))
    df = df.dropna(subset=["in.sqft", CLIMATE_COL]).copy()
    print("After sqft+climate dropna:", len(df))
    df = df[(df["in.sqft"] > 0) & (df["heating_energy_kwh"] >= 0)].copy()
    print("After positive sqft/heating filter:", len(df))

    if "upgrade" in df.columns:
        u_num = pd.to_numeric(df["upgrade"], errors="coerce")
        u_str = df["upgrade"].astype("string").str.strip().str.lower()
        is_base = (
            (u_num == 0).fillna(False)
            | u_str.isin(["0", "00", "baseline", "base", "upgrade00"]).fillna(False)
            | u_str.str.fullmatch(r"0+").fillna(False)
        )
        base_count = int(is_base.sum())
        print("Baseline rows detected:", base_count)
        if base_count >= 1000:
            df = df[is_base].copy()
            print("After baseline filter:", len(df))
        else:
            print("Baseline filter skipped (too few baseline rows).")

    df["floor_area_m2"] = df["in.sqft"] * 0.09290304
    df["specific_heat_demand_kwh_m2"] = df["heating_energy_kwh"] / df["floor_area_m2"]
    df = df[np.isfinite(df["specific_heat_demand_kwh_m2"])].copy()
    df = df[df["specific_heat_demand_kwh_m2"] > 0].copy()
    print("After SHD validity filter:", len(df))

    if len(df) >= 500:
        q1 = df["specific_heat_demand_kwh_m2"].quantile(0.005)
        q2 = df["specific_heat_demand_kwh_m2"].quantile(0.995)
        df = df[(df["specific_heat_demand_kwh_m2"] >= q1) & (df["specific_heat_demand_kwh_m2"] <= q2)].copy()
        print("After SHD outlier trim:", len(df))

    if len(df) < 100:
        raise ValueError(f"Too few rows after filtering ({len(df)}). Detected heating columns: {heating_cols}")

    # ----- engineered -----
    if "in.weekday_operating_hours" in df.columns:
        df["in.weekday_operating_hours_num"] = df["in.weekday_operating_hours"].apply(parse_num_from_text)
    if "in.weekend_operating_hours" in df.columns:
        df["in.weekend_operating_hours_num"] = df["in.weekend_operating_hours"].apply(parse_num_from_text)

    if {"in.weekday_operating_hours_num", "in.weekend_operating_hours_num"}.issubset(df.columns):
        df["operating_hours_week"] = (
            5 * safe_numeric(df["in.weekday_operating_hours_num"])
            + 2 * safe_numeric(df["in.weekend_operating_hours_num"])
        )

    # add missing flags for candidate in.* columns (after tsv-driven selection we’ll keep relevant)
    return df, ddf, heating_cols


def run_training_block(df_block, features, tag):
    X_raw = df_block[features].copy()
    y = df_block["specific_heat_demand_kwh_m2"].to_numpy(dtype=np.float64)

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_STATE
    )

    Xtr_df, enc = fit_label_encoders(Xtr_raw)
    Xte_df = transform_with_encoders(Xte_raw, enc)

    Xtr = Xtr_df.to_numpy(np.float64)
    Xte = Xte_df.to_numpy(np.float64)

    models = train_models(Xtr, ytr, features)
    stats = evaluate(models, Xtr, ytr, Xte, yte)
    best = best_by_gen(stats)

    print(f"\n--- {tag} ---")
    for n, s in stats.items():
        print(f"{n:9s} | MAE:{s['mae']:8.3f} | RMSE:{s['rmse']:8.3f} | R²:{s['r2']:7.4f} | Gap:{s['gap']:7.3f} | GenScore:{s['gen_score']:8.3f}")
    print("Best:", best)

    return models, enc, stats, best


def save_bundle(out_dir, prefix, models, enc, features, stats, best, extra):
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(models["XGBoost"], out_dir / f"xgb_{prefix}.joblib")
    joblib.dump(models["CatBoost"], out_dir / f"catboost_{prefix}.joblib")
    joblib.dump(models["LightGBM"], out_dir / f"lightgbm_{prefix}.joblib")
    joblib.dump(models["Stacking"], out_dir / f"stacking_{prefix}.joblib")
    joblib.dump(enc, out_dir / f"encoders_{prefix}.joblib")
    joblib.dump(features, out_dir / f"features_{prefix}.joblib")

    summary = {
        "selected_model": best,
        "model_stats": stats,
        "use_log_target": USE_LOG_TARGET,
        "use_monotonic_constraints": USE_MONOTONIC_CONSTRAINTS,
    }
    summary.update(extra)

    with open(out_dir / f"summary_{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    CLIMATE_COL = "in.building_america_climate_zone"

    df, ddf, heating_cols = prepare_dataset()

    # TSV-driven features (global includes climate)
    base_features_global = build_tsv_driven_feature_list(df, ddf, include_climate=True)

    # missing flags for selected features
    df = add_missing_flags(df, base_features_global)

    engineered = [c for c in ["floor_area_m2", "in.weekday_operating_hours_num", "in.weekend_operating_hours_num", "operating_hours_week"] if c in df.columns]
    engineered += [f"is_missing__{c}" for c in base_features_global if f"is_missing__{c}" in df.columns]

    global_features = list(dict.fromkeys(base_features_global + engineered))

    if len(global_features) == 0:
        raise ValueError("No usable global features found.")

    df = df.dropna(subset=["specific_heat_demand_kwh_m2", CLIMATE_COL]).copy()
    df[global_features] = df[global_features].astype("string").fillna("__MISSING__")

    print("\nRows ready:", len(df))
    print("Global feature count:", len(global_features))
    print("First 30 global features:", global_features[:30])

    # ===== Global model =====
    models_g, enc_g, stats_g, best_g = run_training_block(df, global_features, tag="GLOBAL MODEL")
    save_bundle(
        MODEL_DIR,
        "global",
        models_g,
        enc_g,
        global_features,
        stats_g,
        best_g,
        extra={
            "rows": int(len(df)),
            "detected_heating_columns": heating_cols,
            "feature_mode": "resstock_tsv_driven",
            "uses_climate_as_feature": True,
        },
    )

    # ===== Zone-specific models =====
    zones = sorted(df[CLIMATE_COL].astype("string").dropna().unique().tolist())
    zone_count = 0

    for zone in zones:
        d = df[df[CLIMATE_COL].astype("string") == zone].copy()
        if len(d) < MIN_SAMPLES_PER_ZONE:
            print(f"Skipping zone '{zone}' ({len(d)} samples < {MIN_SAMPLES_PER_ZONE})")
            continue

        zone_features = [f for f in global_features if f != CLIMATE_COL]
        d[zone_features] = d[zone_features].astype("string").fillna("__MISSING__")

        safe_zone = str(zone).replace(" ", "_").replace("-", "_").replace("/", "_")
        models_z, enc_z, stats_z, best_z = run_training_block(
            d, zone_features, tag=f"ZONE MODEL: {zone} (n={len(d)})"
        )

        save_bundle(
            MODEL_DIR,
            f"zone_{safe_zone}",
            models_z,
            enc_z,
            zone_features,
            stats_z,
            best_z,
            extra={
                "zone": zone,
                "rows": int(len(d)),
                "detected_heating_columns": heating_cols,
                "feature_mode": "resstock_tsv_driven",
                "uses_climate_as_feature": False,
            },
        )
        zone_count += 1

    print("\n=== RESSTOCK TRAINING COMPLETE ===")
    print(f"Output folder: {MODEL_DIR}")
    print("Global model saved: yes")
    print(f"Zone models trained: {zone_count}")
    print(f"USE_LOG_TARGET: {USE_LOG_TARGET}")
    print(f"USE_MONOTONIC_CONSTRAINTS: {USE_MONOTONIC_CONSTRAINTS}")


if __name__ == "__main__":
    main()