import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostRegressor
except ImportError:
    print("CatBoost not installed; skipping. Install with: pip install catboost")
    CatBoostRegressor = None
import time
import gc
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────
BDG2_FOLDER    = Path("A:/College/thesis/Datasets/archive")
ASHRAE_FOLDER  = Path("A:/College/thesis/Datasets/ashrae")
MODELS_FOLDER  = Path("A:/College/thesis/Models/dataset_archive")

os.makedirs(MODELS_FOLDER, exist_ok=True)

RANDOM_STATE  = 42

np.random.seed(RANDOM_STATE)

overall_start = time.time()
print("Loading and processing BDG2 and ASHRAE data...")

# ─── Load BDG2 Data ──────────────────────────────────────────────
load_start = time.time()
meta    = pd.read_csv(BDG2_FOLDER / "metadata.csv")
weather = pd.read_csv(BDG2_FOLDER / "weather.csv", parse_dates=["timestamp"])

# Optimize dtypes early
meta = meta.astype({col: 'int32' for col in meta.select_dtypes(include='int64').columns})
weather = weather.astype({col: 'float32' for col in weather.select_dtypes(include='float64').columns})
weather = weather.astype({col: 'int32' for col in weather.select_dtypes(include='int64').columns})

print(f"BDG2 metadata shape: {meta.shape}")
print(f"BDG2 weather shape: {weather.shape}")

# Process hotwater
hotwater_wide = pd.read_csv(BDG2_FOLDER / "hotwater_cleaned.csv", parse_dates=["timestamp"])
hotwater = pd.melt(hotwater_wide, id_vars=["timestamp"], var_name="building_id", value_name="value")
hotwater["meter_type"] = "hotwater"
del hotwater_wide
gc.collect()

# Process steam
steam_wide = pd.read_csv(BDG2_FOLDER / "steam_cleaned.csv", parse_dates=["timestamp"])
steam = pd.melt(steam_wide, id_vars=["timestamp"], var_name="building_id", value_name="value")
steam["meter_type"] = "steam"
del steam_wide
gc.collect()

# Process gas
gas_wide = pd.read_csv(BDG2_FOLDER / "gas_cleaned.csv", parse_dates=["timestamp"])
gas = pd.melt(gas_wide, id_vars=["timestamp"], var_name="building_id", value_name="value")
gas["meter_type"] = "gas"
del gas_wide
gc.collect()

heating_bdg2 = pd.concat([hotwater, steam, gas], ignore_index=True)
heating_bdg2 = heating_bdg2[heating_bdg2["value"] > 0].dropna(subset=["value"])
print(f"BDG2 heating data shape after melt and filter: {heating_bdg2.shape}")
del hotwater, steam, gas
gc.collect()

meta_cols = ["building_id", "site_id", "primaryspaceusage", "sqm", "yearbuilt", "numberoffloors", "heatingtype"]
df_bdg2 = heating_bdg2.merge(meta[meta_cols], on="building_id", how="left")
df_bdg2 = df_bdg2.merge(weather, on=["timestamp", "site_id"], how="left")
df_bdg2["dataset"] = "BDG2"

print(f"BDG2 merged data shape: {df_bdg2.shape}")

# Memory optimization
df_bdg2 = df_bdg2.astype({col: 'float32' for col in df_bdg2.select_dtypes(include='float64').columns})
df_bdg2 = df_bdg2.astype({col: 'int32' for col in df_bdg2.select_dtypes(include='int64').columns})
del heating_bdg2, meta, weather
gc.collect()
bdg2_time = time.time() - load_start
print(f"BDG2 loading and processing took {bdg2_time:.2f}s")

# ─── Load ASHRAE Data ────────────────────────────────────────────
ashrae_start = time.time()
train_ashrae     = pd.read_csv(ASHRAE_FOLDER / "train.csv", parse_dates=["timestamp"])
meta_ashrae      = pd.read_csv(ASHRAE_FOLDER / "building_metadata.csv")
weather_ashrae   = pd.read_csv(ASHRAE_FOLDER / "weather_train.csv", parse_dates=["timestamp"])

print(f"ASHRAE train shape: {train_ashrae.shape}")
print(f"ASHRAE meta shape: {meta_ashrae.shape}")
print(f"ASHRAE weather shape: {weather_ashrae.shape}")

# Optimize dtypes early
train_ashrae = train_ashrae.astype({col: 'float32' for col in train_ashrae.select_dtypes(include='float64').columns})
train_ashrae = train_ashrae.astype({col: 'int32' for col in train_ashrae.select_dtypes(include='int64').columns})
meta_ashrae = meta_ashrae.astype({col: 'float32' for col in meta_ashrae.select_dtypes(include='float64').columns})
meta_ashrae = meta_ashrae.astype({col: 'int32' for col in meta_ashrae.select_dtypes(include='int64').columns})
weather_ashrae = weather_ashrae.astype({col: 'float32' for col in weather_ashrae.select_dtypes(include='float64').columns})
weather_ashrae = weather_ashrae.astype({col: 'int32' for col in weather_ashrae.select_dtypes(include='int64').columns})
gc.collect()

# Rename columns to match BDG2 and convert sq ft to sq m
meta_ashrae = meta_ashrae.rename(columns={
    "primary_use": "primaryspaceusage",
    "year_built": "yearbuilt",
    "floor_count": "numberoffloors"
})
meta_ashrae["sqm"] = meta_ashrae["square_feet"] * 0.092903  # Convert sq ft to sq m
meta_ashrae["heatingtype"] = "Unknown"

# Rename weather columns to match BDG2
weather_ashrae = weather_ashrae.rename(columns={
    "air_temperature": "airTemperature",
    "cloud_coverage": "cloudCoverage",
    "dew_temperature": "dewTemperature",
    "precip_depth_1_hr": "precipDepth",
    "sea_level_pressure": "seaLevelPressure",
    "wind_direction": "windDirection",
    "wind_speed": "windSpeed"
})

train_ashrae = train_ashrae.rename(columns={"meter_reading": "value"})
train_ashrae["meter_type"] = train_ashrae["meter"].map({0: "electricity", 1: "chilledwater", 2: "steam", 3: "hotwater"})
train_ashrae = train_ashrae.drop(columns=["meter"])

# Optimized merge
df_ashrae = train_ashrae.merge(meta_ashrae.set_index("building_id"), left_on="building_id", right_index=True, how="left")
df_ashrae = df_ashrae.merge(weather_ashrae.set_index(["timestamp", "site_id"]), left_on=["timestamp", "site_id"], right_index=True, how="left")
df_ashrae["dataset"] = "ASHRAE"
df_ashrae = df_ashrae[df_ashrae["value"] > 0].dropna(subset=["value"])

print(f"ASHRAE merged data shape: {df_ashrae.shape}")

del train_ashrae, meta_ashrae, weather_ashrae
gc.collect()
ashrae_time = time.time() - ashrae_start
print(f"ASHRAE loading and processing took {ashrae_time:.2f}s")

# ─── Combine Datasets ────────────────────────────────────────────
combine_start = time.time()
df = pd.concat([df_bdg2, df_ashrae], ignore_index=True, copy=False)
print(f"Combined data shape: {df.shape}")
del df_bdg2, df_ashrae
gc.collect()
combine_time = time.time() - combine_start
print(f"Combining datasets took {combine_time:.2f}s")

# ─── Convert to consistent energy units (kWh) ────────────────────
convert_start = time.time()
df["value_kwh"] = df.apply(
    lambda row: row["value"] * (0.293 if row["meter_type"] in ["hotwater", "gas"] else 
                                0.349 if row["meter_type"] == "steam" else 
                                1.0),
    axis=1
)
convert_time = time.time() - convert_start
print(f"Energy unit conversion took {convert_time:.2f}s")

# ─── Target (hourly specific heating demand in kWh/m²/hour) ──────
target_start = time.time()
df["sqm"] = df["sqm"].clip(lower=1)
df["specific_heat"] = df["value_kwh"] / df["sqm"]  # Hourly specific heat

epsilon = 0.01
df["log_specific_heat"] = np.log(df["specific_heat"] + epsilon)

# ─── Relaxed outlier removal ──────────────────────────────────────
upper = df["specific_heat"].quantile(0.999)  # Back to 99.9% for hourly (less aggressive)
df = df[df["specific_heat"] <= upper].copy()
print(f"Data shape after outlier removal: {df.shape}")
gc.collect()

print(f"Buildings total: {df['building_id'].nunique()}")

# ─── Enhanced Feature Engineering (Hourly-Level) ─────────────────
feat_start = time.time()
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear

# Cyclical features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

df["yearbuilt"] = df["yearbuilt"].fillna(2000)
df["building_age"] = 2017 - df["yearbuilt"]
df["heatingtype"] = df["heatingtype"].fillna("Unknown")
df["numberoffloors"] = df["numberoffloors"].fillna(df["numberoffloors"].median())

base_temp = 15.5
df["hdd"] = (base_temp - df["airTemperature"]).clip(lower=0)
df["cdd"] = (df["airTemperature"] - base_temp).clip(lower=0)

# Fill missing weather FIRST (before creating derived features)
weather_cols = ["airTemperature", "cloudCoverage", "dewTemperature", "precipDepth", "seaLevelPressure", "windDirection", "windSpeed"]
for col in weather_cols:
    if col in df.columns:
        df[col] = df.groupby("site_id")[col].transform(lambda x: x.fillna(x.median()))

# Global fillna for robustness
df["airTemperature"] = df["airTemperature"].fillna(df["airTemperature"].median())
df["dewTemperature"] = df["dewTemperature"].fillna(df["dewTemperature"].median())
df["windSpeed"] = df["windSpeed"].fillna(df["windSpeed"].median())

# Now create derived features
df["temp_dew_diff"] = df["airTemperature"] - df["dewTemperature"]
df["wind_chill"] = 13.12 + 0.6215 * df["airTemperature"] - 11.37 * (df["windSpeed"] ** 0.16) + 0.3965 * df["airTemperature"] * (df["windSpeed"] ** 0.16)

# Fill derived
derived_cols = ["temp_dew_diff", "wind_chill"]
for col in derived_cols:
    if col in df.columns:
        df[col] = df.groupby("site_id")[col].transform(lambda x: x.fillna(x.median()))

df["sqm"] = df["sqm"].fillna(df["sqm"].median())
df["building_age"] = df["building_age"].fillna(df["building_age"].median())

# Polynomial features (after filling)
df = df.dropna(subset=["airTemperature", "hdd"]).copy()  # Ensure no NaNs
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[["airTemperature", "hdd"]])
poly_cols = [f"poly_{i}" for i in range(poly_features.shape[1])]
df_poly = pd.DataFrame(poly_features, columns=poly_cols, index=df.index)
df = pd.concat([df, df_poly], axis=1)

# Hourly lags and rolling (adjusted for hourly)
df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
for lag in [24, 48, 72, 168]:  # 1h, 2h, 3h, 1 week (168h) lags
    df[f"lag_{lag}h"] = df.groupby("building_id")["log_specific_heat"].shift(lag)
    df[f"lag_{lag}h"] = df[f"lag_{lag}h"].fillna(df[f"lag_{lag}h"].median())

df["rolling_mean_24h"] = df.groupby("building_id")["log_specific_heat"].rolling(24, min_periods=1).mean().reset_index(0, drop=True)
df["rolling_std_24h"] = df.groupby("building_id")["log_specific_heat"].rolling(24, min_periods=1).std().reset_index(0, drop=True)

gc.collect()

cat_cols = ["primaryspaceusage", "heatingtype", "meter_type"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

base_features = [
    "airTemperature", "hour_sin", "hour_cos", "month_sin", "month_cos", "day_sin", "day_cos",
    "hdd", "cdd", "building_age", "numberoffloors", "sqm",
    "temp_dew_diff", "wind_chill",
    "rolling_mean_24h", "rolling_std_24h"
] + [f"lag_{lag}h" for lag in [24,48,72,168]] + poly_cols

dummy_features = [c for c in df.columns if any(c.startswith(p + '_') for p in cat_cols) or c in (weather_cols + derived_cols)]

feature_cols = list(dict.fromkeys(base_features + dummy_features))

target_col = "log_specific_heat"

df = df.dropna(subset=feature_cols + [target_col]).copy()
print(f"Final data shape after feature engineering and dropna: {df.shape}")
gc.collect()
feat_time = time.time() - feat_start
print(f"Feature engineering took {feat_time:.2f}s")

# ─── Evaluation Function ──────────────────────────────────────────
def evaluate_daily(y_true_daily, y_pred_daily):
    mae = mean_absolute_error(y_true_daily, y_pred_daily)
    rmse = np.sqrt(mean_squared_error(y_true_daily, y_pred_daily))
    r2 = r2_score(y_true_daily, y_pred_daily)
    cvrmse = rmse / y_true_daily.mean() * 100
    return mae, rmse, r2, cvrmse

# ─── Per-Site Modeling with Tuning and Advanced Ensemble ─────────
train_start = time.time()
sites = df["site_id"].unique()
results = {}
predictions = {}

print(f"Total unique sites: {len(sites)}")
print(f"Unique site IDs: {sorted(sites)}")

total_models_trained = 0

for site in sites:
    site_start = time.time()
    print(f"\nTraining models for site {site} ...")
    site_data = df[df["site_id"] == site]
    print(f"  Site {site} data shape: {site_data.shape}")
    print(f"  Unique buildings in site {site}: {site_data['building_id'].nunique()}")
    if len(site_data) < 1000:  # Threshold for processed data
        print(f"  Skipping site {site}: insufficient data ({len(site_data)} samples)")
        del site_data
        gc.collect()
        continue
    
    # Updated split: Train Jan-Jul (months <8), Test Aug-Dec (months >=8) to reduce seasonal bias
    train_site = site_data[site_data["timestamp"].dt.month < 8]
    test_site = site_data[site_data["timestamp"].dt.month >= 8]
    
    print(f"  Train samples: {len(train_site)}, Test samples: {len(test_site)}")
    print(f"  Train unique buildings: {train_site['building_id'].nunique()}, Test unique buildings: {test_site['building_id'].nunique()}")
    if len(test_site) == 0:
        print(f"  Skipping site {site}: no test data")
        del site_data, train_site, test_site
        gc.collect()
        continue
    
    # Scale features (cast to float32 after scaling)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_site[feature_cols]).astype(np.float32)
    test_features_scaled = scaler.transform(test_site[feature_cols]).astype(np.float32)
    
    # Time-series CV with 3 splits (no purge/embargo built-in, but temporal order maintained)
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Hyperparameter-tuned models with time-series CV
    xgb_param_grid = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [6, 8, 10],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_lambda': [1.0, 1.5],
        'reg_alpha': [0.0, 0.1]
    }
    
    lgb_param_grid = {
        'n_estimators': [200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [8, 10, 12],
        'num_leaves': [50, 70, 100],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_lambda': [1.0, 1.5],
        'reg_alpha': [0.0, 0.1]
    }
    
    models_site = {}
    
    # Tune XGBoost with time-series CV
    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, verbosity=0),
        xgb_param_grid, n_iter=5, cv=tscv, scoring='neg_mean_squared_error', random_state=RANDOM_STATE, n_jobs=1
    )
    xgb_search.fit(train_features_scaled, train_site[target_col])
    models_site["XGBoost"] = xgb_search.best_estimator_
    del xgb_search
    gc.collect()
    
    # Tune LightGBM with time-series CV
    lgb_search = RandomizedSearchCV(
        lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
        lgb_param_grid, n_iter=5, cv=tscv, scoring='neg_mean_squared_error', random_state=RANDOM_STATE, n_jobs=1
    )
    lgb_search.fit(train_features_scaled, train_site[target_col])
    models_site["LightGBM"] = lgb_search.best_estimator_
    del lgb_search
    gc.collect()
    
    if CatBoostRegressor:
        cat_param_grid = {
            'iterations': [200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [6, 8, 10],
            'l2_leaf_reg': [1.0, 1.5, 3.0]
        }
        cat_search = RandomizedSearchCV(
            CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
            cat_param_grid, n_iter=5, cv=tscv, scoring='neg_mean_squared_error', random_state=RANDOM_STATE, n_jobs=1
        )
        cat_search.fit(train_features_scaled, train_site[target_col])
        models_site["CatBoost"] = cat_search.best_estimator_
        del cat_search
        gc.collect()
    
    preds_site = {}
    site_models_trained = 0
    for name, model in models_site.items():
        preds_site[name] = model.predict(test_features_scaled)
        
        model_path = MODELS_FOLDER / f"heating_model_site{site}_{name}.joblib"
        joblib.dump(model, model_path)
        print(f"    Saved tuned {name} model to {model_path}")
        site_models_trained += 1
        total_models_trained += 1
    
    # Advanced Ensemble: Stacking with Ridge as meta-learner (use k-fold CV for stacking to avoid issues with TSCV)
    if len(models_site) > 1:
        estimators = [(name, model) for name, model in models_site.items()]
        stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge(random_state=RANDOM_STATE), cv=3, n_jobs=1)  # Changed to cv=3
        stacking.fit(train_features_scaled, train_site[target_col])
        preds_site["StackingEnsemble"] = stacking.predict(test_features_scaled)
        
        model_path = MODELS_FOLDER / f"heating_model_site{site}_StackingEnsemble.joblib"
        joblib.dump(stacking, model_path)
        print(f"    Saved stacking ensemble to {model_path}")
        total_models_trained += 1
    else:
        # Fallback to simple average if only one model
        preds_site["Ensemble"] = preds_site[list(models_site.keys())[0]]
        print(f"    Fallback to single model ensemble for site {site}")
    
    # Aggregate predictions to daily for evaluation (FIXED)
    test_site_copy = test_site.copy()
    for name in preds_site:
        test_site_copy[f"pred_{name}"] = preds_site[name]
    
    # Correct daily aggregation: sum hourly specific_heat for actual, sum exp(pred) for predictions
    daily_actual = test_site_copy.groupby(["building_id", test_site_copy["timestamp"].dt.date])["specific_heat"].sum().reset_index(name="actual_daily")
    daily_preds = {}
    for name in preds_site:
        # Sum exp(pred) for daily predicted total
        test_site_copy[f"pred_{name}_exp"] = np.exp(test_site_copy[f"pred_{name}"]) - epsilon
        daily_preds[name] = test_site_copy.groupby(["building_id", test_site_copy["timestamp"].dt.date])[f"pred_{name}_exp"].sum().reset_index(name=f"pred_{name}_daily")
        daily_preds[name] = daily_preds[name].merge(daily_actual, on=["building_id", "timestamp"])
    
    # Store daily results
    results[site] = {}
    for name in daily_preds:
        mae, rmse, r2, cvrmse = evaluate_daily(daily_preds[name]["actual_daily"], daily_preds[name][f"pred_{name}_daily"])
        results[site][name] = {
            "MAE (kWh/m²/day)": mae, "RMSE (kWh/m²/day)": rmse,
            "R²": r2, "CV-RMSE (%)": cvrmse,
        }
        print(f"    {name} daily metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, CV-RMSE={cvrmse:.4f}%")
    
    predictions[site] = daily_preds  # Store daily predictions
    site_time = time.time() - site_start
    print(f"  Site {site} completed in {site_time:.2f}s (trained {site_models_trained + (1 if 'StackingEnsemble' in preds_site else 0)} models)")
    # Aggressive cleanup to prevent leaks
    del site_data, train_site, test_site, test_site_copy, scaler, train_features_scaled, test_features_scaled, preds_site, daily_actual, daily_preds, results[site], predictions[site]
    gc.collect()

train_time = time.time() - train_start
print(f"Total training time: {train_time:.2f}s")
print(f"Total models trained: {total_models_trained}")
print(f"Models saved: {len(list(MODELS_FOLDER.glob('*.joblib')))}")
print(f"\nTrained models for {len(results)} sites out of {len(sites)} total sites.")

# ─── Print Detailed Results for Each Model ─────────────────────────
print("\nDetailed Results per Site and Model:")
for site in results:
    print(f"\nSite {site}:")
    for model in results[site]:
        print(f"  {model}: {results[site][model]}")

# ─── Aggregate Results Across Sites ───────────────────────────────
all_mae = []
all_rmse = []
all_r2 = []
all_cvrmse = []
for site in results:
    for model in results[site]:
        all_mae.append(results[site][model]["MAE (kWh/m²/day)"])
        all_rmse.append(results[site][model]["RMSE (kWh/m²/day)"])
        all_r2.append(results[site][model]["R²"])
        all_cvrmse.append(results[site][model]["CV-RMSE (%)"])

if not all_mae:
    print("No sites were trained; no metrics to aggregate.")
else:
    print("\n" + "═"*110)
    print("Aggregated Performance – SPECIFIC HEATING DEMAND (kWh/m²/day) — Hourly Models, Daily Evaluation (Jan-Jul Train, Aug-Dec Test)")
    print("═"*110)
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("─"*110)
    print(f"{'MAE (kWh/m²/day)':<20} {np.mean(all_mae):10.4f} {np.std(all_mae):10.4f} {np.min(all_mae):10.4f} {np.max(all_mae):10.4f}")
    print(f"{'RMSE (kWh/m²/day)':<20} {np.mean(all_rmse):10.4f} {np.std(all_rmse):10.4f} {np.min(all_rmse):10.4f} {np.max(all_rmse):10.4f}")
    print(f"{'R²':<20} {np.mean(all_r2):10.4f} {np.std(all_r2):10.4f} {np.min(all_r2):10.4f} {np.max(all_r2):10.4f}")
    print(f"{'CV-RMSE (%)':<20} {np.mean(all_cvrmse):10.4f} {np.std(all_cvrmse):10.4f} {np.min(all_cvrmse):10.4f} {np.max(all_cvrmse):10.4f}")
    print("═"*110)

total_time = time.time() - overall_start
print(f"\nSummary:")
print(f" • Total runtime:   {total_time:.1f} seconds")
print(f" • Models saved in: {MODELS_FOLDER}")
print(" • Hourly modeling with daily aggregation for evaluation.")

# Plot example from a site
example_site = 0
if example_site in predictions:
    daily_example = predictions[example_site]["StackingEnsemble"]
    ex_building = daily_example["building_id"].sample(1).iloc[0]
    mask = daily_example["building_id"] == ex_building
    actual = daily_example.loc[mask, "actual_daily"]
    pred = daily_example.loc[mask, "pred_StackingEnsemble_daily"]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily_example.loc[mask, "timestamp"], actual, label="Actual", color="#1f77b4", lw=1.3)
    ax.plot(daily_example.loc[mask, "timestamp"], pred, label="Predicted (Stacking Ensemble)", color="#d62728", lw=1.2)
    ax.set_title(f"Example Building — {ex_building} (Site {example_site}) - Daily Aggregated")
    ax.set_ylabel("kWh / m² / day")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.xticks(rotation=45)
    plt.show()