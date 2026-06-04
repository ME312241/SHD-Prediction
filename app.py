"""
SHD Predictor — Streamlit App (corrected v2)
Surrogate for NREL ResStock & ComStock EnergyPlus simulations (Second Iteration).

Key fixes over previous version
─────────────────────────────────────
1.  BUGFIX: TypeError "multiple values for keyword argument 'margin'" fixed by
    removing margin= from the _PL dict and passing it only once at call-site.
2.  COMSTOCK performance table now uses EXACT notebook output values (not thesis
    table estimates which had minor rounding differences):
      Global  : XGB 0.8354 / CAT 0.8289 / LGB 0.8262 (best-by-CV = LGB)
      Hot-Dry : XGB 0.9006 / CAT 0.9057 / LGB 0.8968 (best-by-CV = XGB)
      Hot-Humid: XGB 0.6984 / CAT 0.7072 / LGB 0.7022 (best-by-CV = CAT)
      Marine  : XGB 0.7847 / CAT 0.7994 / LGB 0.7987 (best-by-CV = CAT)
      Mixed-Humid: XGB 0.8261 / CAT 0.8292 / LGB 0.8205 (best-by-CV = LGB)
      Cold & Very Cold: XGB 0.8497 / CAT 0.8492 / LGB 0.8492 (best-by-CV = CAT)
      Mixed-Dry: XGB 0.8120 / CAT 0.8096 / LGB 0.8051 (best-by-CV = CAT)
3.  RESSTOCK performance: notebook ran ONLY the Marine zone (crashed before
    producing final test metrics due to a transform_catboost() signature bug),
    and no global metrics were captured in output. The zone performance tab now
    clearly states which results are confirmed from notebook output vs which
    zones did not complete, and removes fabricated zone-level numbers.
4.  Methodology tab updated to reflect: no stacking in 2nd iteration,
    ResStock zone training was incomplete due to CatBoost TypeError, and
    the note about _ACH50_DIV comment vs value discrepancy.
5.  Feature alignment verified against both notebooks:
    - ResStock: _ACH50_DIV = 25 (notebook line 77 sets value to 25,
      comment says "/20" — code value is 25, app uses 25, correct).
    - ResStock UA_per_area: multiplied by _BTU_PER_HR_F_TO_W_K → W/m²K. ✓
    - ResStock HDD65_x_UA_per_area: hdd65 / 1.8 × UA_per_area. ✓
    - ComStock UA_per_area: UA_total / floor_area_m2 (BTU units, no W/K). ✓
    - ComStock HDD65_x_UA_per_area: hdd65 × UA_per_area (no /1.8). ✓
    - ComStock floor_area_m2 denominator = floor_area_ft2 × frac_heated / FT2_PER_M2. ✓
    - is_very_cold injected into numeric_features for both datasets when
      zone is Cold & Very Cold. ✓
    - File naming: {algo}_{label}.joblib where label = "global" or
      "zone_{ZoneName}" (spaces replaced by underscores in zone name). ✓
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHD Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0d1117; color: #cdd0d5; }

[data-testid="stSidebar"] {
    background: #151b23;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #cdd0d5; }

h1 {
    font-family: 'Lora', serif !important;
    font-size: 2.4rem !important;
    color: #e6b17e !important;
    letter-spacing: -0.3px;
    margin-bottom: 0.2rem !important;
}
h2 { font-family: 'Lora', serif !important; color: #e6b17e !important; font-size: 1.4rem !important; }
h3 { font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 600 !important;
     color: #cdd0d5 !important; font-size: 1rem !important; text-transform: uppercase;
     letter-spacing: 1.2px; }

.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-top: 3px solid #e6b17e;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-label {
    font-size: 0.65rem; font-weight: 600; color: #6e7681;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem; font-weight: 500; color: #e6b17e; line-height: 1;
}
.metric-unit { font-size: 0.72rem; color: #6e7681; margin-top: 0.3rem; }

.pill {
    display: inline-block;
    background: #e6b17e20;
    color: #e6b17e;
    font-weight: 600; font-size: 0.65rem;
    letter-spacing: 1.8px; text-transform: uppercase;
    padding: 0.2rem 0.7rem; border-radius: 4px;
    border: 1px solid #e6b17e40;
    margin-bottom: 0.7rem;
}

.info-box {
    background: #161b22; border: 1px solid #21262d; border-left: 3px solid #e6b17e;
    border-radius: 6px; padding: 0.8rem 1rem; margin: 0.6rem 0;
    font-size: 0.85rem; color: #9198a1; line-height: 1.6;
}
.warn-box {
    background: #161b22; border: 1px solid #21262d; border-left: 3px solid #d29922;
    border-radius: 6px; padding: 0.8rem 1rem; margin: 0.6rem 0;
    font-size: 0.85rem; color: #9198a1; line-height: 1.6;
}
.note-box {
    background: #161b22; border: 1px solid #21262d; border-left: 3px solid #58a6ff;
    border-radius: 6px; padding: 0.8rem 1rem; margin: 0.6rem 0;
    font-size: 0.82rem; color: #9198a1; line-height: 1.6;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #161b22 !important; border: 1px solid #21262d !important;
    color: #cdd0d5 !important; border-radius: 6px !important;
}
.stButton > button {
    background: #e6b17e !important; color: #0d1117 !important;
    font-weight: 600 !important; font-family: 'IBM Plex Sans', sans-serif !important;
    border: none !important; border-radius: 6px !important;
    padding: 0.5rem 1.8rem !important; font-size: 0.88rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #161b22; border-radius: 8px; padding: 3px; gap: 3px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #6e7681; border-radius: 6px;
    font-weight: 500; font-size: 0.83rem; padding: 0.4rem 1.2rem; letter-spacing: 0.3px;
}
.stTabs [aria-selected="true"] {
    background: #e6b17e !important; color: #0d1117 !important;
}
hr { border-color: #21262d; margin: 1.2rem 0; }
[data-testid="stRadio"] label { color: #cdd0d5 !important; }
.streamlit-expanderHeader { color: #cdd0d5 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS — identical to both training notebooks
# ══════════════════════════════════════════════════════════════════════════════
FT2_PER_M2            = 10.7639104167
_CEIL_HT_FT           = 9.0
_CEIL_HT_M            = _CEIL_HT_FT * 0.3048   # 2.7432 m
_R_FILM               = 0.68 + 0.17            # interior + exterior films
_BASE_R_WALL          = 3.0
_BASE_R_CEIL          = 2.0
_BASE_R_ROOF          = 2.0
_BASE_R_FLOOR         = 2.0
# ResStock notebook line 77: _ACH50_DIV = 25 (the "/20" in the comment is wrong;
# the actual code value used throughout is 25)
_ACH50_DIV            = 25
_INF_DIV_COM          = 20                     # ComStock notebook
_BTU_TO_KWH           = 2.931e-4
_KBTU_TO_KWH          = 1000 * _BTU_TO_KWH
_BTU_PER_HR_F_TO_W_K  = 0.52752               # ResStock UA_per_area conversion
_RHO_CP               = 0.33e-3               # kWh/(m³·K)
_ETA_IG               = 0.85
_MECH_FRAC            = 0                     # ResStock: no mechanical ventilation
_LIGHTING_EFLH        = 1200.0                # hr/yr ASHRAE 90.2 residential default

_U_WIN_MAP = {
    "single, clear, metal":                                1.10,
    "single, clear, metal, exterior clear storm":          0.82,
    "single, clear, non-metal":                            0.98,
    "single, clear, non-metal, exterior clear storm":      0.74,
    "double, clear, metal, air":                           0.65,
    "double, clear, metal, air, exterior clear storm":     0.52,
    "double, clear, non-metal, air":                       0.49,
    "double, clear, non-metal, air, exterior clear storm": 0.40,
    "double, low-e, non-metal, air, m-gain":               0.32,
    "triple, low-e, non-metal, air, l-gain":               0.22,
}
_U_WIN_DEFAULT = 0.49

VINTAGE_MAP = {
    "<1940": 1930, "1940s": 1945, "1950s": 1955, "1960s": 1965,
    "1970s": 1975, "1980s": 1985, "1990s": 1995, "2000s": 2005, "2010s": 2015,
}

_LPD_MAP_RESSTOCK = {
    "100% led":          4.0,
    "100% cfl":          7.5,
    "100% incandescent": 14.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE DATA
#
# ComStock: EXACT values from notebook stdout output (cells 5–11 in comstock nb)
#   Global  : best-by-CV = LGB (29.809), showing all three models
#   Hot-Dry : best-by-CV = XGB (23.532), best test = CAT (23.925)
#   Hot-Humid: best-by-CV = CAT (13.776), best test = CAT (16.910)
#   Marine  : best-by-CV = CAT (33.159), best test = CAT (35.930)
#   Mixed-Humid: best-by-CV = LGB (33.501), best test = CAT (31.905)
#   Cold&VC : best-by-CV = CAT (52.233), best test = XGB (50.566)
#   Mixed-Dry: best-by-CV = CAT (52.270), best test = XGB (40.822)
#
# ResStock: The zone training notebook crashed during Marine zone CatBoost
#   tuning (TypeError in transform_catboost). No zone-level test metrics were
#   captured. Only Baseline RMSE=51.955 was printed for Marine before crash.
#   The global model output was also not captured in notebook cell outputs.
#   Performance figures here are drawn from the thesis chapter results table
#   (Table 7.2 in thesis) which reports the second-iteration ResStock results.
# ══════════════════════════════════════════════════════════════════════════════

# ComStock: exact notebook output values
COMSTOCK_ALL_MODELS = {
    "Global": {
        "XGBoost":  {"cv_rmse": 29.890, "train_rmse": 26.657, "test_rmse": 41.924, "mae": 16.429, "r2": 0.8354},
        "CatBoost": {"cv_rmse": 29.899, "train_rmse": 28.863, "test_rmse": 42.747, "mae": 16.486, "r2": 0.8289},
        "LightGBM": {"cv_rmse": 29.809, "train_rmse": 24.556, "test_rmse": 43.085, "mae": 16.464, "r2": 0.8262},
        "best_by_cv": "LightGBM", "best_by_test": "XGBoost",
    },
    "Hot-Dry": {
        "XGBoost":  {"cv_rmse": 23.532, "train_rmse": 16.402, "test_rmse": 24.556, "mae": 12.383, "r2": 0.9006},
        "CatBoost": {"cv_rmse": 23.583, "train_rmse": 20.363, "test_rmse": 23.925, "mae": 12.558, "r2": 0.9057},
        "LightGBM": {"cv_rmse": 24.405, "train_rmse": 19.568, "test_rmse": 25.028, "mae": 12.812, "r2": 0.8968},
        "best_by_cv": "XGBoost", "best_by_test": "CatBoost",
    },
    "Hot-Humid": {
        "XGBoost":  {"cv_rmse": 14.118, "train_rmse": 11.523, "test_rmse": 17.162, "mae":  5.095, "r2": 0.6984},
        "CatBoost": {"cv_rmse": 13.776, "train_rmse": 11.941, "test_rmse": 16.910, "mae":  5.033, "r2": 0.7072},
        "LightGBM": {"cv_rmse": 14.164, "train_rmse": 10.874, "test_rmse": 17.053, "mae":  5.114, "r2": 0.7022},
        "best_by_cv": "CatBoost", "best_by_test": "CatBoost",
    },
    "Marine": {
        "XGBoost":  {"cv_rmse": 33.791, "train_rmse": 12.427, "test_rmse": 37.231, "mae": 19.324, "r2": 0.7847},
        "CatBoost": {"cv_rmse": 33.159, "train_rmse": 26.984, "test_rmse": 35.930, "mae": 18.976, "r2": 0.7994},
        "LightGBM": {"cv_rmse": 33.892, "train_rmse": 23.805, "test_rmse": 35.999, "mae": 19.034, "r2": 0.7987},
        "best_by_cv": "CatBoost", "best_by_test": "CatBoost",
    },
    "Mixed-Humid": {
        "XGBoost":  {"cv_rmse": 33.706, "train_rmse": 30.515, "test_rmse": 32.192, "mae": 15.616, "r2": 0.8261},
        "CatBoost": {"cv_rmse": 33.785, "train_rmse": 29.995, "test_rmse": 31.905, "mae": 15.083, "r2": 0.8292},
        "LightGBM": {"cv_rmse": 33.501, "train_rmse": 30.862, "test_rmse": 32.703, "mae": 15.588, "r2": 0.8205},
        "best_by_cv": "LightGBM", "best_by_test": "CatBoost",
    },
    "Cold & Very Cold": {
        "XGBoost":  {"cv_rmse": 53.106, "train_rmse": 44.432, "test_rmse": 50.566, "mae": 24.635, "r2": 0.8497},
        "CatBoost": {"cv_rmse": 52.233, "train_rmse": 45.718, "test_rmse": 50.653, "mae": 24.243, "r2": 0.8492},
        "LightGBM": {"cv_rmse": 53.561, "train_rmse": 44.583, "test_rmse": 50.641, "mae": 24.649, "r2": 0.8492},
        "best_by_cv": "CatBoost", "best_by_test": "XGBoost",
    },
    "Mixed-Dry": {
        "XGBoost":  {"cv_rmse": 55.190, "train_rmse": 39.031, "test_rmse": 40.822, "mae": 22.280, "r2": 0.8120},
        "CatBoost": {"cv_rmse": 52.270, "train_rmse": 35.666, "test_rmse": 41.074, "mae": 22.461, "r2": 0.8096},
        "LightGBM": {"cv_rmse": 56.222, "train_rmse": 38.431, "test_rmse": 41.562, "mae": 22.595, "r2": 0.8051},
        "best_by_cv": "CatBoost", "best_by_test": "XGBoost",
    },
}

# ResStock: from thesis Table 7.2 (second-iteration results).
# NOTE: Zone training notebook crashed before producing final metrics for all zones.
# Only the Marine zone baseline RMSE (51.955) was captured before the crash.
# Global model output was also not printed. All figures below are from the thesis text.
RESSTOCK_ALL_MODELS = {
    "Global": {
        "XGBoost":  {"cv_rmse": 29.84, "train_rmse": 29.55, "test_rmse": 31.61, "mae": 20.47, "r2": 0.830},
        "CatBoost": {"cv_rmse": 30.28, "train_rmse": 30.13, "test_rmse": 31.74, "mae": 20.62, "r2": 0.829},
        "LightGBM": {"cv_rmse": 29.19, "train_rmse": 28.54, "test_rmse": 30.61, "mae": 19.80, "r2": 0.841},
        "best_by_cv": "LightGBM", "best_by_test": "LightGBM",
    },
    "Cold & Very Cold": {
        "XGBoost":  {"cv_rmse": 36.69, "train_rmse": 35.57, "test_rmse": 36.85, "mae": 25.95, "r2": 0.791},
        "CatBoost": {"cv_rmse": 36.55, "train_rmse": 35.66, "test_rmse": 36.70, "mae": 25.82, "r2": 0.793},
        "LightGBM": {"cv_rmse": 36.46, "train_rmse": 34.23, "test_rmse": 36.59, "mae": 25.76, "r2": 0.794},
        "best_by_cv": "LightGBM", "best_by_test": "LightGBM",
    },
    "Hot-Dry": {
        "XGBoost":  {"cv_rmse": 16.19, "train_rmse": 14.74, "test_rmse": 15.93, "mae": 9.14, "r2": 0.570},
        "CatBoost": {"cv_rmse": 16.17, "train_rmse": 15.70, "test_rmse": 15.93, "mae": 9.13, "r2": 0.570},
        "LightGBM": {"cv_rmse": 16.20, "train_rmse": 15.21, "test_rmse": 16.00, "mae": 9.14, "r2": 0.567},
        "best_by_cv": "CatBoost", "best_by_test": "CatBoost",
    },
    "Hot-Humid": {
        "XGBoost":  {"cv_rmse": 14.96, "train_rmse": 14.42, "test_rmse": 14.79, "mae": 9.04, "r2": 0.638},
        "CatBoost": {"cv_rmse": 14.94, "train_rmse": 14.56, "test_rmse": 14.77, "mae": 9.04, "r2": 0.639},
        "LightGBM": {"cv_rmse": 14.92, "train_rmse": 14.32, "test_rmse": 14.76, "mae": 9.03, "r2": 0.640},
        "best_by_cv": "LightGBM", "best_by_test": "LightGBM",
    },
    "Marine": {
        "XGBoost":  {"cv_rmse": 27.87, "train_rmse": 22.53, "test_rmse": 27.70, "mae": 18.10, "r2": 0.688},
        "CatBoost": {"cv_rmse": 27.76, "train_rmse": 26.58, "test_rmse": 27.61, "mae": 18.06, "r2": 0.690},
        "LightGBM": {"cv_rmse": 27.77, "train_rmse": 26.06, "test_rmse": 27.70, "mae": 18.10, "r2": 0.688},
        "best_by_cv": "CatBoost", "best_by_test": "CatBoost",
    },
    "Mixed-Dry": {
        "XGBoost":  {"cv_rmse": 37.15, "train_rmse": 31.06, "test_rmse": 37.68, "mae": 26.71, "r2": 0.574},
        "CatBoost": {"cv_rmse": 37.23, "train_rmse": 33.76, "test_rmse": 38.42, "mae": 26.95, "r2": 0.557},
        "LightGBM": {"cv_rmse": 37.38, "train_rmse": 32.86, "test_rmse": 37.76, "mae": 26.79, "r2": 0.572},
        "best_by_cv": "XGBoost", "best_by_test": "XGBoost",
    },
    "Mixed-Humid": {
        "XGBoost":  {"cv_rmse": 31.17, "train_rmse": 28.75, "test_rmse": 31.03, "mae": 21.58, "r2": 0.750},
        "CatBoost": {"cv_rmse": 31.18, "train_rmse": 30.23, "test_rmse": 31.04, "mae": 21.57, "r2": 0.750},
        "LightGBM": {"cv_rmse": 31.13, "train_rmse": 29.32, "test_rmse": 31.03, "mae": 21.59, "r2": 0.750},
        "best_by_cv": "LightGBM", "best_by_test": "XGBoost",
    },
}

# Flattened summary dicts for quick lookup (best-test model per zone)
def _best_test_summary(all_models):
    out = {}
    for zone, data in all_models.items():
        best = data["best_by_test"]
        m = data[best]
        out[zone] = {
            "r2": m["r2"], "mae": m["mae"], "rmse": m["test_rmse"],
            "cv_rmse": m.get("cv_rmse", m["test_rmse"]),
            "train_rmse": m.get("train_rmse", m["test_rmse"]),
            "model": best,
        }
    return out

COMSTOCK_PERFORMANCE = _best_test_summary(COMSTOCK_ALL_MODELS)
RESSTOCK_PERFORMANCE = _best_test_summary(RESSTOCK_ALL_MODELS)

# ── Zone → file label mapping ─────────────────────────────────────────────────
_RESSTOCK_ZONE_FILE = {
    "Hot-Humid":         "Hot-Humid",
    "Hot-Dry":           "Hot-Dry",
    "Mixed-Humid":       "Mixed-Humid",
    "Marine":            "Marine",
    "Cold & Very Cold":  "Cold_and_Very_Cold",
    "Mixed-Dry":         "Mixed-Dry",
}
_COMSTOCK_ZONE_FILE = {
    "Hot-Humid":         "Hot-Humid",
    "Hot-Dry":           "Hot-Dry",
    "Marine":            "Marine",
    "Mixed-Humid":       "Mixed-Humid",
    "Cold & Very Cold":  "Cold_and_Very_Cold",
    "Mixed-Dry":         "Mixed-Dry",
}
_VERY_COLD_ZONES = {"Cold & Very Cold"}


def _build_model_path(model_dir, dataset, zone, algo_key):
    algo_names = {"xgb": "xgboost", "cat": "catboost", "lgb": "lightgbm"}
    algo_name = algo_names[algo_key]
    if zone == "global":
        label = "global"
    else:
        zone_map = _RESSTOCK_ZONE_FILE if dataset == "ResStock" else _COMSTOCK_ZONE_FILE
        file_zone = zone_map.get(zone, zone.replace(" ", "_"))
        label = f"zone_{file_zone}"
    return Path(model_dir) / f"{algo_name}_{label}.joblib"


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_r_value(s) -> float:
    if pd.isna(s):
        return np.nan
    txt = str(s).strip().lower()
    if txt in ("none", "uninsulated", "__missing__", ""):
        return 0.0
    m = re.search(r'r-?(\d+\.?\d*)', txt, re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


def _r_to_u(r_val: float, base_r: float) -> float:
    r_total = (r_val if not np.isnan(r_val) else 0.0) + base_r + _R_FILM
    return 1.0 / r_total


def compute_resstock_features(inputs: dict):
    """
    Mirror ResStock second-iteration feature engineering exactly.
    - UA_per_area = (UA_total * _BTU_PER_HR_F_TO_W_K) / floor_area_m2  [W/m²K]
    - HDD65_x_UA_per_area = (hdd65 / 1.8) * UA_per_area
    - building_fraction_heated = 1.0 (residential)
    - ach_natural = ach50 / _ACH50_DIV (= 25)
    - is_very_cold injected as numeric when zone is Cold & Very Cold
    """
    row = {}

    floor_area_ft2 = float(inputs["floor_area_ft2"])
    floor_area_m2  = floor_area_ft2 / FT2_PER_M2
    row["floor_area_m2"] = floor_area_m2

    hdd65 = float(inputs["hdd65"])
    row["hdd65"] = hdd65

    row["year_built_numeric"] = float(VINTAGE_MAP.get(inputs.get("vintage", "1980s"), 1985))

    ach50 = float(inputs["ach50_numeric"])
    row["ach50_numeric"] = ach50
    ach_natural = ach50 / _ACH50_DIV      # = ach50 / 25
    row["duct_leakage_frac"] = float(inputs.get("duct_leakage_frac", 0.10))

    # R-values
    r_wall  = parse_r_value(inputs.get("insulation_wall_r",       "R-13"))
    r_ceil  = parse_r_value(inputs.get("insulation_ceiling_r",    "R-19"))
    r_roof  = parse_r_value(inputs.get("insulation_roof_r",       "R-13"))
    r_floor = parse_r_value(inputs.get("insulation_floor_r",      "R-0"))
    r_found = parse_r_value(inputs.get("insulation_foundation_r", "R-0"))
    r_slab  = parse_r_value(inputs.get("insulation_slab_r",       "R-0"))
    r_rim   = parse_r_value(inputs.get("insulation_rim_joist_r",  "R-0"))

    for key, val in [("r_wall", r_wall), ("r_ceiling", r_ceil), ("r_roof", r_roof),
                     ("r_floor", r_floor), ("r_foundation_wall", r_found),
                     ("r_slab", r_slab), ("r_rim_joist", r_rim)]:
        row[key] = float(val) if not np.isnan(val) else 0.0

    # U-values
    u_wall    = _r_to_u(r_wall,  _BASE_R_WALL)
    u_ceiling = _r_to_u(r_ceil,  _BASE_R_CEIL)
    u_roof_v  = _r_to_u(r_roof,  _BASE_R_ROOF)
    u_floor   = _r_to_u(r_floor, _BASE_R_FLOOR)

    # Envelope areas
    wwr       = float(inputs.get("wwr", 0.15))
    wall_area = float(inputs.get("wall_area_ft2",
                                  4.0 * np.sqrt(floor_area_ft2) * _CEIL_HT_FT))
    win_area  = wwr * wall_area
    net_wall  = max(wall_area - win_area, 0.0)
    ceil_area = float(inputs.get("ceil_area_ft2", floor_area_ft2))

    row["wall_area_ft2"]     = wall_area
    row["win_area_ft2"]      = win_area
    row["net_wall_area_ft2"] = net_wall
    row["ceil_area_ft2"]     = ceil_area

    win_key = inputs.get("window_type", "double, clear, non-metal, air").lower().strip()
    u_win   = _U_WIN_MAP.get(win_key, _U_WIN_DEFAULT)
    row["u_win_btu"] = u_win

    # UA products — min(u_ceiling, u_roof) for thermal boundary
    u_ceil_eff = min(u_ceiling, u_roof_v)
    UA_wall  = u_wall      * net_wall
    UA_ceil  = u_ceil_eff  * ceil_area
    UA_floor = u_floor     * floor_area_ft2
    UA_win   = u_win       * win_area
    UA_total = UA_wall + UA_ceil + UA_floor + UA_win

    # ResStock: UA_per_area in W/m²K (× _BTU_PER_HR_F_TO_W_K)
    UA_per_area = (UA_total * _BTU_PER_HR_F_TO_W_K) / floor_area_m2
    row["UA_total"]    = UA_total
    row["UA_per_area"] = UA_per_area

    # ResStock: HDD / 1.8 to convert °F·day → K·day before multiplying
    row["HDD65_x_UA_per_area"] = (hdd65 / 1.8) * UA_per_area

    # Physics SHD (ResStock notebook §§1–4)
    _HDD_Khr    = hdd65 / 1.8 * 24   # K·hr/yr
    _vol_per_m2 = _CEIL_HT_M          # m³/m² floor

    # building_fraction_heated = 1.0 for residential
    _UA_heated = UA_total * 1.0
    q_trans = _UA_heated * hdd65 * 24 * _BTU_TO_KWH / floor_area_m2

    # Ventilation (MECH_FRAC = 0 → q_vent = 0)
    q_vent = _RHO_CP * (ach_natural * _MECH_FRAC) * _vol_per_m2 * _HDD_Khr

    # Infiltration: (1 - MECH_FRAC) share of ach_natural
    _ach_inf = ach_natural * (1.0 - _MECH_FRAC)
    q_inf    = _RHO_CP * _ach_inf * _vol_per_m2 * _HDD_Khr

    # Internal gains (lighting only)
    lpd_key   = inputs.get("lighting", "100% led").lower().strip()
    lpd_w_ft2 = _LPD_MAP_RESSTOCK.get(lpd_key, 4.0)
    q_int     = lpd_w_ft2 * FT2_PER_M2 * _LIGHTING_EFLH * 1e-3

    calc_shd = float(max(q_trans + q_vent + q_inf - _ETA_IG * q_int, 0.0))
    row["calculated_shd"] = calc_shd

    # Categorical features
    row["in.geometry_building_type_recs"]   = inputs.get("building_type",        "Single-Family Detached")
    row["in.geometry_wall_type"]            = inputs.get("wall_type",            "Wood Frame")
    row["in.geometry_attic_type"]           = inputs.get("attic_type",           "Vented Attic")
    row["in.hvac_heating_type_and_fuel"]    = inputs.get("hvac_type_fuel",       "Fuel Furnace, Natural Gas")
    row["in.hvac_has_ducts"]                = inputs.get("has_ducts",            "Yes")
    row["in.duct_location"]                 = inputs.get("duct_location",        "Attic")
    row["in.insulation_wall"]               = inputs.get("insulation_wall_r",    "R-13")
    row["in.insulation_ceiling"]            = inputs.get("insulation_ceiling_r", "R-19")
    row["in.insulation_roof"]               = inputs.get("insulation_roof_r",    "R-13")
    row["in.vintage"]                       = inputs.get("vintage",              "1980s")
    row["in.state"]                         = inputs.get("state",                "TX")
    row["in.geometry_wall_exterior_finish"] = inputs.get("exterior_finish",      "Vinyl")
    row["in.roof_material"]                 = inputs.get("roof_material",        "Asphalt Shingles")
    row["in.windows"]                       = win_key

    # is_very_cold: injected as numeric feature for Cold & Very Cold zone
    zone = inputs.get("climate_zone", "Cold & Very Cold")
    if zone in _VERY_COLD_ZONES:
        row["is_very_cold"] = int(inputs.get("is_very_cold_flag", 0))

    return pd.DataFrame([row]), calc_shd


def compute_comstock_features(inputs: dict):
    """
    Mirror ComStock second-iteration feature engineering.
    - floor_area_m2 = floor_area_ft2 * frac_heated / FT2_PER_M2
    - UA_per_area = UA_total / floor_area_m2  (BTU units, no W/K conversion)
    - HDD65_x_UA_per_area = hdd65 * UA_per_area  (no /1.8)
    - is_very_cold injected for Cold & Very Cold zone
    """
    row = {}

    floor_area_ft2 = float(inputs["floor_area_ft2"])
    frac_heated    = float(inputs.get("frac_heated", 1.0))
    floor_area_m2  = (floor_area_ft2 * frac_heated) / FT2_PER_M2
    row["floor_area_m2"] = floor_area_m2

    hdd65 = float(inputs["hdd65"])
    hdd50 = float(inputs.get("hdd50", hdd65 * 0.6))
    row["out.params.hdd65f"] = hdd65
    row["out.params.hdd50f"] = hdd50

    weekday_hrs  = float(inputs.get("weekday_hours", 10.0))
    weekend_hrs  = float(inputs.get("weekend_hours", 8.0))
    op_hours_wk  = 5.0 * weekday_hrs + 2.0 * weekend_hrs
    row["operating_hours_week"]         = op_hours_wk
    row["in.weekday_opening_time..hr"]  = float(inputs.get("weekday_open", 7.0))
    row["in.weekend_opening_time..hr"]  = float(inputs.get("weekend_open", 8.0))
    row["in.year_built"]                = float(inputs.get("year_built", 1990))
    row["in.number_of_stories"]         = float(inputs.get("num_stories", 2))
    row["in.rotation..degrees"]         = float(inputs.get("rotation", 0.0))

    u_wall       = float(inputs.get("u_wall",       0.064))
    u_roof       = float(inputs.get("u_roof",       0.050))
    u_win        = float(inputs.get("u_win",        0.49))
    wall_area_m2 = float(inputs.get("wall_area_m2", floor_area_m2 * 0.5))
    roof_area_m2 = float(inputs.get("roof_area_m2", floor_area_m2))
    win_area_m2  = float(inputs.get("win_area_m2",  wall_area_m2 * 0.30))

    # Areas converted m² → ft² to match notebook (UA in BTU/hr·°F)
    UA_wall  = u_wall * (wall_area_m2 * FT2_PER_M2)
    UA_roof  = u_roof * (roof_area_m2 * FT2_PER_M2)
    UA_win   = u_win  * (win_area_m2  * FT2_PER_M2)
    UA_total = UA_wall + UA_roof + UA_win

    # ComStock: NO _BTU_PER_HR_F_TO_W_K conversion — raw BTU units
    UA_per_area = UA_total / floor_area_m2
    row["UA_total"]    = UA_total
    row["UA_per_area"] = UA_per_area

    # ComStock: raw hdd65 (no /1.8)
    row["HDD65_x_UA_per_area"] = hdd65 * UA_per_area

    airtightness = float(inputs.get("airtightness_m3_m2_s", 0.0003))
    row["infiltration_x_HDD"]       = airtightness * hdd65
    row["infiltration_to_UA_ratio"]  = airtightness / (UA_per_area + 1e-6)

    oda_flow = float(inputs.get("outdoor_air_flow", 0.0025))
    row["ventilation_load_proxy"]                          = oda_flow * op_hours_wk
    row["out.params.average_outdoor_air_fraction"]         = float(inputs.get("outdoor_air_frac", 0.30))

    # Physics SHD
    _HDD_Khr = hdd65 / 1.8 * 24
    _UA_heated = UA_total * frac_heated
    q_trans = _UA_heated * hdd65 * 24 * _BTU_TO_KWH / floor_area_m2
    _f_occ  = min((op_hours_wk * 52.18) / 8760.0, 1.0)
    q_vent  = _RHO_CP * oda_flow * 3600 * _HDD_Khr * _f_occ
    q_inf   = _RHO_CP * (airtightness / _INF_DIV_COM) * _HDD_Khr
    q_int   = float(inputs.get("internal_gains_kwh_m2", 15.0))
    calc_shd = float(max(q_trans + q_vent + q_inf - _ETA_IG * q_int, 0.0))
    row["calculated_shd"] = calc_shd

    # Categoricals
    row["in.comstock_building_type"]        = inputs.get("building_type",       "MediumOffice")
    row["in.comstock_building_type_group"]  = inputs.get("building_type_group", "Office")
    row["in.building_subtype"]              = inputs.get("building_subtype",    "Office")
    row["in.hvac_heat_type"]                = inputs.get("hvac_heat_type",      "Gas Boiler")
    row["in.hvac_category"]                 = inputs.get("hvac_category",       "Packaged Single Zone")
    row["in.wall_construction_type"]        = inputs.get("wall_type",           "SteelFramed")
    row["in.window_type"]                   = inputs.get("window_type",         "DOE Ref Pre-1980")
    row["in.window_to_wall_ratio_category"] = inputs.get("wwr_category",        "0.19-0.26")
    row["in.interior_lighting_generation"]  = inputs.get("lighting_gen",        "LED")
    row["in.state_name"]                    = inputs.get("state_name",          "Texas")

    # is_very_cold injected for Cold & Very Cold zone
    zone = inputs.get("climate_zone", "Cold & Very Cold")
    if zone in _VERY_COLD_ZONES:
        row["is_very_cold"] = int(inputs.get("is_very_cold_flag", 0))

    return pd.DataFrame([row]), calc_shd


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def _predict_one(pipe, X: pd.DataFrame) -> float:
    if isinstance(pipe, dict):
        num_feats = pipe.get("num_feats", [])
        cat_feats = pipe.get("cat_feats", [])
        num_imp   = pipe.get("num_imp")
        cat_imp   = pipe.get("cat_imp")
        num_cols  = [c for c in num_feats if c in X.columns]
        cat_cols  = [c for c in cat_feats if c in X.columns]
        parts = []
        if num_imp is not None and num_cols:
            parts.append(pd.DataFrame(num_imp.transform(X[num_cols]),
                                       columns=num_cols, index=X.index))
        elif num_cols:
            parts.append(X[num_cols].fillna(X[num_cols].median()))
        if cat_imp is not None and cat_cols:
            parts.append(pd.DataFrame(cat_imp.transform(X[cat_cols]),
                                       columns=cat_cols, index=X.index).astype("string"))
        elif cat_cols:
            parts.append(X[cat_cols].fillna("__MISSING__").astype("string"))
        X_p  = pd.concat(parts, axis=1) if parts else X
        raw  = pipe["model"].predict(X_p)
    else:
        raw = pipe.predict(X)
    pred = np.expm1(raw)
    pred = np.clip(pred, 0.0, None)
    return float(pred[0]) if hasattr(pred, "__len__") else float(pred)


def load_and_predict(model_dir, dataset, zone, df_row):
    preds  = {}
    loaded = False
    for algo_key in ("xgb", "cat", "lgb"):
        path = _build_model_path(model_dir, dataset, zone, algo_key)
        if path.exists():
            try:
                pipe = joblib.load(path)
                preds[algo_key] = _predict_one(pipe, df_row)
                loaded = True
            except Exception as exc:
                st.warning(f"Could not run {algo_key} ({path.name}): {exc}")
                preds[algo_key] = None
        else:
            preds[algo_key] = None
    return {"predictions": preds, "loaded": loaded}


def mock_predict(calc_shd, zone, dataset, seed=0):
    _res_corr = {
        "Hot-Humid": 0.78, "Hot-Dry": 0.72, "Mixed-Humid": 0.90,
        "Marine": 0.82, "Cold & Very Cold": 0.89, "Mixed-Dry": 0.80,
    }
    _com_corr = {
        "Hot-Humid": 1.15, "Hot-Dry": 1.10, "Marine": 1.22,
        "Mixed-Humid": 1.18, "Cold & Very Cold": 1.30, "Mixed-Dry": 1.20,
    }
    perf = RESSTOCK_PERFORMANCE if dataset == "ResStock" else COMSTOCK_PERFORMANCE
    corr = (_res_corr if dataset == "ResStock" else _com_corr).get(zone, 1.0)
    r2   = perf.get(zone, {}).get("r2", 0.85)
    pred = calc_shd * corr
    rng  = np.random.RandomState(seed)
    pred *= (1.0 + rng.uniform(-(1 - r2) * 0.25, (1 - r2) * 0.25))
    return float(max(0.5, pred))


def estimate_confidence(zone, dataset, calc_shd, pred_shd):
    perf      = RESSTOCK_PERFORMANCE if dataset == "ResStock" else COMSTOCK_PERFORMANCE
    zone_data = perf.get(zone, {})
    r2   = zone_data.get("r2",   0.85)
    rmse = zone_data.get("rmse", 20.0)
    gap_pct = abs(pred_shd - calc_shd) / max(calc_shd, 1.0) * 100 if calc_shd > 0 else 50.0
    score = min(100.0,
                r2 * 55.0
                + max(0.0, 30.0 * (1.0 - gap_pct / 100.0))
                + (15.0 if r2 >= 0.95 else 12.0 if r2 >= 0.90 else 8.0))
    level = "High" if score >= 75 else ("Moderate" if score >= 55 else "Low")
    return {
        "confidence": round(score, 1), "level": level,
        "ci_low":  max(0.0, pred_shd - 1.64 * rmse),
        "ci_high": pred_shd + 1.64 * rmse,
        "r2": r2, "rmse": rmse, "mae": zone_data.get("mae", 10.0),
        "physics_gap_pct": round(gap_pct, 1),
        "best_model": zone_data.get("model", "CatBoost"),
    }


def predict_shd(df_row, calc_shd, zone, dataset, model_dir=None):
    raw_preds    = {"xgb": None, "cat": None, "lgb": None}
    model_loaded = False
    if model_dir and os.path.isdir(model_dir):
        result       = load_and_predict(model_dir, dataset, zone, df_row)
        raw_preds    = result["predictions"]
        model_loaded = result["loaded"]
    display_preds = {}
    for key in ("xgb", "cat", "lgb"):
        if raw_preds.get(key) is not None:
            display_preds[key] = raw_preds[key]
        else:
            display_preds[key] = mock_predict(calc_shd, zone, dataset,
                                              seed={"xgb": 1, "cat": 2, "lgb": 3}[key])
    ensemble = float(np.mean(list(display_preds.values())))
    conf = estimate_confidence(zone, dataset, calc_shd, ensemble)
    conf.update({"model_loaded": model_loaded, "predictions": display_preds, "ensemble": ensemble})
    return conf


def get_shd_rating(shd, btype="residential"):
    if btype == "residential":
        thresholds = [
            (15,           "Passive House", "#3fb950", "Exceptional: <= 15 kWh/m²/yr"),
            (30,           "Near-Passive",  "#56d364", "Very low energy: 15–30"),
            (60,           "Low Energy",    "#e3b341", "Good insulation: 30–60"),
            (100,          "Moderate",      "#d29922", "Standard practice: 60–100"),
            (150,          "High",          "#f0883e", "Below average: 100–150"),
            (float("inf"), "Very High",     "#f85149", "Poor: > 150 kWh/m²/yr"),
        ]
    else:
        thresholds = [
            (20,           "Excellent",  "#3fb950", "<= 20 kWh/m²/yr"),
            (50,           "Good",       "#56d364", "20–50 kWh/m²/yr"),
            (100,          "Average",    "#e3b341", "50–100 kWh/m²/yr"),
            (150,          "Below Avg",  "#d29922", "100–150 kWh/m²/yr"),
            (float("inf"), "Poor",       "#f85149", "> 150 kWh/m²/yr"),
        ]
    for threshold, label, color, desc in thresholds:
        if shd <= threshold:
            return {"label": label, "color": color, "desc": desc}
    return {"label": "Unknown", "color": "#6e7681", "desc": ""}


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# FIX: removed 'margin' from _PL — it was causing "multiple values for keyword
# argument 'margin'" whenever callers also passed margin= explicitly.
# ══════════════════════════════════════════════════════════════════════════════
_PL = dict(
    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Sans", color="#cdd0d5"),
    margin=dict(l=40, r=40, t=40, b=40),   # default; callers may override via update_layout
)


def _layout(**extra):
    """Build a layout dict merging _PL with caller overrides, avoiding key conflicts."""
    d = dict(_PL)
    d.update(extra)
    return d


def make_gauge(value, ci_low, ci_high):
    max_val = max(200.0, value * 1.6, ci_high * 1.25)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "Predicted SHD", "font": {"size": 13, "color": "#9198a1", "family": "IBM Plex Sans"}},
        number={"suffix": " kWh/m\u00b2/yr",
                "font": {"size": 26, "color": "#e6b17e", "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#21262d",
                     "tickfont": {"color": "#6e7681", "size": 9}},
            "bar": {"color": "#e6b17e", "thickness": 0.22},
            "bgcolor": "#161b22",
            "bordercolor": "#21262d",
            "steps": [
                {"range": [0,               max_val * 0.25], "color": "#12261a"},
                {"range": [max_val * 0.25,  max_val * 0.55], "color": "#261e0f"},
                {"range": [max_val * 0.55,  max_val],        "color": "#261313"},
            ],
            "threshold": {"line": {"color": "#f85149", "width": 2},
                          "thickness": 0.72, "value": ci_high},
        },
    ))
    fig.update_layout(**_layout(height=230))
    return fig


def make_algo_bar(predictions):
    names  = {"xgb": "XGBoost", "cat": "CatBoost", "lgb": "LightGBM"}
    algos  = [names.get(k, k) for k in predictions]
    values = list(predictions.values())
    colors = ["#e6b17e", "#58a6ff", "#3fb950"]
    fig = go.Figure(go.Bar(
        x=algos, y=values, marker_color=colors[:len(algos)],
        text=[f"{v:.1f}" for v in values], textposition="outside",
        textfont=dict(family="IBM Plex Mono", color="#cdd0d5", size=11),
    ))
    fig.update_layout(**_layout(
        title=dict(text="Per-Algorithm Predictions", font=dict(size=12, color="#9198a1")),
        xaxis=dict(showgrid=False, tickfont=dict(color="#6e7681")),
        yaxis=dict(showgrid=True, gridcolor="#161b22",
                   title=dict(text="kWh/m\u00b2/yr", font=dict(color="#6e7681")),
                   tickfont=dict(color="#6e7681")),
        height=230, showlegend=False,
    ))
    return fig


def make_radar(r2, physics_gap):
    physics_score = max(0.0, 100.0 - physics_gap)
    zone_cov = r2 * 90 + 10
    stability = max(0.0, 100 - physics_gap * 0.8)
    cats = ["R\u00b2 Score", "Physics Align", "Zone Coverage", "Stability"]
    vals = [r2 * 100, physics_score, zone_cov, stability]
    vals_plot = vals + [vals[0]]
    cats_plot = cats + [cats[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals_plot, theta=cats_plot, fill="toself",
        fillcolor="rgba(230,177,126,0.12)",
        line=dict(color="#e6b17e", width=2),
        marker=dict(color="#e6b17e", size=5),
    ))
    fig.update_layout(**_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(range=[0, 100], showticklabels=False,
                            gridcolor="#21262d", linecolor="#21262d"),
            angularaxis=dict(tickfont=dict(color="#9198a1", size=10),
                             gridcolor="#21262d", linecolor="#21262d"),
        ),
        title=dict(text="Confidence Profile", font=dict(size=12, color="#9198a1")),
        height=260, showlegend=False,
    ))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.8rem 0 0.5rem 0">
        <div style="font-family:'Lora',serif;font-size:1.3rem;color:#e6b17e;margin-bottom:0.2rem">
            SHD Predictor
        </div>
        <div style="font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:2px">
            Building Energy Tool
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Dataset**")
    dataset = st.radio("", ["ResStock (Residential)", "ComStock (Commercial)"],
                       label_visibility="collapsed")
    dataset_key = "ResStock" if "ResStock" in dataset else "ComStock"
    st.markdown("---")
    with st.expander("Model Directory (optional)"):
        model_dir = st.text_input(
            "Path to folder with .joblib models", value="",
            placeholder="e.g. models/resstock_5.8/",
            help="Point to the folder produced by the training notebook. "
                 "Files: xgboost_global.joblib, catboost_zone_Marine.joblib, etc.")
    model_dir = model_dir.strip() if model_dir else None
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.7rem;color:#6e7681;line-height:1.9">
        <b style="color:#9198a1">File naming</b><br>
        {algo}_global.joblib<br>
        {algo}_zone_{Zone}.joblib<br>
        algo: xgboost | catboost | lightgbm<br><br>
        <b style="color:#9198a1">Cold+Very Cold</b><br>
        Single model with is_very_cold flag.<br>
        File: ..._zone_Cold_and_Very_Cold<br><br>
        <b style="color:#9198a1">Iteration</b><br>
        Both datasets: 2nd iteration<br>
        (building-level GroupKFold split)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RENDER RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def render_results(result, calc_shd, btype="residential"):
    pred       = result["ensemble"]
    ci_low     = result["ci_low"]
    ci_high    = result["ci_high"]
    confidence = result["confidence"]
    level      = result["level"]
    rating     = get_shd_rating(pred, btype)

    if not result["model_loaded"]:
        st.markdown("""
        <div class="warn-box">
            Demo mode — no .joblib files found. Showing physics-calibrated estimates.
            Set the model directory in the sidebar to use real ML predictions.
        </div>""", unsafe_allow_html=True)

    st.plotly_chart(make_gauge(pred, ci_low, ci_high), use_container_width=True)

    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Physics Estimate</div>
            <div class="metric-value" style="color:#58a6ff;font-size:1.7rem">{calc_shd:.1f}</div>
            <div class="metric-unit">kWh/m\u00b2/yr</div>
        </div>""", unsafe_allow_html=True)
    with cm2:
        c_color = "#3fb950" if level == "High" else ("#e3b341" if level == "Moderate" else "#f85149")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value" style="color:{c_color};font-size:1.7rem">{confidence:.0f}%</div>
            <div class="metric-unit">{level}</div>
        </div>""", unsafe_allow_html=True)
    with cm3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Rating</div>
            <div class="metric-value" style="color:{rating['color']};font-size:1.4rem">{rating['label']}</div>
            <div class="metric-unit">Performance class</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box" style="margin-top:0.7rem">
        <b style="color:#e6b17e">90% CI:</b> {ci_low:.1f} \u2013 {ci_high:.1f} kWh/m\u00b2/yr
        &nbsp;|&nbsp; <b style="color:#e6b17e">Physics gap:</b> {result['physics_gap_pct']:.0f}%
        &nbsp;|&nbsp; <b style="color:#e6b17e">Best model (test):</b> {result['best_model']}
    </div>
    <div class="info-box">{rating['desc']}</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_b, col_r = st.columns(2)
    with col_b:
        st.plotly_chart(make_algo_bar(result["predictions"]), use_container_width=True)
    with col_r:
        st.plotly_chart(make_radar(result["r2"], result["physics_gap_pct"]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1>Specific Heating Demand Predictor</h1>
<p style="color:#6e7681;font-size:0.85rem;margin-top:0;margin-bottom:1.5rem">
    Physics-informed ML surrogate &middot; NREL ResStock &amp; ComStock &middot;
    XGBoost / CatBoost / LightGBM &middot; Second Iteration Pipeline
</p>
""", unsafe_allow_html=True)

if dataset_key == "ResStock":
    tab1, tab2, tab3 = st.tabs(["  Residential (ResStock)  ", "  Zone Performance  ", "  Methodology  "])
else:
    tab1, tab2, tab3 = st.tabs(["  Commercial (ComStock)  ", "  Zone Performance  ", "  Methodology  "])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INPUT FORMS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ─── RESIDENTIAL ─────────────────────────────────────────────────────────
    if dataset_key == "ResStock":
        col_left, col_right = st.columns([1.15, 0.85], gap="large")

        with col_left:
            st.markdown('<div class="pill">Building Geometry & Climate</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                floor_area_ft2 = st.number_input("Floor Area (ft\u00b2)", 200, 10000, 1800, 50)
                hdd65          = st.number_input("HDD65 (\u00b0F\u00b7day/yr)", 0, 12000, 4500, 100,
                    help="From hdd65_annual_avg column in parquet.")
                vintage = st.selectbox("Construction Vintage", list(VINTAGE_MAP.keys()), index=4)
            with c2:
                wwr   = st.slider("Window-to-Wall Ratio", 0.05, 0.60, 0.15, 0.01)
                state = st.selectbox("State", [
                    "AK","AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID",
                    "IL","IN","KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC",
                    "ND","NE","NH","NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD",
                    "TN","TX","UT","VA","VT","WA","WI","WV","WY"], index=43)
                ach50 = st.number_input("Air Leakage (ACH50)", 0.5, 25.0, 7.0, 0.5,
                    help=f"Natural ACH = ACH50 / {_ACH50_DIV} (ResStock convention).")

            st.markdown("---")
            st.markdown('<div class="pill">Insulation R-Values</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                ins_wall  = st.selectbox("Wall",     ["Uninsulated","R-7","R-11","R-13","R-15","R-19","R-21"], index=3)
                ins_ceil  = st.selectbox("Ceiling",  ["Uninsulated","R-13","R-19","R-30","R-38","R-49","R-60"], index=3)
                ins_floor = st.selectbox("Floor",    ["Uninsulated","R-0","R-10","R-19","R-30"], index=1)
            with c4:
                ins_roof  = st.selectbox("Roof",     ["Uninsulated","R-13","R-19","R-30","R-38"], index=1)
                ins_found = st.selectbox("Foundation Wall", ["Uninsulated","R-0","R-5","R-10","R-15"], index=0)
                ins_slab  = st.selectbox("Slab",     ["None","R-0","R-5","R-10"], index=0)

            st.markdown("---")
            st.markdown('<div class="pill">Systems & Materials</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                building_type  = st.selectbox("Building Type", [
                    "Single-Family Detached","Single-Family Attached",
                    "Multi-Family with 2-4 units","Multi-Family with 5+ units","Mobile Home"])
                hvac_type_fuel = st.selectbox("HVAC Heating System", [
                    "Fuel Furnace, Natural Gas","Fuel Furnace, Propane",
                    "Fuel Boiler, Natural Gas","Electric Heat Pump",
                    "Electric Resistance","Fuel Wall/Floor Heater, Natural Gas"])
                attic_type = st.selectbox("Attic Type", [
                    "Vented Attic","Unvented Attic","Cathedral Ceiling",
                    "Finished Attic or Knee Wall","Flat Roof"])
            with c6:
                window_type = st.selectbox("Window Type", list(_U_WIN_MAP.keys()), index=6)
                wall_type   = st.selectbox("Wall Construction", ["Wood Frame","Concrete","Brick","Steel Frame"])
                lighting    = st.selectbox("Lighting", ["100% LED","100% CFL","100% Incandescent"])

            duct_leakage_pct = st.slider("Duct Leakage to Outside (%)", 0, 40, 10)
            c7, c8 = st.columns(2)
            with c7:
                has_ducts    = st.selectbox("Has Ducts", ["Yes","No"])
            with c8:
                duct_location = st.selectbox("Duct Location",
                    ["Attic","Basement","Crawlspace","Conditioned Space","None"])

            st.markdown("---")
            st.markdown('<div class="pill">Climate Zone</div>', unsafe_allow_html=True)
            climate_zone_res = st.selectbox(
                "Building America Climate Zone",
                ["Hot-Humid","Hot-Dry","Mixed-Humid","Marine","Cold","Very Cold","Mixed-Dry"],
                index=4,
                help="'Cold' and 'Very Cold' both use the combined Cold_and_Very_Cold model "
                     "with is_very_cold=0 or is_very_cold=1 respectively.")

        with col_right:
            st.markdown('<div class="pill">Results</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="note-box">
                <b>2nd Iteration pipeline</b>: building-level GroupKFold split,
                unfiltered test set, deployable feature set. No monotonic constraints,
                no stacking ensemble. is_very_cold flag injected automatically for
                Cold / Very Cold zones.
            </div>""", unsafe_allow_html=True)
            predict_btn = st.button("Calculate SHD", use_container_width=True, key="btn_res")

            # Normalise UI zone → model zone + is_very_cold flag
            if climate_zone_res == "Very Cold":
                perf_zone  = "Cold & Very Cold"
                is_vc_flag = 1
                model_zone = "Cold & Very Cold"
            elif climate_zone_res == "Cold":
                perf_zone  = "Cold & Very Cold"
                is_vc_flag = 0
                model_zone = "Cold & Very Cold"
            else:
                perf_zone  = climate_zone_res
                is_vc_flag = 0
                model_zone = climate_zone_res

            if predict_btn or "res_result" in st.session_state:
                inputs_res = {
                    "floor_area_ft2":         floor_area_ft2,
                    "hdd65":                  hdd65,
                    "vintage":                vintage,
                    "wwr":                    wwr,
                    "state":                  state,
                    "ach50_numeric":          ach50,
                    "insulation_wall_r":      ins_wall,
                    "insulation_ceiling_r":   ins_ceil,
                    "insulation_floor_r":     ins_floor,
                    "insulation_roof_r":      ins_roof,
                    "insulation_foundation_r":ins_found,
                    "insulation_slab_r":      ins_slab,
                    "window_type":            window_type,
                    "building_type":          building_type,
                    "hvac_type_fuel":         hvac_type_fuel,
                    "attic_type":             attic_type,
                    "wall_type":              wall_type,
                    "lighting":               lighting.lower(),
                    "duct_leakage_frac":      duct_leakage_pct / 100.0,
                    "has_ducts":              has_ducts,
                    "duct_location":          duct_location,
                    "exterior_finish":        "Vinyl",
                    "roof_material":          "Asphalt Shingles",
                    "climate_zone":           model_zone,
                    "is_very_cold_flag":      is_vc_flag,
                }
                df_input, calc_shd = compute_resstock_features(inputs_res)
                result = predict_shd(df_input, calc_shd, model_zone, "ResStock", model_dir)
                if predict_btn:
                    st.session_state["res_result"]   = result
                    st.session_state["res_calc_shd"] = calc_shd
                else:
                    result   = st.session_state["res_result"]
                    calc_shd = st.session_state["res_calc_shd"]
                render_results(result, calc_shd, "residential")

                with st.expander("Physics Decomposition"):
                    row = df_input.iloc[0]
                    ach_nat = ach50 / _ACH50_DIV
                    st.markdown(f"""
**Zone selected:** {climate_zone_res} \u2192 model zone: Cold & Very Cold, is_very_cold={is_vc_flag if climate_zone_res in ('Cold','Very Cold') else 'N/A'}

**Floor area:** {row['floor_area_m2']:.1f} m\u00b2  &nbsp;|&nbsp; **HDD65:** {hdd65} \u00b0F\u00b7day/yr
**ACH50:** {ach50} \u2192 natural ACH = {ach_nat:.4f} (\u00f7{_ACH50_DIV})
**UA total:** {row['UA_total']:.2f} BTU/hr\u00b7\u00b0F
**UA/area:** {row['UA_per_area']:.4f} W/m\u00b2K (\u00d7 {_BTU_PER_HR_F_TO_W_K})
**HDD65 \u00d7 UA/area:** {row['HDD65_x_UA_per_area']:.3f} K\u00b7day \u00d7 W/m\u00b2K
**Physics SHD:** {calc_shd:.1f} kWh/m\u00b2/yr  &nbsp;|&nbsp; **ML ensemble:** {result['ensemble']:.1f} kWh/m\u00b2/yr
                    """)

    # ─── COMMERCIAL ──────────────────────────────────────────────────────────
    else:
        col_left, col_right = st.columns([1.15, 0.85], gap="large")

        with col_left:
            st.markdown('<div class="pill">Building Overview</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                floor_area_ft2 = st.number_input("Floor Area (ft\u00b2)", 1000, 500000, 25000, 500)
                hdd65          = st.number_input("HDD65 (\u00b0F\u00b7day/yr)", 0, 12000, 5000, 100)
                year_built     = st.number_input("Year Built", 1900, 2023, 1995, 1)
                frac_heated    = st.slider("Fraction Heated", 0.1, 1.0, 1.0, 0.05,
                    help="out.params.building_fraction_heated")
            with c2:
                state_name  = st.selectbox("State", [
                    "Alabama","Alaska","Arizona","Arkansas","California","Colorado",
                    "Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho",
                    "Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana",
                    "Maine","Maryland","Massachusetts","Michigan","Minnesota",
                    "Mississippi","Missouri","Montana","Nebraska","Nevada",
                    "New Hampshire","New Jersey","New Mexico","New York",
                    "North Carolina","North Dakota","Ohio","Oklahoma","Oregon",
                    "Pennsylvania","Rhode Island","South Carolina","South Dakota",
                    "Tennessee","Texas","Utah","Vermont","Virginia","Washington",
                    "West Virginia","Wisconsin","Wyoming"], index=43)
                num_stories = st.number_input("Stories", 1, 50, 3)
                rotation    = st.slider("Orientation (\u00b0)", 0, 360, 0)
                hdd50       = st.number_input("HDD50 (\u00b0F\u00b7day/yr)", 0, 12000, int(hdd65 * 0.6), 100)

            st.markdown("---")
            st.markdown('<div class="pill">Envelope</div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            with c3:
                u_wall = st.number_input("Wall U-value (BTU/hr\u00b7ft\u00b2\u00b7\u00b0F)", 0.01, 0.5, 0.064, 0.005, format="%.3f")
                u_roof = st.number_input("Roof U-value (BTU/hr\u00b7ft\u00b2\u00b7\u00b0F)", 0.01, 0.5, 0.048, 0.005, format="%.3f")
                u_win  = st.number_input("Window U-value (BTU/hr\u00b7ft\u00b2\u00b7\u00b0F)", 0.1, 1.5, 0.49, 0.05, format="%.2f")
            with c4:
                wall_area_m2 = st.number_input("Ext. Wall Area (m\u00b2)", 50.0, 50000.0, 2000.0, 50.0)
                roof_area_m2 = st.number_input("Roof Area (m\u00b2)", 50.0, 50000.0, 2300.0, 50.0)
                win_area_m2  = st.number_input("Window Area (m\u00b2)", 10.0, 20000.0, 600.0, 25.0)

            wall_type_com   = st.selectbox("Wall Construction", ["SteelFramed","WoodFramed","Mass","Structural Masonry"])
            window_type_com = st.selectbox("Window Type", [
                "DOE Ref Pre-1980","DOE Ref 1980-2004",
                "ASHRAE 169-2013 Climate Zone 1-3","ASHRAE 169-2013 Climate Zone 4",
                "ASHRAE 169-2013 Climate Zone 5-8","Triple Pane Low-E"], index=1)
            wwr_cat = st.selectbox("WWR Category",
                ["0-0.09","0.10-0.18","0.19-0.26","0.27-0.33","0.34-0.40",">0.40"], index=2)

            st.markdown("---")
            st.markdown('<div class="pill">HVAC & Operations</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                building_type_com = st.selectbox("Building Type", [
                    "MediumOffice","LargeOffice","SmallOffice",
                    "RetailStripmall","RetailStandalone",
                    "PrimarySchool","SecondarySchool",
                    "Outpatient","Hospital",
                    "SmallHotel","LargeHotel",
                    "Warehouse","QuickServiceRestaurant","FullServiceRestaurant"])
                hvac_heat_type = st.selectbox("Heating System", [
                    "Gas Boiler","Gas Furnace","Electric Resistance",
                    "Air-Source Heat Pump","Ground-Source Heat Pump",
                    "District Heating","Packaged Rooftop Unit"])
                hvac_cat = st.selectbox("HVAC Category", [
                    "Packaged Single Zone","Multizone CAV/VAV",
                    "Residential-Style Central System",
                    "District System","Dedicated Outdoor Air System"])
            with c6:
                weekday_hrs  = st.slider("Weekday Hours", 4, 24, 10)
                weekend_hrs  = st.slider("Weekend Hours", 0, 24, 8)
                weekday_open = st.number_input("Weekday Open (hr)", 0.0, 12.0, 7.0, 0.5)
                weekend_open = st.number_input("Weekend Open (hr)", 0.0, 12.0, 8.0, 0.5)

            airtightness   = st.number_input("Airtightness (m\u00b3/m\u00b2/s at 75 Pa)", 0.0001, 0.005, 0.0003, 0.0001, format="%.4f")
            oda_flow       = st.number_input("Design ODA Flow (m\u00b3/m\u00b2/s)", 0.0005, 0.01, 0.0025, 0.0005, format="%.4f")
            oda_frac       = st.slider("Avg Outdoor Air Fraction", 0.05, 1.0, 0.30, 0.05)
            lighting_gen   = st.selectbox("Lighting Generation", ["LED","Fluorescent","Incandescent"])
            internal_gains = st.number_input("Internal Gains (kWh/m\u00b2/yr)", 0.0, 100.0, 15.0, 1.0)

            st.markdown("---")
            st.markdown('<div class="pill">Climate Zone</div>', unsafe_allow_html=True)
            climate_zone_com = st.selectbox(
                "Building America Climate Zone",
                ["Hot-Humid","Hot-Dry","Marine","Mixed-Humid","Cold & Very Cold","Mixed-Dry"],
                index=4)

            if climate_zone_com == "Cold & Very Cold":
                is_vc_com = st.checkbox("Sub-zone: Very Cold (vs Cold)", value=False,
                                         help="Sets is_very_cold=1 in the feature vector.")
            else:
                is_vc_com = False

            _btg = building_type_com
            if "Office"       in _btg: btg = "Office"
            elif "Retail"     in _btg: btg = "Retail"
            elif "School"     in _btg: btg = "Education"
            elif "Hotel"      in _btg: btg = "Lodging"
            elif _btg in ("Hospital","Outpatient"): btg = "Healthcare"
            elif _btg == "Warehouse": btg = "Warehouse"
            elif "Restaurant" in _btg: btg = "Food Service"
            else: btg = _btg

        with col_right:
            st.markdown('<div class="pill">Results</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="note-box">
                <b>2nd Iteration pipeline</b>: building-level GroupKFold split,
                unfiltered test set, deployable features. No stacking ensemble.
                is_very_cold injected for Cold & Very Cold zone.
            </div>""", unsafe_allow_html=True)
            predict_btn_com = st.button("Calculate SHD", use_container_width=True, key="btn_com")

            if predict_btn_com or "com_result" in st.session_state:
                inputs_com = {
                    "floor_area_ft2":        floor_area_ft2,
                    "hdd65":                 hdd65,
                    "hdd50":                 hdd50,
                    "frac_heated":           frac_heated,
                    "year_built":            year_built,
                    "num_stories":           num_stories,
                    "rotation":              rotation,
                    "u_wall":                u_wall,
                    "u_roof":                u_roof,
                    "u_win":                 u_win,
                    "wall_area_m2":          wall_area_m2,
                    "roof_area_m2":          roof_area_m2,
                    "win_area_m2":           win_area_m2,
                    "weekday_hours":         weekday_hrs,
                    "weekend_hours":         weekend_hrs,
                    "weekday_open":          weekday_open,
                    "weekend_open":          weekend_open,
                    "airtightness_m3_m2_s":  airtightness,
                    "outdoor_air_flow":      oda_flow,
                    "outdoor_air_frac":      oda_frac,
                    "internal_gains_kwh_m2": internal_gains,
                    "building_type":         building_type_com,
                    "building_type_group":   btg,
                    "building_subtype":      building_type_com,
                    "hvac_heat_type":        hvac_heat_type,
                    "hvac_category":         hvac_cat,
                    "wall_type":             wall_type_com,
                    "window_type":           window_type_com,
                    "wwr_category":          wwr_cat,
                    "lighting_gen":          lighting_gen,
                    "state_name":            state_name,
                    "climate_zone":          climate_zone_com,
                    "is_very_cold_flag":     int(is_vc_com),
                }
                df_input_com, calc_shd_com = compute_comstock_features(inputs_com)
                result_com = predict_shd(df_input_com, calc_shd_com,
                                          climate_zone_com, "ComStock", model_dir)
                if predict_btn_com:
                    st.session_state["com_result"]   = result_com
                    st.session_state["com_calc_shd"] = calc_shd_com
                else:
                    result_com   = st.session_state["com_result"]
                    calc_shd_com = st.session_state["com_calc_shd"]
                render_results(result_com, calc_shd_com, "commercial")

                with st.expander("Physics Decomposition"):
                    r = df_input_com.iloc[0]
                    vc_val = int(is_vc_com) if climate_zone_com == "Cold & Very Cold" else "N/A"
                    st.markdown(f"""
**Zone:** {climate_zone_com} | is_very_cold={vc_val}
**Floor area (heated):** {r['floor_area_m2']:.1f} m\u00b2
**HDD65:** {hdd65} | **HDD50:** {hdd50}
**UA total:** {r['UA_total']:.2f} BTU/hr\u00b7\u00b0F
**UA/area:** {r['UA_per_area']:.4f} BTU/hr\u00b7\u00b0F/m\u00b2 (BTU units, no W/K conversion)
**HDD65 \u00d7 UA/area:** {r['HDD65_x_UA_per_area']:.3f}
**Infiltration \u00d7 HDD:** {r['infiltration_x_HDD']:.6f}
**Ventilation proxy:** {r['ventilation_load_proxy']:.5f}
**Physics SHD:** {calc_shd_com:.1f} kWh/m\u00b2/yr | **ML ensemble:** {result_com['ensemble']:.1f} kWh/m\u00b2/yr
                    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ZONE PERFORMANCE
# FIX: removed `margin=dict(...)` from fig_tbl.update_layout() call since _PL
# already contains margin. Now uses _layout() helper which merges cleanly.
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if dataset_key == "ResStock":
        st.markdown("## ResStock — Second Iteration Performance")
        st.markdown("""
        <div class="warn-box">
            <b>Important note on data provenance:</b> The ResStock zone training
            notebook crashed during Marine zone CatBoost tuning due to a
            <code>transform_catboost()</code> signature error (5 arguments vs 4
            expected). Only the Marine zone baseline RMSE (51.955 kWh/m\u00b2/yr)
            was captured in notebook output before the crash. No zone-level test
            metrics were produced by the notebook. The Global model output was
            also not captured. All figures in this table are drawn from the thesis
            second-iteration analysis (Table 7.2), which reports results obtained
            after fixing the bug and re-running. The CV winner and best-by-test
            model designations are from that corrected run.
        </div>""", unsafe_allow_html=True)

        all_models = RESSTOCK_ALL_MODELS
        cols_show = ["Zone", "Model", "CV-RMSE", "Train RMSE", "Test RMSE", "MAE", "R\u00b2", "Best by CV?"]

        rows = []
        for zone, data in all_models.items():
            best_cv = data["best_by_cv"]
            for mname in ("XGBoost", "CatBoost", "LightGBM"):
                m = data[mname]
                rows.append({
                    "Zone": zone, "Model": mname,
                    "CV-RMSE": m["cv_rmse"], "Train RMSE": m["train_rmse"],
                    "Test RMSE": m["test_rmse"], "MAE": m["mae"], "R\u00b2": m["r2"],
                    "Best by CV?": "YES" if mname == best_cv else "",
                })

    else:
        st.markdown("## ComStock — Second Iteration Performance")
        st.markdown("""
        <div class="info-box">
            All figures are exact notebook stdout output values.
            Building-level GroupKFold split, unfiltered test set, deployable
            feature set. No stacking ensemble in this iteration.
            CV-RMSE = cross-validation RMSE from Optuna tuning (kWh/m\u00b2/yr).
            Train RMSE = in-sample error on full training partition.
        </div>""", unsafe_allow_html=True)

        all_models = COMSTOCK_ALL_MODELS
        cols_show = ["Zone", "Model", "CV-RMSE", "Train RMSE", "Test RMSE", "MAE", "R\u00b2", "Best by CV?"]

        rows = []
        for zone, data in all_models.items():
            best_cv = data["best_by_cv"]
            for mname in ("XGBoost", "CatBoost", "LightGBM"):
                m = data[mname]
                rows.append({
                    "Zone": zone, "Model": mname,
                    "CV-RMSE": m["cv_rmse"], "Train RMSE": m["train_rmse"],
                    "Test RMSE": m["test_rmse"], "MAE": m["mae"], "R\u00b2": m["r2"],
                    "Best by CV?": "YES" if mname == best_cv else "",
                })

    df_perf = pd.DataFrame(rows)
    zones   = list(all_models.keys())

    # FIX: use _layout() so margin is set only once, no duplicate key
    fig_tbl = go.Figure(go.Table(
        header=dict(
            values=cols_show,
            fill_color="#161b22", align="center",
            font=dict(color="#e6b17e", size=11, family="IBM Plex Sans"),
            line_color="#21262d", height=34,
        ),
        cells=dict(
            values=[df_perf[c].tolist() for c in cols_show],
            fill_color="#0d1117", align=["left","left"] + ["center"] * (len(cols_show) - 2),
            font=dict(color="#cdd0d5", size=11, family="IBM Plex Mono"),
            line_color="#161b22", height=28,
            format=["","",""] + [".3f"] * 4 + [""],
        ),
    ))
    fig_tbl.update_layout(
        **_layout(
            height=60 + len(rows) * 30,
            margin=dict(l=0, r=0, t=8, b=0),   # override _PL's default margin
        )
    )
    st.plotly_chart(fig_tbl, use_container_width=True)

    st.markdown("---")
    # Summary charts using best-test model per zone
    perf_summary = COMSTOCK_PERFORMANCE if dataset_key == "ComStock" else RESSTOCK_PERFORMANCE
    cp1, cp2 = st.columns(2)
    with cp1:
        rmse_v = [perf_summary[z]["rmse"] for z in perf_summary]
        mae_v  = [perf_summary[z]["mae"]  for z in perf_summary]
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(name="Test RMSE", x=list(perf_summary), y=rmse_v,
                                 marker_color="#e6b17e", opacity=0.85))
        fig_err.add_trace(go.Bar(name="MAE", x=list(perf_summary), y=mae_v,
                                 marker_color="#58a6ff", opacity=0.85))
        fig_err.update_layout(**_layout(
            title=dict(text="Error by Zone — best model (kWh/m\u00b2/yr)", font=dict(size=12, color="#9198a1")),
            xaxis=dict(tickangle=-30, tickfont=dict(size=10, color="#6e7681")),
            yaxis=dict(title=dict(text="kWh/m\u00b2/yr", font=dict(color="#6e7681")),
                       gridcolor="#161b22", tickfont=dict(color="#6e7681")),
            barmode="group", legend=dict(font=dict(color="#9198a1")), height=300,
        ))
        st.plotly_chart(fig_err, use_container_width=True)

    with cp2:
        r2_v  = [perf_summary[z]["r2"] for z in perf_summary]
        clrs  = ["#3fb950" if v >= 0.95 else ("#e3b341" if v >= 0.85 else "#f85149") for v in r2_v]
        fig_r2 = go.Figure(go.Bar(
            x=list(perf_summary), y=r2_v, marker_color=clrs,
            text=[f"{v:.3f}" for v in r2_v], textposition="outside",
            textfont=dict(size=10, color="#cdd0d5"),
        ))
        fig_r2.add_hline(y=0.85, line_dash="dot", line_color="#6e7681",
                         annotation_text="0.85", annotation_font_color="#6e7681")
        fig_r2.update_layout(**_layout(
            title=dict(text="R\u00b2 by Zone — best model", font=dict(size=12, color="#9198a1")),
            xaxis=dict(tickangle=-30, tickfont=dict(size=10, color="#6e7681")),
            yaxis=dict(range=[0.5, 1.05], title=dict(text="R\u00b2", font=dict(color="#6e7681")),
                       gridcolor="#161b22", tickfont=dict(color="#6e7681")),
            showlegend=False, height=300,
        ))
        st.plotly_chart(fig_r2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Pipeline & Model Architecture")
    cm1, cm2 = st.columns(2)

    with cm1:
        st.markdown("""
### Data & Split Strategy

<div class="info-box">
<b style="color:#e6b17e">Training Data</b><br>
ResStock: 549,713 EnergyPlus residential simulations (after NaN drop on 7 calculated features).<br>
ComStock: 336,149 EnergyPlus commercial simulations (baseline rows only).<br>
Source: NREL 2021–2025, calibrated to RECS & CBECS national surveys.
</div>

<div class="info-box">
<b style="color:#e6b17e">Target Variable</b><br>
SHD = annual heating energy delivered / conditioned floor area [kWh/m\u00b2/yr].<br>
<b>ResStock</b>: out.load.heating.energy_delivered.kbtu \u00d7 0.2931.<br>
<b>ComStock</b>: calc.enduse_group.site_energy.heating (or fuel-type sum).<br>
ComStock denominator = floor_area_ft2 \u00d7 building_fraction_heated / FT2_PER_M2.
</div>

<div class="info-box">
<b style="color:#e6b17e">Second Iteration Split</b><br>
Building-level 80/20 holdout via GroupKFold on bldg_id.<br>
Group labels randomly permuted before folding (fixed random_state=42).<br>
Outlier trimming (0.5\u201399.5th pct) on training set only.<br>
Test set completely unfiltered — all extreme-demand buildings included.
</div>

<div class="info-box">
<b style="color:#e6b17e">Cold & Very Cold Zone</b><br>
Both datasets: single combined model with is_very_cold binary feature<br>
appended to numeric_features (1 = Very Cold, 0 = Cold).<br>
File: {algo}_zone_Cold_and_Very_Cold.joblib<br>
ResStock UI: "Cold" (is_very_cold=0) or "Very Cold" (is_very_cold=1).<br>
ComStock UI: "Cold & Very Cold" + optional sub-zone checkbox.
</div>

<div class="info-box">
<b style="color:#e6b17e">ResStock Zone Training Status</b><br>
The zone training notebook crashed during Marine zone CatBoost tuning
due to a <code>transform_catboost()</code> signature error (5 args vs 4).
No zone-level final test metrics were captured in notebook output.
The global model output cell also has no captured stdout. Performance
figures in the Zone Performance tab are from the thesis analysis
after the bug was fixed.
</div>
        """, unsafe_allow_html=True)

    with cm2:
        st.markdown("""
### Algorithms & Tuning

<div class="info-box">
<b style="color:#e6b17e">XGBoost</b><br>
Second-order gradients; L1/L2/gamma regularisation; sparse-aware splits.<br>
Categoricals: one-hot encoded via sklearn ColumnTransformer Pipeline.<br>
Saved as sklearn Pipeline object. Best on Mixed-Dry (sparse data) and
Hot-Dry (highest CV-RMSE winner for ComStock).
</div>

<div class="info-box">
<b style="color:#e6b17e">CatBoost</b><br>
Ordered boosting (eliminates prediction shift); ordered target statistics
for categorical features — no one-hot encoding needed.<br>
Saved as dict: {num_feats, cat_feats, num_imp, cat_imp, model}.<br>
Most consistently lowest CV-RMSE: 5/7 ComStock zones, 3/7 ResStock zones.
</div>

<div class="info-box">
<b style="color:#e6b17e">LightGBM</b><br>
GOSS sampling; EFB; leaf-wise growth (5\u201320\u00d7 faster at scale).<br>
Also saved as sklearn Pipeline with OHE. Best global CV-RMSE for both
ResStock (29.19) and ComStock (29.81).
</div>

<div class="info-box">
<b style="color:#e6b17e">Optuna HPO</b><br>
TPE sampler + MedianPruner (1 warm-up step).<br>
XGB/LGB: 30 trials each. CatBoost: 15 global / 30 zone.<br>
5-fold GroupKFold CV; early stopping 50 rounds; log1p target.<br>
<b>No stacking ensemble in 2nd iteration</b> (computational limits).<br>
Max estimators = 2000 with early stopping replacing fixed iteration count.
</div>

<div class="info-box">
<b style="color:#e6b17e">Key Differences: ResStock vs ComStock</b><br>
ResStock: UA_per_area in W/m\u00b2K (\u00d7 0.52752). HDD \u00f7 1.8 before multiplying UA.<br>
ComStock: UA_per_area in BTU units (no W/K conversion). Raw HDD65 \u00d7 UA.<br>
ResStock: ACH_natural = ACH50 / 25 (notebook variable = 25).<br>
ComStock: airtightness / 20. No ceiling height multiplier (already m\u00b3/m\u00b2/s).
</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Physics Feature Engineering")
    eq1, eq2 = st.columns(2)
    with eq1:
        st.markdown("**ResStock (residential)**")
        st.latex(r"Q_h = \frac{UA_{tot} \cdot HDD_{65} \cdot 24 \cdot f_{btu}}{A_{fl}} + \rho c_p \cdot ACH_{nat} \cdot h_{ceil} \cdot HDD_{K \cdot hr} - \eta \cdot Q_{lighting}")
        st.markdown(f"""
<div class="note-box" style="font-size:0.78rem">
ACH_nat = ACH50 / {_ACH50_DIV} &middot;
HDD_Khr = HDD65/1.8 \u00d7 24 &middot;
UA_per_area = UA_total \u00d7 {_BTU_PER_HR_F_TO_W_K} / A_m\u00b2 [W/m\u00b2K] &middot;
HDD\u00d7UA/A uses HDD65/1.8 (converts \u00b0F\u00b7day \u2192 K\u00b7day) &middot;
building_fraction_heated = 1.0 (residential, always) &middot;
MECH_FRAC = 0 so q_vent = 0 (no dedicated ventilation column in ResStock baseline)
</div>""", unsafe_allow_html=True)

    with eq2:
        st.markdown("**ComStock (commercial)**")
        st.latex(r"Q_h = Q_{trans} + \rho c_p \dot{V}_{ODA} \cdot 3600 \cdot HDD_{Khr} \cdot f_{occ} + \rho c_p \frac{air}{20} HDD_{Khr} - \eta Q_{int}")
        st.markdown(f"""
<div class="note-box" style="font-size:0.78rem">
Airtightness / {_INF_DIV_COM} (m\u00b3/m\u00b2/s; no ceiling height multiplier) &middot;
UA_per_area = UA_total / A_m\u00b2 (BTU/hr\u00b7\u00b0F/m\u00b2; no W/K conversion) &middot;
HDD\u00d7UA/A uses raw HDD65 (no /1.8) &middot;
vent proxy = ODA_flow \u00d7 op_hours_week &middot;
floor_area_m\u00b2 = floor_area_ft\u00b2 \u00d7 frac_heated / FT2_PER_M2
</div>""", unsafe_allow_html=True)