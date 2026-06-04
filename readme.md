# SHD-Prediction

Machine Learning Surrogate Models for Specific Heating Demand Prediction

This repository contains the code for the thesis "Machine Learning Approaches to the Prediction of Specific Heating Demand" (Mohamed Elsayed, Spring 2026).

It provides training pipelines for gradient boosting surrogate models (XGBoost, CatBoost, LightGBM) and a Streamlit demonstration app. The models predict Specific Heating Demand (SHD) in kWh/m²/yr using physics-informed feature engineering and advanced hyperparameter tuning.

---

## Repository Structure

```
SHD-Prediction/
├── final_final_restock.ipynb       # Residential (ResStock) training pipeline
├── final_final_comstock.ipynb      # Commercial (ComStock) training pipeline
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
├── models/                         # Folder for trained .joblib models
├── BDG2.ipynb                      # Additional analysis notebook
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ME312241/SHD-Prediction.git
cd SHD-Prediction
pip install -r requirements.txt
```

**Requirements:** Python 3.10+

---

## Data Requirements

You need to download the baseline Parquet files separately:

- **ResStock:** `baseline_metadata_and_annual_results_hdd65.parquet`
- **ComStock:** Baseline ComStock Parquet file

Then update the `DATA_PATH` variable in the respective notebooks.

---

## Running the Training Notebooks

Open the notebooks in Jupyter and run them:

### Residential Models
- `final_final_restock.ipynb`

### Commercial Models
- `final_final_comstock.ipynb`

### Important Settings

Set these parameters at the top of each notebook:

- `DATA_PATH` — path to the downloaded Parquet file
- `OUTPUT_DIR` — directory to save trained models
- `N_TRIALS_*` — reduce for faster testing (e.g., to 10)

### Features

The notebooks include:
- Physics-informed feature engineering
- Optuna hyperparameter tuning
- GroupKFold cross-validation
- SHAP explainability analysis

Trained models will be saved as `.joblib` files in the specified output directory (or `models/` folder).

---

## Running the Streamlit App

1. Place the trained model files in the `models/` folder
2. Run the app:

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

### Features

- **Predict** — Estimate SHD from building inputs
- **Zone Performance** — View model metrics by climate zone
- **Methodology** — Project overview

The app runs in limited mode if models are not found. You can specify your own full directory path where you pulled the repository.

---

## Notes

- Trained model files are not included in the repository due to their size
- You must generate them locally by running the notebooks
