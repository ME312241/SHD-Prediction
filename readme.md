SHD-Prediction
Machine Learning Surrogate Models for Specific Heating Demand Prediction
This repository contains the code for the thesis "Machine Learning Approaches to the Prediction of Specific Heating Demand" (Mohamed Elsayed, Spring 2026).
It provides training pipelines for gradient boosting surrogate models (XGBoost, CatBoost, LightGBM) and a Streamlit demonstration app. The models predict Specific Heating Demand (SHD) in kWh/m²/yr using building characteristics from NREL’s ResStock and ComStock datasets.

Repository Structure
textSHD-Prediction/
├── final_final_restock.ipynb          # Residential (ResStock) training pipeline
├── final_final_comstock.ipynb         # Commercial (ComStock) training pipeline
├── app.py                             # Streamlit web app
├── requirements.txt
├── models/                            # Folder for trained .joblib models
├── BDG2.ipynb
└── README.md

Installation
Bashgit clone https://github.com/ME312241/SHD-Prediction.git
cd SHD-Prediction
pip install -r requirements.txt
Python 3.10+ is recommended.

Data Requirements
You need to download the baseline Parquet files separately:

ResStock: baseline_metadata_and_annual_results_hdd65.parquet
ComStock: Baseline ComStock Parquet file

Then update the DATA_PATH variable in the respective notebooks.

Running the Training Notebooks
Open the notebooks in Jupyter and run them:

final_final_restock.ipynb — Residential models
final_final_comstock.ipynb — Commercial models

Important settings (at the top of each notebook):

DATA_PATH → path to the downloaded Parquet file
OUTPUT_DIR → directory to save trained models
Reduce N_TRIALS_* (e.g., to 10) for faster testing

The notebooks include physics-informed feature engineering, Optuna hyperparameter tuning, GroupKFold validation, and SHAP explainability.
Trained models will be saved as .joblib files in the specified output directory (or models/ folder).

Running the Streamlit App

Place the trained model files in the models/ folder.
Run the app:

Bashstreamlit run app.py
The app will be available at http://localhost:8501.
Features:

Predict: Estimate SHD from building inputs
Zone Performance: View model metrics by climate zone
Methodology: Project overview

The app runs in limited mode if models are not found, you should input your own full directory where you pulled the repository.

Note: Trained model files are not included in the repository due to their size. You must generate them locally by running the notebooks.