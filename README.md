# AgriLST-ML — Statewide Thermal Sharpening for Land Surface Temperature

Machine learning pipeline that downscales MODIS Land Surface Temperature (LST) from **1 km to 10 m resolution** using Sentinel-2 imagery and engineered environmental features, generalized across **36 districts of Maharashtra**.

> This project was built for personal learning and portfolio purposes — to explore thermal downscaling, spatial generalization, and gradient-boosted regression at scale. The trained model is included for reference/demo use, not production deployment.

---

## Overview

Thermal bands from satellites like MODIS offer frequent revisits but coarse spatial resolution (~1 km), which is too coarse for field-level agricultural decisions. This project trains an XGBoost regression model to predict fine-resolution LST (10 m) using Sentinel-2 derived vegetation indices, temperature, rainfall, soil, and terrain features — effectively "sharpening" coarse thermal data to a resolution useful for crop and irrigation stress analysis.

The project evolved in two stages:
1. **Regional model** — trained on 10 Maharashtra districts.
2. **Statewide generalized model** — retrained on all 36 districts (~3.19M rows) with spatial cross-validation and per-district bias calibration, to test generalization beyond the training region.

---

## Data

- **Source imagery:** Sentinel-2 (NDVI, NDWI) and MODIS LST
- **Auxiliary features:** air temperature, rainfall (0-day and 15-day), soil texture/clay/sand fraction, elevation, district mean elevation, coastline distance proxy, day-of-year cyclic encodings (`doy_sin`, `doy_cos`)
- **Scale:** 36 districts, ~3,189,628 rows in the statewide dataset
- **Seasons covered:** kharif, early_rabi, rabi, summer

## Methodology

- **Model:** XGBoost regression
- **Validation:** Leave-One-District-Out (LOGO) spatial cross-validation, to test how well the model generalizes to districts unseen during training
- **Calibration:** Per-district bias correction applied post-prediction
- **Feature engineering:** polynomial (`airtemp_C_squared`), interaction (`NDVI_x_AirTemp`), and ratio (`NDWI_div_Clay`) terms in addition to raw features

## Model Performance (Statewide Generalized Model)

| Metric | Value |
|---|---|
| RMSE | 3.06 °C |
| MAE | 2.29 °C |
| R² | 0.780 |
| Mean Bias | -0.08 °C |

Performance breakdown (see `reports/`):
- RMSE is lowest in `early_rabi` and highest in `summer`, consistent with harder prediction at temperature extremes.
- Residuals show mild heteroscedasticity at higher predicted LST.
- Per-district RMSE and bias are broadly stable across the 36 districts, indicating the model generalizes reasonably well spatially.

**Top features by gain:** `airtemp_C_squared`, `airtemp_C`, `ndvi`, `doy_cos`
**Top features by split weight:** `doy_sin`, `airtemp_C`, `doy_cos`, `district_mean_elevation`

---

## Repository Structure

```
AgriLST-ML/
├── notebooks/
│   ├── Data_Preparation_Pipeline_v9_statewide.ipynb   # feature engineering & dataset assembly
│   └── XgBoost_prod_v2_district_calibrated.ipynb       # training, LOGO CV, bias calibration
├── model/
│   ├── agrilst_final_xgboost.pkl      # trained XGBoost model
│   └── agrilst_final_metadata.pkl     # feature list, calibration params, metadata
│   ├── 01_residual_analysis.png
│   ├── 02_feature_importance.png
│   └── 03_summer_analysis.png
├── requirements.txt
└── README.md
```

## Usage

```bash
git clone https://github.com/Suyash-Bhosale2112/AgriLST-ML.git
cd AgriLST-ML
pip install -r requirements.txt
```

Load the trained model for inference:

```python
import pickle

with open("model/agrilst_final_xgboost.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/agrilst_final_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# metadata contains feature order, per-district bias correction, etc.
preds = model.predict(X[metadata["feature_names"]])
```

Open the notebooks in Jupyter or Colab to see the full pipeline — from raw Sentinel-2/MODIS inputs through feature engineering, spatial CV, and evaluation.

---

## Notes

- This is an educational/portfolio project — the model was trained on data current as of the training period and is not maintained for production inference.
- Spatial cross-validation (LOGO) was used specifically to give an honest estimate of out-of-region generalization, rather than relying on random train/test splits which tend to overstate performance on spatially correlated geospatial data.

