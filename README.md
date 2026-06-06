# AQI-PIPELINE
# Pearls AQI Predictor
### End-to-End Machine Learning Pipeline for Air Quality Forecasting — Karachi, Pakistan

**Submitted by:** Rana Mohsin  
**Organization:** 10Pearls  
**University:** Iqra University, Karachi | CGPA: 3.81  
**Date:** June 2026

---

## What This Project Does

This system collects real-time air quality data for Karachi, trains machine learning models to predict AQI, and serves forecasts through an interactive dashboard and REST API — fully automated with no manual intervention.

- Collects hourly pollutant data from OpenWeatherMap API
- Trains 3 ML models: Random Forest, Ridge Regression, TensorFlow Neural Network
- Generates 3-day (72-hour) AQI forecasts
- Displays results on a Streamlit dashboard with SHAP explainability
- Exposes predictions via a Flask REST API
- Runs entirely on GitHub Actions CI/CD — 328+ automated commits over 6 months

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| Scikit-learn | Random Forest + Ridge Regression |
| TensorFlow/Keras | Neural Network model |
| Flask | REST API |
| Streamlit | Interactive dashboard |
| GitHub Actions | CI/CD automation |
| OpenWeatherMap API | Live AQI + weather data |
| SHAP | Model explainability |

---

## Project Structure

```
AQI-PIPELINE/
├── AQI_Project.py          # Hourly data collection + feature engineering
├── train_model.py          # Train Random Forest + Ridge Regression
├── train_tf_model.py       # Train TensorFlow Neural Network
├── dashboard.py            # Streamlit dashboard
├── app.py                  # Flask REST API
├── eda.py                  # Exploratory data analysis + plots
├── aqi_model.pkl           # Trained Random Forest model
├── aqi_tf_model.keras      # Trained TensorFlow model
├── scaler.pkl              # StandardScaler for TF model
├── karachi_raw_data.csv    # Raw collected data
├── karachi_clean_dataset.csv  # Engineered feature dataset
├── eda_plots/              # 7 EDA visualizations
└── .github/workflows/      # GitHub Actions CI/CD pipelines
```

---

## Dataset

- **Source:** OpenWeatherMap Air Pollution API
- **Coverage:** November 2025 – May 2026 (6 months)
- **Total Records:** 4,410 hourly samples
- **Pollutants:** PM2.5, PM10, NO2, CO, O3, SO2, NH3
- **Collection:** Automated every hour via GitHub Actions

---

## Features Used (13 Total)

| Feature | Type |
|---------|------|
| pm25, pm10, no2, co, o3, so2, nh3 | Raw pollutants |
| hour, day_of_week, month | Time-based |
| aqi_lag_1h, aqi_lag_3h | Lag features |
| aqi_change | Derived (rate of change) |

---

## Model Results

| Model | RMSE | R² Score | Accuracy |
|-------|------|----------|----------|
| Random Forest | 0.0583 | 0.9955 | 99.66% |
| Ridge Regression | 0.0009 | 1.0000* | 100%* |
| TensorFlow Neural Net | 0.1700 | 0.8699 | 86.99% |

> *Ridge R²=1.0 indicates overfitting on lag-dominated data. See Limitations section below.

---

## Automated Pipelines (GitHub Actions)

| Workflow | Schedule | What It Does |
|----------|----------|--------------|
| Feature Pipeline | Every hour | Fetches live data, engineers features, saves to CSV |
| Training Pipeline | Daily | Retrains all 3 models on latest data |
| CI Check | On every push | Validates code before merging |

---

## Flask API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/current` | GET | Live AQI and pollutant levels |
| `/predict` | POST | Accepts pollutant JSON, returns AQI prediction |
| `/forecast` | GET | 72-hour AQI forecast as JSON array |

---

## Dashboard Features

- Real-time AQI display with health advisory
- Live pollutant cards: PM2.5, PM10, NO2, O3
- 3-day forecast with day-by-day summary
- Model selector: switch between all 3 models
- Historical trends with Plotly charts
- SHAP feature importance visualization
- Hazardous AQI alert system (triggers at AQI 4 or 5)

---

## Honest Limitations

**1. High accuracy is misleading**  
The 99.66% accuracy is driven by lag features (AQI barely changes hour to hour), not by the model discovering complex patterns. True multi-day forecasting accuracy without lag data would be significantly lower.

**2. No Hopsworks/Vertex AI**  
Billing access was not available. GitHub CSV is used as a lightweight feature store instead.

**3. No historical weather data**  
OpenWeatherMap free tier does not provide historical temperature/humidity/wind data, which limits the model's ability to capture weather-driven pollution events.

**4. Single city only**  
Model is trained on Karachi data and will not generalize to other cities without retraining.

---

## How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENWEATHER_KEY=your_key_here

# Collect one data sample
python AQI_Project.py

# Train models
python train_model.py
python train_tf_model.py

# Run dashboard
streamlit run dashboard.py

# Run Flask API
python app.py
```

---

## Report

Full project report is available in `AQI_Predictor_Report_Final.pdf` in this repository.

---

*Rana Mohsin | Data Sciences Intern | 10Pearls | June 2026*
