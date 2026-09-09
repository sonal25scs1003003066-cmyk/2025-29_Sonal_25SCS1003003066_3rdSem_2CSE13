🌦️ Weather Data Analysis and Next-Day Temperature Prediction

 AI/ML Internship Project

**Student:** Sonal 
**Roll Number:** 25SCS1003003066  
**Program:** B.Tech CSE (AI & ML)  
**Semester:** 3rd Semester  
**Section:** 2CSE13  
**Batch:** 2025–2029  
**University:** IILM University, Greater Noida  

**Organization:** Codec Technologies Pvt. Ltd.  
**Role:** Artificial Intelligence Intern  
**Duration:** 6 July 2026 – 6 August 2026

---

## 📌 Project Overview

This project was developed as part of my Artificial Intelligence and
Machine Learning internship at Codec Technologies Pvt. Ltd.

The project focuses on analyzing historical daily weather data and
predicting the next day's temperature using Machine Learning regression
models.

The project demonstrates a complete end-to-end AI/ML workflow:

**Raw Data → Data Cleaning → Feature Engineering → Exploratory Data
Analysis → Model Training → Prediction → Evaluation → Visualization**

---

## 🎯 Objectives

- Analyze historical weather data.
- Clean missing and inconsistent weather readings.
- Perform exploratory data analysis.
- Identify seasonal weather patterns.
- Create time-aware features.
- Predict next-day temperature.
- Implement Linear Regression as a baseline model.
- Implement Random Forest Regression.
- Compare model performance.
- Evaluate predictions using MAE, RMSE and R².
- Visualize the results.

---

## 🧠 Methodology

### 1. Data Generation / Collection

The repository contains a reproducible historical weather dataset so that
the complete project can run without requiring an external API.

The dataset contains:

- Date
- City
- Temperature
- Humidity
- Wind Speed
- Precipitation

### 2. Data Preprocessing

The preprocessing stage includes:

- Date parsing and chronological sorting
- Detection of inconsistent values
- Handling of missing values
- Time-based interpolation
- Removal of duplicate dates

### 3. Feature Engineering

Time-aware features are created to improve prediction:

- Day of year
- Month
- Seasonal sine/cosine features
- 1-day temperature lag
- 2-day temperature lag
- 3-day temperature lag
- 1-day humidity lag
- 2-day humidity lag
- 3-day humidity lag
- 7-day rolling temperature mean
- 7-day rolling humidity mean
- 7-day rolling wind-speed mean

The target variable is:

**Next-Day Temperature**

---

## 🤖 Machine Learning Models

### Linear Regression

Linear Regression is used as a baseline regression model.

### Random Forest Regression

Random Forest Regression is used to model non-linear relationships
between weather variables and the next day's temperature.

Both models are trained using a chronological train/test split.

---

## 📊 Model Evaluation

The models are evaluated using three standard regression metrics.

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted
values.

**Lower is better.**

### RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

**Lower is better.**

### R² — Coefficient of Determination

Measures how well the model explains the variation in the target
variable.

**Higher is better.**

---

## 📈 Results

The current saved evaluation results are:

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---:|---:|---:|
| Linear Regression | 1.294 | 1.649 | 0.974 |
| **Random Forest** | **1.292** | **1.639** | **0.975** |

Based on the saved results, Random Forest performs slightly better than
the Linear Regression baseline.

These results are specific to the included reproducible dataset and
configuration. They should not be interpreted as production weather
forecasting accuracy.

---

## 📊 Visualizations

### Seasonal Temperature Trends

![Seasonal Trends](outputs/figures/seasonal_trends.png)

### Correlation Heatmap

![Correlation Heatmap](outputs/figures/correlation_heatmap.png)

### Actual vs Predicted Temperature

![Actual vs Predicted](outputs/figures/actual_vs_predicted.png)

---
````

## 📂 Project Structure


Weather-Internship-Project/
│
├── data/
│   ├── historical_weather.csv
│   ├── historical_weather_clean.csv
│   └── weather_features.csv
│
├── notebooks/
│   └── weather_analysis_pipeline.ipynb
│
├── outputs/
│   ├── metrics.json
│   ├── test_predictions.csv
│   └── figures/
│       ├── seasonal_trends.png
│       ├── actual_vs_predicted.png
│       └── correlation_heatmap.png
│
├── src/
│   ├── generate_data.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── visualize.py
│   └── main.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
