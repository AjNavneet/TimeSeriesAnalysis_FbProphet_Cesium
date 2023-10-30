# Time Series Analysis with FbProphet and Cesium

## Business Objective

Working with time series data can be challenging, especially for those without in-depth knowledge of various machine learning or deep learning models. To make time series analysis more accessible to data analysts and data scientists, two open-source libraries, FbProphet and Cesium, were introduced. These libraries empower users to perform powerful and intuitive time-series modeling without requiring expertise in forecasting techniques.

FbProphet, developed by Facebook, offers a user-friendly forecasting tool available in Python and R. It simplifies time-series forecasting with easily tunable parameters, making it accessible even to users without deep expertise.

Cesium is another open-source library that allows users to extract features from raw time series data, build machine learning models, and generate predictions for new data.

In this project, we aim to build models using the FbProphet and Cesium libraries for time series analysis.

---

## Data Description

The dataset is "Call-centres" data, recorded at a monthly level. Calls are categorized by domain as the call center operates for various domains. The dataset also includes external regressors, such as the number of channels and phone lines, indicating traffic predictions by in-house analysts and resource availability. The dataset contains a total of 132 rows and 8 columns:

- Month, healthcare, telecom, banking, technology, insurance, number of phone lines, and number of channels.

---

## Aim

The primary objectives of this project are to:
- Build a prophet model using the FbProphet library.
- Create a multi-layer perceptron (MLP) model using the Cesium library on the provided time series dataset.

---

## Tech Stack

- **Language:** `Python`
- **Libraries:** `pandas`, `NumPy`, `matplotlib`, `seaborn`, `TensorFlow`, `Keras`, `FbProphet`, `Cesium`

---

## Approach

1. Import the required libraries and read the dataset.
2. Perform descriptive analysis.
3. Exploratory Data Analysis (EDA):
   - Visualize the data.
4. FbProphet:
   - Define a prophet model.
   - Fit the model.
   - Create a dataframe with future date values.
   - Make predictions within a defined range.
   - Create models with varying changepoints parameters.
   - Plot the results.
5. Cesium:
   - Convert month to date timestamp format.
   - Reshape the data.
   - Extract features from raw time-series data.
   - Perform train-test split on the data.
   - Build a machine learning model (MLP) using these features.
   - Fit and train the MLP.
   - Generate predictions for the test data.
   - Plot the results.

---

## Modular Code Overview

- **Input folder:** Contains the data for analysis (e.g., CallCenterData.xlsx).

- **Src folder:** Contains modularized code for all project steps, including:
  - Engine.py
  - ML_Pipeline (folder): Contains Python functions organized into appropriately named files, called within the engine.py file.

- **Output folder:** Contains two subfolders:
  - Plots: Includes various visualization plots.
  - Models: Contains four models saved in pickle format.

- **Lib folder:** Includes an IPython notebook

---

## Key Concepts Explored

1. FbProphet and its parameters.
2. Building and fine-tuning FbProphet models.
3. Introduction to Cesium and feature extraction.
4. Building and training a Multi-Layer Perceptron (MLP) model.
5. Making predictions on test data.
6. Visualizing the results.

---