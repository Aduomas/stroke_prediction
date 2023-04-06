# Stroke Prediction App

This GitHub repository contains the code for a Stroke Prediction App. The app is built using Streamlit, and it predicts the likelihood of a stroke based on real-life data. The model used for predictions is trained on a dataset of healthcare records.

## Features

- Predicts the likelihood of a stroke based on input features such as age, gender, hypertension, heart disease, ever married, work type, residence type, average glucose level, BMI, and smoking status.
- The app uses a machine learning model (XGBoost) trained on a dataset of healthcare records for predictions.
- The app can be hosted and used on Streamlit.

## Usage

To run the app locally, follow these steps:

1. Clone the repository to your local machine.
2. Make sure you have the required Python packages installed: `streamlit`, `pandas`, `scikit-learn`, `xgboost`, and `numpy`.
3. Run the following command in your terminal: `streamlit run app.py`
4. Open the URL displayed in your terminal (usually http://localhost:8501) to access the app.

## Data

The data used for training the model is provided in the `healthcare-dataset-stroke-data.csv` file. It contains information about patients, including their age, gender, hypertension, heart disease, work type, residence type, average glucose level, BMI, and smoking status, as well as whether they had a stroke.

## Model

The XGBoost classifier model is trained on the healthcare dataset to predict the likelihood of a stroke. The trained model is saved as `model.json` and is used in the app to make predictions.

## Dependencies

- streamlit
- pandas
- scikit-learn
- xgboost
- numpy
