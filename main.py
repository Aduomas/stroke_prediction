import streamlit as st
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np

st.header("Stroke Prediction App")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

encoder_gender = LabelEncoder()
encoder_gender.classes_ = np.load('genders.npy',allow_pickle=True)

encoder_married = LabelEncoder()
encoder_married.classes_ = np.load('married.npy',allow_pickle=True)

encoder_residence = LabelEncoder()
encoder_residence.classes_ = np.load('residencetypes.npy',allow_pickle=True)

encoder_work = LabelEncoder()
encoder_work.classes_ = np.load('worktypes.npy',allow_pickle=True)

encoder_smoking = LabelEncoder()
encoder_smoking.classes_ = np.load('smokingstatus.npy',allow_pickle=True)

best_xgboost_model = xgb.XGBClassifier()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show dataframe'):
    data
    
st.subheader("Please select relevant features of your fish!")
left_column, right_column = st.columns(2)
with left_column:
    inp_species = st.radio(
        'Name of the fish:',
        np.unique(data['stroke']))