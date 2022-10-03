import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

st.header("Stroke Prediction App")
data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data.dropna(inplace=True)
data = data[data['gender'] != 'Other']

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
best_xgboost_model.load_model("model.json")
    
left_column, right_column = st.columns(2)
with left_column:
    input_gender = st.radio(
        'Gender type:',
        np.unique(data['gender']))
    
input_age = st.slider('Age (years)', 0.0, max(data["age"]), 40.0)
input_hypertension = st.checkbox('Hypertension', False)
input_heart_disease = st.checkbox('Heart Disease',False)
input_married = st.checkbox('Ever Married', False)

left_column, right_column = st.columns(2)
with left_column:
    input_worktype = st.radio(
        'Work type:',
        np.unique(data['work_type']))
    
left_column, right_column = st.columns(2)
with left_column:
    input_residence = st.radio(
        'Residence type:',
        np.unique(data['Residence_type']))

input_glucose = st.slider('Average Glucose Level', min(data["avg_glucose_level"]), max(data["avg_glucose_level"]), 100.0)
input_bmi = st.slider('BMI', min(data["bmi"]), max(data["bmi"]), 22.0)

left_column, right_column = st.columns(2)
with left_column:
    input_smoking = st.radio(
        'Smoking status:',
        np.unique(data['smoking_status']))
    
data['gender'] = encoder_gender.transform(data['gender'])
data['ever_married'] = encoder_married.transform(data['ever_married'])
data['work_type'] = encoder_work.transform(data['work_type'])
data['Residence_type'] = encoder_residence.transform(data['Residence_type'])
data['smoking_status'] = encoder_smoking.transform(data['smoking_status'])

x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0,5,9])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

   
if st.button('Make Prediction'):
    input_gender = encoder_gender.transform(np.expand_dims(input_gender, -1))
    married_dict = {False: "No", True: "Yes"}
    input_married = encoder_married.transform(np.expand_dims(married_dict[input_married], -1))
    input_residence = encoder_residence.transform(np.expand_dims(input_residence, -1))
    input_smoking = encoder_smoking.transform(np.expand_dims(input_smoking, -1))
    input_worktype = encoder_work.transform(np.expand_dims(input_worktype, -1))
    inputs = np.array([[input_gender[0], input_age, int(input_hypertension), int(input_heart_disease), int(input_married), input_worktype[0], input_residence[0], input_glucose, input_bmi, input_smoking[0]]])
    inputs = ct.transform(inputs)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    prediction_dict = {0: "Negative", 1: "Positive"}
    st.write(f"Your stroke prediction is: **{prediction_dict[int(np.squeeze(prediction, -1))]}!**")