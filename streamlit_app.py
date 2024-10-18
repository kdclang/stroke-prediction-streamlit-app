import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
import sklearn


def get_data():
    model, scaler = pickle.load(open('stroke_mdl.pkl', 'rb'))
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    return model, scaler, df

def YesNo(n):
    if n == 'Yes':
        return 1
    else:
        return 0


model, scaler, df = get_data()
st.title("Stroke Predictor App")
st.write(
    'Fill in the information to predict your risk of stroke')

age = st.slider('Age', 0, 100, 50)
gender = st.radio(label="Gender", options=['Male', 'Female', 'Other'])
married = st.radio(label="Ever married?", options=['Yes', 'No'])
hypertension = st.radio(label="Hypertension?", options=['Yes', 'No'])
heart_disease = st.radio(label="Heart disease?", options=['Yes', 'No'])
glucose = st.slider('Average Glucose Level', 0, 300, value=int(df['avg_glucose_level'].mean()))
bmi = st.slider('Body Mass Index', 0, 70, value=int(df['bmi'].mean()))
smoked = st.selectbox('Smoking Status', df['smoking_status'].unique())

if gender == "Male":
    gender = 1
elif gender == 'Female':
    gender = 0
else:
    gender = random.choice([0,1])

value_list = [1,2,3,4]
smoking_dict = {key:value for key, value in zip(df['smoking_status'].unique(), value_list)}

user_values = [[gender, age, YesNo(hypertension), YesNo(heart_disease), YesNo(married),
                 glucose, bmi, smoking_dict.get(smoked)]]


if st.button('Predict'):
    result = model.predict(scaler.transform(user_values))
    risk = "The stroke risk is "
    if result == 1:
        risk += "HIGH"
    else:
        risk += "LOW"
    st.subheader(risk)
