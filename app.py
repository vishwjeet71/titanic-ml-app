import streamlit as st
import pandas as pd
import numpy as np
import joblib


#Random Forest model
model = joblib.load('RF.pkl')

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("Titanic Survival Predictor")
st.markdown("Enter passenger details to predict if they would survive the Titanic disaster.")

# user input
with st.form("prediction_form"):
    pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=8, value=0)
    parch = st.number_input("Number of Parents/Children", min_value=0, max_value=6, value=0)
    fare = st.number_input("Fare Paid (in $)", min_value=0.0, max_value=600.0, value=30.0)
    cabin = st.text_input("Cabin (Leave blank if unknown)", "")
    sex = st.selectbox("Gender", ["male", "female"])
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    submit = st.form_submit_button("Predict")

# Input preprocessing function — aligned with your model features
def preprocess_input(pclass, age, sibsp, parch, cabin, sex, embarked, fare):
    family_group = 1 if (sibsp + parch) > 0 else 0
    Cabin = 1 if cabin.strip() else 0
    female = 1 if sex == "female" else 0
    male = 1 if sex == "male" else 0
    Cherbourg = 1 if embarked == "C" else 0
    Queenstown = 1 if embarked == "Q" else 0
    Southampton = 1 if embarked == "S" else 0
    Fare_Transformed_Capped = np.log1p(fare)

    data = pd.DataFrame([[
        pclass, age, family_group, Cabin, female, male,
        Cherbourg, Queenstown, Southampton, Fare_Transformed_Capped
    ]], columns=[
        'Pclass', 'Age', 'family_group', 'Cabin', 'female', 'male',
        'Cherbourg', 'Queenstown', 'Southampton', 'Fare_Transformed_Capped'
    ])
    return data

# Run prediction
if submit:
    input_df = preprocess_input(pclass, age, sibsp, parch, cabin, sex, embarked, fare)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100  # Survival probability

    if prediction == 1:
        st.success(f"🎉 Prediction: Survived ({probability:.2f}% confidence)")
    else:
        st.error(f"💀 Prediction: Did Not Survive ({100 - probability:.2f}% confidence)")
