import streamlit as st
import pandas as pd
import joblib

# load model
model = joblib.load("titanic_survival_model.pkl")

# app title
st.title("Titanic Survival Prediction App")
st.write("Enter passenger information below to predict survival.")

# inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
fare = st.number_input("Fare", min_value=0.0, value=32.0, step=1.0)

# prediction button
if st.button("Predict"):
    input_df = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "Fare": [fare]
    })

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did Not Survive")
