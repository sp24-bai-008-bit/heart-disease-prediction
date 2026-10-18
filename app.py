import streamlit as st
import pandas as pd
import pickle

# load the saved model and encoders
model = pickle.load(open("heart_svc_model.pkl", "rb"))
sex_le = pickle.load(open("sex_encoder.pkl", "rb"))
cp_le = pickle.load(open("cp_encoder.pkl", "rb"))

st.title("Heart Disease Prediction System")
st.write("Enter 4 details to predict heart disease risk.")

# the 4 inputs
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

if st.button("Predict"):
    # same steps as training: build row -> encode text -> predict
    row = pd.DataFrame({"age": [age], "sex": [sex_le.transform([sex])[0]],
                        "cp": [cp_le.transform([cp])[0]], "thalach": [thalach]})
    result = model.predict(row)[0]
    if result == 1:
        st.error("Prediction: HEART DISEASE LIKELY")
    else:
        st.success("Prediction: NO HEART DISEASE")
