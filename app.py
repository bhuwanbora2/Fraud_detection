import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details:")

amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Predict"):

    # Create dummy feature array (30 features total)
    features = np.zeros(30)
    features[-1] = amount

    prediction = model.predict([features])

    if prediction[0] == 1:
        st.error("⚠ Fraud Transaction Detected!")
    else:
        st.success("✅ Normal Transaction")