import streamlit as st
import numpy as np
import joblib

# Load trained objects
model = joblib.load("models/wine_model.joblib")
scaler = joblib.load("models/wine_scaler.joblib")

st.title("🍷 Wine Classifier")
st.write("Enter the wine's chemical properties to predict its class.")

# Input features
alcohol = st.number_input("Alcohol", value=13.0)
malic_acid = st.number_input("Malic Acid", value=2.0)
ash = st.number_input("Ash", value=2.3)
alcalinity_of_ash = st.number_input("Alcalinity of Ash", value=19.0)
magnesium = st.number_input("Magnesium", value=100.0)
total_phenols = st.number_input("Total Phenols", value=2.5)
flavanoids = st.number_input("Flavanoids", value=2.5)
nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", value=0.3)
proanthocyanins = st.number_input("Proanthocyanins", value=1.5)
color_intensity = st.number_input("Color Intensity", value=5.0)
hue = st.number_input("Hue", value=1.0)
od280_od315 = st.number_input("OD280/OD315 of Diluted Wines", value=3.0)
proline = st.number_input("Proline", value=1000.0)

if st.button("Predict"):

    # Put inputs in the same order as training features
    X = np.array([[
        alcohol,
        malic_acid,
        ash,
        alcalinity_of_ash,
        magnesium,
        total_phenols,
        flavanoids,
        nonflavanoid_phenols,
        proanthocyanins,
        color_intensity,
        hue,
        od280_od315,
        proline
    ]])

    # Apply the SAME scaler used during training
    X_scaled = scaler.transform(X)

    # Make prediction
    prediction = model.predict(X_scaled)

    st.success(f"Predicted Wine Class: {prediction[0]}")