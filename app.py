import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model dan scaler dari file lokal (gunakan joblib)
model_path = 'best_obesity_model.pkl'
scaler_path = 'scaler.pkl'
ordinal_mapping_path = 'ordinal_mappings.pkl'
encoders_path = {
    'Gender': 'Gender_encoder.pkl',
    'HighCalorieFood': 'HighCalorieFood_encoder.pkl',
    'CalorieMonitoring': 'CalorieMonitoring_encoder.pkl',
    'FamilyHistoryOverweight': 'FamilyHistoryOverweight_encoder.pkl',
    'Transportation': 'Transportation_encoder.pkl'
}

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    ordinal_mappings = joblib.load(ordinal_mapping_path)

    encoders = {}
    for col, enc_path in encoders_path.items():
        encoders[col] = joblib.load(enc_path)

except FileNotFoundError:
    st.error("Pastikan semua file .pkl berada di direktori yang sama dengan aplikasi Streamlit.")
    st.stop()

# Judul Aplikasi
st.title("Prediksi Tingkat Obesitas")
st.write("Aplikasi ini memprediksi tingkat obesitas berdasarkan karakteristik individu.")

# Input Form
st.header("Masukkan Data Individu")

gender = st.selectbox("Gender", ['Female', 'Male'])
age = st.slider("Age", 10, 100, 25)
height = st.number_input("Height (in meters)", min_value=0.5, max_value=3.0, value=1.70, step=0.01)
weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
family_history = st.selectbox("Family History with Overweight", ['no', 'yes'])
favc = st.selectbox("High Calorie Food Consumption (FAVC)", ['no', 'yes'])
fcvc = st.slider("Vegetable Consumption (FCVC)", 1, 3, 2)
ncp = st.slider("Meal Frequency (NCP)", 1, 4, 3)
caec = st.selectbox("Snack Consumption (CAEC)", ['No', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox("Smoking (SMOKE)", ['no', 'yes'])
ch2o = st.slider("Water Intake (CH2O)", 1, 3, 2)
scc = st.selectbox("Calorie Monitoring (SCC)", ['no', 'yes'])
faf = st.slider("Physical Activity (FAF)", 0, 3, 1)
tue = st.slider("Technology Use (TUE)", 0, 2, 1)
calc = st.selectbox("Alcohol Consumption (CALC)", ['No', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox("Transportation (MTRANS)", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# Buat DataFrame dari input
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'FamilyHistoryOverweight': [family_history],
    'HighCalorieFood': [favc],
    'VegetableConsumption': [fcvc],
    'MealFrequency': [ncp],
    'SnackConsumption': [caec],
    'Smoking': [smoke],
    'WaterIntake': [ch2o],
    'CalorieMonitoring': [scc],
    'PhysicalActivity': [faf],
    'TechnologyUse': [tue],
    'AlcoholConsumption': [calc],
    'Transportation': [mtrans]
})

# Proses Encoding
for col, mapping in ordinal_mappings.items():
    input_data[col] = input_data[col].map(mapping)

for col, encoder in encoders.items():
    try:
        input_data[col] = encoder.transform(input_data[col])
    except ValueError:
        st.warning(f"Nilai '{input_data[col][0]}' pada kolom '{col}' tidak dikenal oleh model.")
        input_data[col] = 0

# Drop kolom Smoking karena tidak digunakan dalam training
if 'Smoking' in input_data.columns:
    input_data = input_data.drop(columns=['Smoking'])

# Pastikan urutan kolom sesuai dengan training
expected_cols = [
    'Gender', 'Age', 'Height', 'Weight', 'FamilyHistoryOverweight',
    'HighCalorieFood', 'VegetableConsumption', 'MealFrequency', 'SnackConsumption',
    'WaterIntake', 'CalorieMonitoring', 'PhysicalActivity', 'TechnologyUse',
    'AlcoholConsumption', 'Transportation'
]
try:
    input_data = input_data[expected_cols]
except KeyError as e:
    st.error(f"Kolom hilang: {e}")
    st.stop()

# Scaling
numeric_columns = [col for col in expected_cols if col not in [
    'AlcoholConsumption', 'SnackConsumption', 'Gender',
    'FamilyHistoryOverweight', 'HighCalorieFood', 'CalorieMonitoring', 'Transportation'
]]
input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

# Prediksi
if st.button("Predict Obesity Level"):
    prediction_encoded = model.predict(input_data)

    obesity_level_mapping = {
        0: 'Clinical_Obesity_I',
        1: 'Clinical_Obesity_II',
        2: 'Clinical_Obesity_III',
        3: 'Insufficient_Weight',
        4: 'Normal_Weight',
        5: 'Overweight_Level_I',
        6: 'Overweight_Level_II'
    }

    predicted_level = obesity_level_mapping.get(prediction_encoded[0], "Unknown")

    st.header("Hasil Prediksi")
    st.success(f"Tingkat obesitas yang diprediksi adalah: **{predicted_level}**")
