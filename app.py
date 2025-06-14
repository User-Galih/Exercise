# prompt: kemudian buatkan code deploy ke streamlit nya dengan akses file .pkl nya lokal bukan gdrive

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load the model and scaler from local files
# Make sure 'best_obesity_model.pkl' and 'scaler.pkl' are in the same directory as your Streamlit app script
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
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(ordinal_mapping_path, 'rb') as f:
        ordinal_mappings = pickle.load(f)

    encoders = {}
    for col, enc_path in encoders_path.items():
         with open(enc_path, 'rb') as f:
            encoders[col] = pickle.load(f)

except FileNotFoundError:
    st.error("Pastikan file 'best_obesity_model.pkl', 'scaler.pkl', 'ordinal_mappings.pkl', dan encoder files berada di direktori yang sama dengan aplikasi Streamlit.")
    st.stop()

# Streamlit App Title and Description
st.title("Prediksi Tingkat Obesitas")
st.write("Aplikasi ini memprediksi tingkat obesitas berdasarkan karakteristik individu.")

# Input Fields
st.header("Masukkan Data Individu")

gender = st.selectbox("Gender", ['Female', 'Male'])
age = st.slider("Age", 10, 100, 25)
height = st.number_input("Height (in meters)", min_value=0.5, max_value=3.0, value=1.70, step=0.01)
weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.1)
family_history = st.selectbox("Family History with Overweight", ['no', 'yes'])
favc = st.selectbox("High Calorie Food Consumption (FAVC)", ['no', 'yes'])
fcvc = st.slider("Vegetable Consumption (FCVC): How often do you eat vegetables? (1: Never, 2: Sometimes, 3: Always)", 1, 3, 2)
ncp = st.slider("Meal Frequency (NCP): How many main meals do you have daily? (1 to 4)", 1, 4, 3)
caec = st.selectbox("Snack Consumption (CAEC): How often do you eat between meals?", ['No', 'Sometimes', 'Frequently', 'Always'])
smoke = st.selectbox("Smoking (SMOKE)", ['no', 'yes']) # Smoking was dropped but include for completeness
ch2o = st.slider("Water Intake (CH2O): How many liters of water do you drink daily? (1 to 3)", 1, 3, 2)
scc = st.selectbox("Calorie Monitoring (SCC): Do you monitor your calorie intake?", ['no', 'yes'])
faf = st.slider("Physical Activity (FAF): How often do you engage in physical activity? (0: Never, 1: 1-2 days/week, 2: 2-4 days/week, 3: 4-5 days/week)", 0, 3, 1)
tue = st.slider("Technology Use (TUE): How much time do you spend using technology? (0: 0-2 hours, 1: 3-5 hours, 2: >5 hours)", 0, 2, 1)
calc = st.selectbox("Alcohol Consumption (CALC): How often do you consume alcohol?", ['No', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox("Transportation (MTRANS)", ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])

# Create a DataFrame from inputs
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
    'Smoking': [smoke], # Include Smoking for now, drop later
    'WaterIntake': [ch2o],
    'CalorieMonitoring': [scc],
    'PhysicalActivity': [faf],
    'TechnologyUse': [tue],
    'AlcoholConsumption': [calc],
    'Transportation': [mtrans]
})

# Preprocessing the input data
# Apply encoding
for col, mapping in ordinal_mappings.items():
    input_data[col] = input_data[col].map(mapping)

for col, encoder in encoders.items():
    # Check if the input value exists in the encoder's classes
    if input_data[col][0] in encoder.classes_:
        input_data[col] = encoder.transform(input_data[col])
    else:
        # Handle cases where the input value is not in the training data classes
        # This is a simplified approach; a robust solution might require more
        # sophisticated handling of unseen categories, like using a default or OHE.
        st.warning(f"Input value '{input_data[col][0]}' for '{col}' was not seen in training data. Prediction may be inaccurate.")
        # For simplicity, let's use the mode of the training data for this category
        # You would need to load the mode from your training data or handle it differently
        # For now, we'll just use 0 as a placeholder (assuming 0 is a valid class)
        # A better approach is to use OneHotEncoder and handle unknown values
        try:
             # Try to find the index if the value exists
             input_data[col] = encoder.transform(input_data[col])
        except ValueError:
            # If transform fails (unseen value), replace with a placeholder or handle appropriately
             # For this example, let's assume a placeholder of 0 is acceptable for unseen categories
             input_data[col] = 0
             st.warning(f"Handling unseen category for {col}. Using placeholder value.")


# Drop Smoking column as it was dropped during training
if 'Smoking' in input_data.columns:
    input_data = input_data.drop(columns=['Smoking'])

# Ensure column order is the same as the training data (X_train from notebook)
# Get the original column order from your training script's X_train or saved column list
# Assuming the column order from your notebook's X_train is available or consistent
# You need the exact list of columns X was trained on *before* scaling
# Based on your notebook:
# X = df.drop(columns=['ObesityLevel', 'Smoking'])
# The columns should be in the order they appeared in the original df after initial processing
# Let's reconstruct the expected column order based on your notebook
expected_cols = [
    'Gender', 'Age', 'Height', 'Weight', 'FamilyHistoryOverweight',
    'HighCalorieFood', 'VegetableConsumption', 'MealFrequency', 'SnackConsumption',
    'WaterIntake', 'CalorieMonitoring', 'PhysicalActivity', 'TechnologyUse',
    'AlcoholConsumption', 'Transportation'
]

# Reindex the input data to match the expected column order
try:
    input_data = input_data[expected_cols]
except KeyError as e:
    st.error(f"Missing column in input data: {e}. Please check input fields.")
    st.stop()


# Scale the numerical features
# Identify numerical columns (same as in your notebook preprocessing)
numeric_columns = [col for col in expected_cols if col not in ['AlcoholConsumption', 'SnackConsumption', 'Gender',
                                                               'FamilyHistoryOverweight', 'HighCalorieFood',
                                                               'CalorieMonitoring', 'Transportation']] # Add other non-numeric columns dropped from scaling

# Apply scaling
input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])


# Make prediction
if st.button("Predict Obesity Level"):
    prediction_encoded = model.predict(input_data)

    # Reverse the encoding for the prediction
    # Need the mapping from encoded label back to original ObesityLevel string
    # Based on your notebook's label encoding for ObesityLevel
    obesity_level_mapping = {
        0: 'Clinical_Obesity_I',
        1: 'Clinical_Obesity_II',
        2: 'Clinical_Obesity_III',
        3: 'Insufficient_Weight',
        4: 'Normal_Weight',
        5: 'Overweight_Level_I',
        6: 'Overweight_Level_II'
    } # Adjust based on your actual ObesityLevel encoder classes order

    predicted_level = obesity_level_mapping.get(prediction_encoded[0], "Unknown")

    st.header("Prediction Result")
    st.success(f"Based on the input data, the predicted Obesity Level is: **{predicted_level}**")
