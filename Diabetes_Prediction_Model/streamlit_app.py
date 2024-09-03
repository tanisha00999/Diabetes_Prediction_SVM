import streamlit as st
import numpy as np
import pickle

# Load the trained classifier model and scaler
model = pickle.load(open('classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title and description for the web app
st.title("Diabetes Prediction App")
st.write("""
This app predicts whether a person is diabetic based on the following health parameters:
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
""")

# Input fields for user data
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=79)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input('Age', min_value=0, max_value=120, value=33)

# Prepare the input for prediction
input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Scale the input data
std_data = scaler.transform(input_data_as_numpy_array)

# Make a prediction
prediction = model.predict(std_data)

# Display the result
if prediction[0] == 1:
    st.error("The person is diabetic.")
else:
    st.success("The person is non-diabetic.")
