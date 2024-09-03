# Diabetes Prediction Web App

## Overview
This project is a web application for predicting diabetes based on user-provided health data. It uses a Support Vector Machine (SVM) model trained on medical data to classify whether a person is diabetic or not. The web interface is built using Flask.

## Features
- **User Input**: Enter health parameters like glucose level, blood pressure, BMI, etc.
- **Prediction**: The app uses a trained SVM model to predict if a person is diabetic.
- **User-Friendly Interface**: A simple and intuitive HTML form for input and display of results.

  ![Diabetes Prediction Web App Screenshot](screenshot.png)


## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.6 or higher
- pip (Python package installer)
- Flask
- NumPy
- Scikit-learn
- Pickle

## Installation

### 1. Clone the Repository
- git clone https://github.com/yourusername/diabetes-prediction.git
- cd diabetes-prediction

### 2. Run the Flask App
- python app.py

### 3. Open Browser
- Open in Browser: Go to http://127.0.0.1:5000/ to use the app.

## Usage
- Input the required health data into the form fields (like glucose level, blood pressure, BMI, etc.).
- Click on the "Predict" button.
- The app will display whether the person is diabetic or not.

## Data
The model was trained on the Pima Indians Diabetes Database, which is publicly available on Kaggle.




