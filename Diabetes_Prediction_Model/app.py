from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and scaler
with open('model/classifier.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_features = [float(x) for x in request.form.values()]
    input_data_as_numpy_array = np.asarray(input_features).reshape(1, -1)
    
    # Standardize input data
    std_data = scaler.transform(input_data_as_numpy_array)
    
    # Make prediction
    prediction = classifier.predict(std_data)
    
    if prediction[0] == 0:
        result = "The person is non-diabetic."
    else:
        result = "The person is diabetic."
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
