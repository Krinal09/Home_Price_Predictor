from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Define file paths
DATA_FILE = 'Clean_Data.csv'
MODEL_FILE = 'Model.pkl'
DATA_PATH = os.path.join(os.path.dirname(__file__), DATA_FILE)
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILE)

# Load data and model
try:
    data = pd.read_csv(DATA_PATH)
    with open(MODEL_PATH, 'rb') as model_file:
        pipe = pickle.load(model_file)
except FileNotFoundError:
    print("Error: Data or model file not found.")
    exit()

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location, bhk, bath, sqft)

    input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    
    try:
        prediction = pipe.predict(input_data)[0] + 1e5
        return str(np.round(prediction, 2))
    except Exception as e:
        print("Error:", e)
        return "Error: Unable to make prediction."

if __name__ == "__main__":
    app.run(debug=True, port=5002)
