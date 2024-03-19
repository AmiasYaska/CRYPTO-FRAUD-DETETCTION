import json

import requests
import pandas as pd

# Load the sample dataset
sample_data = pd.read_csv("test.csv")

# URL of your Flask API endpoint
url = 'http://127.0.0.1:5000/predict'  # Replace this with your actual API endpoint URL

# Convert sample data to JSON format
data_json = sample_data.to_json(orient='records')

# Send POST request to the API endpoint
response = requests.post(url, json=json.loads(data_json))

# Check the response status code
if response.status_code == 200:
    predictions = response.json()['predictions']
    print('Predictions:', predictions)
else:
    print('Error occurred:', response.json()['error'])
