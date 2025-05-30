import sys
import pickle
import pytest
import pandas as pd
import numpy as np
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/runner/work/energy_consumption/energy_consumption')
from ml_model.flask_app import app


def test_home():
    response = app.test_client().get('/')
    assert response.status_code == 200

# Load the trained model
@pytest.fixture(scope="module")
def model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

# Sample input for prediction
def test_model_prediction(model):
    feature_names = ['Building Type', 'Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week']
    test_data = pd.DataFrame([[1, 1200.0, 4, 10, 22.5, 0]], columns=feature_names)

    # Predict
    prediction = model.predict(test_data)

    # Check that prediction is a numeric value and within a sensible range
    assert prediction is not None
    assert isinstance(prediction[0], (int, float, np.integer, np.floating))
    assert prediction[0] >= 0  # Energy consumption should not be negative
