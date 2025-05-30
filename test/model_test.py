import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/runner/work/energy_consumption/energy_consumption')
from ml_model.flask_app import app
import json

def test_home():
    response = app.test_client().get('/')
    assert response.status_code == 200
    #assert b"Iris Prediction API is running" in response.data

def test_prediction():
    response = app.test_client().post('/predict', 
        data=json.dumps({'features': [1,5000, 35, 14, 20,0]}),
        content_type='application/json')
    json_data = response.get_json()
    assert response.status_code == 200
    assert 'prediction' in json_data
    assert json_data['prediction'] in [0, 1, 2]