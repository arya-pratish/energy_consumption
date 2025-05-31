import sys
import pandas as pd

sys.path.insert(1, '/home/runner/work/energy_consumption/energy_consumption')
from ml_model.preprocessing import pre_process


def test_preprocess():
    cols_name = ['Building Type', 'Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week','Energy Consumption']
    df = pd.DataFrame([['Residential', 4323  ,20, 56,  '22  '  , 'Weekday'  , 4212]], columns=cols_name )          
    assert pre_process(df)== [0, 4323  ,20, 56,  22  ,  0 , 4212], "Fail to pre-processed"
