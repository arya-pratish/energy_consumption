import sys
import pandas as pd

sys.path.insert(1, '/home/runner/work/energy_consumption/energy_consumption')
from ml_model.preprocessing import con_str_num


def test_func_digit():
    assert con_str_num('9000')== 9000, "Fail to convert into numeric"

def test_func_null():
    assert con_str_num('') is pd.NA, "Fail to identify NULL value"

def test_func_str():
    assert con_str_num('Jack')== 'Jack', "Fail to identify string value"