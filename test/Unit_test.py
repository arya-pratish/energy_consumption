import model
import pandas as pd

def check_func_digit():
    assert model.con_str_num('9000')== 9000, "Fail to convert into numeric"

def check_func_null():
    assert model.con_str_num('') is pd.NA, "Fail to identify NULL value"

def check_func_str():
    assert model.con_str_num('Jack')== 'Jack', "Fail to identify string value"