from sklearn_pandas import DataFrameMapper

#from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
  from sklearn.impute import SimpleImputer
except ImportError:
  from sklearn.preprocessing import Imputer

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype, is_categorical_dtype

import pandas as pd
import numpy as np
import sklearn
import re, warnings

def join_df(left, right, left_on, right_on=None, suffix='_y', how='left'):
    if right_on is None: right_on = left_on
    return left.merge(right, how=how, left_on=left_on, right_on=right_on, 
                      suffixes=("", suffix))
