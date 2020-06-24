# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Import and save dataset

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.impute import SimpleImputer

# Treatment NAs
# Transform NaN/Empty/Undefined/Null values into mean of the column

imputer = SimpleImputer(missing_values = np.NAN, strategy = "mean")
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encode category data
# Split dependent and independent variables and transform text values into non-ordinal categories.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
x = np.array(ct.fit_transform(x), dtype=np.float)
y = LabelEncoder().fit_transform(y)

# Split dataset betweenn training and testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Variables Scale
# Values are scaled because the Euclidean distance that results from comparing values 
# with large differences (such as salary and age) can be problematic

from sklearn.preprocessing import StandardScaler

scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.fit_transform(x_test)