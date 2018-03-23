#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:44:04 2018

@author: ugoslight
"""

"""Decision Tree Regressor"""
"""One hot encoding"""
"""Ensemble method"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, explained_variance_score



hosp_data = pd.read_csv('/Users/ugoslight/Downloads/xiitry.xlsx - xiitry.xlsx.csv')
#X values are:   ['MonthYear', 'Provider_Code', 'Diagnosis', 'Drugs', 'Charges_Sent', 'Company Code']] 
X = hosp_data.iloc[:, 2:8].values
Y = hosp_data.iloc[:, 12:13].values


le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:,2].astype(str))
X[:, 3] = le.fit_transform(X[:,3].astype(str))
X[:, 4] = le.fit_transform(X[:,4].astype(str))

"""One hot Encode"""
onehot_encoder = OneHotEncoder(sparse=False)
X = onehot_encoder.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""Model-dree"""
dree = DecisionTreeRegressor(random_state = 0)
kfold = KFold(n_splits = 5, shuffle = False, random_state = None)
dree.fit(X_train, y_train)
score = cross_val_score(estimator = dree, X = X_train, y = y_train, scoring = 'neg_mean_squared_log_error')
cross_val = cross_val_predict(dree, X_test, y_test, cv = kfold)
y_pred = dree.predict(X_test)

"""Error Calculation"""
expl_variance = explained_variance_score(y_test, y_pred) # 1.0 best, lower worse. 
msle = mean_squared_log_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2_error = r2_score(y_test, y_pred)

from sklearn.ensemble import RandomForestRegressor

"""Ensemble Bagging method"""
model = RandomForestRegressor(random_state=0, n_jobs=-1)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
model.fit(X_train, y_train)
y_pred = model.predict

"""Errors"""
expl_variance = explained_variance_score(y_test, y_pred) # 1.0 best, lower worse. 
msle = mean_squared_log_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2_error = r2_score(y_test, y_pred)

