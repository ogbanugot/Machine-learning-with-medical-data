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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, explained_variance_score


hosp_data = pd.read_csv('/Users/ugoslight/Downloads/2015_Private.csv')
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

"""Feature Scaling"""
sc = StandardScaler()
X = sc.fit_transform(X)


"""Training and Testing"""
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
dree = DecisionTreeRegressor(random_state=0)
dree.fit(X_train, y_train)
kfold = KFold(n_splits=5, shuffle=False, random_state= None)

score = cross_val_score(estimator = dree, X  = X_train, y = y_train, cv = kfold, scoring='neg_mean_squared_log_error')
prediction = cross_val_predict(knn, X_test, y_test, cv= kfold, n_jobs=-1)
y_pred = cross_val_predict(dree, X_test, y_test, cv= kfold, n_jobs=-1)
"""Ensemble - Adaboost"""
ada_boost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng)
ada_boost.fit(X_train, y_train)
ada_boost_predict = ada_boost.predict(X_test)

"""Errors - Decision Tree and AdaBoost"""
expl_variance = explained_variance_score(y_test, y_pred) # 1.0 best, lower worse. 
msle = mean_squared_log_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2_error = r2_score(y_test, y_pred)

expl_variance_boost = explained_variance_score(y_test, ada_boost_predict) # 1.0 best, lower worse. 
msle_boost = mean_squared_log_error(y_test, ada_boost_predict)
mse_boost = mean_squared_error(y_test, ada_boost_predict)
r2_error_boost = r2_score(y_test, ada_boost_predict)

"""Visual representation"""
plt.figure()
plt.plot(y_test, color="cornflowerblue", label="Actual", linewidth=2)
plt.plot(y_pred, color="red", label="Model", linewidth=0.5)
plt.title("Time lapse interval prediction")
plt.legend(loc = 'A')
plt.show()

dot_data = tree.export_graphviz(dree)
graph = graphviz.Source(dot_data)
graph.render("Decision Tree")

