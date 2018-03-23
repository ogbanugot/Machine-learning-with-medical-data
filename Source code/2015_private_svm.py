"""
===================================

Support Vector Machine regression
===================================

"""

# Author: Ogban-Asuquo Ugot <ogbanugot@gmail.com>

# #############################################################################
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error


dataset = pd.read_csv('../data/xtry4.csv')
X = dataset.iloc[:, 2:8].values
y = dataset.iloc[:, 12].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

labelencoder_X_2 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])

labelencoder_X_2 = LabelEncoder()
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# #############################################################################
# Fit regression model

svm = svm.SVR()
kfold = KFold(n_splits=5, shuffle=False, random_state= None)


#Bagging Ensemble 
from sklearn.ensemble import BaggingRegressor


bagging = BaggingRegressor(svm, max_samples=0.5, max_features=0.5, n_jobs=-1)

kfold = KFold(n_splits=5, shuffle=False, random_state= None)

bagging.fit(X_train, y_train)

score = cross_val_score(estimator = svm, X  = X_train, y = y_train, cv = kfold, scoring='neg_mean_squared_log_error')

prediction = cross_val_predict(svm, X_test, y_test, cv= kfold, n_jobs=-1)

mse = mean_squared_log_error(y_test, prediction)

mse2 = mean_squared_error(y_test,prediction)

from math import sqrt
sqrt(mse2)
#Visualization
#Visualization
plt.plot(y_test, color = 'red', label = 'Real time lapse')
plt.plot(prediction, color = 'blue', label = 'Predicted time lapse')
plt.title('Time lapse interval predictions')
plt.xlabel('Frequency')
plt.ylabel('Time lapse')
plt.legend()
plt.show()

