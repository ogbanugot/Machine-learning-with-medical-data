"""
============================
Nearest Neighbors regression
============================

"""

# Author: Ogban-Asuquo Ugot <ogbanugot@gmail.com>

# #############################################################################
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time.time
from sklearn import neighbors
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
# =============================================================================
# Import the dataframe
# =============================================================================

dataset = pd.read_csv('2015_Private.csv')
X = dataset.iloc[:, [2,3,4,5,6,7]].values #Individually select the  columns
y = dataset.iloc[:, 12].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 0] = labelencoder_X1.fit_transform(X[:, 0])

labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

labelencoder_X3 = LabelEncoder()
X[:, 3] = labelencoder_X3.fit_transform(X[:, 3])

labelencoder_X4 = LabelEncoder()
X[:, 4] = labelencoder_X4.fit_transform(X[:, 4])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# #############################################################################
# Fit regression model
n_neighbors = 10
weights = "distance"
kfold = KFold(n_splits=5, shuffle=False, random_state= None)
knn = neighbors.KNeighborsRegressor(n_neighbors, weights = weights, algorithm='ball_tree', n_jobs =-1)
t0 = time.time()
knn.fit(X_train, y_train)
print ("training time:", round(time.time()-t0, 3), "s")
# =============================================================================
# #Bagging Ensemble 
# from sklearn.ensemble import BaggingRegressor
# bagging = BaggingRegressor(knn, max_samples=0.5, max_features=0.5)
# kfold = KFold(n_splits=5, shuffle=False, random_state= None)
# bagging.fit(X_train, y_train)
# =============================================================================

# =============================================================================
# Grid search 
# =============================================================================
parameters = {'n_neighbors':[5,10],'algorithm':('ball_tree','kd_tree')}
knn = neighbors.KNeighborsRegressor(weights = 'distance')
reg = GridSearchCV(knn, parameters, scoring = 'r2', n_jobs = -1)
reg.fit(X_train, y_train)
best_score = reg.best_score_
best_param = reg.best_params_
best_model = reg.best_estimator_ #Can use in Cross_val_score & predict

#Cross validated estimate on training and test data
score = cross_val_score(estimator = knn, X  = X_train, y = y_train, cv = kfold, scoring='r2')
prediction = cross_val_predict(knn, X_test, y_test, cv= kfold, n_jobs=-1)



#Variants of scoring
msle = mean_squared_log_error(y_test, prediction)
mse = mean_squared_error(y_test,prediction)
mae = mean_absolute_error(y_test,prediction)
r2 = r2_score(y_test,prediction)
from math import sqrt
rmse = sqrt(mse) #root mean --

#mode Visualization
plt.plot(y_test, color = 'red', label = 'Real time lapse')
plt.plot(prediction, color = 'blue', label = 'Predicted time lapse')
plt.title('Time lapse interval predictions')
plt.xlabel('Frequency')
plt.ylabel('Time lapse')
plt.legend()
plt.show()