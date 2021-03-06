# Regression Template

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#import the dataset
dataset = pd.read_csv('Container_Crane_Controller_Data_Set.csv', sep='[;]',engine='python' )
print(dataset)
X = dataset.iloc[:,0:2].values
y = dataset.iloc[:,2].values
for i in range(len(y)):
    y[i]=int(list(y[i])[2])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)


# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.reshape(-1,1))


# Fitting the Regression Model to the dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
print(y_pred)
from sklearn.metrics import mean_squared_error,r2_score
print("mean_squared_error=",mean_squared_error(y_test,y_pred))
print("r square is=",r2_score(y_test,y_pred))