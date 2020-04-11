# for numerical computing
import numpy as np

# for dataframes
import pandas as pd

# for easier visualization
import seaborn as sns

# for visualization and to display plots
from matplotlib import pyplot as plt
# %matplotlib inline

# import color maps
from matplotlib.colors import ListedColormap

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from math import sqrt

# to split train and test set
from sklearn.model_selection import train_test_split

# to perform hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import Ridge  # Linear Regression + L2 regularization
from sklearn.linear_model import Lasso  # Linear Regression + L1 regularization
from sklearn.svm import SVR # Support Vector Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

# Evaluation Metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as rs
from sklearn.metrics import mean_absolute_error as mae

#import xgboost
# import os
# mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-seh-rt_v5-rev0\\mingw64\\bin'
# os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
# from xgboost import XGBRegressor
# from xgboost import plot_importance  # to plot feature importance

# to save the final model on disk
from sklearn.externals import joblib

np.set_printoptions(precision=2, suppress=True) #for printing floating point numbers upto  precision 2
df = pd.read_csv('real_estate_data.csv')
print("df.shape=",df.shape)
print("df.columns",df.columns)
pd.set_option('display.max_columns', 20)# display max 20 columns
print(df.head())
print(df.dtypes[df.dtypes=='object'])

# Plot histogram grid
df.hist(figsize=(16,16), xrot=-45) # Display the labels rotated by 45 degress

# Clear the text "residue"
# plt.show()
print(df.describe())
print(df.describe(include=['object']))

plt.figure(figsize=(8,8))
sns.countplot(y='exterior_walls', data=df)



plt.figure(figsize=(5,2))
sns.countplot(y='property_type', data=df)

sns.boxplot(y='property_type', x='tx_price', data=df)
# plt.show()

print(df.groupby('property_type').mean())

sns.boxplot(y='property_type', x='sqft', data=df)

print(df.groupby('property_type').agg([np.mean, np.std]))
print("correlation",df.corr())

plt.figure(figsize=(20,20))
sns.heatmap(df.corr())
# plt.show()

mask=np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10,10))
with sns.axes_style("white"):
    ax = sns.heatmap(df.corr()*100, mask=mask, fmt='.0f', annot=True, lw=1, cmap=ListedColormap(['green', 'yellow', 'red','blue']))
plt.show()
