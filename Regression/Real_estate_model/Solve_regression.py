# importing the required libraries
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for easier visualization
import seaborn as sns
# import color maps
from matplotlib.colors import ListedColormap

# Ignore Warnings
import warnings

from numpy.ma import sqrt

warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import Ridge  # Linear Regression + L2 regularization
from sklearn.linear_model import Lasso  # Linear Regression + L1 regularization
from sklearn.svm import SVR  # Support Vector Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Evaluation Metrics
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.metrics import r2_score as rs
from sklearn.metrics import mean_absolute_error as mae

# from xgboost import XGBRegressor
# from xgboost import plot_importance  # to plot feature importance

# to save the final model on disk
from sklearn.externals import joblib

# for printing floating point numbers upto  precision 2
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:.2f}'.format

# Load real estate data from CSV or Excel file
print("Loading ........... data from CSV or Excel file")
df = pd.read_excel("real_estate.xlsx")
print("----------------------------------------------------------------------------")

# Display the dimensions of the dataset.
print("Displaying the dimensions of the dataset--")
print("df.shape=", df.shape)
print("----------------------------------------------------------------------------")
print("RENAME SPECIFIC COLUMNS")
# df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
# # Or rename the existing DataFrame (rather than creating a copy)
# df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)
df.rename(columns={'X1 transaction date': 'tx_date', 'X2 house age': 'house_age',
                   'X3 distance to the nearest MRT station': "Metro_dist",
                   "X4 number of convenience stores": "no_of_store", "X5 latitude": "latitude"}, inplace=True)
df.rename(columns={'X6 longitude': 'longitude', 'Y house price of unit area': "price"}, inplace=True)

# Columns of the dataset
print("Columns of the dataset--")
print("Columns of the dataset", df.columns)
print("----------------------------------------------------------------------------")

# Display the first 5 rows to see example observations.
print("Display the first 5 rows to see example observations--")
pd.set_option('display.max_columns', 20)  # display max 20 columns
print("data head=")
print(df.head())
print("----------------------------------------------------------------------------")

# Some feaures are numeric and some are categorical
# Filtering the categorical features:
print("Some feaures are numeric and some are categorical Filtering the categorical features:")
print(df.dtypes[df.dtypes == 'object'])

# Distributions of numeric features
# One of the most enlightening data exploration tasks is plotting the distributions of your features.

# Plot histogram grid
print("Plotting histogram grid")
df.hist(figsize=(16, 16), xrot=-45)  # Display the labels rotated by 45 degress

# Clear the text "residue"
# plt.show()
print("----------------------------------------------------------------------------")

# Display summary statistics for the numerical features.
print("Display summary statistics for the numerical features")
print("data describe()", df.describe())
print("----------------------------------------------------------------------------")

# Distributions of categorical features
# Display summary statistics for categorical features
print("Distributions of categorical features,Display summary statistics for categorical features")
# print(" describe the categorical feature=",df.describe(include=['object']))
print("----------------------------------------------------------------------------")

# Bar plots for categorical Features
# Plot bar plot for the 'exterior_walls' feature.
print("Bar plots for categorical Features Plot bar plot for the 'exterior_walls' feature.")
# plt.figure(figsize=(8,8))
# sns.countplot(y='feature_name like categorical feature', data=df)
# do this for multiple feature
print("----------------------------------------------------------------------------")

print("taking care of sparse class if the categorical feature is present")
print("Sparse classes are classes in categorical features that have a very small number of observations.")
print("They tend to be problematic when we get to building models.")
print(
    "In the best case, they don't influence the model much.In the worst case, they can cause the model to be overfit.")
print("Let's make a mental note to combine or reassign some of these classes later using drawn bar plot")
# code here to remove the 
print("----------------------------------------------------------------------------")

print("Segmentation")
print("Segmentation are powerful ways to cut the data to observe the relationship between categorical features and ")
print("numeric features.")
print("Segmenting the target variable by key categorical features.")
# sns.boxplot(y='any categorical feature', x='target variable', data=df)
print("----------------------------------------------------------------------------")

print("Let's compare the any catorical feateare across other features as well")
print("displaying the means and standard deviations within each class")
# df.groupby('property_type').mean()
print("----------------------------------------------------------------------------")

print("Correlations")
print(
    "Positive correlation means that as one feature increases, the other increases; eg. a child's age and her height.")
print("Negative correlation means that as one feature increases, the other decreases;")
print("Correlations near -1 or 1 indicate a strong relationship.")
print("Those closer to 0 indicate a weak relationship.")
print("0 indicates no relationship.")

print("correlation=", df.corr())
print("----------------------------------------------------------------------------")

print("A lot of numbers make things difficult to read. So let's visualize this.")
plt.figure(figsize=(20, 20))
sns.heatmap(df.corr())
# plt.show()
print("Dark colors indicate strong negative correlations and light colors indicate strong positive correlations")
print("he most helpful way to interpret this correlation heatmap")
print("is to first find features that are correlated with our target variable by scanning the first column")
print("----------------------------------------------------------------------------")

print("Plotting the more elobrative way")
mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(10, 10))
with sns.axes_style("white"):
    ax = sns.heatmap(df.corr() * 100, mask=mask, fmt='.0f', annot=True, lw=1,
                     cmap=ListedColormap(['green', 'yellow', 'red', 'blue']))
# plt.show()
print("----------------------------------------------------------------------------")

print("##########################################################################################")
print("Untill now we have anylased the data Now we will begin to Clean the data")
print("##########################################################################################")

print("Dropping the duplicates (De-duplication)")
df = df.drop_duplicates()
print("is any shape change=", df.shape)
print("*****************************************************************************")

print("Fix structural errors")
print("First find the if any nan value is there")
# print(df."Column name".unique())
print("*****************************************************************************")

print("fixing NaN values ")
# df.column Name.fillna("here value to inserted to replace", inplace=True)
print("cheching for confirmation")
# print(df."Column name".unique())
print("*****************************************************************************")

# Typos and capitalization
print("Typos and capitalization")
# sns.countplot(y='any column name prefarabley categorical value', data=df)
print("composition' should be 'Composition'")
print("renaming if any typos is there")
# df.COLUMN NAME.replace('composition', 'Composition', inplace=True)
print("again plotting")
# sns.countplot(y='COLUMN NAME', data=df)
print("*****************************************************************************")

print("Mislabeled classes")
print("Finally, we'll check for classes that are labeled as separate classes when they should really be the same")
print("e.g. If 'N/A' and 'Not Applicable' appear as two separate classes, we should combine them")
print("let's plot the class distributions for 'ANY COLUMN NAME'")
# sns.countplot(y='exterior_walls', data=df)
# df.exterior_walls.replace(['Rock, Stone'], 'Masonry', inplace=True)
# df.exterior_walls.replace(['Concrete', 'Block'], 'Concrete Block', inplace=True)
# sns.countplot(y='exterior_walls', data=df)
print("*****************************************************************************")

print("Removing Outliers------->")
print("Outliers can cause problems with certain types of models.")
print("Boxplots are a nice way to detect outliers")
print("Let's start with a box plot of your target variable, since that's what you're actually trying to predict.")

# sns.boxplot(df.Metro_dist)
# sns.boxplot(df.no_of_store)
# sns.boxplot(df.latitude)
# sns.boxplot(df.longitude)
# sns.boxplot(df.price)
# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(df['Metro_dist'], df['price'])
# ax.set_xlabel('Proportion of non-retail business acres per town')
# ax.set_ylabel('Full-value property-tax rate per $10,000')
# plt.show()
print("*****************************************************************************")

# Label missing categorical data
print("Label missing categorical data")
print("You cannot simply ignore missing values in your dataset. You must handle them in some way")
print("for the very practical reason that Scikit-Learn algorithms do not accept missing values")
# Display number of missing values by categorical feature

# print(df.select_dtypes(include=['object']).isnull().sum())
# he best way to handle missing data for categorical features is to simply label them as 'Missing'¶
print("he best way to handle missing data for categorical features is to simply label them as 'Missing'")

# df['exterior_walls'] = df['exterior_walls'].fillna('Missing')
# df['roof'] = df['roof'].fillna('Missing')
# df.select_dtypes(include=['object']).isnull().sum()
# Flag and fill missing numeric data
print("Flag and fill missing numeric data")
print(df.select_dtypes(exclude=['object']).isnull().sum())
print("*****************************************************************************")

print("Before we move on to the next module, let's save the new dataframe we worked hard to clean.")
# Save cleaned dataframe to new file
df.to_csv('cleaned_real_estate.csv', index=None)

# Feature Engineering
print("Feature Engineering")
print(
    "For example, let's say you knew that homes with 2 bedrooms and 2 bathrooms are especially popular for investors.")

# Display percent of rows where two_and_two == 1
# df[df['two_and_two']==1].shape[0]/df.shape[0]

df['cheap'] = (df.price < 100).astype(int)
# display the percentage of properties the are cheap
# df[df['cheap']==1].shape[0]/df.shape[0]
print("shape=", df['cheap'])
print("Columns of the dataset--")
print("Columns of the dataset", df.columns)
print(df.head(5))
print("*****************************************************************************")

# dropping any column
print("dropping any column")
# dropping the basement column
df = df.drop(['No', 'cheap'], axis=1)
# df = df.drop(['cheap'], axis=1)
print(df.columns)
print("*****************************************************************************")

# Handling Sparse Classes
print("he easiest way to check for sparse classes is simply by plotting the distributions of your categorical features")
# Bar plot for exterior_walls
# sns.countplot(y='Column Name', data=df)
# For example Group 'Wood Siding', 'Wood Shingle', and 'Wood' together. Label all of them as 'Wood'.
# df.exterior_walls.replace(['Wood Siding', 'Wood Shingle', 'Wood'], 'Wood', inplace=True)
print("*****************************************************************************")

# Encode dummy variables (One Hot Encoding)
print("Encode dummy variables (One Hot Encoding) if any")
# df = pd.get_dummies(df, columns=['exterior_walls', 'roof', 'property_type'])
print("*****************************************************************************")

# Remove unused or redundant features
print("Remove unused or redundant features if any")
print("*****************************************************************************")

print("lets save this model")
print("initially, before we move on to the next module, let's save our new DataFrame we that augmented through feature")
print(" engineering.We'll call it the analytical base table because we'll be building our models on it.")
print("Remember to set the argument index=None to save only the data.")
df.to_csv('analytical_base_table.csv', index=None)

print("#####################################Machine Learning Models#################################################")
df = pd.read_csv("analytical_base_table.csv")
print("shape=", df.shape)
print("#################################################################")

print("Train and Test Splits")
# Create separate object for target variable
y = df.price
# Create separate object for input features
X = df.drop('price', axis=1)

# Split X and y into train and test sets: 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print("print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)")
print("#################################################################")

# Data standardization
print("Data standardization---")
print("n Data Standardization we perform zero mean centring and unit scaling; i.e. we make the mean of all the ")
print("features as zero and the standard deviation as 1.")
print("Thus we use mean and standard deviation of each feature.It is very important to save the mean and standard")
print("deviation for each of the feature from the training set, because we use the same mean and standard deviation")
print("in the test set.")

train_mean = X_train.mean()
train_std = X_train.std()
# Standardize the train data set
X_train = (X_train - train_mean) / train_std
print("X_train descreption")
print(X_train.describe())

# Note: We use train_mean and train_std_dev to standardize test data set
X_test = (X_test - train_mean) / train_std

# Check for mean and std dev. - not exactly 0 and 1
print("X_test descreption")
print(X_test.describe())
print("#################################################################")

# Model 1 - Baseline Model
# In this model, for every test data point, we will simply predict the average of the train labels as the output.
# We will use this simple model to perform hypothesis testing for other complex models.
print("Model 1 - Baseline Model")

# Predict Train results
y_train_pred = np.ones(y_train.shape[0]) * y_train.mean()
# Predict Test results
y_pred = np.ones(y_test.shape[0]) * y_train.mean()

print("Train Results for Baseline Model:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))
print("#################################################################")

# Model-2 Ridge Regression
print("Model-2 Ridge Regression")
tuned_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
model = GridSearchCV(Ridge(), tuned_params, scoring='neg_mean_absolute_error', cv=10, n_jobs=-1)
model.fit(X_train, y_train)
print("model.best_estimator=", model.best_estimator_)
# Predict Train results
y_train_pred = model.predict(X_train)

# Predict Test results
y_pred = model.predict(X_test)

print("Train Results for Ridge Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))

# Feature Importance
print("Feature Importance")
# Building the model again with the best hyperparameters
model = Ridge(alpha=100)
model.fit(X_train, y_train)
indices = np.argsort(-abs(model.coef_))
print("The features in order of importance are:")
print(5 * '-')
for feature in X.columns[indices]:
    print(feature)
print("#################################################################")

# Model-3 Support Vector Regression
print("Model-3 Support Vector Regression")
tuned_params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
model = GridSearchCV(SVR(), tuned_params, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
model.fit(X_train, y_train)
print("model.best_estimator=", model.best_estimator_)
# Building the model again with the best hyperparameters
model = SVR(C=100000, gamma=0.01)
model.fit(X_train, y_train)
# Predict Test results
y_pred = model.predict(X_test)
print("Train Results for Support Vector Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))
print("#################################################################")

# Model-4 Random Forest Regression
print("Model-4 Random Forest Regressio")
tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]}
model = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter=20, scoring='neg_mean_absolute_error', cv=5,
                           n_jobs=-1)
model.fit(X_train, y_train)
print("model.best_estimator=", model.best_estimator_)
# Building the model again with the best hyperparameters
model = SVR(C=100000, gamma=0.01)
model.fit(X_train, y_train)

# Predict Train results
y_train_pred = model.predict(X_train)

# Predict Test results
y_pred = model.predict(X_test)

print("Train Results for Random forest Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))

# Feature Importance
print("Feature Importance")
# Building the model again with the best hyperparameters
model = RandomForestRegressor(n_estimators=200, min_samples_split=10, min_samples_leaf=2)
model.fit(X_train, y_train)
indices = np.argsort(-model.feature_importances_)
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)

print("#################################################################")


# Model-5 XGBoost Regression
print("Model-5 XGBoost Regression")
# Reference for random search on xgboost
# https://gist.github.com/wrwr/3f6b66bf4ee01bf48be965f60d14454d
# tuned_params = {'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300, 400, 500], 'reg_lambda': [0.001, 0.1, 1.0, 10.0, 100.0]}
# model = RandomizedSearchCV(XGBRegressor(), tuned_params, n_iter=20, scoring = 'neg_mean_absolute_error', cv=5, n_jobs=-1)
# model.fit(X_train, y_train)
#
# print("model.best_estimator",model.best_estimator_)
#
# ## Predict Train results
# y_train_pred = model.predict(X_train)
# ## Predict Test results
# y_pred = model.predict(X_test)
# print("Train Results for XGBoost Regression:")
# print("*******************************")
# print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
# print("R-squared: ", rs(y_train.values, y_train_pred))
# print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))
#
# # Feature Importance
# print("Feature Importance")
#
# ## Building the model again with the best hyperparameters
# model = XGBRegressor(max_depth=2,learning_rate=0.05,n_estimators=400, reg_lambda=0.001)
# model.fit(X_train, y_train)
# # Function to include figsize parameter
# # Reference: https://stackoverflow.com/questions/40081888/xgboost-plot-importance-figure-size
# def my_plot_importance(booster, figsize, **kwargs):
#     from matplotlib import pyplot as plt
#     from xgboost import plot_importance
#     fig, ax = plt.subplots(1,1,figsize=figsize)
#     return plot_importance(booster=booster, ax=ax, **kwargs)
# my_plot_importance(model, (10,10))

print("#################################################################")


# Model-6 Lasso Regression
print("Model-6 Lasso Regression")
tuned_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
model = GridSearchCV(Lasso(), tuned_params, scoring = 'neg_mean_absolute_error', cv=20, n_jobs=-1)
model.fit(X_train, y_train)
print("model.best_estimator",model.best_estimator_)
## Predict Train results
y_train_pred = model.predict(X_train)
## Predict Test results
y_pred = model.predict(X_test)

print("Train Results for Lasso Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", rs(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))

# Feature Importance
print("Feature Importance")
## Building the model again with the best hyperparameters
model = Lasso(alpha=1000)
model.fit(X_train, y_train)
indices = np.argsort(-abs(model.coef_))
print("The features in order of importance are:")
print(50*'-')
for feature in X.columns[indices]:
    print(feature)
print("#################################################################")



# Model-7 Descision Tree Regressio
print("Model-7 Descision Tree Regression")

tuned_params = {'min_samples_split': [2, 3, 4, 5, 7], 'min_samples_leaf': [1, 2, 3, 4, 6], 'max_depth': [2, 3, 4, 5, 6, 7]}
model = RandomizedSearchCV(DecisionTreeRegressor(), tuned_params, n_iter=20, scoring = 'neg_mean_absolute_error', cv=10, n_jobs=-1)
model.fit(X_train, y_train)
print("model.best_estimator=",model.best_estimator_)
## Predict Train results
y_train_pred = model.predict(X_train)
## Predict Test results
y_pred = model.predict(X_test)
print("Train Results for Decision Tree Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", rs(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))

print("#################################################################")

# Model-8 KN Regression
print("Model-8 KN Regression")
# creating odd list of K for KNN
neighbors = list(range(1,50,2))
# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('\nThe optimal number of neighbors is %d.' % optimal_k)
model = KNeighborsRegressor(n_neighbors = optimal_k)
model.fit(X_train, y_train)
## Predict Train results
y_train_pred = model.predict(X_train)


## Predict Test results
y_pred = model.predict(X_test)
print("Train Results for KN Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))
print("R-squared: ", rs(y_train.values, y_train_pred))
print("Mean Absolute Error: ", mae(y_train.values, y_train_pred))

print("Test Results for KN Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", rs(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))

print("#################################################################")

print("Compare all model::")
print("best model is Decision Tree Regression")

print("saving this model")

win_model =RandomizedSearchCV(DecisionTreeRegressor(), tuned_params, n_iter=20, scoring = 'neg_mean_absolute_error', cv=10, n_jobs=-1)
win_model.fit(X_train, y_train)
with open('rfr_real_estate.pkl', 'wb') as pickle_file:
       joblib.dump(win_model, 'rfr_real_estate.pkl')

print("predicting the score")
pickle_model=joblib.load('rfr_real_estate.pkl')
score = pickle_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * score))
y_pred = pickle_model.predict(X_test)

print("Test Results for Decisiont tree Regression:")
print("*******************************")
print("Root mean squared error: ", sqrt(mse(y_test, y_pred)))
print("R-squared: ", rs(y_test, y_pred))
print("Mean Absolute Error: ", mae(y_test, y_pred))
