#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# for easier visualization
import seaborn as sns
# import color maps
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")




#for printing floating point numbers upto  precision 2
np.set_printoptions(precision=2, suppress=True)
pd.options.display.float_format = '{:.2f}'.format

# Load real estate data from CSV or Excel file
print("Loading ........... data from CSV or Excel file")
column_list=["id","Clump_Thickness","Cell_Size","Cell_shape","Marginal_Adhesion","Single_Epithelial_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses","Class"]
df = pd.read_csv("breast-cancer.data",names=column_list)
print("----------------------------------------------------------------------------")


# Display the dimensions of the dataset.
print("Displaying the dimensions of the dataset--")
print("df.shape=",df.shape)
print("----------------------------------------------------------------------------")
print("RENAME SPECIFIC COLUMNS")

# Columns of the dataset
print("Columns of the dataset--")
print("Columns of the dataset",df.columns)
print("----------------------------------------------------------------------------")




# Display the first 5 rows to see example observations.
print("Display the first 5 rows to see example observations--")
pd.set_option('display.max_columns', 20) ## display max 20 columns
print("data head=")
print(df.head())
print("----------------------------------------------------------------------------")


# # Some feaures are numeric and some are categorical
# Filtering the categorical features:
print("Some feaures are numeric and some are categorical Filtering the categorical features:")
print(df.dtypes[df.dtypes=='object'])


# Distributions of numeric features
# One of the most enlightening data exploration tasks is plotting the distributions of your features.

# Plot histogram grid
print("Plotting histogram grid")
# df.hist(figsize=(16,16), xrot=-45) ## Display the labels rotated by 45 degress
#
# # Clear the text "residue"
# plt.show()
print("----------------------------------------------------------------------------")

# Display summary statistics for the numerical features.
print("Display summary statistics for the numerical features")
print("data describe()",df.describe())
print("----------------------------------------------------------------------------")


# Distributions of categorical features
# Display summary statistics for categorical features
print("Distributions of categorical features,Display summary statistics for categorical features")
print(" describe the categorical feature=",df.describe(include=['object']))
print("----------------------------------------------------------------------------")


# Bar plots for categorical Features
# Plot bar plot for the 'exterior_walls' feature.
print("Bar plots for categorical Features Plot bar plot for the 'exterior_walls' feature.")
# plt.figure(figsize=(20,20))
# sns.countplot(y='Bare_Nuclei', data=df)
# plt.show()
# do this for multiple feature
print("----------------------------------------------------------------------------")


print("taking care of sparse class if the categorical feature is present")
print("Sparse classes are classes in categorical features that have a very small number of observations.")
print("They tend to be problematic when we get to building models.")
print("In the best case, they don't influence the model much.In the worst case, they can cause the model to be overfit.")
print("Let's make a mental note to combine or reassign some of these classes later using drawn bar plot")
# code here to remove the
print("----------------------------------------------------------------------------")


print("Segmentations")
print("Segmentations are powerful ways to cut the data to observe the relationship between categorical features and numeric features.")
print("Segmenting the target variable by key categorical features.")
 # sns.boxplot(y='any categorical feature', x='target variable', data=df)
print("----------------------------------------------------------------------------")


print("Let's compare the any catorical feateare across other features as well")
print("displaying the means and standard deviations within each class")
print(df.groupby('Bare_Nuclei').mean())

print("----------------------------------------------------------------------------")



print("Correlations")
print("Positive correlation means that as one feature increases, the other increases; eg. a child's age and her height.")
print("Negative correlation means that as one feature increases, the other decreases;")
print("Correlations near -1 or 1 indicate a strong relationship.")
print("Those closer to 0 indicate a weak relationship.")
print("0 indicates no relationship.")

print("correlation=",df.corr())
print("----------------------------------------------------------------------------")

print("A lot of numbers make things difficult to read. So let's visualize this.")
# plt.figure(figsize=(20,20))
# sns.heatmap(df.corr())
# plt.show()
print("Dark colors indicate strong negative correlations and light colors indicate strong positive correlations")
print("he most helpful way to interpret this correlation heatmap is to first find features that are correlated with our target variable by scanning the first column")
print("----------------------------------------------------------------------------")

print("Plotting the more elobrative way")
mask=np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True
# plt.figure(figsize=(20,20))
# with sns.axes_style("white"):
#     ax = sns.heatmap(df.corr()*100, mask=mask, fmt='.0f', annot=True, lw=1, cmap=ListedColormap(['green', 'yellow', 'red','blue']))
# plt.show()
print("----------------------------------------------------------------------------")

print("##########################################################################################")
print("Untill now we have anylased the data Now we will begin to Clean the data")
print("##########################################################################################")

print("Dropping the duplicates (De-duplication)")
df = df.drop_duplicates()
print("is any shape change=",df.shape )
print("*****************************************************************************")

print("Fix structural errors")
print("First find the if any nan value is there")
# print(df["Mitoses"].unique())
print("*****************************************************************************")


print("fixingn NaN values ")
# df.column Name.fillna("here value to inserted to replace", inplace=True)
print("cheching for confirmation")
# print(df."Column name".unique())
print("*****************************************************************************")



print("Removing Outliers------->")
print("Outliers can cause problems with certain types of models.")
print("Boxplots are a nice way to detect outliers")
print("Let's start with a box plot of your target variable, since that's what you're actually trying to predict.")

# sns.boxplot(df.Bland_Chromatin)
# plt.show()

print("*****************************************************************************")


# Label missing categorical data
print("Label missing categorical data")
print("You cannot simply ignore missing values in your dataset. You must handle them in some way for the very practical reason that Scikit-Learn algorithms do not accept missing values")
# Display number of missing values by categorical feature

print(df.select_dtypes(include=['object']).isnull().sum())
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
df.to_csv('cleaned_Breast_cancer.csv', index=None)

#Feature Engineering
print("Feature Engineering")
print("For example, let's say you knew that homes with 2 bedrooms and 2 bathrooms are especially popular for investors.")

print("*****************************************************************************")

# dropping any column
print("dropping any column")
#dropping the basement column
# df = df.drop(['No','cheap'], axis=1)
df = df.drop(['id'], axis=1)
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
df = pd.get_dummies(df, columns=['Bare_Nuclei'])
print("*****************************************************************************")



# Remove unused or redundant features
print("Remove unused or redundant features if any")
print("*****************************************************************************")


print("lets save this model")
print("inally, before we move on to the next module, let's save our new DataFrame we that augmented through feature engineering.")
print(" We'll call it the analytical base table because we'll be building our models on it.")
print("Remember to set the argument index=None to save only the data.")
df.to_csv('analytical_Breast_cancer.csv', index=None)

print("#########################################Machine Learning Models#################################################")
df = pd.read_csv("analytical_Breast_cancer.csv")
print("shape=",df.shape)

print("-------------------------------------start applying machine learning-------------------")
print("Train and Test Splits")
# Create separate object for target variable
y = df.Class
# Create separate object for input features
X = df.drop('Class', axis=1)

# Split X and y into train and test sets: 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
print("print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)")
print("#################################################################")

# Data standardization
print("Data standardization---")
print("n Data Standardization we perform zero mean centring and unit scaling; i.e. we make the mean of all the ")
print("features as zero and the standard deviation as 1.")
print("Thus we use mean and standard deviation of each feature.It is very important to save the mean and standard")
print("deviation for each of the feature from the training set, because we use the same mean and standard deviation")
print("in the test set.")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# Check for mean and std dev. - not exactly 0 and 1
print("X_test descreption")

print("##############################start applying classification model ###################################")

#Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#using confusion matrix to see the result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("1.logistic confusion matrixm")
print(cm)
# accuracy score
from sklearn.metrics import  accuracy_score
print("accuracy_score logistic regression",accuracy_score(y_test, y_pred))
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


#Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#using confusion matrix to see the result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("2.KNeighborsClassifier confusion matrix")
print(cm)
# accuracy score
from sklearn.metrics import  accuracy_score
print("accuracy_score of KNeighbors algorithm",accuracy_score(y_test, y_pred))
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


#Support Vector Machine Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#using confusion matrix to see the result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("3.Support Vector Machine Algorithm confusion matrix")
print(cm)
# accuracy score
from sklearn.metrics import  accuracy_score
print("3.accuracy_score of Support Vector Machine Algorithm",accuracy_score(y_test, y_pred))
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")



#Kernel-Support Vector Machine Algorithm
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#using confusion matrix to see the result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("4.KERNEL SVC confusion matrix")
print(cm)
# accuracy score
from sklearn.metrics import  accuracy_score
print("4.accuracy_score of kernal svc",accuracy_score(y_test, y_pred))
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")



#Naïve Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#using confusion matrix to see the result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("5.Naïve Bayes Algorithm confusion matrix")
print(cm)
# accuracy score
from sklearn.metrics import  accuracy_score
print("accuracy_score of Naïve Bayes Algorithm",accuracy_score(y_test, y_pred))
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


#Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#using confusion matrix to see the result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("6.Decision Tree Algorithm confusion matrix")
print(cm)
# accuracy score
from sklearn.metrics import  accuracy_score
print("accuracy_score of Decision Tree Algorithm",accuracy_score(y_test, y_pred))
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")



# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
#using confusion matrix to see the result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("7.RandomForestClassifier confusion matrix")
print(cm)
# accuracy score
from sklearn.metrics import  accuracy_score
print("accuracy_score of random forest classifier",accuracy_score(y_test, y_pred))
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
