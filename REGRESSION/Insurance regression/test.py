from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male'], ['Female'], ['Female']]
enc.fit(X)
print(enc.categories_)
enc.transform(X)

print("X=",X)

# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# A=[1, 2, 2, 6]
# le.fit(A)
#
# print(le.classes_)
#
# print(le.transform([1, 1, 2, 6]))
#
# print(le.inverse_transform([0, 0, 1, 2]))
enc = OrdinalEncoder()
X = [['male', 'from US', 'uses Safari',1], ['female', 'from Europe', 'uses Firefox',2]]
enc.fit(X)

print(enc.transform([['female', 'from US', 'uses Safari',2]]))
enc = OneHotEncoder(handle_unknown="error")
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)

print(enc.transform([['female', 'from US', 'uses Safari'],
               ['male', 'from Europe', 'uses Safari']]).toarray())