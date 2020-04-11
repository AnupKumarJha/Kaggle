import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("insurance.csv")
print(data.head(5))
print(data.describe())
print(data.info())
print(data.corr())
print(data.corr(method='pearson'))
print(data.corr(method='spearman'))
print(data.corr(method='kendall'))
print(data.shape)
print(data.columns)
f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
# plt.show()
data.age.plot(kind='line', color='g', label='Age', linewidth=1, alpha=0.5, grid=True, linestyle='-')
data.bmi.plot(kind='line', color='r', label='BMI', linewidth=1, alpha=0.5, grid=True, linestyle=':')
plt.legend('upper left')
plt.xlabel('age')
plt.ylabel('bmi')
plt.title('Line Plot')
plt.show()
data.plot(kind='scatter', x='age', y='children',alpha = 0.5,color = 'red')
plt.xlabel('Age')
plt.ylabel('Children')
plt.title('Age Children Scatter Plot')
plt.show()
data.charges.plot(kind='hist', bins=50, figsize=(10,10))
plt.show()



data1=data['sex']=='female'
data_female=data[data1]
data2=data['sex']=='male'
data_male=data[data2]
data_female.charges.plot(kind='hist', bins=50, figsize=(10,10))
plt.show()
