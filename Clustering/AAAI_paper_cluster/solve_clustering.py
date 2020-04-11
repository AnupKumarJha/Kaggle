#implementing the clustering algorithms

#importing the library
import numpy as np
import pandas as pd

#reading the file
df=pd.read_csv("papers.csv")
print(df.shape)
print(df.columns)
print(df['Title'])