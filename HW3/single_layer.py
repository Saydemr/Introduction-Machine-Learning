import numpy as np
import pandas as pd



matrix = pd.read_csv('testing.csv')
#print(matrix.head())

y = matrix[0]
X = matrix.drop('label', axis=1)

#print(X.head())
print(y.head())