import numpy as np
import pandas as pd



matrix = pd.read_csv('testing.csv')
#print(matrix.head())

y = matrix[0]
X = matrix.drop('label', axis=1)

#print(X.head())
print(y.head())

v_matrix = np.random.uniform(low=-0.01, high=0.01, size=(10))
w_matrix = np.random.uniform(low=-0.01, high=0.01, size=(101))