import numpy as np
import pandas as pd
import math
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)
   
def one_hot_encode(y):
    return np.transpose(np.eye(10)[y-1])

learning_rate = 0.01
K = 10
d = 100

matrix = pd.read_csv('training.csv', header=None)
#print(matrix.head())

R = matrix.iloc[:,0]
X = matrix.iloc[:,1:] / 255
N = matrix.shape[0]
#print(X.head())
print(R.head())
print(X.head())
print(N)

w_matrix = np.random.uniform(low=-0.01, high=0.01, size=(d+1,K))
y_matrix = np.zeros((N,K))

for epochs in range(50):
    delta_w = np.zeros(w_matrix.shape)

    for t in range(N):
        x_t = X.iloc[t,:] / 255
        x_t = np.append(x_t, 1)
        x_t = np.transpose(x_t)

        r_t = one_hot_encode(R.iloc[t])
        o = np.zeros(K, dtype=np.float64)
        for i in range(K):
            o[i] = np.dot(w_matrix[:,i], x_t)

        y_matrix[t] = softmax(o)
        for i in range(K):
            for j in range(d+1):
                delta_w[j,i] += (r_t[i] - y_matrix[t][i])  * x_t[j]

        for i in range(K):
            for j in range(d+1):
                w_matrix[j,i] += delta_w[j,i] * learning_rate
