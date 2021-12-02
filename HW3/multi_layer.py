import numpy as np
import pandas as pd

def one_hot_encode(y):
    return np.transpose(np.eye(10)[y-1])

def sigmoid(x):
    return 1/(1+np.exp(-x))

matrix = pd.read_csv('testing.csv', header=None)
print(matrix.shape)
Y = matrix.iloc[:,0]
X = matrix.iloc[:,1:]
print(Y.shape)
print(X.shape)
for H in [5, 10, 25, 50, 75]:
    v_matrix = np.random.uniform(low=-0.01, high=0.01, size=(H+1, 10))
    w_matrix = np.random.uniform(low=-0.01, high=0.01, size=(101,H))
    for i in range(10): # X.shape[1]
        x_t = np.transpose(X.iloc[i,:])
        r_t = one_hot_encode(Y.iloc[i])
        
        z_t = np.zeros(H+1)
        z_t[0] = 1
        print(x_t.shape)
        print(r_t.shape)

        for h in range(1,H):
            w_h_T  = np.transpose(w_matrix[:,h-1]) 
            z_t[h] = np.dot(w_h_T, x_t)

        os = np.zeros(10)
        for i in range(10):
            os[i] = np.dot(v_matrix[:,i], z_t)


            print(v_matrix.shape)
            print(w_matrix.shape)

            for i in range(H):
                pass