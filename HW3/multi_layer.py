import numpy as np
import pandas as pd
import random

def one_hot_encode(y):
    return np.transpose(np.eye(10)[y-1])

def one_hot_decode(y):
    return np.argmax(y) + 1

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

learning_rate = 0.01
K = 10

matrix = pd.read_csv('training.csv', header=None)
#print(matrix.shape)
R = matrix.iloc[:,0]
X = matrix.iloc[:,1:] / 255

#print(R.shape)
#print(X.shape)

for H in [5, 10, 25, 50, 75]:
    v_matrix =[]
    w_matrix =[]
    for epochs in range(20):
        v_matrix = np.random.uniform(low=-0.01, high=0.01, size=(H+1, 10))
        w_matrix = np.random.uniform(low=-0.01, high=0.01, size=(101,H))

        random_list = list(range(X.shape[1]))
        random.shuffle(random_list)

        for instance in random_list:
            x_t = X.iloc[instance,:]
            x_t = np.append(x_t, 1)
            x_t = np.transpose(x_t)
            print(x_t.shape)
            #print(x_t)
            r_t = one_hot_encode(R.iloc[instance])

            print(x_t.shape)

            z_t = np.zeros(H)
            z_t = np.append(z_t, 1)

            print(z_t.shape)

            for h in range(H):
                w_h_T  = np.transpose(w_matrix[:,h]) 
                z_t[h] = sigmoid(np.dot(w_h_T, x_t))

            os = np.zeros(K)
            for i in range(K):
                os[i] = np.dot(np.transpose(v_matrix[:,i]), z_t)

            y_t = softmax(os)

            delta_v_matrix = np.zeros(v_matrix.shape)
            delta_w_matrix = np.zeros(w_matrix.shape)

            for i in range(K):
                delta_v_matrix[:,i] = learning_rate*(y_t[i]-r_t[i])*z_t

            for h in range(H):
                summa = 0.0
                for i in range(10):
                    summa += (r_t[i] - y_t[i])*v_matrix[h,i]
                delta_w_matrix[:,h] = learning_rate*z_t[h]*(1-z_t[h])*summa*x_t
            
            for i in range(K):
                v_matrix[:,i] = v_matrix[:,i] + delta_v_matrix[:,i]

            for h in range(H):
                w_matrix[:,h] = w_matrix[:,h] + delta_w_matrix[:,h]