import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
   
def one_hot_encode(y):
    return np.transpose(np.eye(10)[y-1])

def accuracy_confusion_matrix(x):
    correct = 0
    total   = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            total += x[i][j]
            if i == j:
                correct += x[i][j]
    return correct/total

learning_rate = 0.07
K = 10
d = 100

matrix = pd.read_csv('training.csv', header=None)

R = matrix.iloc[:,0]
X = matrix.iloc[:,1:] / 255
N = matrix.shape[0]

confusion_matrix_training = np.zeros((K,K))
w_matrix = np.random.uniform(low=-0.01, high=0.01, size=(d+1,K))
y_matrix = np.zeros((N,K))


for epochs in range(50):
    delta_w = np.zeros(w_matrix.shape)

    for t in range(N):
        x_t = X.iloc[t,:]
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

    for t in range(N):
        confusion_matrix_training[R.iloc[t]-1, np.argmax(y_matrix[t])] += 1

    print("Epoch: ", epochs+1, "Accuracy: ", accuracy_confusion_matrix(confusion_matrix_training))

    if epochs == 49:
        print(w_matrix.shape)
        w_matrix_final = np.transpose(w_matrix[:,1][1:]).reshape(K,K)
        print(w_matrix_final.shape)
        plt.imshow(w_matrix_final, cmap='gray')
        plt.show()


matrix_test = pd.read_csv('testing.csv', header=None)

R = matrix_test.iloc[:,0]
X = matrix_test.iloc[:,1:] / 255
N = matrix_test.shape[0]

y_matrix_test = np.zeros((N,K))
confusion_matrix_test = np.zeros((K,K))

for t in range(N):
    x_t = X.iloc[t,:]
    x_t = np.append(x_t, 1)
    x_t = np.transpose(x_t)

    r_t = one_hot_encode(R.iloc[t])
    o = np.zeros(K, dtype=np.float64)
    for i in range(K):
        o[i] = np.dot(w_matrix[:,i], x_t)

    y_matrix_test[t] = softmax(o)

for t in range(N):
    confusion_matrix_test[R.iloc[t]-1, np.argmax(y_matrix[t])] += 1

print("Accuracy Testing: ", accuracy_confusion_matrix(confusion_matrix_test))

# Probabilities for each image
#print(y_matrix)