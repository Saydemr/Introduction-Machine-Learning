import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random

# Taken from Mahir, the TA
def plot_mean_images(weights):
    # Creating 4*3 subplots
    fig, axes = plt.subplots(4, 3)
    # Set the height of plat 8px*8px
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.suptitle('Mean Images')
    # For each subplot run the code inside loop
    for label in range(12):
        # If the subplot index is a label (0,1,2...9)
        if label<10:
            axes[label//3][label%3].imshow(weights[:,label].reshape(10,10),)
        # Do not show the axes of subplots
        axes[label//3][label%3].axis('off') 
    # Showing the plot
    plt.show()


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

learning_rate = 0.1
K = 10
d = 100
E = 50

if len(sys.argv) == 2:
    learning_rate = float(sys.argv[1])

matrix = pd.read_csv('training.csv', header=None, skiprows=1)

R = matrix.iloc[:,0]
X = matrix.iloc[:,1:] / 255
N = matrix.shape[0]

confusion_matrix_training = np.zeros((K,K))
confusion_matrix_test = np.zeros((K,K))
w_matrix = np.random.uniform(low=-0.01, high=0.01, size=(d+1,K))
y_matrix = np.zeros((N,K))

matrix_test = pd.read_csv('testing.csv', header=None, skiprows=1)
R_test = matrix_test.iloc[:,0]
X_test = matrix_test.iloc[:,1:] / 255
N_test = matrix_test.shape[0]
y_matrix_test = np.zeros((N_test,K))

accuracy_best = 0
w_matrix_best = np.zeros((d+1,K))
confusion_matrix_training_best = np.zeros((K,K))
confusion_matrix_test_best = np.zeros((K,K))


print("-----------------------------------------------------------------")
print("K = ", K)
print("E = ", E)
print("d = ", d)
print("N = ", N)
print("N_test = ", N_test)
print("Learning rate = ", learning_rate)
print("Starting the process...\n")


training_accuracies = []
test_accuracies = []
for epochs in range(E):

    delta_w = np.zeros(w_matrix.shape)
    
    for t in range(N):
        x_t = X.iloc[t,:]
        x_t = np.append(x_t, 1)
        x_t = np.transpose(x_t)

        o = np.zeros(K)
        for i in range(K):
            o[i] = np.dot(w_matrix[:,i], x_t)
        
        y_matrix[t] = softmax(o)
        r_t = one_hot_encode(R.iloc[t])

        for i in range(K):
            for j in range(d+1):
                delta_w[j,i] += (r_t[i] - y_matrix[t][i])  * x_t[j]

        for i in range(K):
            for j in range(d+1):
                w_matrix[j,i] += delta_w[j,i] * learning_rate

    for t in range(N):
        x_t = X.iloc[t,:]
        x_t = np.append(x_t, 1)
        x_t = np.transpose(x_t)

        o = np.zeros(K, dtype=np.float64)
        for i in range(K):
            o[i] = np.dot(w_matrix[:,i], x_t)

        y_matrix[t] = softmax(o)

    for t in range(N):
        x_t_test = X_test.iloc[t,:]
        x_t_test = np.append(x_t_test, 1)
        x_t_test = np.transpose(x_t_test)

        o = np.zeros(K, dtype=np.float64)
        for i in range(K):
            o[i] = np.dot(w_matrix[:,i], x_t_test)

        y_matrix_test[t] = softmax(o)


    confusion_matrix_training = np.zeros((K,K))
    confusion_matrix_test = np.zeros((K,K))

    for t in range(N):
        confusion_matrix_training[R.iloc[t]-1, np.argmax(y_matrix[t])]       += 1
    for t in range(N_test):
        confusion_matrix_test[R_test.iloc[t]-1, np.argmax(y_matrix_test[t])] += 1

    accuracy_training = accuracy_confusion_matrix(confusion_matrix_training)
    accuracy_test     = accuracy_confusion_matrix(confusion_matrix_test)
    test_accuracies.append(accuracy_test)
    training_accuracies.append(accuracy_training)

    print("Epoch: ", epochs+1, "Training Accuracy: ", accuracy_training, "Testing Accuracy: ", accuracy_test)

    if accuracy_test > accuracy_best:
        accuracy_best = accuracy_test
        w_matrix_best = w_matrix
        confusion_matrix_test_best     = confusion_matrix_test
        confusion_matrix_training_best = confusion_matrix_training


print("\nConfusion Matrix Training:")
print(*confusion_matrix_training_best, sep='\n', end="\n\n")


print("Confusion Matrix Test:")
print(*confusion_matrix_test_best, sep='\n', end="\n\n")
print("Best Accuracy: ", accuracy_best)

plot_mean_images(w_matrix_best[:-1,:])
#print("10 Dimensional Probabilities :", y_matrix)

plt.plot(training_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Testing Accuracy')
plt.legend()
plt.show()