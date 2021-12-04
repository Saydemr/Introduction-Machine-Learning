from os import sep
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys

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


def accuracy_confusion_matrix(x):
    correct = 0
    total   = 0
    for i in range(len(x)):
        for j in range(len(x[i])):
            total += x[i][j]
            if i == j:
                correct += x[i][j]
    return correct/total


def one_hot_encode(y):
    return np.transpose(np.eye(10)[y-1])


def one_hot_decode(y):
    return np.argmax(y) + 1


def sigmoid(x):
    return 1. / (1.+ np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


learning_rate = 0.1
K = 10
E = 50
d = 100

if len(sys.argv) == 2:
    learning_rate = float(sys.argv[1])

matrix = pd.read_csv('training.csv', header=None, skiprows=1)
R = matrix.iloc[:,0]
X = matrix.iloc[:,1:] / 255
N = X.shape[0]
y_matrix = np.zeros((N,K))

matrix_test = pd.read_csv('testing.csv', header=None, skiprows=1)
R_test = matrix_test.iloc[:,0]
X_test = matrix_test.iloc[:,1:] / 255
N_test = matrix_test.shape[0]
y_matrix_test = np.zeros((N_test,K))

Hs              = [5, 10, 25, 50, 75]
best_H_accuracy = 0
best_H          = -1
best_H_accuracies = []

print("K = ", K)
print("E = ", E)
print("d = ", d)
print("N = ", N)
print("N_test = ", N_test)
print("Learning rate = ", learning_rate)
print("H values are: ", *Hs)

print("Starting the process...\n")
print("-----------------------------------------------------------------")

for H in Hs:
    accuracy_best = 0
    w_matrix_best = np.zeros((d+1,H))
    confusion_matrix_training_best = np.zeros((K,K))
    confusion_matrix_test_best     = np.zeros((K,K))

    training_accuracies = []
    test_accuracies     = []

    v_matrix = np.random.uniform(low=-0.01, high=0.01, size=(H+1, K))
    w_matrix = np.random.uniform(low=-0.01, high=0.01, size=(d+1 ,H))
    for epochs in range(E):
        random_list = list(range(N))
        random.shuffle(random_list)

        for instance in random_list:
            x_t = X.iloc[instance,:]
            x_t = np.append(1, x_t)
            x_t = np.transpose(x_t)

            z = np.zeros(H)

            for h in range(H):
                w_h_T  = np.transpose(w_matrix[:,h]) 
                z[h]   = np.dot(w_h_T, x_t)
            
            z = sigmoid(z)
                        
            z = np.append(1, z)
            z = np.transpose(z)

            os = np.zeros(K)
            for i in range(K):
                os[i] = np.dot(np.transpose(v_matrix[:,i]), z)
            

            y_matrix[instance] = softmax(os)

            delta_v_matrix = np.zeros(v_matrix.shape)
            delta_w_matrix = np.zeros(w_matrix.shape)

            r_t = one_hot_encode(R.iloc[instance])

            for i in range(K):
                delta_v_matrix[:,i] = learning_rate*(r_t[i] - y_matrix[instance][i]) * z

            for h in range(H):
                summa = 0.0
                for i in range(K):
                    summa += (r_t[i] - y_matrix[instance][i])*v_matrix[h,i]
                delta_w_matrix[:,h] = learning_rate*summa*z[h+1]*(1-z[h+1])*x_t
            
            for i in range(K):
                v_matrix[:,i] += delta_v_matrix[:,i]

            for h in range(H):
                w_matrix[:,h] += delta_w_matrix[:,h]
        
        for instance in random_list:
            x_t = X.iloc[instance,:]
            x_t = np.append(1, x_t)
            x_t = np.transpose(x_t)

            z = np.zeros(H)
            for h in range(H):
                w_h_T  = np.transpose(w_matrix[:,h]) 
                z[h]   = np.dot(w_h_T, x_t)
            
            z = sigmoid(z)

            z = np.append(1, z)
            z = np.transpose(z)

            os = np.zeros(K)
            for i in range(K):
                os[i] = np.dot(np.transpose(v_matrix[:,i]), z)
            
            y_matrix[instance] = softmax(os)
        
        for instance in random_list:
            x_t_test = X_test.iloc[instance,:]
            x_t_test = np.append(1, x_t_test)
            x_t_test = np.transpose(x_t_test)

            z_test = np.zeros(H)
            for h in range(H):
                w_h_T  = np.transpose(w_matrix[:,h]) 
                z_test[h]   = np.dot(w_h_T, x_t_test)
            
            z_test = sigmoid(z_test)

            z_test = np.append(1, z_test)
            z_test = np.transpose(z_test)

            os = np.zeros(K)
            for i in range(K):
                os[i] = np.dot(np.transpose(v_matrix[:,i]), z_test)

            y_matrix_test[instance] = softmax(os)

        confusion_matrix_training = np.zeros((K,K))
        confusion_matrix_test     = np.zeros((K,K))

        for t in range(N):
            confusion_matrix_training[np.argmax(y_matrix[t])][R.iloc[t]-1] += 1
        for t in range(N_test):
            confusion_matrix_test[np.argmax(y_matrix_test[t])][R_test.iloc[t]-1] += 1

        accuracy_training = accuracy_confusion_matrix(confusion_matrix_training)
        accuracy_test     = accuracy_confusion_matrix(confusion_matrix_test)
        test_accuracies.append(accuracy_test)
        training_accuracies.append(accuracy_training)

        print("H: ", H," Epoch: ", epochs+1, "Training Accuracy: ", accuracy_training, "Testing Accuracy: ", accuracy_test)

        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
            w_matrix_best = w_matrix
            confusion_matrix_test_best     = confusion_matrix_test
            confusion_matrix_training_best = confusion_matrix_training
            best_H = H

    best_H_accuracies.append(accuracy_best)
    print("H: ", H, " Best Accuracy: ", accuracy_best)
    print("-----------------------------------------------------------------")
    #print("10 Dimensional Probabilities (Training) :", y_matrix)
    #print("10 Dimensional Probabilities (Testing) :", y_matrix_test)

    plt.title("Accuracy vs Epochs for H = " + str(H))
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.legend()
    plt.show()

print(*best_H_accuracies, sep = "\n")