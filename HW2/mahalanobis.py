import csv
import matplotlib.pyplot as plt
from operator import mul
import numpy as np

lookup = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def read_csv_file(file_name):
    """
    Reads a csv file after second line and returns list of dictionaries with the data. Initial values of predictions are -1
    """
    numbers = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for item in list(csv_reader):
            my_dict = { 'petal_length': float(item[0]), 'petal_width': float(item[1]) , 'species' : lookup.index(item[2]), 'prediction' : -1}
            numbers.append(my_dict)
    return numbers


def sum_numbers(numbers):
    """
    Sums all the numbers in a list
    """
    sum = 0.0
    for number in numbers:
        sum += float(number)
    return sum
    

def my_mean(numbers):
    """
    Calculates the mean of a list of numbers
    """
    return sum_numbers(numbers) / float(len(numbers))


def my_transpose(matrix):
    """
    Transposes a matrix
    """
    return [matrix[j][i] for j in range(len(matrix)) for i in range(len(matrix[j]))]


def standard_deviation(numbers):
    """
    Calculates the variance of a list of numbers
    """
    average = my_mean(numbers)
    variance = sum_numbers([float((x['value']) - average) ** 2 for x in numbers]) / float(len(numbers))
    return variance

def covariance(numbers):
    """
    Calculates the covariance of a list of numbers
    """


def mahalanobis_distance(fx,fy,sx,sy):
    """
    Calculates the mahalanobis distance between two points
    """



training_data = read_csv_file('training.csv')

length_mean_setosa = my_mean([instance['petal_length'] for instance in training_data if instance['species'] == 0])
length_mean_versicolor = my_mean([instance['petal_length'] for instance in training_data if instance['species'] == 1])
length_mean_virginica = my_mean([instance['petal_length'] for instance in training_data if instance['species'] == 2])

width_mean_setosa = my_mean([instance['petal_width'] for instance in training_data if instance['species'] == 0])
width_mean_versicolor = my_mean([instance['petal_width'] for instance in training_data if instance['species'] == 1])
width_mean_virginica = my_mean([instance['petal_width'] for instance in training_data if instance['species'] == 2])

mean_setosa = [length_mean_setosa, width_mean_setosa]
mean_versicolor = [length_mean_versicolor, width_mean_versicolor]
mean_virginica = [length_mean_virginica, width_mean_virginica]

covariance_setosa = [[0.0, 0.0], [0.0, 0.0]]
covariance_versicolor = [[0.0, 0.0], [0.0, 0.0]]
covariance_virgina = [[0.0, 0.0], [0.0, 0.0]]


for flower in training_data:
    x_minus_mean_setosa = [flower['petal_length'] - length_mean_setosa, flower['petal_width'] - width_mean_setosa] 
    x_minus_mean_versicolor = [flower['petal_length'] - length_mean_versicolor, flower['petal_width'] - width_mean_versicolor]
    x_minus_mean_virginica = [flower['petal_length'] - length_mean_virginica, flower['petal_width'] - width_mean_virginica]

    x_minus_mean_setosa_transpose = my_transpose(x_minus_mean_setosa)
    x_minus_mean_versicolor_transpose = my_transpose(x_minus_mean_versicolor)
    x_minus_mean_virginica_transpose = my_transpose(x_minus_mean_virginica)

    # (x-m)T . COV-1 . (x-m)
