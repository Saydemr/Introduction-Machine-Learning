import csv
import matplotlib.pyplot as plt

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
            my_dict = { 'petal_length': float(item[0]), 'petal_width': float(item[1]) , 'species' : lookup.index(item[2]), 'prediction' : -1, 'nearest_nine' :[]}
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


def euclidian_distance(fx,fy,sx,sy):
    """
    Calculates the euclidian distance between two points
    """
    return (((fx-sx)**2 + (fy-sy)**2)**0.5)


training_data = read_csv_file('training.csv')

adjacency_matrix = [[0 for j in range(len(training_data))] for i in range(len(training_data))]

for flower in training_data:
    for flower2 in training_data:
        adjacency_matrix[training_data.index(flower)][training_data.index(flower2)] = euclidian_distance(flower['petal_length'],flower['petal_width'],flower2['petal_length'],flower2['petal_width'])

    for i in range(len(training_data)):
        adjacency_matrix[training_data.index(flower)][i] = []


