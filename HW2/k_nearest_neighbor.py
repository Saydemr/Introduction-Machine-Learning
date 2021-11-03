import csv
import matplotlib.pyplot as plt
import math

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
            my_dict = { 'petal_length': float(item[0]), 'petal_width': float(item[1]) , 'species' : lookup.index(item[2]), 'prediction' : -1, 'nearest_nine_index' :[]}
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
    return (((float(fx)-float(sx))**2 + (float(fy)-float(sy))**2.0)**0.5)


def k_nearest_neighbor_types(k, flower, data):
    """
    Returns the species of k nearest neighbors of a flower
    """
    species = []
    for i in flower['nearest_nine_index']:
        species.append(data[i]['species'])
    return species[:k]

def my_mode(numbers):
    """
    Returns the most common element in a list
    """
    return max(set(numbers), key=numbers.count)

print('---Training---')
training_data = read_csv_file('training.csv')

adjacency_matrix_training = [[0 for j in range(len(training_data))] for i in range(len(training_data))]
accuracy_training = []
for flower in training_data:
    for flower2 in training_data:
        adjacency_matrix_training[training_data.index(flower)][training_data.index(flower2)] = (euclidian_distance(flower['petal_length'],flower['petal_width'],flower2['petal_length'],flower2['petal_width']) if flower != flower2 else math.inf)

    flower['nearest_nine_index'] = sorted(range(len(adjacency_matrix_training[training_data.index(flower)])), key=lambda k: adjacency_matrix_training[training_data.index(flower)][k])[:9]

for k in range(1,10,2):
    confusion_matrix_training = [[0,0,0],[0,0,0],[0,0,0]]
    for flower in training_data:
        flower['prediction'] = my_mode(k_nearest_neighbor_types(k, flower, training_data))
        confusion_matrix_training[flower['prediction']][flower['species']] += 1
    

    print('k = ', k, end='\n')
    #print('Confusion Matrix', *confusion_matrix_training, sep='\n',end='\n\n')
    print('Accuracy: ', (confusion_matrix_training[0][0] + confusion_matrix_training[1][1] + confusion_matrix_training[2][2]) / float(sum_numbers([sum_numbers(x) for x in confusion_matrix_training])),end='\n')
    accuracy_training.append((confusion_matrix_training[0][0] + confusion_matrix_training[1][1] + confusion_matrix_training[2][2]) / float(sum_numbers([sum_numbers(x) for x in confusion_matrix_training])))


print('\n\n---Testing---')


test_data = read_csv_file('testing.csv')

adjacency_matrix_test = [[0 for j in range(len(test_data))] for i in range(len(test_data))]
accuracy_test = []

for flower in test_data:
    for flower2 in test_data:
        adjacency_matrix_test[test_data.index(flower)][test_data.index(flower2)] = (euclidian_distance(flower['petal_length'],flower['petal_width'],flower2['petal_length'],flower2['petal_width']) if flower != flower2 else math.inf)

    flower['nearest_nine_index'] = sorted(range(len(adjacency_matrix_test[test_data.index(flower)])), key=lambda k: adjacency_matrix_test[test_data.index(flower)][k])[:9]

for k in range(1,10,2):
    confusion_matrix_test = [[0,0,0],[0,0,0],[0,0,0]]
    for flower in test_data:
        flower['prediction'] = my_mode(k_nearest_neighbor_types(k, flower, test_data))
        confusion_matrix_test[flower['prediction']][flower['species']] += 1
    
    accuracy_test.append((confusion_matrix_test[0][0] + confusion_matrix_test[1][1] + confusion_matrix_test[2][2]) / float(sum_numbers([sum_numbers(x) for x in confusion_matrix_test])))
    
    print('k = ', k, end='\n')
    print('Accuracy: ', (confusion_matrix_test[0][0] + confusion_matrix_test[1][1] + confusion_matrix_test[2][2]) / float(sum_numbers([sum_numbers(x) for x in confusion_matrix_test])),end='\n')
    print('Confusion Matrix', *confusion_matrix_test, sep='\n',end='\n\n')
    
