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
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def matrix_multiplication(m1, m2):
    """
    Multiplies two matrices
    """
    result = [[sum(a*b for a,b in zip(row, col)) for col in zip(*m2)] for row in m1]
    return result

def inverse_two_by_two(matrix):
    """
    Calculates the inverse of a 2x2 matrix
    """
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return [[matrix[1][1] / det, -matrix[0][1] / det], [-matrix[1][0] / det, matrix[0][0] / det]]


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


c_s  = [ [x['petal_length'] - mean_setosa[0] for x in training_data if x['species'] == 0] , 
         [x['petal_width'] - mean_setosa[1]  for x in training_data if x['species'] == 0] ]
c_ve = [ [x['petal_length'] - mean_versicolor[0] for x in training_data if x['species'] == 1] , 
         [x['petal_width'] - mean_versicolor[1]  for x in training_data if x['species'] == 1] ]

c_vi = [ [x['petal_length'] - mean_virginica[0] for x in training_data if x['species'] == 2] , 
         [x['petal_width'] - mean_virginica[1]  for x in training_data if x['species'] == 2] ]

covariance_setosa     = matrix_multiplication(c_s, my_transpose(c_s))
covariance_setosa_1   = [[ x / 30.0 for x in covariance_setosa[i]] for i in range(len(covariance_setosa))]
covariance_versicolor = matrix_multiplication(c_ve, my_transpose(c_ve))  
covariance_versicolor_1 = [[ x / 30.0 for x in covariance_versicolor[i]] for i in range(len(covariance_versicolor))]
covariance_virginica  = matrix_multiplication(c_vi, my_transpose(c_vi)) 
covariance_virginica_1  = [[ x / 30.0 for x in covariance_virginica[i]] for i in range(len(covariance_virginica))]


covariance_setosa_i     = inverse_two_by_two(covariance_setosa_1)
covariance_versicolor_i = inverse_two_by_two(covariance_versicolor_1)
covariance_virginica_i  = inverse_two_by_two(covariance_virginica_1)


confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]
for flower in training_data:
    x_minus_mean_setosa = [[flower['petal_length'] - length_mean_setosa, flower['petal_width'] - width_mean_setosa]] 
    x_minus_mean_versicolor = [[flower['petal_length'] - length_mean_versicolor, flower['petal_width'] - width_mean_versicolor]]
    x_minus_mean_virginica = [[flower['petal_length'] - length_mean_virginica, flower['petal_width'] - width_mean_virginica]]

    x_minus_mean_setosa_transpose = my_transpose(x_minus_mean_setosa)
    x_minus_mean_versicolor_transpose = my_transpose(x_minus_mean_versicolor)
    x_minus_mean_virginica_transpose = my_transpose(x_minus_mean_virginica)

    mahalanobis_setosa     = matrix_multiplication(matrix_multiplication(x_minus_mean_setosa_transpose    , covariance_setosa_i)    , x_minus_mean_setosa)
    mahalanobis_versicolor = matrix_multiplication(matrix_multiplication(x_minus_mean_versicolor_transpose, covariance_versicolor_i), x_minus_mean_versicolor)
    mahalanobis_virginica  = matrix_multiplication(matrix_multiplication(x_minus_mean_virginica_transpose , covariance_virginica_i) , x_minus_mean_virginica)

    if mahalanobis_setosa < mahalanobis_versicolor and mahalanobis_setosa < mahalanobis_virginica:
        flower['prediction'] = 0
    elif mahalanobis_versicolor < mahalanobis_setosa and mahalanobis_versicolor < mahalanobis_virginica:
        flower['prediction'] = 1
    elif mahalanobis_virginica < mahalanobis_setosa and mahalanobis_virginica < mahalanobis_versicolor:
        flower['prediction'] = 2

    confusion_matrix[flower['prediction']][flower['species']] += 1

print(*confusion_matrix, sep='\n')
print('\n')

testing_data = read_csv_file('testing.csv')
confusion_matrix_test = [[0,0,0],[0,0,0],[0,0,0]]

for flower in testing_data:
    x_minus_mean_setosa     = [[flower['petal_length'] - length_mean_setosa, flower['petal_width'] - width_mean_setosa]]
    x_minus_mean_versicolor = [[flower['petal_length'] - length_mean_versicolor, flower['petal_width'] - width_mean_versicolor]]
    x_minus_mean_virginica  = [[flower['petal_length'] - length_mean_virginica, flower['petal_width'] - width_mean_virginica]]

    x_minus_mean_setosa_transpose     = my_transpose(x_minus_mean_setosa)
    x_minus_mean_versicolor_transpose = my_transpose(x_minus_mean_versicolor)
    x_minus_mean_virginica_transpose  = my_transpose(x_minus_mean_virginica)

    mahalanobis_setosa     = matrix_multiplication(matrix_multiplication(x_minus_mean_setosa_transpose    , covariance_setosa_i)    , x_minus_mean_setosa)
    mahalanobis_versicolor = matrix_multiplication(matrix_multiplication(x_minus_mean_versicolor_transpose, covariance_versicolor_i), x_minus_mean_versicolor)
    mahalanobis_virginica  = matrix_multiplication(matrix_multiplication(x_minus_mean_virginica_transpose , covariance_virginica_i) , x_minus_mean_virginica)

    if mahalanobis_setosa < mahalanobis_versicolor and mahalanobis_setosa < mahalanobis_virginica:
        flower['prediction'] = 0
    elif mahalanobis_versicolor < mahalanobis_setosa and mahalanobis_versicolor < mahalanobis_virginica:
        flower['prediction'] = 1
    elif mahalanobis_virginica < mahalanobis_setosa and mahalanobis_virginica < mahalanobis_versicolor:
        flower['prediction'] = 2
    
    confusion_matrix_test[flower['prediction']][flower['species']] += 1

print(*confusion_matrix_test, sep='\n')