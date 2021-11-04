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


def euclidian_distance(fx,fy,sx,sy):
    """
    Calculates the euclidian distance between two points
    """
    return (((fx-sx)**2 + (fy-sy)**2)**0.5)


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

confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]

for flower in training_data:
    euclidian_setosa = euclidian_distance(flower['petal_length'], flower['petal_width'], mean_setosa[0], mean_setosa[1])
    euclidian_versicolor = euclidian_distance(flower['petal_length'], flower['petal_width'], mean_versicolor[0], mean_versicolor[1])
    euclidian_virginica = euclidian_distance(flower['petal_length'], flower['petal_width'], mean_virginica[0], mean_virginica[1])

    if euclidian_setosa < euclidian_versicolor and euclidian_setosa < euclidian_virginica:
        flower['prediction'] = 0
    elif euclidian_versicolor < euclidian_setosa and euclidian_versicolor < euclidian_virginica:
        flower['prediction'] = 1
    elif euclidian_virginica < euclidian_setosa and euclidian_virginica < euclidian_versicolor:
        flower['prediction'] = 2

    confusion_matrix[flower['prediction']][flower['species']] += 1


testing_data = read_csv_file('testing.csv')
confusion_matrix_test = [[0,0,0],[0,0,0],[0,0,0]]

for flower in testing_data:
    euclidian_setosa = euclidian_distance(flower['petal_length'], flower['petal_width'], mean_setosa[0], mean_setosa[1])
    euclidian_versicolor = euclidian_distance(flower['petal_length'], flower['petal_width'], mean_versicolor[0], mean_versicolor[1])
    euclidian_virginica = euclidian_distance(flower['petal_length'], flower['petal_width'], mean_virginica[0], mean_virginica[1])

    if euclidian_setosa < euclidian_versicolor and euclidian_setosa < euclidian_virginica:
        flower['prediction'] = 0
    elif euclidian_versicolor < euclidian_setosa and euclidian_versicolor < euclidian_virginica:
        flower['prediction'] = 1
    elif euclidian_virginica < euclidian_setosa and euclidian_virginica < euclidian_versicolor:
        flower['prediction'] = 2

    confusion_matrix_test[flower['prediction']][flower['species']] += 1


print('Confusion matrix for training data:', *confusion_matrix,sep='\n', end='\n\n')
print('Confusion matrix for testing data:', *confusion_matrix_test,sep='\n', end='\n\n')


plt.title("Iris Dataset - Training")
plt.scatter([x['petal_length'] for x in training_data if x['species']==0] , [x['petal_width'] for x in training_data if x['species']==0], c='r', label='Iris-Setosa', marker="x")
plt.scatter([y['petal_length'] for y in training_data if y['species']==1] , [y['petal_width'] for y in training_data if y['species']==1], c='g', label='Iris-Versicolor', marker="o")
plt.scatter([z['petal_length'] for z in training_data if z['species']==2] , [z['petal_width'] for z in training_data if z['species']==2], c='b', label='Iris-Virginica', marker="+")

plt.scatter(mean_setosa[0], mean_setosa[1], c='c', marker='^', label='Mean Setosa')
plt.scatter(mean_versicolor[0], mean_versicolor[1], c='m', marker='d', label='Mean Versicolor')
plt.scatter(mean_virginica[0], mean_virginica[1], c='y', marker='p', label='Mean Virginica')

plt.legend(loc='upper left')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()


plt.title("Iris Dataset - Testing")
plt.scatter([x['petal_length'] for x in testing_data if x['species']==0] , [x['petal_width'] for x in testing_data if x['species']==0], c='r', label='Iris-Setosa', marker="x")
plt.scatter([y['petal_length'] for y in testing_data if y['species']==1] , [y['petal_width'] for y in testing_data if y['species']==1], c='g', label='Iris-Versicolor', marker="o")
plt.scatter([z['petal_length'] for z in testing_data if z['species']==2] , [z['petal_width'] for z in testing_data if z['species']==2], c='b', label='Iris-Virginica', marker="+")

plt.scatter(mean_setosa[0]    , mean_setosa[1]    , c='c', marker='^', label='Mean Setosa')
plt.scatter(mean_versicolor[0], mean_versicolor[1], c='m', marker='d', label='Mean Versicolor')
plt.scatter(mean_virginica[0] , mean_virginica[1] , c='y', marker='p', label='Mean Virginica')

plt.legend(loc='upper left')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()