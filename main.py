import csv
import math


def read_csv_file(file_name):
    """
    Reads a csv file after second line and returns three list of lists based on the second column
    """
    numbers = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for item in list(csv_reader):
            my_dict = { 'value': int(item[0]), 'class': int(item[1]) , 'prediction' : -1 }
            numbers.append(my_dict)
    return numbers


def calculate_appearances(numbers, class_id):
    """
    Calculates how many times an item that belonging to a class appears in a list
    """
    appear = 0.0
    for dict in numbers:
        if dict['class'] == class_id:
            appear += 1.0
    return appear


def sum_numbers(numbers, class_id = None):
    """
    Sums all the numbers in a list
    """
    sum = 0.0
    if class_id == None:
        for number in numbers:
            sum += float(number)
    else:
        interested_class = [x for x in numbers if x['class'] == class_id]
        for number in interested_class:
            sum += float(number['value'])
    return sum
    

def my_mean(numbers, class_id = None):
    """
    Calculates the mean of a list of numbers
    """
    if class_id == None:
        return sum_numbers(numbers) / float(len(numbers))
    else:
        return sum_numbers(numbers, class_id) / calculate_appearances(numbers, class_id)


def standard_deviation(numbers, class_id = None):
    """
    Calculates the standard deviation of a list of numbers
    """
    average = 0.0
    variance = 0.0
    if class_id == None:
        average = my_mean(numbers)
        variance = sum_numbers([float((x['value']) - average) ** 2 for x in numbers]) / float(len(numbers))
    else:
        average = my_mean(numbers, class_id)
        variance = sum_numbers([(float(x['value']) - average) ** 2 for x in numbers if x['class'] == class_id]) / calculate_appearances(numbers, class_id)
    return variance ** 0.5




def calculate_likelihood(number, mean, std_dev):
    """
    Calculates the likelihood
    """
    likelihood = -1.0 / (std_dev * (2.0 * math.pi) ** 0.5) * (math.exp(-((number - mean) ** 2) / (2.0 * std_dev ** 2)))
    return likelihood


numbers = read_csv_file("training.csv")
count = float(len(numbers))

prior_class_1 = calculate_appearances(numbers, 1) / count
prior_class_2 = calculate_appearances(numbers, 2) / count
prior_class_3 = calculate_appearances(numbers, 3) / count

#print(prior_class_1, prior_class_2, prior_class_3)

mean_class_1 = my_mean(numbers, 1)
mean_class_2 = my_mean(numbers, 2)
mean_class_3 = my_mean(numbers, 3)

#print(mean_class_1,mean_class_2,mean_class_3)

std_class_1 = standard_deviation(numbers, 1)
std_class_2 = standard_deviation(numbers, 2)
std_class_3 = standard_deviation(numbers, 3)

#print(std_class_1,std_class_2,std_class_3)

for i in range(len(numbers)):
    posterior_one   = calculate_likelihood(numbers[i]['value'], mean_class_1, std_class_1) * prior_class_1
    posterior_two   = calculate_likelihood(numbers[i]['value'], mean_class_2, std_class_2) * prior_class_2
    posterior_three = calculate_likelihood(numbers[i]['value'], mean_class_3, std_class_3) * prior_class_3

    if posterior_one > posterior_two and posterior_one > posterior_three:
        numbers[i]['prediction'] = 1
    elif posterior_two > posterior_one and posterior_two > posterior_three:
        numbers[i]['prediction'] = 2
    elif posterior_three > posterior_one and posterior_three > posterior_two:
        numbers[i]['prediction'] = 3


print(*numbers, sep='\n')