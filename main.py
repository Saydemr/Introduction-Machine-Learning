import csv


def read_csv_file(file_name):
    """
    Reads a csv file after second line and returns three list of lists based on the second column
    """
    numbers = []
    with open(file_name, 'r') as csv_file:

        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        
        for item in list(csv_reader):
            my_dict = { 'value': item[0], 'class': item[1] , 'prediction' : -1 }
            numbers.append(my_dict)
    return numbers


def sum_numbers(numbers):
    """
    Sums all the numbers in a list
    """
    sum = 0
    for number in numbers:
        sum += number
    return sum
    

def my_mean(numbers):
    """
    Calculates the mean of a list of numbers
    """

    return sum_numbers(numbers) / len(numbers)


def standard_deviation(numbers):
    """
    Calculates the standard deviation of a list of numbers
    """
    average = my_mean(numbers)
    variance = sum_numbers([(x - average) ** 2 for x in numbers]) / len(numbers)
    return variance ** 0.5


def calculate_appearances(numbers, class_id):
    """
    Calculates how many times an item that belongs to a class appears in a list
    """
    appear = 0
    for dict in numbers:
        if dict['class'] == class_id:
            appear += 1
    return appear


def calculate_prior(numbers, class_id):
    """
    Calculates the prior of a class
    """
    appearances = calculate_appearances(numbers, class_id)
    total = len(numbers)
    return appearances / total


numbers = read_csv_file("testing.csv")
