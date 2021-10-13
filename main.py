import csv
import math


def read_csv_file(file_name):
    """
    Reads a csv file after second line and returns three list of lists based on the second column
    """
    ones = []
    twos = []
    threes = []
    with open(file_name, 'r') as csv_file:

        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        
        for item in list(csv_reader):
            if item[1] == 1:
                ones.append(int(item[1]))
            elif item[1] == 2:
                twos.append(int(item[1]))
            elif item[1] == 3:
                threes.append(int(item[1]))

    return ones, twos, threes


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


def calculate_likelihoods(numbers):
    """
    Calculates the likelihoods of a list of list of numbers
    """
    likelihoods = []
    for a_set in numbers:
        mn = my_mean(a_set)
        std = standard_deviation(a_set)

        # Calculate the likelihood of each number in the set
        likelihoods.append([1 / (std * (2 * 3.14) ** 0.5) * (math.exp(-((x - mn) ** 2) / (2 * std ** 2))) for x in a_set])
    return likelihoods


ones, twos, threes = read_csv_file("testing.csv")
count = len(ones) + len(twos) + len(threes)
data = [ones, twos, threes]

prior_one = len(ones) / count
prior_two = len(twos) / count
prior_three = len(threes) / count

mean_one = my_mean(ones)
mean_two = my_mean(twos)
mean_three = my_mean(threes)

likelihoods = calculate_likelihoods(data)

for i in range(len(data[0])):
    highest_posterior = -1.0

    for j in range(len(data)):
        posterior = prior_one * likelihoods[j][i]
        if posterior > highest_posterior:
            highest_posterior = posterior
        

