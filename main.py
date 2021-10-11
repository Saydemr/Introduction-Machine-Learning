import csv

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


def standard_deviation(numbers):
    """
    Calculates the standard deviation of a list of numbers
    """
    average = sum(numbers) / len(numbers)
    variance = sum([(x - average) ** 2 for x in numbers]) / len(numbers)
    return variance ** 0.5

def mean(numbers):
    """
    Calculates the mean of a list of numbers
    """
    return sum(numbers) / len(numbers)



ones, twos, threes = read_csv_file("testing.csv")
count = len(ones) + len(twos) + len(threes)
data = [ones, twos, threes]

for items in data:
    mn = mean(items)
    std = standard_deviation(items)
    print("The mean of the data is: {}".format(mn))
    print("The standard deviation of the data is: {}".format(std))

