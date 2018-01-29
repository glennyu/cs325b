from collections import defaultdict
import csv 
import numpy as np

CITY_COL = 5
FOOD_TYPE_COL = 7
MONTH_COL = 14
YEAR_COL = 15
FOOD_PRICE_COL = 16

START_MONTH = 1
START_YEAR = 2014
END_MONTH = 11
END_YEAR = 2016

PATH = './../../'

transportation = ['transport', 'strike', 'hike', 'import', 'sack', 'scam', 'rail', 'export', 'import']

# Input: A = array of true values, F = array of corresponding forecasted values
# Returns MAPE metric
def mape(A, F):
	total = 0
	for i in range(len(A)):
		total += abs(1.0 * (A[i] - F[i]) / A[i])
	return 1.0 * total / len(A)

# Returns all prices in city from 01/2014 to 11/2016
def get_prices(city):
    with open(PATH + 'India_Food_Prices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        food_to_prices = defaultdict(list)
        for row in reader:
            if (row[CITY_COL] == city):
                month, year = int(row[MONTH_COL]), int(row[YEAR_COL])
                if ((year >= START_YEAR and year < END_YEAR) or (year == END_YEAR and month <= END_MONTH)):
                    food = row[FOOD_TYPE_COL]
                    price = float(row[FOOD_PRICE_COL])
                    food_to_prices[food].append(price)

        city_prices = []
        for food in food_to_prices:    
            city_prices.append([food] + food_to_prices[food])
        city_prices = np.asarray(city_prices)
        print city_prices.shape
        assert(city_prices.shape == (21, 36))
        return city_prices
