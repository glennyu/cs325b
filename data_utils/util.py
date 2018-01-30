import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

PATH = '../../' #'../data_utils/'

# Input: A = array of true values, F = array of corresponding forecasted values
# Returns MAPE metric
def mape(A, F):
	total = 0
	for i in range(A.shape[0]):
		total += abs(1.0 * (A[i] - F[i]) / A[i])
	return 1.0 * total / A.shape[0]

food_to_predict = ['Lentils', 'Wheat', 'Salt (iodised)', 'Lentils (masur)', 'Sugar', 'Tea (black)', 'Potatoes', 'Oil (mustard)', 'Rice', 'Onions', 'Milk (pasteurized)', 'Tomatoes']

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
        for food in food_to_predict:
            city_prices.append(food_to_prices[food])
        city_prices = np.asarray(city_prices)
        assert(city_prices.shape == (len(food_to_predict), 35))
        return city_prices

def plot_price_trend(fignum, prediction, true_price, title):
    plt.figure(fignum)
    plt.plot(np.arange(1, prediction.shape[0] + 1), prediction, '-b', label='Predictions')
    plt.plot(np.arange(1, true_price.shape[0] + 1), true_price, '-r', label='True Price')
    plt.legend()
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.title(title)
    plt.savefig(title + '.png')
