import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import csv 
import dateutil.parser
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

def count_prices():
    foods_to_predict = ['Lentils', 'Salt (iodised)', 'Sugar', 'Tea (black)', 'Potatoes', 'Rice', 'Onions', 'Tomatoes']
    cities_to_predict = ['Mumbai', 'Delhi', 'Bengaluru', 'Kolkata', 'Hyderabad', 'Lucknow', 'Jaipur', 'Chandigarh', 'Chennai', 'Bhubaneshwar', 'Jammu', 'Bhopal', 'Patna', 'Indore', 'Nagpur', 'Kanpur', 'Ludhiana', 'Varanasi', 'Guwahati', 'Dehradun', 'Jodhpur', 'Cuttack']
    
    with open('India_Food_Prices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        food_to_prices = defaultdict(int)
        for row in reader:
            if (row[CITY_COL] in cities_to_predict):
                month, year = int(row[MONTH_COL]), int(row[YEAR_COL])
                if ((year >= START_YEAR and year < END_YEAR) or (year == END_YEAR and month <= END_MONTH)):
                    food = row[FOOD_TYPE_COL]
                    if row[CITY_COL] == "Varanasi" and 'Salt' in food:
                        print row
                    if food in foods_to_predict:
                        food_to_prices[food] += 1
        for k, v in food_to_prices.iteritems():
            print k, v

def count_tweets():
    foods_to_predict = ['lentil', 'salt', 'sugar', 'black tea', 'tea', 'potato', 'rice', 'onion', 'tomato']
    cities_to_predict = ['Mumbai', 'Delhi', 'Bengaluru', 'Kolkata', 'Hyderabad', 'Lucknow', 'Jaipur', 'Chandigarh', 'Chennai', 'Bhubaneshwar', 'Jammu', 'Bhopal', 'Patna', 'Indore', 'Nagpur', 'Kanpur', 'Ludhiana', 'Varanasi', 'Guwahati', 'Dehradun', 'Jodhpur', 'Cuttack']
    
    f_cnt = open("food_count.txt", "w")
    for city in cities_to_predict:
        f_cnt.write("%s\n" % city)
        food_cnt = np.zeros((len(foods_to_predict), 35), dtype=np.int32)
        with open(city + "_tweets.csv") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                date = dateutil.parser.parse(row["postedTime"])
                date_index = (date.year - 2014)*12 + (date.month - 1)
                tweet = row["tweet"].lower()
                for i in range(len(foods_to_predict)):
                    if foods_to_predict[i] in tweet:
                        food_cnt[i][date_index] += 1

        for i in range(len(foods_to_predict)):
            f_cnt.write("%s:" % foods_to_predict[i])
            for j in range(35):
                f_cnt.write(" %d" % food_cnt[i][j])
            f_cnt.write("\n")
        print "finished", city
    f_cnt.close()

#count_prices()
count_tweets()
