import csv
from collections import defaultdict

DIR = '/mnt/mounted_bucket/'
STATE_COL = 3
CITY_COL = 5
FOOD_TYPE_COL = 7
DISTRIB_TYPE_COL = 11
MONTH_COL = 14
YEAR_COL = 15
FOOD_PRICE_COL = 16

START_MONTH = 1
START_YEAR = 2014
END_MONTH = 11
END_YEAR = 2016


def parse_file():
    with open(DIR + 'WFPVAM_FoodPrices_24-01-2017.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        india_food_prices = []
        for row in reader:
            if (row[1] == 'India'):
                india_food_prices += [row]
        return india_food_prices

def output_stats(india_food_prices):
    food_to_freq = defaultdict(int) # Food type frequencies
    food_to_prices = defaultdict(list) # Price trends
    #food_to_avg_price = dict() # Average monthly prices 
    city_to_freq = defaultdict(int) # City frequencies
    state_to_freq = defaultdict(int) # State frequencies
    distrib_type_to_freq = defaultdict(int) # Distribution type frequencies
    for row in india_food_prices:
        month, year = int(row[MONTH_COL]), int(row[YEAR_COL])
        if ((year >= START_YEAR and year < END_YEAR) 
                or (year == END_YEAR and month <= END_MONTH)):
            food_type = row[FOOD_TYPE_COL]
            food_to_freq[food_type] += 1
            food_to_prices[food_type].append(float(row[FOOD_PRICE_COL]))
            city_to_freq[row[CITY_COL]] += 1
            state_to_freq[row[STATE_COL]] += 1
            distrib_type_to_freq[row[DISTRIB_TYPE_COL]] += 1
    #for food in food_to_prices:
    #    food_to_avg_price[food] = sum(food_to_prices[food]) / food_to_freq[food]

def main():
    india_food_prices = parse_file()
    output_stats(india_food_prices)

if __name__ == '__main__':
    main()
