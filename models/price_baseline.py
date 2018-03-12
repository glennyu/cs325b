from collections import defaultdict
import csv
import numpy as np
from sklearn.linear_model import RidgeClassifier as RC

DIR = '/Users/glennyu/Downloads/'
NUM_MONTHS = 38
START_MONTH = 10
START_YEAR = 2013
END_MONTH = 11
END_YEAR = 2016
WINDOW = 3 # Number of previous months to use

# Return map from city to array of prices by month
def get_prices():
    city_to_prices = defaultdict(lambda: [-1 for i in range(NUM_MONTHS)])
    with open(DIR + 'India_Food_Prices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            if ('National' in row[5]): continue
            if ('Onion' not in row[7]): continue
            city = row[5]
            month = int(row[14])
            year = int(row[15])
            if ((year == START_YEAR and month >= START_MONTH) or (year > START_YEAR and year < END_YEAR) or (year == END_YEAR and month <= END_MONTH)):
                idx = (year - START_YEAR)*12 + (month - START_MONTH)
                price = float(row[16])
                city_to_prices[city][idx] = price
    return city_to_prices

# Split prices into X, y for linear regression
# by using previous three months' prices to 
# predict next month's price direction
def split_data(city_to_prices):
    X = []
    y = []
    total = 0
    found = 0
    decrease = 0
    no_change = 0
    increase = 0

    for city in city_to_prices:
        prices = city_to_prices[city]
        for i in range(WINDOW, len(prices)):
            historical_prices = np.array(prices[i - 3: i])
            cur_price = prices[i]
            if (-1 not in historical_prices and cur_price != -1):
                X.append(historical_prices)
                price_deviation = (cur_price - prices[i - 1]) / prices[i - 1]
                #print cur_price, prices[i - 1], price_deviation
                if (price_deviation <= 0.05 and price_deviation >= -0.05):
                    price_direction = 1
                    no_change += 1
                elif (price_deviation <= -0.05):
                    price_direction = 0
                    decrease += 1
                else:
                    price_direction = 2
                    increase += 1
                y.append(price_direction)
                found += 1
            total += 1

    print 'total: %d, found: %d' % (total, found)
    print 'decrease: %d, no change: %d, increase: %d' % (decrease, no_change, increase)
    return np.array(X), np.array(y)

# Output training and validation accuracies using 
# Ridge Classifier
def train_model(X, y):
    X_train, y_train = X[:len(X)/2], y[:len(y)/2]
    X_val, y_val = X[len(X)/2:], y[len(y)/2:]
    clf = RC()
    clf.fit(X_train, y_train)
    mean_train_acc = clf.score(X_train, y_train)
    mean_val_acc = clf.score(X_val, y_val)
    #print clf.predict(X_val)
    print 'train acc: %f, val acc: %f' % (mean_train_acc, mean_val_acc)

def main():
    city_to_prices = get_prices()
    X, y = split_data(city_to_prices)
    train_model(X, y)


if __name__ == '__main__':
    main()


            
