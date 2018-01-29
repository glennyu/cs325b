import csv
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

from util import *

#DIR = '/mnt/mounted_bucket/'
DIR = './../../'
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

SPIKE = 0.1
SPIKE_SCALE = 0.1

tweet_cnt = [218054, 219862, 214713, 250476, 261245, 328003, 335138, 340330, 331155, 373601, 386361, 415478, 443676, 267091]

SOUTHERN_CITIES = ['Port Blair', 'T.Puram', 'Ernakulam', 'Dindigul', 'Kozhikode', 'Tiruchirappalli', 'Puducherry', 'Bengaluru', 'Chennai',
                    'Srinagar', 'Hyderabad', 'Dharwad', 'Panaji', 'Mumbai', 'Raipur', 'Bhubaneswar', 'Sambalpur', 'Nagpur', 'Rajkot', 'Ahmedabad'
                    'Bhopal', 'Jabalpur', 'Rourkela', 'Kolkata', 'Ranchi', 'Agartala', 'Aizwal']

def parse_file():
    with open(DIR + 'WFPVAM_FoodPrices_24-01-2017.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = ', '.join(reader.next()) + '\n'
        india_food_prices = []
        with open(DIR + 'India_Food_Prices.csv', 'w') as output:
            output.write(header)
            for row in reader:
                if (row[1] == 'India'):
                    india_food_prices += [row]
                    output.write(','.join(row) + '\n')
            return india_food_prices

def read_file():
    with open(DIR + 'India_Food_Prices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        india_food_prices = []
        for row in reader:
            if (row[1] == 'India'):
                india_food_prices += [row]
        return india_food_prices

def output_price_plots(food_to_prices):
    def plot_prices(food, prices, fig_num):
        plt.figure(fig_num)
        x = [i for i in range(1, len(prices) + 1)]
        plt.xlabel('Month')
        plt.ylabel('National Average Price (INR)')
        plt.title('Price Trend for ' + food)
        plt.plot(x, prices)
        food = food.replace('/', '_')
        plt.savefig(food + '.png')

    fig_num = 10
    for food in food_to_prices:
        city_to_prices = food_to_prices[food]
        national_prices = city_to_prices['National Average'] 
        if (len(national_prices) > 0):
            plot_prices(food, national_prices, fig_num)
            fig_num += 1

def output_spike_histogram(spike_freqs):
    max_spike = 2.0
    freqs = np.zeros(int(max_spike / 0.1) + 1)
    for spike in spike_freqs:
        if (spike < 2.0):
            freqs[int(spike / 0.1)] += spike_freqs[spike]
        else:
            freqs[-1] += spike_freqs[spike]
    x = np.arange(len(freqs))
    x_labels = [str(i * 0.1) for i in range(len(freqs) - 1)] + ['>' + str(max_spike)]

    def plot_histogram(freqs, x, x_labels, start, fig_num):
        plt.figure(fig_num)
        plt.bar(x, freqs)
        plt.xticks(x, x_labels)
        plt.xlabel('Spike')
        plt.ylabel('Frequency')
        title = 'Spike histogram starting from ' + str(start)        
        plt.title(title)
        plt.savefig(title + '.png')

    plot_histogram(freqs, x, x_labels, 0.0, 0)
    plot_histogram(freqs[1:], x[1:], x_labels[1:], 0.1, 1)

def output_food_correlations(food_to_prices):
    def plot(x, y, food1, food2):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        print food1 + ' ' + food2 + ' r-squared: ', r_value**2
        plt.plot(x, y, 'o', label='original data')
        plt.plot(x, intercept + slope*x, 'r', label='fitted line')
        plt.legend(loc='best')
        plt.savefig(food1 + '_' + food2 + '.png')

    def get_prices(food1, food2):
        # return np.array(food_to_prices[food1]['National Average'] 
                            # + food_to_prices[food2]['National Average'])
        food1 = food_to_prices[food1]['National Average'] 
        food2 = food_to_prices[food2]['National Average'] 
        food1_arr = []
        food2_arr = []
        for i in range(1, len(food1)):
            if i == 3: continue
            food1_arr.append(food1[i] - food1[i - 1])
            food2_arr.append(food2[i] - food2[i - 1])
        return np.array(food1_arr + food2_arr)

    x = [i for i in range(1, 13)]
    food1 = 'Wheat'
    food2 = 'Rice'
    plot(np.array(x + x), get_prices(food1, food2), food1, food2)

def output_pca(food_to_prices):
    # Get common cities between all foods
    common_cities = set()
    for food in food_to_prices:
        if (len(common_cities) == 0):
            common_cities = set(food_to_prices[food].keys())
        else:
            common_cities &= set(food_to_prices[food].keys())
    common_cities = list(common_cities)
    foods = sorted(food_to_prices.keys())

    # 2-d array where A[i][j] = average price of food i in city j
    A = [[0 for col in range(len(common_cities))] for row in range(len(food_to_prices))]
    for i, food in enumerate(foods):
        for j, city in enumerate(common_cities):
            A[i][j] = float(sum(food_to_prices[food][city]) / len(food_to_prices[food][city]))
    A = np.array(A)

    # 2-d array where B[i][j] = average price of food j in city i
    B = [[0 for col in range(len(food_to_prices))] for row in range(len(common_cities))]
    for i, city in enumerate(common_cities):
        for j, food in enumerate(foods):
            B[i][j] = float(sum(food_to_prices[food][city]) / len(food_to_prices[food][city]))
    B = np.array(B)

    pca = PCA(n_components=2)
    B_new = pca.fit_transform(B)
    X = B_new[:,0]
    Y = B_new[:,1]
    fig = plt.figure(20)
    ax = fig.add_subplot(111)
    for x, y, label in zip(X, Y, common_cities):
        ax.scatter(x, y, label=label)
    colormap = plt.cm.gist_ncar 
    colorst = [colormap(i) for i in np.linspace(0, 0.9,len(ax.collections))]       
    for t,j1 in enumerate(ax.collections):
        j1.set_color(colorst[t])
    ax.legend(fontsize='small')
    plt.show()

def output_correlation(food_to_prices):
    fig_num = 100
    for food in food_to_prices:
        city_to_prices = food_to_prices[food]
        national_prices = city_to_prices['National Average'] 
        print food
        r,p = pearsonr(tweet_cnt, national_prices)
        print r, p
        
        if food == "Sugar" or food == "Potatoes":
            plt.figure(fig_num)
            fig_num += 1
            fig, ax1 = plt.subplots()
            t = np.arange(1, 15)
            ax1.plot(t, national_prices, 'b-')
            ax1.set_xlabel('Month')
            ax1.set_ylabel('Price', color='b')
            ax1.tick_params('y', colors='b')

            ax2 = ax1.twinx()
            ax2.plot(t, tweet_cnt, 'r-')
            ax2.set_ylabel('# of Tweets', color='r')
            ax2.tick_params('y', colors='r')

            fig.tight_layout()
            plt.title(food + " vs. Tweet Volume")
            plt.savefig(food + "_tweetcnt_correlation.png")

def output_food_city_correlations(food_to_prices):
    for food in food_to_prices:
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        for city in food_to_prices[food]:
            if (city == 'National Average'): continue
            prices = food_to_prices[food][city]
            if (city in SOUTHERN_CITIES):
                x1 += list(range(1, len(prices) + 1))
                y1 += prices
            else:
                x2 += list(range(1, len(prices) + 1))
                y2 += prices

    slope, intercept, r_value, p_value, std_err = stats.linregress(x1, y1)
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)
    print 'Southern', r_value, p_value
    print 'Northern', r_value2, p_value2

def output_stats(india_food_prices):
    # Extract basic data
    food_to_freq = defaultdict(int) # Food type frequencies
    food_to_prices = defaultdict(lambda : defaultdict(list)) # {Food : {City : Price}}
    city_to_freq = defaultdict(int) # City frequencies
    state_to_freq = defaultdict(int) # State frequencies
    distrib_type_to_freq = defaultdict(int) # Distribution type frequencies
    for row in india_food_prices:
        month, year = int(row[MONTH_COL]), int(row[YEAR_COL])
        if ((year >= START_YEAR and year < END_YEAR) 
                or (year == END_YEAR and month <= END_MONTH)):
            food_type = row[FOOD_TYPE_COL]
            price = float(row[FOOD_PRICE_COL])
            city = row[CITY_COL]
            state = row[STATE_COL]
            distrib_type = row[DISTRIB_TYPE_COL]

            food_to_freq[food_type] += 1
            food_to_prices[food_type][city].append(price)
            city_to_freq[city] += 1
            state_to_freq[state] += 1
            distrib_type_to_freq[distrib_type] += 1

    # print 'Number of food types: ' + str(len(food_to_freq))
    # print 'Number of cities: ' + str(len(city_to_freq.keys()))
    # print 'Number of states: ' + str(len(state_to_freq.keys()))
    # print 'Number of food-city pairs: ' + str(sum([len(food_to_prices[food]) for food in food_to_prices]))

    # Get price spikes and spike frequencies
    food_to_spikes = defaultdict(lambda : defaultdict(int))
    spike_percent_to_freq = defaultdict(int)
    for food in food_to_prices:
        city_to_prices = food_to_prices[food]
        for city in city_to_prices:
            prices = city_to_prices[city]
            for idx in range(1, len(prices)):
                quotient = float(prices[idx]) / prices[idx - 1]
                spike_percent_to_freq[int(abs(quotient - 1) / SPIKE_SCALE) * SPIKE_SCALE] += 1  
                if (quotient <= (1.0 - SPIKE) or quotient >= (1.0 + SPIKE)):
                    food_to_spikes[food][city] += 1

    #output_price_plots(food_to_prices)
    #output_correlation(food_to_prices)
    #output_spike_histogram(spike_percent_to_freq)
    #output_food_correlations(food_to_prices)
    #output_pca(food_to_prices)
    output_food_city_correlations(food_to_prices)


def main():
    # india_food_prices = parse_file()
    india_food_prices = read_file()
    output_stats(india_food_prices)
    print get_prices('Delhi')

if __name__ == '__main__':
    main()
