import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

#DIR = '/mnt/mounted_bucket/'
DIR = './'
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

def parse_file():
    with open(DIR + 'WFPVAM_FoodPrices_24-01-2017.csv') as csvfile:
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
    output_correlation(food_to_prices)
    #output_spike_histogram(spike_percent_to_freq)

def main():
    india_food_prices = parse_file()
    output_stats(india_food_prices)

if __name__ == '__main__':
    main()
