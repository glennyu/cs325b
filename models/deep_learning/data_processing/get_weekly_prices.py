import math
import os
import csv
from datetime import date, timedelta
from collections import defaultdict

PATH = '../data/Onion Weekly By Market/'
BATCHES_DIR = '../data/'
START = date(2014, 1, 1)
END = date(2016, 11, 30)
VAL_CITIES = ['bathinda', 'siliguri', 'nagpur', 'varanasi', 'lucknow', 'kota', 'panchkula', 'raipur']
N = 6 # Number of previous prices required

cities = 'Ernakulam, Raipur, Ranchi, Jaipur, Thiruchirapalli, Dehradun, Shillong, Amritsar, Bengaluru, Patna, Trivandrum, Itanagar, Kolkata, Dimapur, Mandi, Bhopal, Puducherry, Chandigarh, Gurgaon, Chennai, Ludhiana, Jodhpur, Jammu, Shimla, Delhi, Bhubaneshwar, Jabalpur, Karnal, Bathinda, Hisar, Vijaywada, Srinagar, Ahmedabad, Kota, Varanasi, Lucknow, Kohima, Dharwad, Nagpur, Gwalior, Cuttack, Port Blair, Siliguri, Aizwal, Indore, Rajkot, Dindigul, Hyderabad, Kozhidoke, Kanpur, Rourkela, Panchkula, Agartala, Bhagalpur, Mumbai, Panaji, Guwahati, Sambalpur, Agra'
cities = [word.lower() for word in cities.split(', ')]

# Return map from city to list [date, price]
def get_city_to_weeks():
    city_to_weeks = defaultdict(list)
    city_set = set(cities)
    for filename in os.listdir(PATH):
        filename = filename.lower()
        for city in city_set:
            if city in filename:
                with open(PATH + filename) as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)
                    for row in reader:
                        time = map(int, row[1:4])
                        cur = date(time[0], time[1], time[2])
                        price = float(row[13])
                        if (cur >= START and cur <= END and not math.isnan(price)):
                            city_to_weeks[city].append(time + [price])
                    city_set.remove(city)
                break

    for city in city_to_weeks:
        city_to_weeks[city] = sorted(city_to_weeks[city], key=lambda x: x[0])

    return city_to_weeks

# Returns map from city to map of valid date (of (n + 1)th week) to prices (of n previous weeks plus current week))
def get_valid_city_weeks():
    city_to_weekly_prices = get_city_to_weeks()
    #print sum([len(city_to_weekly_prices[city]) for city in city_to_weekly_prices])
    city_to_valid_weeks = defaultdict(lambda: defaultdict(list))
    for city in city_to_weekly_prices:  
        all_weeks = city_to_weekly_prices[city]
        for week in range(len(all_weeks)):
            if (week < N): continue
            cur = all_weeks[week]
            cur_date = date(cur[0], cur[1], cur[2])

            is_valid = True
            for i in range(N - 1):
                prev_week = all_weeks[week - i - 1]
                if cur_date - date(prev_week[0], prev_week[1], prev_week[2]) > timedelta(days = (i + 1)*7):
                    is_valid = False
                    break
            if (is_valid):
                prices = [all_weeks[j][3] for j in range(week - N, week + 1)]
                city_to_valid_weeks[city][cur_date] = prices

    return city_to_valid_weeks

# Writes to training and evaluation data files
def write_data(city_to_valid_weeks):
    with open(BATCHES_DIR + 'weekly_prices_train.txt', 'a') as tf:
        with open(BATCHES_DIR + 'weekly_prices_val.txt', 'a') as vf:   
            for city in city_to_valid_weeks:
                all_weeks = city_to_valid_weeks[city]
                for week in all_weeks:
                    prices = map(str, all_weeks[week])
                    if (city in VAL_CITIES):
                        vf.write(' '.join(prices) + '\n')
                    else:
                        tf.write(' '.join(prices) + '\n')

# Main
def main():
    city_to_valid_weeks = get_valid_city_weeks()
    write_data(city_to_valid_weeks)

if __name__ == '__main__':
    main()

