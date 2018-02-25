import csv
import os

PATH = '../data/'
NUM_MONTHS = 35
TRAIN = 'batches_train'
VAL = 'batches_val'
TEST = 'batches_test'

def update_batch_files(city_to_trends, city_to_spikes):
    for filename in os.listdir(PATH + TRAIN):
        with open(PATH + TRAIN

def get_data():
    city_to_trends = {}
    city_to_spikes = {}

    with open(PATH + 'India_Onion_Prices_Vector_Updated.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            city = row[0]
            trends = map(lambda x: int(x) + 1, city[2:2 + NUM_MONTHS])
            spikes = map(int, city[3 + NUM_MONTHS:3 + 2*NUM_MONTHS])
            city_to_trends[city] = trends
            city_to_spikes[city] = spikes

    return city_to_trends, city_to_spikes

def main():
    city_to_trends, city_to_spikes = get_data()
    update_batch_files(city_to_trends, city_to_spikes)

if __name__ == '__main__':
    main()
