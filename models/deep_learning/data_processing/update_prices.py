import csv
import os

PATH = '../data/'
NUM_MONTHS = 35
TRAIN = 'batches_train/'
VAL = 'batches_val/'
TEST = 'batches_test/'

def update_folder(folder, city_to_trends, city_to_spikes):
    for filename in os.listdir(PATH + folder):
        with open(PATH + folder + filename, 'w') as f:  
            filename = filename.split('_')
            city = filename[0]
            month = int(filename[1])
            f.write('%d,%d' % (city_to_trends[city][month], city_to_spikes[city][month]))

def update_batch_files(city_to_trends, city_to_spikes):
    update_folder(TRAIN, city_to_trends, city_to_spikes)
    update_folder(VAL, city_to_trends, city_to_spikes)
    update_folder(TEST, city_to_trends, city_to_spikes)

def get_data():
    city_to_trends = {}
    city_to_spikes = {}

    with open(PATH + 'India_Onion_Prices_Vector_Updated.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for row in reader:
            city = row[0]
            trends = map(lambda x: int(x) + 1 if x != 'NA' else -1, row[2:2 + NUM_MONTHS])
            spikes = map(lambda x: int(x) if x != 'NA' else -1, row[3 + NUM_MONTHS:3 + 2*NUM_MONTHS])
            city_to_trends[city] = trends
            city_to_spikes[city] = spikes

    return city_to_trends, city_to_spikes

def main():
    city_to_trends, city_to_spikes = get_data()
    update_batch_files(city_to_trends, city_to_spikes)

if __name__ == '__main__':
    main()
