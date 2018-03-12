import csv
import os

PATH = '../data/'
PRICE_VECTOR_FILE = 'India_Onion_Prices_Vector_5%.csv'
NUM_MONTHS = 35
TRAIN = 'batches_train/'
VAL = 'batches_val/'
TEST = 'batches_test/'

def update_folder(folder, city_to_trends, city_to_spikes):
    for filename in os.listdir(PATH + folder):
        old_file = PATH + folder + filename
        new_file = PATH + folder + filename + '_updated'
        is_first_line = True
        with open(old_file, 'r') as rf: 
            with open(new_file, 'w') as wf:
                filename = filename.split('_')
                city = filename[0]
                month = int(filename[1])
                for line in rf:
                    if is_first_line:
                        wf.write('%d,%d,%s\n' % (city_to_trends[city][month], city_to_spikes[city][month], ','.join(line.strip().split(',')[-2:])))
                        is_first_line = False
                    else:
                        wf.write(line)
        os.rename(new_file, old_file) 

def update_batch_files(city_to_trends, city_to_spikes):
    update_folder(TRAIN, city_to_trends, city_to_spikes)
    update_folder(VAL, city_to_trends, city_to_spikes)
    update_folder(TEST, city_to_trends, city_to_spikes)

def get_data():
    city_to_trends = {}
    city_to_spikes = {}

    with open(PATH + PRICE_VECTOR_FILE) as csvfile:
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
