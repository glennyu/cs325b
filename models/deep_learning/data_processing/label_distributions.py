import csv

PATH = '../data/'
NUM_MONTHS = 35

city_to_batches = {
                    'Bengaluru': 17545,
                    'Chandigarh': 1290,
                    'Chennai': 1835,
                    'Delhi': 23050,
                    'Gurgaon': 4015,
                    'Hyderabad': 3840,
                    'Jaipur': 1270,
                    'Kolkata': 6035,
                    'Lucknow': 1280,
                    'Mumbai': 42000
                  }

def print_distributions():
    #assert sum(city_to_batches.values()) == 121560
    with open(PATH + 'India_Onion_Prices_Vector.csv') as csvfile:
        f = csv.reader(csvfile)
        trend_dist = [0 for i in range(4)]
        spike_dist = [0 for i in range(3)]
        next(f, None)
        for row in f:
            city = row[0]
            if (city not in city_to_batches): continue
            trends = row[2:2 + NUM_MONTHS]
            spikes = row[3 + NUM_MONTHS:3 + 2*NUM_MONTHS]
            for i in range(NUM_MONTHS):
                if trends[i] == 'NA': 
                    trend_dist[0] += 1
                else:
                    trend_dist[int(trends[i]) + 2] += city_to_batches[city]

                if spikes[i] == 'NA':
                    spike_dist[0] += 1
                else:
                    spike_dist[int(spikes[i]) + 1] += city_to_batches[city]
        print trend_dist
        print spike_dist
                
def main():
    print_distributions()

if __name__ == '__main__':
    main()
