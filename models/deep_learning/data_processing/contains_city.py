import os
import csv
from datetime import date, timedelta
from collections import defaultdict

PATH = 'Onion Weekly By Market/'
START = date(2014, 1, 1)
END = date(2016, 11, 30)

cities = 'Ernakulam, Raipur, Ranchi, Jaipur, Thiruchirapalli, Dehradun, Shillong, Amritsar, Bengaluru, Patna, Trivandrum, Itanagar, Kolkata, Dimapur, Mandi, Bhopal, Puducherry, Chandigarh, Gurgaon, Chennai, Ludhiana, Jodhpur, Jammu, Shimla, Delhi, Bhubaneshwar, Jabalpur, Karnal, Bathinda, Hisar, Vijaywada, Srinagar, Ahmedabad, Kota, Varanasi, Lucknow, Kohima, Dharwad, Nagpur, Gwalior, Cuttack, Port Blair, Siliguri, Aizwal, Indore, Rajkot, Dindigul, Hyderabad, Kozhidoke, Kanpur, Rourkela, Panchkula, Agartala, Bhagalpur, Mumbai, Panaji, Guwahati, Sambalpur, Agra'

cities = [word.lower() for word in cities.split(', ')]
city_to_weeks = defaultdict(list)
res = set()
total_bytes = 0

def get_city_to_weeks():
    for filename in os.listdir(PATH):
        filename = filename.lower()
        for city in cities:
            if city in filename:
                '''
                statinfo = os.stat(PATH + filename)
                size = statinfo.st_size
                res.add(city)
                total_bytes += size
                '''
                with open(PATH + filename) as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)
                    prevDate = None
                    prevPrice = None
                    for row in reader:
                        time = map(int, row[1:4])
                        cur = date(time[0], time[1], time[2])
                        price = float(row[13])
                        if not prevDate or not prevPrice:
                            prevDate = cur
                            prevPrice = price
                            continue
                        time_diff = cur - prevDate
                        if (cur >= START and cur <= END and time_diff < timedelta(days=8)):
                            price_change = 1
                            if (price == prevPrice):
                                price_change = 0
                            elif (price < prevPrice):
                                price_change = -1
                            city_to_weeks[city].append(time + [price_change])
                        prevDate = cur
                        prevPrice = price
                break
    return city_to_weeks

total = 0
for city in city_to_weeks:
    print city, len(city_to_weeks[city])
    total += len(city_to_weeks[city])
print 'Total weeks: %d' % total

'''
for city in res:
    print city
	
print len(res)
print total_bytes / 1e6
'''

