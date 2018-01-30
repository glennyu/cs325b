import csv
import dateutil.parser
import HTMLParser
from linear_model import read_features, get_prices
from sklearn.linear_model import LinearRegression
import sys
sys.path.insert(0, '../data_utils')
from util import *

NUM_MONTHS = 35

transportation = ['transport', 'strike', 'hike', 'sack', 'scam', 'rail']
food_to_predict = ['Lentils', 'Wheat', 'Salt (iodised)', 'Lentils (masur)', 'Sugar', 'Tea (black)', 'Potatoes', 'Oil (mustard)', 'Rice', 'Onions', 'Milk (pasteurized)', 'Tomatoes']
food_to_index = [0, 2, 3, 0, 6, 7, 8, 1, 10, 11, 5, 12]

def get_keywords():
    transport_cnt = [0 for i in range(NUM_MONTHS)]
    t_cnt = open("Delhi_transport_count.txt", "w")
    with open("./../../Delhi_tweets.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = dateutil.parser.parse(row["postedTime"])
            date_index = (date.year - 2014)*12 + (date.month - 1)
            
            tweet = ' '.join([word for word in HTMLParser.HTMLParser().unescape(row["tweet"]).split() if "@" not in word and "http" not in word])
            tweet = tweet.lower()
            for i in range(len(transportation)):
                if transportation[i] in tweet:
                	transport_cnt[date_index] += 1

    for i in range(NUM_MONTHS):
        t_cnt.write("%d\n" % transport_cnt[i])
    t_cnt.close()

def read_transport():
	transport_feat = np.zeros((NUM_MONTHS, 1), dtype=np.float32)
	with open("Delhi_transport_count.txt") as t_cnt:
		for i in range(NUM_MONTHS):
			transport_feat[i][0] = float(t_cnt.readline())
	return transport_feat

def lin_reg(sentiment_feat, food_cnt, transport_feat, prices):
	feat = np.hstack((sentiment_feat, np.expand_dims(food_cnt, axis=1), transport_feat))
	train_feat, test_feat = feat[:24], feat[24:]
	train_prices, test_prices = prices[:24], prices[24:]

	reg = LinearRegression()
	reg.fit(train_feat, train_prices)
	#print(reg.coef_)
	#print("score: %f" % reg.score(train_feat, train_prices))
	pred = np.squeeze(reg.predict(test_feat))
	#print("predictions:")
	#print(pred)
	#print("actual:")
	#print(np.squeeze(test_prices))
	print("MAPE: %f" % mape(np.squeeze(test_prices), pred)) 

def main():
    get_keywords()
    # transport_feat = read_transport()
    # sentiment_feat, food_cnt = read_features()
    # prices = get_prices('Delhi')
    # for i in range(len(food_to_predict)):
    #     idx = food_to_index[i]
    #     print(food_to_predict[i])
    #     lin_reg(np.squeeze(sentiment_feat[idx:(idx + 1)]), np.squeeze(food_cnt[idx:(idx + 1)]), transport_feat, prices[i:(i + 1)].T)

if __name__ == '__main__':
	main()