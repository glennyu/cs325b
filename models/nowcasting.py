import csv
import nltk
import html
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

class NowcastingModel(object):
    def __init__(self, city):
        self.city = city
        self.tweets = None
        self.food_words = set(['lentil', 'lentils', 'oil', 'soybean', 
            'soybeans', 'moong', 'wheat', 'salt', 'masur', 
            'flour', 'milk', 'pasteurized', 'sugar', 'palm', 
            'sunflower', 'jaggery', 'gur', 'tea', 'urad', 'potato', 
            'potatoes', 'ghee', 'vanaspati', 'mustard', 'rice', 'onion', 
            'onions', 'tomato', 'tomatoes', 'groundnut', 'groundnuts'])
        self.price_words = set(['price', 'priced', 'sell', 'sells', 'sold', 'buy', 'buys', 'bought', 
                                'rs', 'rps', 'rupee', 'rupees', 'inr', 'dollar', 'euro', 
                                'kg', 'gram', 'grams', 'kilo', 'lit', 'liter', 'l'])
        self.price_units = set(['rs', 'rps', 'rupee', 'rupees', 'inr', 'dollar', 'euro'])
        self.commodity_units = set(['kg', 'gram', 'grams', 'kilo', 'lit', 'liter', 'l'])



    def create_tweets_matrix(self, filename):
        '''
        Returns: a list of tweets represented as a list of words. Each tweet was tokenized and lowercased.
        '''
        tweets = []
        tokenizer = TweetTokenizer()

        with open(filename) as csvfile:
            csvreader = csv.DictReader(csvfile)
            # i = 0
            for row in csvreader:
                tokens = tokenizer.tokenize(html.unescape(row["tweet"]).lower().replace('/', ' ')) # replaces '/'' with ' ' so 'price/unit' can get tokenized into a price and a unit
                tweet_object = {
                    "tweet": tokens, 
                    "postedTime": row["postedTime"],
                    "actorLoc": row["actorLoc"],
                    "areakm2": row["areakm2"],
                    "lng": row["lng"],
                    "lat": row["lat"]
                }
                tweets.append(tweet_object)
                # i += 1
                # if i == 10:
                #     break
        self.tweets = tweets



    def get_relevant_tweets(self):
        ''' 
        Returns: list of strings, where each string is a lowercase tweet
        '''
        relevant_tweets = []
        for tweet_object in tqdm(self.tweets):
            tweet = tweet_object["tweet"]
            contains_food_word = False
            contains_digit = False
            contains_unit = False
            for word in tweet:
                if word in self.food_words:
                    contains_food_word = True
                if word.isnumeric():
                    contains_digit = True
                if word in self.price_units or word in self.commodity_units:
                    contains_unit = True
            
            if contains_food_word and contains_digit and contains_unit:
                tweet_object["tweet"] = ' '.join(tweet)
                relevant_tweets.append(tweet_object)
        return relevant_tweets


    def write_tweets_to_file(self, tweets, filename):
        with open("data/csv/" + filename, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            count = 0
            for tweet_object in tweets:
                if count == 0:
                    header = tweet_object.keys()
                    writer.writerow(header)
                    count += 1

                writer.writerow(tweet_object.values())


def main():
    model = NowcastingModel("Delhi")
    model.create_tweets_matrix('data/csv/Delhi_tweets.csv')
    relevant_tweets = model.get_relevant_tweets()

    print(model.get_relevant_tweets())
    model.write_tweets_to_file(relevant_tweets, "Relevant_tweets.csv")

if __name__ == "__main__":
    main()