from collections import defaultdict
import csv
import numpy as np
import itertools
from sklearn.linear_model import RidgeClassifier as RC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DIR = '../data/'
NUM_MONTHS = 38
START_MONTH = 10
START_YEAR = 2013
END_MONTH = 11
END_YEAR = 2016
WINDOW = 3 # Number of previous months to use
model = "Random Forest"

# Return map from city to array of prices by month
def get_prices():
    city_to_prices = defaultdict(lambda: [-1 for i in range(NUM_MONTHS)])
    with open(DIR + 'India_Food_Prices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        for row in reader:
            if ('National' in row[5]): continue
            if ('Onion' not in row[7]): continue
            city = row[5]
            month = int(row[14])
            year = int(row[15])
            if ((year == START_YEAR and month >= START_MONTH) or (year > START_YEAR and year < END_YEAR) or (year == END_YEAR and month <= END_MONTH)):
                idx = (year - START_YEAR)*12 + (month - START_MONTH)
                price = float(row[16])
                city_to_prices[city][idx] = price
    return city_to_prices

# Split prices into X, y for linear regression
# by using previous three months' prices to 
# predict next month's price direction and spike
def split_data(city_to_prices):
    X = []
    y_dir = []
    y_spike = []

    total = 0
    found = 0
    decrease = 0
    no_change = 0
    increase = 0

    for city in city_to_prices:
        prices = city_to_prices[city]
        for i in range(WINDOW, len(prices)):
            historical_prices = np.array(prices[i - 3: i])
            cur_price = prices[i]
            if (-1 not in historical_prices and cur_price != -1):
                X.append(historical_prices)
                price_deviation = (cur_price - prices[i - 1]) / prices[i - 1]
                price_spike = int(abs(price_deviation) >= 0.1)
                #print cur_price, prices[i - 1], price_deviation
                if (price_deviation <= 0.05 and price_deviation >= -0.05):
                    price_direction = 1
                    no_change += 1
                elif (price_deviation <= -0.05):
                    price_direction = 0
                    decrease += 1
                else:
                    price_direction = 2
                    increase += 1
                y_dir.append(price_direction)
                y_spike.append(price_spike)
                found += 1
            total += 1

    print 'total: %d, found: %d' % (total, found)
    print 'decrease: %d, no change: %d, increase: %d' % (decrease, no_change, increase)
    return np.array(X), np.array(y_dir), np.array(y_spike)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title + '.png')

def get_conf_matrix(pred, labels, classes, title):
    plt.figure()
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=np.float32)
    for i in range(labels.shape[0]):
        conf_matrix[labels[i]][pred[i]] += 1
    plot_confusion_matrix(conf_matrix,
                          classes=classes,
                          normalize=True,
                          title=title)


def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

def gradient_boosting_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4)
    clf.fit(features, target)
    return clf

# Output training and validation accuracies using 
# Ridge Classifier
def train_model(X, y, predictSpike):
    X_train, y_train = X[:len(X)/2], y[:len(y)/2]
    X_val, y_val = X[len(X)/2:], y[len(y)/2:]

    if model == "Random Forest":
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_val)
        mean_train_acc = accuracy_score(y_train, rf.predict(X_train))
        mean_val_acc = accuracy_score(y_val, predictions)
        if predictSpike:
            get_conf_matrix(rf.predict(X_train), y_train, 
                    ['no spike', 'spike'], 'Past Prices Random Forest Price Spike Train')
            get_conf_matrix(rf.predict(X_val), y_val, 
                    ['no spike', 'spike'], 'Past Prices Random Forest Price Spike Validation')
        else:
            get_conf_matrix(rf.predict(X_train), y_train, 
                    ['decrease', 'no change', 'increase'], 'Past Prices Random Forest Price Direction Train')
            get_conf_matrix(rf.predict(X_val), y_val, 
                    ['decrease', 'no change', 'increase'], 'Past Prices Random Forest Price Direction Validation')

    elif model == "Ridge Classifier":
        clf = RC()
        clf.fit(X_train, y_train)
        mean_train_acc = clf.score(X_train, y_train)
        mean_val_acc = clf.score(X_val, y_val)
    elif model == "Gradient Boosting":
        trained_model = gradient_boosting_classifier(X_train, y_train)
        predictions = trained_model.predict(X_val)
        mean_train_acc = accuracy_score(y_train, trained_model.predict(X_train))
        mean_val_acc = accuracy_score(y_val, predictions)
    else:
        print "Model not found"
    print 'train acc: %f, val acc: %f' % (mean_train_acc, mean_val_acc)

def main():
    city_to_prices = get_prices()
    X, y_dir, y_spike = split_data(city_to_prices)
    print 'Predicting price direction...'
    train_model(X, y_dir, False)
    print 'Predicting price spike...'
    train_model(X, y_spike, True)
    

if __name__ == '__main__':
    main()


            
