import csv 

MONTH_COL = 14
YEAR_COL = 15
START_MONTH = 1
START_YEAR = 2014
END_MONTH = 11
END_YEAR = 2016

# Input: A = array of true values, F = array of corresponding forecasted values
# Returns MAPE metric
def mape(A, F):
	total = 0
	for i in range(len(A)):
		total += abs(1.0 * (A[i] - F[i]) / A[i])
	return 1.0 * total / len(A)

# Returns all prices in India from 01/2014 to 11/2016
def get_prices():
    with open(DIR + 'India_Food_Prices.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)
        india_food_prices = []
        for row in reader:
            month, year = int(row[MONTH_COL]), int(row[YEAR_COL])
        	if ((year >= START_YEAR and year < END_YEAR) 
                or (year == END_YEAR and month <= END_MONTH)):
                india_food_prices += [row]
        return india_food_prices