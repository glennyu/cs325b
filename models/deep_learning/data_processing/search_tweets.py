from collections import defaultdict
import csv
import dateutil.parser

PATH = '/Users/glennyu/Downloads/'
MONTH_IDX = ''
YEAR = ''
NUM_MONTHS = 35

2014-01
tweets = defaultdict(list)
with open(PATH + "Delhi_tweets.csv") as csvfile:
	reader = csv.DictReader(csvfile)
        for row in reader:
        	time = row['postedTime']
        	month = int(time[5:7])
        	year = int(time[:4])
        	idx = (year - 2014)*12 + (month - 1)
        	tweets[idx].append(row['tweet'].lower())

print '- Done with reading tweets'

for month in range(NUM_MONTHS):
	MONTH_IDX = str(month)
	YEAR = str(int(month / 12) + 2014) 

	print 'month: ' + MONTH_IDX
	print 'year: ' + YEAR

	with open('data/batches_train/' + MONTH_IDX + '_batches_delhi.txt') as f:
		maxCnt = 0
		batchNum = 0
		cur = 0
		for line in f:
			cnt = 0
			idxes = [int(num) for num in line.strip().split('\t')]
			for idx in idxes:
				if ('onion' in tweets[month][idx]):
					print tweets[month][idx]
					cnt += 1
			if (cnt > maxCnt):
				maxCnt = cnt
				batchNum = cur
			cur += 1
			
		print 'count %d' % maxCnt
		print 'batch num %d' % batchNum
