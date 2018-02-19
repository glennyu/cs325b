import random
import os
import shutil

PATH = '../data/'
TRAIN_PATH = PATH + 'batches_train/'
VAL_PATH = PATH + 'batches_val/'
TEST_PATH = PATH + 'batches_test/'
TOTAL = 102584

random.seed(10)

train_set = set()
val_set = set()
test_set = set()

with open(PATH + 'batch_counts.txt', 'r') as f:
    labels = []
    cnts = []
    for row in f:
        line = row.split(',')
        city = line[0]
        month = line[1]
        cnt = line[2]
        
        labels.append(city + '_' + month + '_batch.txt')
        cnts.append(cnt)

    tmp = list(zip(labels, cnts))
    random.shuffle(tmp)
    
    labels, cnts = zip(*tmp)
    cnt_sum = 0
    folder = 'Train'
    start = 0
    for i in range(len(cnts)):
        if (cnt_sum > 0.7 * TOTAL and folder == 'Train'):
            with open(PATH + 'train_labels.txt', 'w') as output:
                output.write(str(cnt_sum) + '\n')
                for j in range(start, i):
                    output.write(labels[j] + '\n')
                    train_set.add(labels[j])
            folder = 'Val'
            cnt_sum = 0
            start = i
        elif (cnt_sum > 0.2 * TOTAL and folder == 'Val'):
            with open(PATH + 'val_labels.txt', 'w') as output:
                output.write(str(cnt_sum) + '\n')
                for j in range(start, i):
                    output.write(labels[j] + '\n')
                    val_set.add(labels[j])
            folder = 'Test'
            cnt_sum = 0
            start = i
        cnt_sum += int(cnts[i])

    with open(PATH + 'test_labels.txt', 'w') as output: 
        output.write(str(cnt_sum) + '\n')
        for j in range(start, len(cnts)):
            output.write(labels[j] + '\n')
            test_set.add(labels[j])

batch_folder = PATH + 'batches/'
for filename in os.listdir(batch_folder):
    dest = TRAIN_PATH
    if filename in val_set:
        dest = VAL_PATH
    elif filename in test_set:
        dest = TEST_PATH
    shutil.move(batch_folder + filename, dest + filename)


