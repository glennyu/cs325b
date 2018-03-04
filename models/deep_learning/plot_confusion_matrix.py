print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
#class_names = ['decrease', 'no change', 'increase']
#class_names = ['no spike', 'spike']
class_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
#train_cnf_matrix = np.array([[113, 1, 1, 0, 1, 16, 1],
#                             [3, 83, 4, 3, 12, 21, 21],
#                             [3, 2, 68, 3, 17, 32, 22],
#                             [3, 0, 1, 63, 24, 23, 30],
#                             [1, 0, 1, 0, 118, 21, 7],
#                             [5, 1, 0, 0, 2, 124, 11],
#                             [0, 1, 0, 0, 2, 13, 132]])
train_cnf_matrix = np.array([
[
1841
,
69
,
59
,
19
,
110
,
3635
,
335
]
,
[
158
,
841
,
146
,
57
,
233
,
4325
,
340
]
,
[
70
,
90
,
807
,
73
,
293
,
4541
,
298
]
,
[
89
,
30
,
69
,
729
,
356
,
4498
,
403
]
,
[
103
,
28
,
25
,
41
,
1570
,
3911
,
386
]
,
[
171
,
23
,
34
,
13
,
170
,
5105
,
557
]
,
[
80
,
15
,
31
,
6
,
41
,
2793
,
3125
]
])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=class_names, normalize=True,
                      title='Day of Week Prediction Matrix')
plt.savefig('day_of_week_all_conf_matrix.png')

#plt.figure()
#plot_confusion_matrix(eval_cnf_matrix, classes=class_names, normalize=True,
#                      title='Price Direction Validation Confusion Matrix')
#plt.savefig('price_dir_val_conf_matrix.png')
