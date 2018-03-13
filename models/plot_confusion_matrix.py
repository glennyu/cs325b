print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
#class_names = ['decrease', 'no change', 'increase']
class_names = ['no spike', 'spike']
#class_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

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
                 horizontalalignment="center", fontsize=16,
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
#train_cnf_matrix = np.array([[7617, 538, 817], [236, 1615, 302], [710, 490, 4752]]).T
#eval_cnf_matrix = np.array([[1967, 1385, 1381], [0, 0, 0], [99, 74, 95]]).T 
train_cnf_matrix = np.array([[8463, 966], [771, 6877]]).T
eval_cnf_matrix = np.array([[1488, 1568], [889, 1056]]).T
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=class_names, normalize=True,
                      title='Single Tweet Model Price Spike Train')
plt.savefig('single_price_spike_train_conf_matrix.png')

plt.figure()
plot_confusion_matrix(eval_cnf_matrix, classes=class_names, normalize=True,
                      title='Single Tweet Model Price Spike Validation')
plt.savefig('single_price_spike_val_conf_matrix.png')
