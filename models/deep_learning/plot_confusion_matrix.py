print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# import some data to play with
class_names = ['decrease', 'no change', 'increase']

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
train_cnf_matrix = np.array([[13131, 59, 198], [49, 2287, 90], [203, 136, 13847]]).T
eval_cnf_matrix = np.array([[3379, 148, 2886], [212, 167, 182], [1408, 110, 1508]]).T
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(train_cnf_matrix, classes=class_names, normalize=True,
                      title='Weekly Model Train Confusion Matrix')
plt.savefig('weekly_model_train_conf_matrix.png')

plt.figure()
plot_confusion_matrix(eval_cnf_matrix, classes=class_names, normalize=True,
                      title='Weekly Model Validation Confusion Matrix')
plt.savefig('weekly_model_val_conf_matrix.png')
