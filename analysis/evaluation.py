import itertools
import numpy as np
import glob
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scikitplot.helpers import validate_labels

import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

list_test = []
list_pred = []
P_ext = []
P_true = []

loadresult = sorted(glob.glob('result*'))
print loadresult
for name in loadresult:
  print name		
  pred, test, IDs = np.genfromtxt(fname=name, delimiter=',', dtype=np.str, unpack=True, usecols=(0,1,4))
  perext, pertrue = np.genfromtxt(fname=name, delimiter=',', unpack=True, usecols=(2,3))
  

  labtest = [y for y in test]
  labpred = [x for x in pred]
  T_ext = [t for t in perext]
  T_true = [m for m in pertrue]

  list_test.append(labtest)
  list_pred.append(labpred)
  P_ext.append(T_ext)
  P_true.append(T_true)

y_true = list(itertools.chain.from_iterable(list_test))
y_pred = list(itertools.chain.from_iterable(list_pred))

#y_true = [f.translate(' ', '_') for f in y_tru]
#y_pred = [f.translate(' ', '_') for f in y_pre]

y_true = [s.replace('EB_EELL', 'ELL') for s in y_true]
y_true = [s.replace('EB_ENC', 'EB NC') for s in y_true]
y_true = [s.replace('_', ' ') for s in y_true]

y_pred = [s.replace('EB_EELL', 'ELL') for s in y_pred]
y_pred = [s.replace('EB_ENC', 'EB NC') for s in y_pred]
y_pred = [s.replace('_', ' ') for s in y_pred]

#print y_true

initru = pd.Series(y_true)
inipred = pd.Series(y_pred)
#print list_test,list_pred
print len(y_true), len(y_pred)
cnf_matrix = pd.crosstab(initru, inipred, rownames=['True'], colnames=['Predicted'], margins=True)

print cnf_matrix
#print percent_cnf_matrix


def plot_confusion_matrix(y_true, y_pred, labels=None, true_labels=None,
                          pred_labels=None, title=None, normalize=False,
                          hide_zeros=False, x_tick_rotation=0, ax=None,
                          figsize=None, cmap='Blues', title_fontsize="large",
                          text_fontsize="medium"):
    """Generates confusion matrix plot from predictions and true labels
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_pred (array-like, shape (n_samples)):
            Estimated targets as returned by a classifier.
        labels (array-like, shape (n_classes), optional): List of labels to
            index the matrix. This may be used to reorder or select a subset
            of labels. If none is given, those that appear at least once in
            ``y_true`` or ``y_pred`` are used in sorted order. (new in v0.2.5)
        true_labels (array-like, optional): The true labels to display.
            If none is given, then all of the labels are used.
        pred_labels (array-like, optional): The predicted labels to display.
            If none is given, then all of the labels are used.
        title (string, optional): Title of the generated plot. Defaults to
            "Confusion Matrix" if `normalize` is True. Else, defaults to
            "Normalized Confusion Matrix.
        normalize (bool, optional): If True, normalizes the confusion matrix
            before plotting. Defaults to False.
        hide_zeros (bool, optional): If True, does not plot cells containing a
            value of zero. Defaults to False.
        x_tick_rotation (int, optional): Rotates x-axis tick labels by the
            specified angle. This is useful in cases where there are numerous
            categories and the labels overlap each other.
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> rf = RandomForestClassifier()
        >>> rf = rf.fit(X_train, y_train)
        >>> y_pred = rf.predict(X_test)
        >>> skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_confusion_matrix.png
           :align: center
           :alt: Confusion matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    
    #Move the ticks to top of plot
    ax.xaxis.tick_top() 
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if labels is None:
        classes = unique_labels(y_true, y_pred)
    else:
        classes = np.asarray(labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0

    if true_labels is None:
        true_classes = classes
    else:
        validate_labels(classes, true_labels, "true_labels")

        true_label_indexes = np.in1d(classes, true_labels)

        true_classes = classes[true_label_indexes]
        cm = cm[true_label_indexes]

    if pred_labels is None:
        pred_classes = classes
    else:
        validate_labels(classes, pred_labels, "pred_labels")

        pred_label_indexes = np.in1d(classes, pred_labels)

        pred_classes = classes[pred_label_indexes]
        cm = cm[:, pred_label_indexes]

    if title:
        ax.set_title(title, fontsize=title_fontsize)
    elif normalize:
        ax.set_title('Normalized Confusion Matrix', fontsize=title_fontsize)
    else:
        ax.set_title('Confusion Matrix', fontsize=title_fontsize)

    image = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    plt.colorbar(mappable=image)
    x_tick_marks = np.arange(len(pred_classes))
    y_tick_marks = np.arange(len(true_classes))
    ax.set_xticks(x_tick_marks)
    ax.set_xticklabels(pred_classes, fontsize='medium',
                       rotation=x_tick_rotation)
    ax.set_yticks(y_tick_marks)
    ax.set_yticklabels(true_classes, fontsize='medium')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if not (hide_zeros and cm[i, j] == 0):
            ax.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize='medium',
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('OGLE class', fontsize=text_fontsize)
    ax.set_xlabel('UPSILoN class', fontsize=text_fontsize)
    ax.grid('off')

    return ax
LAB = ['CEPH 1O','CEPH F','CEPH Other','DSCT', 'EB EC','EB ED','EB ESD','EB NC', 'ELL', 'LPV Mira AGB C','LPV Mira AGB O','LPV OSARG AGB','LPV OSARG RGB','LPV SRV AGB C','LPV SRV AGB O','NonVar','RRL ab','RRL c', 'RRL d', 'RRL e', 'T2CEPH','aRRL d'] 
UPS = ['DSCT','RRL ab','RRL c', 'RRL d', 'RRL e','CEPH F','CEPH 1O','CEPH Other','EB EC','EB ED','EB ESD', 'LPV Mira AGB C','LPV Mira AGB O','LPV OSARG AGB','LPV OSARG RGB','LPV SRV AGB C','LPV SRV AGB O','T2CEPH','NonVar']
OGL = ['DSCT','RRL ab','RRL c', 'RRL d', 'aRRL d', 'CEPH F','CEPH 1O','CEPH Other','EB EC','EB NC', 'ELL', 'LPV Mira AGB C','LPV Mira AGB O','LPV OSARG AGB','LPV OSARG RGB','LPV SRV AGB C','LPV SRV AGB O','T2CEPH']
#

plot_confusion_matrix(y_true, y_pred, title= ' ', normalize=False, labels=LAB, text_fontsize = 18, true_labels=OGL, pred_labels=UPS, cmap='RdPu', x_tick_rotation=90, hide_zeros=True)
#plt.savefig("Confusion Matrix ALL.png", dpi=200)
plt.show()

#cnf_matrix.to_csv('CF - NEW.csv', index=True, encoding='utf-8')
