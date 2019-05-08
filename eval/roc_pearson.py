import numpy, os, errno
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re

try:
    os.mkdir("pearson")
except OSError as e:
    if e.errno == errno.EEXIST:
        pass

true_complete = pandas.read_csv('../data/yeast/gnw2000_truenet', sep=' ', index_col=0)

files = os.listdir('../runs/pearson/output')
auroc = []
average_precision = []
for _filename in files:
    filename = '../runs/pearson/output/' + _filename
    pred = pandas.read_csv(filename, sep=' ', index_col=0)
    true = true_complete.loc[list(pred), list(pred)]

    pred = pred[true.columns]
    pred = pred.reindex(true.index).values.flatten()
    true = true.values.flatten()
    pred = numpy.absolute(pred)
    fpr, tpr, thresholds = roc_curve(true, pred)
    precision, recall, _ = precision_recall_curve(true, pred)
    auroc.append(roc_auc_score(true, pred))
    average_precision.append(average_precision_score(true, pred))
    
    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0,1], [0,1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange')
    plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    fig.savefig('pearson/'+_filename.split('-')[0]+'_roc.png')

print("AUROC")
for i in range(len(auroc)):
    print(files[i] + " : " + str(auroc[i]))
print("Average_precision")
for i in range(len(average_precision)):
    print(files[i] + " : " + str(average_precision[i]))
