import numpy, os, errno
from sklearn.metrics import roc_curve, roc_auc_score
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re

try:
    os.mkdir("wgcna")
except OSError as e:
    if e.errno == errno.EEXIST:
        pass

true_complete = pandas.read_csv('../data/yeast/gnw2000_truenet', sep=' ', index_col=0)

print('AUROC')
for _filename in os.listdir('../runs/wgcna/output'):
    filename = '../runs/wgcna/output/' + _filename
    pred = pandas.read_csv(filename, sep=' ', index_col=0)
    true = true_complete.loc[list(pred), list(pred)]

    pred = pred[true.columns]
    pred = pred.reindex(true.index).values.flatten()
    true = true.values.flatten()

    fpr, tpr, thresholds = roc_curve(true, pred)
    auroc = roc_auc_score(true, pred)
    
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0,1], [0,1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    fig.savefig('wgcna/'+_filename+'.png')

    print(filename + " : " + str(auroc))
