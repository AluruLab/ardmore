import numpy, os, errno, sys
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

try:
    os.mkdir("genenet")
except OSError as e:
    if e.errno == errno.EEXIST:
        pass

true_complete = numpy.genfromtxt('../data/yeast/gnw2000_truenet', delimiter=' ', skip_header=1, usecols=range(1, 2001))

directory = '../runs/genenet/output'
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
auroc = []
average_precision = []
for _filename in files:
    filename = os.path.join(directory, 'stats', _filename+'_stats')
    pred = pandas.read_csv(filename, sep=' ', index_col=0)
    pred['node1'] = pred['node1'] - 1
    pred['node2'] = pred['node2'] - 1
    true = true_complete[pred['node1'].tolist(), pred['node2'].tolist()]
    pred = pred['pval'].values
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

    fig.savefig('genenet/'+_filename.split('-')[0]+'_roc.png')

print("AUROC")
for i in range(len(auroc)):
    print(files[i] + " : " + str(auroc[i]))
print("Average_precision")
for i in range(len(average_precision)):
    print(files[i] + " : " + str(average_precision[i]))
