import numpy, pandas, os, errno, matplotlib, sys
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve

try:
    os.mkdir("aracne")
except OSError as e:
    if e.errno == errno.EEXIST:
        pass

true_complete = pandas.read_csv('../data/yeast/gnw2000_truenet', sep=' ', index_col=0)

dirs = os.listdir('../runs/aracne/output')
auroc = []
average_precision = []
for _dir in dirs:
    filename = '../runs/aracne/output/' + _dir + '/bootstrapNetwork_ul3atth75o35ngtur8ibskqq7s.txt'
    pred = pandas.read_csv(filename, sep = '\t')
    true = numpy.zeros((len(pred.index)))
    for i, row in zip(range(len(pred.index)), pred.itertuples()):
        true[i] = true_complete.loc[row.Regulator, row.Target]

    pred = pred['MI'].values

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

    fig.savefig('aracne/'+_dir.split('-')[0]+'_roc.png')

print("AUROC")
for i in range(len(auroc)):
    print(dirs[i] + " : " + str(auroc[i]))
print("Average_precision")
for i in range(len(average_precision)):
    print(dirs[i] + " : " + str(average_precision[i]))
