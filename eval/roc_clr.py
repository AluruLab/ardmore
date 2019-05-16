import numpy, pandas, os, errno, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve

try:
    os.mkdir("clr")
except OSError as e:
    if e.errno == errno.EEXIST:
        pass

true_complete = pandas.read_csv('../data/yeast/gnw2000_truenet', sep=' ', index_col=0)

files = os.listdir('../runs/clr/output')
auroc = []
average_precision = []
for _filename in files:
    filename = '../runs/clr/output/' + _filename
    ngenes = _filename.split('.', 1)[0]
    nsamples = _filename.split('.', 1)[1].split('-',1)[0]
    datafile = '../data/yeast/gnwn' + str(nsamples) + 'X' + str(ngenes)
    genes = []
    with open(datafile) as f:
        line = f.readline().replace('"', '')
        genes = line.split()
    
    pred = pandas.read_csv(filename, sep=',', header=None, names=genes)
    pred.rename(dict(zip(range(len(genes)), genes)), axis='index')
    true = true_complete.loc[genes, genes]
    pred = pred.values.flatten()
    true = true.values.flatten()
    
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

    fig.savefig('clr/'+_filename.split('-')[0]+'_roc.png')

print("AUROC")
for i in range(len(auroc)):
    print(files[i] + " : " + str(auroc[i]))
print("Average_precision")
for i in range(len(average_precision)):
    print(files[i] + " : " + str(average_precision[i]))
