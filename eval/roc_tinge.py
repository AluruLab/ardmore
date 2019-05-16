import numpy, pandas, os, errno, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve

try:
    os.mkdir("tinge")
except OSError as e:
    if e.errno == errno.EEXIST:
        pass

true_complete = numpy.genfromtxt('../data/yeast/gnw2000_truenet', delimiter=' ', skip_header=1, usecols=range(1, 2001))
genes = {}
with open('../data/yeast/gnw2000_truenet') as f:
    g = f.readline().rstrip().split(' ')
    for n, e in zip(range(len(g)), g):
        e = e.replace('"', '')
        genes[e] = n

files = os.listdir('../runs/tinge/output')
auroc = []
average_precision = []
for _filename in files:
    filename = '../runs/tinge/output/' + _filename
    row = []
    col = []
    pred = []
    with open(filename) as f:
        num=0
        for line in f:
            if(num>=30):
                terms = line.rstrip().split('\t')
                row.extend([genes[terms[0]] for i in range(int((len(terms)-1)/2))])
                col.extend([genes[terms[i]] for i in range(1, len(terms)-1, 2)])
                pred.extend([float(terms[i]) for i in range(2, len(terms), 2)])
            else:
                num+=1

    true = true_complete[row, col] 
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

    fig.savefig('tinge/'+_filename+'_roc.png')

print("AUROC")
for i in range(len(auroc)):
    print(files[i] + " : " + str(auroc[i]))
print("Average_precision")
for i in range(len(average_precision)):
    print(files[i] + " : " + str(average_precision[i]))
