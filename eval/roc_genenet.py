import numpy, os, errno, sys, argparse, re
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve
import pandas
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

PRINT_ORDER_SXG =["2000X250", "2000X500", "2000X750", "2000X1000", "2000X1250", "2000X1500", "2000X1750", "2000X2000", "1750X2000", "1500X2000", "1250X2000", "1000X2000", "750X2000", "500X2000", "250X2000"]

def main(input_base_dir = "..", output_base_dir = "."):
    input_base_dir = input_base_dir + "/"
    output_base_dir = output_base_dir + "/"
    try:
        os.mkdir(output_base_dir+"genenet")
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass

    true_complete = numpy.genfromtxt(input_base_dir+'data/yeast/gnw2000_truenet',
                                     delimiter=' ',
                                     skip_header=1, usecols=range(1, 2001))

    directory = input_base_dir + 'runs/genenet/output'
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    auroc = {}
    average_precision = {}
    for _filename in files:
        filename = os.path.join(directory, 'stats', _filename+'_stats')
        pred = pandas.read_csv(filename, sep=' ', index_col=0)
        pred['node1'] = pred['node1'] - 1
        pred['node2'] = pred['node2'] - 1
        true = true_complete[pred['node1'].tolist(), pred['node2'].tolist()]
        pred = pred['pval'].values
        pred = 1 - pred
        #print(max(pred), min(pred))
        bname = os.path.basename(filename)
        mx = re.search('([0-9]+)\.([0-9]+)-', bname)
        ngenes = int(mx.group(1))
        nsamples = int(mx.group(2))
        #print(ngenes, nsamples)
        sxgn = str(nsamples) + "X" + str(ngenes)

        fpr, tpr, thresholds = roc_curve(true, pred)
        precision, recall, _ = precision_recall_curve(true, pred)
        auroc[sxgn] = roc_auc_score(true, pred)
        average_precision[sxgn] = average_precision_score(true, pred)

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

        fig.savefig(output_base_dir+'genenet/'+_filename.split('-')[0]+'_roc.png')
    print("\t".join(["SXG"] + [x for x in PRINT_ORDER_SXG]))
    print("\t".join(["AUROC"] + [str(auroc[x]) if x in auroc else "NA" for x in PRINT_ORDER_SXG]))
    print("\t".join(["AUPR"] + [str(average_precision[x]) if x in average_precision else "NA" for x in PRINT_ORDER_SXG]))


if __name__ == "__main__":
    PROG_DESC = """
    Computes the ROC for genenet networks.
    """
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    PARSER.add_argument("-i", "--input_base_dir", type=str, default="..",
                        help="""Base dir of fastggm runs dir""")
    PARSER.add_argument("-o", "--output_base_dir", type=str, default=".",
                        help="""Base dir of output dir""")

    ARGS = PARSER.parse_args()
    main(ARGS.input_base_dir, ARGS.output_base_dir)
