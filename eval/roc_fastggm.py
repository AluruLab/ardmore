import numpy, pandas, os, errno, matplotlib, argparse
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve

PRINT_ORDER_SXG =["2000X250", "2000X500", "2000X750", "2000X1000", "2000X1250", "2000X1500", "2000X1750", "2000X2000", "1750X2000", "1500X2000", "1250X2000", "1000X2000", "750X2000", "500X2000", "250X2000"]

def main(input_base_dir = "..", output_base_dir = "."):
    input_base_dir = input_base_dir + "/"
    output_base_dir = output_base_dir + "/"
    try:
        os.mkdir(output_base_dir + "fastggm")
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass

    true_complete = pandas.read_csv(input_base_dir + 'data/yeast/gnw2000_truenet', sep=' ', index_col=0)

    files = os.listdir(input_base_dir + 'runs/fastggm/output')
    auroc = {}
    average_precision = {}
    for _filename in files:
        filename = input_base_dir + 'runs/fastggm/output/' + _filename
        ngenes = int(_filename.split('.', 1)[0])
        nsamples = int(_filename.split('.', 1)[1].split('-',1)[0])
        datafile = input_base_dir + 'runs/fastggm/data/format.gnwn' + str(nsamples) + 'X' + str(ngenes)
        genes = []
        with open(datafile) as f:
            line = f.readline().replace('"', '')
            genes = line.split()

        pred = pandas.read_csv(filename, sep=' ', index_col=0, usecols=range(2*ngenes + 1, 3*ngenes + 1))
        true = true_complete.loc[genes, genes]
        pred = pred.values.flatten()
        true = true.values.flatten()
        print(max(pred), min(pred))
        sxgn = str(nsamples) + "X" + str(ngenes)
        pred = 1 - pred

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

        fig.savefig(output_base_dir + 'fastggm/'+_filename.split('-')[0]+'_roc.png')

    print("\t".join(["SXG"] + [x for x in PRINT_ORDER_SXG]))
    print("\t".join(["AUROC"] + [str(auroc[x]) if x in auroc else "NA" for x in PRINT_ORDER_SXG]))
    print("\t".join(["AUPR"] + [str(average_precision[x]) if x in average_precision else "NA" for x in PRINT_ORDER_SXG]))

if __name__ == "__main__":
    PROG_DESC = """
    Computes the ROC for fastggm networks.
    """
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    PARSER.add_argument("-i", "--input_base_dir", type=str, default="..",
                        help="""Base dir of fastggm runs dir""")
    PARSER.add_argument("-o", "--output_base_dir", type=str, default=".",
                        help="""Base dir of output dir""")

    ARGS = PARSER.parse_args()
    main(ARGS.input_base_dir, ARGS.output_base_dir)
