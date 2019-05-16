import numpy, os, errno
import glob
import re
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
import argparse

TRUE_NETWORK_FILE = '../data/yeast/gnw2000_truenet'
TRUE_NETWORK_GENES = '../data/yeast/genes'

def load_mat_network(mat_file: str, wt_attr_name: str = 'wt',
                     delimiter: str = " ") -> pd.DataFrame:
    """
    Load network with adjacency matrix;
    Adjacency matrix lists the target wts for each source node:
    source_node1 source_node2 source_node_3 ...
    target_node1 weight11 weight12 weight13 ...
    target_node2 weight21 weight22 weight23 ...
    target_node3 weight31 weight32 weight33 ...
    ...
    ...

    Parameters
    ----------
    adj_file : Path to the input .adj file

    wt_attr_name : weight column name in the data frame returned

    delimiter : seperators between the fields

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name
    """
    mat_df = pd.read_csv(mat_file, sep=delimiter, index_col=0)
    mat_cnames = mat_df.columns
    mat_size = mat_df.shape[0]
    net_df = pd.DataFrame({
        'source': np.repeat(mat_cnames, mat_size),
        'target': np.tile(mat_cnames, mat_size),
        wt_attr_name: mat_df.values.flatten()})
    return net_df[net_df.source < net_df.target]

def load_tsv_network(eda_file: str, wt_attr_name: str = 'wt') -> pd.DataFrame:
    """
    Load network from edge attribute file file (.eda).
    Eda file lists the edges in the following format
    The first line has the string "Weight". The edges are listed with the following
    format:
    source (itype) target = weight

    where source and target are node ids, itype is the interaction type and weight is
    the edge weight.

    Example:
    Weight
    244901_at (pp) 244902_at = 0.192777
    244901_at (pp) 244926_s_at = 0.0817807


    Parameters
    ----------
    eda_file : Path to the input .eda file

    wt_attr_name : weight column name in the data frame returned

    delimiter : seperators between the fields

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name

    """
    tmp_df = pd.read_csv(eda_file, sep='\t', header=None, 
                         names=['source', 'target', wt_attr_name])
    tmp_rcds = [(x, y, z) if x < y else (y, x, z) for x, y, z in tmp_df.to_dict('split')['data']]
    xdf = pd.DataFrame(tmp_rcds, columns=['source', 'target', wt_attr_name])
    return xdf.sort_values(by=wt_attr_name, ascending=False)


def read_genes(genes_file):
    with open(genes_file) as f:
        rgenes = f.readlines()
        return [x.strip() for x in rgenes]


def make_dir():
   try:
       os.mkdir("grnboost")
   except OSError as e:
       if e.errno == errno.EEXIST:
           pass

def gen_auroc(genes_file, true_net_file, network_files):
    rgenes = read_genes(genes_file)
    tnet = load_mat_network(true_net_file, wt_attr_name="twt")
    print("Loaded ", true_net_file, "with", tnet.shape[0],
          "edges and", len(rgenes), "genes.")
    #for filename in glob.glob("../runs/grnboost/output/*.tsv"): # os.listdir('../runs/grnboost/output'):
    print("Input Files : ", " ".join(network_files))
    print("GENES", "SAMPLES", "AUROC")
    for filename in network_files: 
        bname = os.path.basename(filename)
        mx = re.search('([0-9]+)X([0-9]+)-', bname)
        ngenes = int(mx.group(1))
        nsamples = int(mx.group(2))
        sgenes = rgenes[:ngenes]
        sgenesst = set(sgenes)
        stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                         (tnet.target.isin(sgenesst)), : ]
        rnet = load_tsv_network(filename, wt_attr_name="pwt") 
        rnet = rnet.drop_duplicates(subset=['source','target'])
        jnnet = pd.merge(stnet, rnet, on=['source','target'], how='left')
        #print(ngenes, stnet.shape, rnet.shape, jnnet.shape, 
        #      sum(jnnet.twt.isna()), sum(jnnet.pwt.isna()))
        jnnet.fillna(0, inplace=True)
        true = jnnet.twt 
        pred = jnnet.pwt 
        fpr, tpr, thresholds = roc_curve(true, pred)
        auroc = roc_auc_score(true, pred)
        
        fig = plt.figure()
        plt.plot(fpr, tpr, color='darkorange')
        plt.plot([0,1], [0,1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        fig.savefig('grnboost/'+bname+'.png')
    
        print(ngenes, nsamples, str(auroc))

def main():
    PROG_DESC = """
    Computes the ROC for input networks.
    """
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    PARSER.add_argument("network_files", nargs="+",
                        help="""network build from a reverse engineering methods """)
    ARGS = PARSER.parse_args()
    make_dir()
    #true_net = load_true_network(TRUE_NETWORK_FILE)
    gen_auroc(TRUE_NETWORK_GENES, TRUE_NETWORK_FILE, ARGS.network_files)

if __name__ == "__main__": 
    main()

