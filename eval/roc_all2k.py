import sys
import sklearn.metrics as skm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = "Myriad Pro"


DATA_DIR = '../data/revnet/'
TRUE_NET = '../data/scerevisiae/gnw2000_truenet'
PR_OUTPUT_FILE = 'Simulations-PR-2Kx2K.'
ROC_OUTPUT_FILE = 'Simulations-ROC-2Kx2K.'
ROCPR_OUTPUT_FILE = 'Simulations-ROC-PR-2Kx2K.'
FIG_SIZE = (8, 6)
SUB_FIG_SIZE = (15, 6)
REVNET_FILES = {
    'aracne'  :
        'aracne/2000.2000-2019.05.16-11.19.33/bootstrapNetwork_ul3atth75o35ngtur8ibskqq7s.txt',
    'clr'     :
        'clr/2000.2000-2019.05.05-13.42.52',
    'fastggm' :
        'fastggm/2000.2000-2019.05.03-18.41.16',
    'genenet' :
        'genenet/stats/2000.2000-2019.05.15-15.53.26_stats',
    'genie3'  :
        'genie3/2000.2000-2019.05.06-17.02.21',
    'grnboost' :
        'grnboost/2000X2000-2019.05.16-15.05.30.result.tsv',
    'inferlator' :
        'inferlator/2000.2000-2019.05.16-16.50.04/summary_frac_tp_0_perm_1--frac_fp_0_perm_1_1.tsv',
    'mrnet' :
        'mrnet/2000.2000-2019.05.05-17.56.28',
    'tigress' :
        'tigress/2000.2000-2019.05.06-15.59.22',
    'tinge' :
        'tinge/2000.2000.weight.eda',
    'wgcna' :
        'wgcna/2000.2000-2019.05.03-18.09.55',
    'pcc' :
        'pearson/2000.2000-2019.05.03-17.33.01'
}

CN_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
TAB_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
DIST_COLORS = [
    '#800000', # Maroon
    '#9A6324', # Brown
    '#f58231', # Orange
    '#3cb44b', # Green
    '#911eb4', # Purple
    '#000075', # Navy
    '#808000', # Olive
    '#4363d8', # Blue
    '#e6194B', # Red
    '#ffe119', # Yellow
    '#a9a9a9', # Grey
]
PLOT_COLORS = DIST_COLORS

def remove_dupe_rows(in_df: pd.DataFrame, wt_attr_name: str) -> pd.DataFrame:
    """
    Given a data frame of three columns : "source", "target" and wt_attr_name
    such that only the row with max weight is returned

    Parameters
    ----------
    in_df : Input data frame with columns 'source', 'target', wt_attr_name

    wt_attr_name : weight column name in the data frame returned

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name such that
    only the row with max weight is returned
    """
    in_df = in_df.sort_values(by=[wt_attr_name], ascending=False)
    return in_df.drop_duplicates(subset=['source', 'target'], keep='first')


def order_network_rows(in_df: pd.DataFrame, wt_attr_name: str) -> pd.DataFrame:
    """
    Given a data frame of three columns : "source", "target" and wt_attr_name
    returns rows such that source < target

    Parameters
    ----------
    in_df : Input data frame with columns 'source', 'target', wt_attr_name

    wt_attr_name : weight column name in the data frame returned

    Returns
    -------
    pandas DataFrame with three columns: 'source', 'target', wt_attr_name such that
    source entry < target entry
    """
    if (in_df.source < in_df.target).all():
        return remove_dupe_rows(in_df, wt_attr_name)
    tmp_rcds = [(x, y, z) if x < y else (y, x, z) for x, y, z in in_df.to_dict('split')['data']]
    tmp_df = pd.DataFrame(tmp_rcds, columns=['source', 'target', wt_attr_name])
    return remove_dupe_rows(tmp_df, wt_attr_name)

def load_eda_network(eda_file: str, wt_attr_name: str = 'wt',
                     delimiter: str = r'\s+') -> pd.DataFrame:
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
    tmp_df = pd.read_csv(eda_file, sep=delimiter, usecols=[0, 2, 4], skiprows=[0],
                         names=['source', 'target', wt_attr_name])
    return order_network_rows(tmp_df, wt_attr_name)


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
    with open(genes_file) as ifx:
        rgenes = ifx.readlines()
        return [x.strip() for x in rgenes]



def aracne_old_roc(true_complete_file, aracne_file):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = DATA_DIR + aracne_file
    pred = pd.read_csv(filename, sep='\t')
    true = np.zeros((len(pred.index)))
    for i, row in zip(range(len(pred.index)), pred.itertuples()):
        true[i] = true_complete.loc[row.Regulator, row.Target]
    pred = pred['MI'].values
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    # auroc.append(roc_auc_score(true, pred))
    # average_precision.append(average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def get_lower_triangle(in_mat):
    ill = np.tril_indices(in_mat.shape[0], -1)
    return in_mat[ill]

def aracne_roc(true_complete_file, arance_file, genes_list):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = DATA_DIR + arance_file
    pred = pd.read_csv(filename, sep='\t')
    min_mi = min(pred['MI'].values)
    max_mi = max(pred['MI'].values)
    #print(min_mi, max_mi)
    true_a = true_complete.loc[genes_list, genes_list]
    pred_a = pd.DataFrame(min_mi/2, columns=true_a.columns, index=true_a.index)
    for row in pred.itertuples():
        r_max = max(pred_a.loc[row.Regulator, row.Target],
                    pred_a.loc[row.Target, row.Regulator])
        r_max = max(r_max, row.MI)
        pred_a.loc[row.Regulator, row.Target] = r_max/max_mi
        pred_a.loc[row.Target, row.Regulator] = r_max/max_mi
    true_a = get_lower_triangle(true_a.to_numpy()) # true_a.values.flatten()
    pred_a = get_lower_triangle(pred_a.to_numpy()) # pred_a.values.flatten()
    #print(max(pred_a), min(pred_a))
    fpr, tpr, _ = skm.roc_curve(true_a, pred_a)
    #print(len(thx), max(thx), min(thx))
    precision, recall, _ = skm.precision_recall_curve(true_a, pred_a)
    print("ARACNE (MI) : ", pred_a.shape, true_a.shape,
          skm.roc_auc_score(true_a, pred_a),
          skm.average_precision_score(true_a, pred_a))
    return (fpr, tpr, precision, recall)

def clr_roc(true_complete_file, clr_fname, genes):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = DATA_DIR + clr_fname
    ngenes = len(genes)
    true = true_complete.loc[genes, genes]
    pred = pd.read_csv(filename, sep=',', header=None, names=genes)
    pred.rename(dict(zip(range(ngenes), genes)), axis='index')
    # pred = pred.values.flatten()
    # true = true.values.flatten()
    true = true.to_numpy()
    pred = pred.to_numpy()
    pred = np.fmax(np.transpose(pred), pred)
    true = get_lower_triangle(true) # true_a.values.flatten()
    pred = get_lower_triangle(pred) # pred_a.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("CLR (MI) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def fastggm_roc(true_complete_file, fastggm_fname, genes):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    ngenes = len(genes)
    filename = DATA_DIR + fastggm_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0,
                       usecols=range(2*ngenes + 1, 3*ngenes + 1))
    true = true_complete.loc[genes, genes]
    # pred = pred.values.flatten()
    # true = true.values.flatten()
    true = get_lower_triangle(true.to_numpy()) # true_a.values.flatten()
    pred = get_lower_triangle(pred.to_numpy()) # pred_a.values.flatten()
    #print(max(pred), min(pred))
    #sxgn = str(nsamples) + "X" + str(ngenes)
    pred = 1 - pred
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("FastGGM (COR) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)


def genenet_roc(true_complete_file, genenet_fname):
    true_complete = np.genfromtxt(true_complete_file,
                                  delimiter=' ',
                                  skip_header=1, usecols=range(1, 2001))
    filename = DATA_DIR + genenet_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    pred['node1'] = pred['node1'] - 1
    pred['node2'] = pred['node2'] - 1
    true = true_complete[pred['node1'].tolist(), pred['node2'].tolist()]
    pred = pred['pval'].values
    pred = 1 - pred
    pred = pred.flatten()
    true = true.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("GeneNet (COR) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)


def genie3_old_roc(true_complete_file, g3_fname, genes_list, full_genes_list):
    true_complete = np.genfromtxt(true_complete_file, delimiter=' ',
                                  skip_header=1, usecols=range(1, 2001))
    genes_dct = {}
    genes_list = set(genes_list)
    for idx, gname in zip(range(len(full_genes_list)), full_genes_list):
        gname = gname.replace('"', '')
        if gname in genes_list:
            genes_dct[gname] = idx
    filename = DATA_DIR + g3_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    true = true_complete[[genes_dct[g] for g in pred['regulatoryGene']],
                         [genes_dct[g] for g in pred['targetGene']]]
    pred = pred['weight'].values
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("GENIE3 (COR) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def genie3_roc(true_complete_file, g3_fname, genes_list):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = DATA_DIR + g3_fname
    genes_list = sorted(genes_list)
    true_a = true_complete.loc[genes_list, genes_list]
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    pred = pred.sort_values(by=['regulatoryGene', 'targetGene'])
    min_pred = min(pred['weight'].values)
    pred_a = pd.DataFrame(min_pred/2, columns=true_a.columns, index=true_a.index).to_numpy()
    # print(pred_a[~np.eye(pred_a.shape[0], dtype=bool)].shape)
    # print(pred['weight'].values.shape)
    pred_a[~np.eye(pred_a.shape[0], dtype=bool)] = pred['weight'].values # pylint: disable=E1136  # pylint/issues/3139
    true_a = true_a.to_numpy()
    pred_a = np.fmax(np.transpose(pred_a), pred_a)
    true_a = get_lower_triangle(true_a) # true_a.values.flatten()
    pred_a = get_lower_triangle(pred_a) # pred_a.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true_a, pred_a)
    precision, recall, _ = skm.precision_recall_curve(true_a, pred_a)
    print("GENIE3 (COR) : ", pred_a.shape, true_a.shape,
          skm.roc_auc_score(true_a, pred_a),
          skm.average_precision_score(true_a, pred_a))
    return (fpr, tpr, precision, recall)


def grnboost_roc(true_complete_file, grn_fname, genes_list):
    filename = DATA_DIR + grn_fname
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    rnet = load_tsv_network(filename, wt_attr_name="pwt")
    min_vx = min(rnet.pwt.values)
    rnet = rnet.drop_duplicates(subset=['source', 'target'])
    jnnet = pd.merge(stnet, rnet, on=['source', 'target'], how='left')
    jnnet.fillna(min_vx/2.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("GRNBoost (LR) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)


def inferlator_old_roc(true_complete_file, infr_fname, genes_list, full_genes_list):
    true_complete = np.genfromtxt(true_complete_file, delimiter=' ',
                                  skip_header=1, usecols=range(1, 2001))
    genes_dct = {}
    genes_list = set(genes_list)
    for idx, gname in zip(range(len(full_genes_list)), full_genes_list):
        gname = gname.replace('"', '')
        if gname in genes_list:
            genes_dct[gname] = idx
    filename = DATA_DIR + infr_fname
    pred = pd.read_csv(filename, sep='\t',
                       usecols=['regulator', 'target', 'var.exp.median'])
    true = true_complete[[genes_dct[g] for g in pred['regulator']],
                         [genes_dct[g] for g in pred['target']]]
    pred = pred['var.exp.median'].values
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("Inferelator (LR) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)


def inferlator_roc(true_complete_file, infr_fname, genes_list):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = DATA_DIR + infr_fname
    pred = pd.read_csv(filename, sep='\t',
                       usecols=['regulator', 'target', 'var.exp.median'])
    true = true_complete.loc[genes_list, genes_list]
    min_pred = min(pred['var.exp.median'].values)
    #pred = pred['var.exp.median'].values
    true_a = true
    pred_a = pd.DataFrame(min_pred/2, columns=true_a.columns, index=true_a.index)
    for _, row in pred.iterrows():
        r_max = max(pred_a.loc[row.regulator, row.target], pred_a.loc[row.target, row.regulator])
        r_max = max(r_max, row['var.exp.median'])
        pred_a.loc[row.regulator, row.target] = r_max
        pred_a.loc[row.target, row.regulator] = r_max
    true_a = get_lower_triangle(true_a.to_numpy()) # true_a.values.flatten()
    pred_a = get_lower_triangle(pred_a.to_numpy()) # pred_a.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true_a, pred_a)
    precision, recall, _ = skm.precision_recall_curve(true_a, pred_a)
    print("Inferelator (LR) : ", pred_a.shape, true_a.shape,
          skm.roc_auc_score(true_a, pred_a),
          skm.average_precision_score(true_a, pred_a))
    return (fpr, tpr, precision, recall)


def mrnet_roc(true_complete_file, mrnet_fname):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = DATA_DIR + mrnet_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    true = true_complete.loc[pred.columns, pred.columns]
    #pred = pred[true.columns]
    #pred = pred.reindex(true.index).values.flatten()
    #true = true.values.flatten()
    true = true.to_numpy()
    pred = np.fmax(np.transpose(pred.to_numpy()), pred.to_numpy())
    true = get_lower_triangle(true) # true_a.values.flatten()
    pred = get_lower_triangle(pred) # pred_a.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("mrnet (MI) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def tigress_roc(true_complete_file, tig_fname):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = DATA_DIR + tig_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    ngenes = len(pred.index.values)
    pred = pred[pred.columns[-ngenes:]]
    pred.columns = pred.index.values
    true = true_complete.loc[list(pred), list(pred)]
    pred = pred[true.columns]
    pred = pred.reindex(true.index)
    true = get_lower_triangle(true.to_numpy()) # true_a.values.flatten()
    pred = get_lower_triangle(pred.to_numpy()) # pred_a.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("TIGRESS (LR) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def tinge_old_roc(true_complete_file, tinge_fname, genes_list, full_genes_list):
    filename = DATA_DIR + tinge_fname
    true_complete = np.genfromtxt(true_complete_file, delimiter=' ',
                                  skip_header=1, usecols=range(1, 2001))
    genes_dct = {}
    genes_list = set(genes_list)
    for idx, gname in zip(range(len(full_genes_list)), full_genes_list):
        gname = gname.replace('"', '')
        if gname in genes_list:
            genes_dct[gname] = idx
    row = []
    col = []
    pred = []
    with open(filename) as ifx:
        num = 0
        for line in ifx:
            if num >= 30:
                terms = line.rstrip().split('\t')
                row.extend([genes_dct[terms[0]] for i in range(int((len(terms)-1)/2))])
                col.extend([genes_dct[terms[i]] for i in range(1, len(terms)-1, 2)])
                pred.extend([float(terms[i]) for i in range(2, len(terms), 2)])
            else:
                num += 1
    true = true_complete[row, col]
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("TINGe (MI) : ", len(pred), true.flatten().shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def tinge_roc(true_complete_file, tinge_fname, genes_list):
    filename = DATA_DIR + tinge_fname
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    true_a = true_complete.loc[genes_list, genes_list]
    pred = load_eda_network(filename)
    min_mi = min(pred['wt'].values)
    #print(min_mi)
    true_a = true_complete.loc[genes_list, genes_list]
    pred_a = pd.DataFrame(min_mi/2, columns=true_a.columns, index=true_a.index)
    for row in pred.itertuples():
        pred_a.loc[row.source, row.target] = row.wt
        pred_a.loc[row.target, row.source] = row.wt
    true_a = get_lower_triangle(true_a.to_numpy()) # true_a.values.flatten()
    pred_a = get_lower_triangle(pred_a.to_numpy()) # pred_a.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true_a, pred_a)
    precision, recall, _ = skm.precision_recall_curve(true_a, pred_a)
    print("TINGe (MI) : ", pred_a.shape, true_a.shape,
          skm.roc_auc_score(true_a, pred_a),
          skm.average_precision_score(true_a, pred_a))
    return (fpr, tpr, precision, recall)


def wgcna_roc(true_complete_file, tinge_fname):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = DATA_DIR + tinge_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    true = true_complete.loc[list(pred), list(pred)]
    pred = pred[true.columns]
    pred = pred.reindex(true.index) # .values.flatten()
    true = get_lower_triangle(true.to_numpy()) # true.values.flatten()
    pred = get_lower_triangle(pred.to_numpy()) # pred.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("WGCNA (COR) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def pcc_roc(true_complete_file, tinge_fname):
    true_complete = pd.read_csv(true_complete_file, sep=' ', index_col=0)
    filename = DATA_DIR + tinge_fname
    pred = pd.read_csv(filename, sep=' ', index_col=0)
    true = true_complete.loc[list(pred), list(pred)]
    pred = pred[true.columns]
    pred = pred.reindex(true.index) # .values.flatten()
    true = get_lower_triangle(true.to_numpy()) # true.values.flatten()
    pred = get_lower_triangle(np.abs(pred.to_numpy())) # pred.values.flatten()
    fpr, tpr, _ = skm.roc_curve(true, pred)
    precision, recall, _ = skm.precision_recall_curve(true, pred)
    print("PCC (COR) : ", pred.shape, true.shape,
          skm.roc_auc_score(true, pred),
          skm.average_precision_score(true, pred))
    return (fpr, tpr, precision, recall)

def plot_all_roc2k(true_edgesfx, net_genes, plt_colors):
    fpr_full = []
    tpr_full = []
    method_full = []
    fpr, tpr, _, _ = aracne_roc(true_edgesfx, REVNET_FILES['aracne'], net_genes)
    plt.plot(fpr, tpr, color=plt_colors[0], label='ARACNE-AP')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['ARACNE-AP' for _ in tpr]
    fpr, tpr, _, _ = clr_roc(true_edgesfx, REVNET_FILES['clr'], net_genes)
    plt.plot(fpr, tpr, color=plt_colors[1], label='CLR')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['CLR' for _ in tpr]
    fpr, tpr, _, _ = fastggm_roc(true_edgesfx, REVNET_FILES['fastggm'], net_genes)
    plt.plot(fpr, tpr, color=plt_colors[2], label='FastGGM')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['FastGGM' for _ in tpr]
    fpr, tpr, _, _ = genenet_roc(true_edgesfx, REVNET_FILES['genenet'])
    plt.plot(fpr, tpr, color=plt_colors[3], label='GeneNet')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['GeneNet' for _ in tpr]
    fpr, tpr, _, _ = genie3_roc(true_edgesfx, REVNET_FILES['genie3'], net_genes)
    plt.plot(fpr, tpr, color=plt_colors[4], label='GENIE3')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['GENIE3' for _ in tpr]
    fpr, tpr, _, _ = grnboost_roc(true_edgesfx, REVNET_FILES['grnboost'], net_genes)
    plt.plot(fpr, tpr, color=plt_colors[5], label='GRNBoost')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['GRNBoost' for _ in tpr]
    fpr, tpr, _, _ = inferlator_roc(true_edgesfx, REVNET_FILES['inferlator'], net_genes)
    plt.plot(fpr, tpr, color=plt_colors[6], label='Inferelator')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['Inferelator' for _ in tpr]
    fpr, tpr, _, _ = mrnet_roc(true_edgesfx, REVNET_FILES['mrnet'])
    plt.plot(fpr, tpr, color=plt_colors[7], label='MRNET')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['MRNET' for _ in tpr]
    fpr, tpr, _, _ = tigress_roc(true_edgesfx, REVNET_FILES['tigress'])
    plt.plot(fpr, tpr, color=plt_colors[8], label='TIGRESS')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['TIGRESS' for _ in tpr]
    fpr, tpr, _, _ = tinge_roc(true_edgesfx, REVNET_FILES['tinge'], net_genes)
    plt.plot(fpr, tpr, color=plt_colors[9], label='TINGe')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['TINGe' for _ in tpr]
    fpr, tpr, _, _ = pcc_roc(true_edgesfx, REVNET_FILES['pcc'])
    plt.plot(fpr, tpr, color=plt_colors[10], label='PCC')
    fpr_full = fpr_full + [x for x in fpr]
    tpr_full = tpr_full + [x for x in tpr]
    method_full = method_full + ['PCC' for _ in tpr]
    mdf = pd.DataFrame({"FPR": fpr_full, "TPR": tpr_full, "METHOD": method_full})
    mdf.to_csv("roc.full.2k.csv", index=False)
    # diagonal line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

def plot_all_pr2k(true_edgesfx, net_genes, plt_colors):
    prec_full = []
    recall_full = []
    method_full = []
    _, _, prec, recall = aracne_roc(true_edgesfx, REVNET_FILES['aracne'], net_genes)
    plt.plot(prec, recall, color=plt_colors[0], label='ARACNE-AP')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['ARACNE-AP' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = clr_roc(true_edgesfx, REVNET_FILES['clr'], net_genes)
    plt.plot(prec, recall, color=plt_colors[1], label='CLR')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['CLR' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = fastggm_roc(true_edgesfx, REVNET_FILES['fastggm'], net_genes)
    plt.plot(prec, recall, color=plt_colors[2], label='FastGGM')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['FastGGM' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = genenet_roc(true_edgesfx, REVNET_FILES['genenet'])
    plt.plot(prec, recall, color=plt_colors[3], label='GeneNet')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['GeneNet' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = genie3_roc(true_edgesfx, REVNET_FILES['genie3'], net_genes)
    plt.plot(prec, recall, color=plt_colors[4], label='GENIE3')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['GENIE3' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = grnboost_roc(true_edgesfx, REVNET_FILES['grnboost'], net_genes)
    plt.plot(prec, recall, color=plt_colors[5], label='GRNBoost')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['GRNBoost' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = inferlator_roc(true_edgesfx, REVNET_FILES['inferlator'], net_genes)
    plt.plot(prec, recall, color=plt_colors[6], label='Inferelator')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['Inferelator' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = mrnet_roc(true_edgesfx, REVNET_FILES['mrnet'])
    plt.plot(prec, recall, color=plt_colors[7], label='MRNET')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['MRNET' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = tigress_roc(true_edgesfx, REVNET_FILES['tigress'])
    plt.plot(prec, recall, color=plt_colors[8], label='TIGRESS')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['TIGRESS' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = tinge_roc(true_edgesfx, REVNET_FILES['tinge'], net_genes)
    plt.plot(prec, recall, color=plt_colors[9], label='TINGe')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['TINGe' for _ in prec]
    print(len(prec), len(prec_full))
    _, _, prec, recall = pcc_roc(true_edgesfx, REVNET_FILES['pcc'])
    plt.plot(prec, recall, color=plt_colors[10], label='PCC')
    prec_full = prec_full + [x for x in prec]
    recall_full = recall_full + [x for x in recall]
    method_full = method_full + ['PCC' for _ in prec]
    print(len(prec), len(prec_full))
    mdf = pd.DataFrame({"PRECISION": prec_full, "RECALL": recall_full, "METHOD": method_full})
    mdf.to_csv("pr.full.2k.csv", index=False)
    #
    # diagonal line
    #
    plt.plot([0, 1], [1, 0], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right')

def main(out_format, split_sub):
    true_complete_file = TRUE_NET
    net_genes = []
    plt_colors = PLOT_COLORS
    with open(true_complete_file) as ifx:
        line = ifx.readline().replace('"', '')
        net_genes = line.split()
    matplotlib.style.use('seaborn-muted')
    if split_sub == 'split':
        fig = plt.figure(figsize=FIG_SIZE)
        # plt.subplot(1, 2, 2)
        plot_all_roc2k(true_complete_file, net_genes, plt_colors)
        fig.savefig(ROC_OUTPUT_FILE+out_format)
        fig = plt.figure(figsize=FIG_SIZE)
        plot_all_pr2k(true_complete_file, net_genes, plt_colors)
        fig.savefig(PR_OUTPUT_FILE+out_format)
    else:
        fig = plt.figure(figsize=SUB_FIG_SIZE)
        plt.subplot(1, 2, 1)
        plot_all_roc2k(true_complete_file, net_genes, plt_colors)
        plt.subplot(1, 2, 2)
        plot_all_pr2k(true_complete_file, net_genes, plt_colors)
        fig.savefig(ROCPR_OUTPUT_FILE+out_format)



if __name__ == "__main__":
    if not sys.argv[1:] or len(sys.argv[1:]) < 2:
        print("python", sys.argv[0], 'png/pdf', 'sub/split')
        IN_ARGS = ['pdf', 'split']
    else:
        IN_ARGS = sys.argv[1:]
    if IN_ARGS[0] != 'png' and IN_ARGS[0] != 'pdf':
        print("python", sys.argv[0], 'png/pdf', 'sub/split')
    elif IN_ARGS[1] != 'sub' and IN_ARGS[1] != 'split':
        print("python", sys.argv[0], 'png/pdf', 'sub/split')
    else:
        print(IN_ARGS)
        main(IN_ARGS[0], IN_ARGS[1])
