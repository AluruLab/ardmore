import sys
import json
import pandas as pd
import numpy as np

def remove_dupe_rows(in_df: pd.DataFrame, wt_attr_name: str) -> pd.DataFrame:
    """
    Given a data frame of three columns : "source", "target" and wt_attr_name
    such that only the row with max weight is returned

    Parameters
    ----------
    in_df : Input data frame with columns "source", "target", wt_attr_name

    wt_attr_name : weight column name in the data frame returned

    Returns
    -------
    pandas DataFrame with three columns: "source", "target", wt_attr_name such that
    only the row with max weight is returned
    """
    in_df = in_df.sort_values(by=[wt_attr_name], ascending=False)
    return in_df.drop_duplicates(subset=["source", "target"], keep="first")


def order_network_rows(in_df: pd.DataFrame, wt_attr_name: str) -> pd.DataFrame:
    """
    Given a data frame of three columns : "source", "target" and wt_attr_name
    returns rows such that source < target

    Parameters
    ----------
    in_df : Input data frame with columns "source", "target", wt_attr_name

    wt_attr_name : weight column name in the data frame returned

    Returns
    -------
    pandas DataFrame with three columns: "source", "target", wt_attr_name such that
    source entry < target entry
    """
    if (in_df.source < in_df.target).all():
        return remove_dupe_rows(in_df, wt_attr_name)
    tmp_rcds = [(x, y, z) if x < y else (y, x, z) for x, y, z in in_df.to_dict("split")["data"]]
    tmp_df = pd.DataFrame(tmp_rcds, columns=["source", "target", wt_attr_name])
    return remove_dupe_rows(tmp_df, wt_attr_name)

def load_eda_network(eda_file: str, wt_attr_name: str = "wt",
                     delimiter: str = r"\s+") -> pd.DataFrame:
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
    pandas DataFrame with three columns: "source", "target", wt_attr_name

    """
    tmp_df = pd.read_csv(eda_file, sep=delimiter, usecols=[0, 2, 4], skiprows=[0],
                         names=["source", "target", wt_attr_name])
    return order_network_rows(tmp_df, wt_attr_name)


def load_mat_network(mat_file: str, header_opt="infer", wt_attr_name: str = "wt",
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
    pandas DataFrame with three columns: "source", "target", wt_attr_name
    """
    mat_df = pd.read_csv(mat_file, header=header_opt,
                         sep=delimiter, index_col=0)
    mat_cnames = mat_df.columns
    mat_size = mat_df.shape[0]
    net_df = pd.DataFrame({
        "source": np.repeat(mat_cnames, mat_size),
        "target": np.tile(mat_cnames, mat_size),
        wt_attr_name: mat_df.values.flatten()})
    return net_df[net_df.source < net_df.target]

def load_mat_network_glist_df(mat_df, gene_list, header_opt=None, wt_attr_name: str = "wt",
                          delimiter: str = ",") -> pd.DataFrame:
    mat_cnames = gene_list
    mat_size = mat_df.shape[0]
    net_df = pd.DataFrame({
        "source": np.repeat(mat_cnames, mat_size),
        "target": np.tile(mat_cnames, mat_size),
        wt_attr_name: mat_df.to_numpy().flatten()})
    return net_df[net_df.source < net_df.target]



def load_mat_network_glist(mat_file: str, gene_list, header_opt=None, wt_attr_name: str = "wt",
                          delimiter: str = ",") -> pd.DataFrame:
    mat_df = pd.read_csv(mat_file, header=header_opt,
                         sep=delimiter, names=gene_list)
    mat_cnames = gene_list
    mat_size = mat_df.shape[0]
    net_df = pd.DataFrame({
        "source": np.repeat(mat_cnames, mat_size),
        "target": np.tile(mat_cnames, mat_size),
        wt_attr_name: mat_df.values.flatten()})
    return net_df[net_df.source < net_df.target]

def load_tsv_network(eda_file: str, header_opt = None, wt_attr_name: str = "wt") -> pd.DataFrame:
    """
    Load network from edge attribute file file (.eda).
    Eda file lists the edges in the following format
    The first line has the string "Weight". The edges are listed with the following
    format:
    source  target  weight

    where source and target are node ids, itype is the interaction type and weight is
    the edge weight.

    Example:
    244901_at   244902_at   0.192777
    244901_at   244926_s_at 0.0817807


    Parameters
    ----------
    eda_file : Path to the input .eda file

    wt_attr_name : weight column name in the data frame returned

    delimiter : seperators between the fields

    Returns
    -------
    pandas DataFrame with three columns: "source", "target", wt_attr_name

    """
    tmp_df = pd.read_csv(eda_file, sep="\t", header=header_opt,
                         names=["source", "target", wt_attr_name])
    tmp_rcds = [(x, y, z) if x < y else (y, x, z) for x, y, z in tmp_df.to_dict("split")["data"]]
    xdf = pd.DataFrame(tmp_rcds, columns=["source", "target", wt_attr_name])
    return xdf.sort_values(by=wt_attr_name, ascending=False)


def read_genes(genes_file):
    with open(genes_file) as ifx:
        rgenes = ifx.readlines()
        return [x.strip() for x in rgenes]


def get_lower_triangle(in_mat):
    ill = np.tril_indices(in_mat.shape[0], -1)
    return in_mat[ill]


def aracne_preds_old(true_complete_file, arance_file, genes_list, in_true):
    true_complete = pd.read_csv(true_complete_file, sep=" ", index_col=0)
    filename = arance_file
    pred = pd.read_csv(filename, sep="\t")
    #min_mi = min(pred["MI"].values)
    max_mi = max(pred["MI"].values)
    #print(min_mi, max_mi)
    true_a = true_complete.loc[genes_list, genes_list]
    pred_a = pd.DataFrame(0, columns=true_a.columns, index=true_a.index)
    for row in pred.itertuples():
        r_max = max(pred_a.loc[row.Regulator, row.Target],
                    pred_a.loc[row.Target, row.Regulator])
        r_max = max(r_max, row.MI)
        pred_a.loc[row.Regulator, row.Target] = r_max
        pred_a.loc[row.Target, row.Regulator] = r_max
    true_a = get_lower_triangle(true_a.to_numpy()) # true.values.flatten()
    pred_a = get_lower_triangle(pred_a.to_numpy()) # pred.values.flatten()
    print(np.all(true_a == in_true))
    return pred_a

def clr_preds(true_complete_file, clr_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = clr_fname
    rnet = load_mat_network_glist(filename, genes_list, wt_attr_name="pwt")
    print(filename, rnet.columns)
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred

def genenet_preds(true_complete_file, genenet_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = genenet_fname
    tdf = pd.read_csv(filename, sep=" ")
    tdf = tdf.loc[genes_list, genes_list]
    print(tdf)
    rnet = load_mat_network_glist_df(tdf, genes_list, wt_attr_name="pwt")
    print(filename, rnet.columns, rnet.shape)
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred


def fastggm_preds(true_complete_file, fastggm_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = fastggm_fname
    ngenes = len(genes_list)
    pred_df = pd.read_csv(filename, sep=" ", index_col=0,
                       usecols=range(2*ngenes, 3*ngenes))
    print(pred_df)
    pred_df = 1 - pred_df
    rnet = load_mat_network_glist_df(pred_df, genes_list, wt_attr_name="pwt")
    print(filename, rnet.columns)
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred




def aracne_preds(true_complete_file, arance_file, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = arance_file
    rnet = load_tsv_network(filename, header_opt=0, wt_attr_name="pwt")
    print(filename, rnet.columns)
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred

def grnboost_preds(true_complete_file, grn_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = grn_fname
    rnet = load_tsv_network(filename, header_opt=0, wt_attr_name="pwt")
    print(filename, rnet.columns, rnet.shape)
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    print(rnet.shape)
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred

def genie3_preds(true_complete_file, grn_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = grn_fname
    rnet = pd.read_csv(filename, sep=" ", names=["source", "target", "pwt"], skiprows=1)
    print(rnet.head())
    print(filename, rnet.columns, rnet.shape)
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    print(rnet.shape)
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred

def mrnet_preds(true_complete_file, mrnet_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = mrnet_fname
    pred = pd.read_csv(filename, sep=" ", index_col=0)
    rnet = load_mat_network(filename, wt_attr_name="pwt")
    print(filename, rnet.columns)
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred


def tinge_preds(true_complete_file, tinge_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = tinge_fname
    rnet = load_eda_network(filename, wt_attr_name="pwt")
    print(filename, rnet.columns)
    #min_mi = min(pred["wt"].values)
    #print(min_mi)
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred

def pcc_preds(true_complete_file, pcc_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = pcc_fname
    rnet = load_mat_network(filename, wt_attr_name="pwt")
    print(filename, rnet.columns, rnet.shape, rnet.head())
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred

 

def wgcna_preds(true_complete_file, wgcna_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = wgcna_fname
    rnet = load_mat_network(filename, wt_attr_name="pwt")
    print(filename, rnet.columns, rnet.shape, rnet.head())
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred

def tigress_preds(true_complete_file, tig_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = tig_fname
    mat_df = pd.read_csv(filename, sep=" ", index_col=0)
    ngenes = len(mat_df.index.values)
    mat_df = mat_df[mat_df.columns[-ngenes:]]
    mat_df.columns = mat_df.index.values
    mat_df = mat_df.loc[genes_list, genes_list]
    mat_cnames = genes_list
    mat_size = mat_df.shape[0]
    net_df = pd.DataFrame({
        "source": np.repeat(mat_cnames, mat_size),
        "target": np.tile(mat_cnames, mat_size),
        "pwt": mat_df.values.flatten()})
    rnet = net_df[net_df.source < net_df.target]
    print(filename, rnet.columns, rnet.shape, rnet.head())
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred

def inferelator_preds(true_complete_file, inf_fname, genes_list, in_true):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    filename = inf_fname
    net_df = pd.read_csv(filename, sep="\t",
                          usecols=["regulator", "target", "var.exp.median"])
    net_df = net_df.rename(columns={"regulator": "source",
        "target":"target", "var.exp.median": "pwt"})
    net_df2 = net_df.rename(columns={"source": "target", "target": "source"})
    net_df = pd.concat([net_df, net_df2])
    rnet = net_df[net_df.source < net_df.target]
    print(filename, rnet.columns, rnet.shape, rnet.head())
    rnet = rnet.drop_duplicates(subset=["source", "target"])
    jnnet = pd.merge(stnet, rnet, on=["source", "target"], how="left")
    jnnet.fillna(0.0, inplace=True)
    true = jnnet.twt
    pred = jnnet.pwt
    print(np.all(true.values == in_true.values), len(pred), len(true))
    return pred


def get_true(true_complete_file, genes_list):
    # true_complete = pd.read_csv(true_complete_file, sep=" ", index_col=0)
    # true_a = true_complete.loc[genes_list, genes_list]
    # true = get_lower_triangle(true_a.to_numpy()) # true.values.flatten()
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    true = stnet.twt
    print("TRUE", len(true))
    return true

def get_edge_names2(true_complete_file, genes_list):
    tnet = load_mat_network(true_complete_file, wt_attr_name="twt")
    sgenesst = set(genes_list)
    stnet = tnet.loc[(tnet.source.isin(sgenesst)) &
                     (tnet.target.isin(sgenesst)), : ]
    enames = list(stnet["source"].astype(str) + "-" + stnet["target"])
    print("enames", len(enames), enames[0:8])
    return enames

def get_edge_names(genesl):
    genes_list = genesl
    ngenes = len(genes_list)
    enames = [(x,y) for x in range(ngenes) for y in range(x+1) if x != y]
    enames = [genes_list[x] + "-" + genes_list[y] for x in range(ngenes) for y in range(x+1) if x != y]
    print("enames", len(enames), enames[0:8])
    return enames
    
ROCPR_METHOD_DICT = {
    "aracne"  : aracne_preds,
    "clr"     : clr_preds,
    "grnboost" : grnboost_preds,
    "mrnet" : mrnet_preds,
    "tinge" : tinge_preds,
    "pcc" :  pcc_preds,
    "wgcna" :  wgcna_preds,
    "genenet" : genenet_preds,
    "genie3" : genie3_preds,
    "tigress" : tigress_preds,
    "inferelator" : inferelator_preds,
    "fastggm" : fastggm_preds
}

def collate_grn_nets(jsx):
    true_complete_file = jsx["TRUE_NET"]
    net_genes = []
    with open(true_complete_file) as ifx:
        line = ifx.readline().replace("\"", "")
        net_genes = line.strip().split()
    all_method_pred = {}
    enames = get_edge_names2(true_complete_file, net_genes)
    all_method_pred["edge"] = enames
    true = get_true(true_complete_file, net_genes)
    all_method_pred["prediction"] = list(true)
    methods_list = ["pcc", "clr", "aracne", "grnboost", "mrnet", "tinge", "wgcna",
                    "fastggm", "genenet", "tigress", "genie3", "inferelator"]
    #methods_list = ["tigress", "inferelator"]
    for method_name in methods_list:
        roc_fn = ROCPR_METHOD_DICT[method_name]
        pred_file = jsx_data["DATA_DIR"] + jsx["REVNET_FILES"][method_name]
        pred_method = roc_fn(true_complete_file, pred_file, net_genes, true)
        all_method_pred[method_name] = list(pred_method)
    for x,y in all_method_pred.items():
        print(x, len(y))
    dfx = pd.DataFrame.from_dict(all_method_pred)
    dfx.to_csv(jsx["OUT_FILE"], index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("  Usage: collate_grns.py <INPUT_JSON>")
        print("   <INPUT_JSON> Input configuration with paths to generated networks")
        print("   See collate2k.json for an example")
    else:
        input_json_file = sys.argv[1]
        with open(input_json_file) as f:
            jsx_data = json.load(f)
        collate_grn_nets(jsx_data)
