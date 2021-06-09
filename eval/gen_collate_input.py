import glob
import json

pred_data_dir = "../data/revnet/"
true_data_dir = "../data/ref_yeast/"
data_subset  = ["1000.2000", "1250.2000", "1500.2000", "1750.2000",
               "2000.1000", "2000.1250", "2000.1500", "2000.1750",
               "2000.2000", "2000.250", "2000.500", "2000.750", "250.2000", 
               "500.2000", "750.2000"]

rev_base_dir = { 
        "aracne": "aracne/",
        "clr": "clr/",
        "fastggm": "fastggm/",
        "genenet": "genenet/",
        "genie3": "genie3/",
        "grnboost": "grnboost/",
        "inferelator": "inferlator/",
        "irafnet": "irafnet/",
        "mrnet": "mrnet/",
        "pcc": "pearson/",
        "tigress": "tigress/",
        "tinge":"tinge/",
        "wgcna":"wgcna/"}

rev_file_pattern  = {
        "aracne": "-*/bootstrapNetwork_ul3atth75o35ngtur8ibskqq7s.txt",
        "clr": "-*",
        "fastggm": "-*",
        "genenet":"-*",
        "genie3": "-*",
        "grnboost": "-*.result.tsv",
        "inferelator": "-*/summary_frac_tp_0_perm_1--frac_fp_0_perm_1_1.tsv",
        "irafnet": "-*",
        "pcc": "-*",
        "mrnet": "-*",
        "tigress": "-*",
        "tinge":".weight.eda",
        "wgcna":"-*"}

def pred_files(base_dir, all_methods, dx):
    pred_files = {}
    for method in all_methods:
        m_base_dir = base_dir + rev_base_dir[method]
        file_pattern = m_base_dir + dx + rev_file_pattern[method]
        all_files = glob.glob(file_pattern)
        if len(all_files) == 0:
            print("Can find file for" , method, dx)
            continue
        pred_files[method] = all_files[0].replace(base_dir, "")
    #print(pred_files)
    return pred_files

def true_net(base_dir, data_subset):
    file_pattern = base_dir + "/" + "gnw" + data_subset.split(".")[0] + "_trunet"
    fx = glob.glob(file_pattern)
    #print(fx)
    return fx[0]

if __name__ == "__main__":
    print("Generating Input Files for Collate GRN script...")
    all_methods = rev_base_dir.keys()
    pred_dct = {x: pred_files(pred_data_dir, all_methods, x) for x in data_subset}
    tru_file = {x: true_net(true_data_dir, x) for x in data_subset}
    out_file = {x: "yeast-edges-weights-" + x + ".csv" for x in data_subset}
    dst_dct = { x: {'OUT_FILE': out_file[x], 'DATA_DIR': pred_data_dir, "TRUE_NET"  : tru_file[x], "REVNET_FILES": pred_dct[x]} for x in data_subset }
    for x in data_subset:
        json_fname = "collate" + x + ".json"
        with open(json_fname, "w") as ofx:
            json.dump(dst_dct[x], ofx, indent=4)
    print("Complete generation of the input config files")
    

