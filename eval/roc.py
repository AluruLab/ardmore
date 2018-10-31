import numpy, itertools, pandas
import matplotlib.pyplot as plt
from sklearn import metrics

val = pandas.read_csv('at.tsv', sep='\t', header=None)
val.columns = ['U', 'V']
val['WT'] = 0.0
val['IND'] = 1
val_probes = set(val.U.unique())
val_nodes = set(pandas.concat([val.U, val.V], ignore_index=True).unique())
val_targets = val_nodes - val_probes

universe_tfs_df = pandas.DataFrame(list(val_probes))
universe_targets_df = pandas.DataFrame(list(val_targets))
universe_tfs_df.columns = ["U"]
universe_targets_df.columns = ["V"]
universe_targets_df['key'] = 1
universe_tfs_df['key'] = 1
universe_cart = pandas.merge(universe_tfs_df, universe_targets_df, on="key", how="outer")
universe_tfs = numpy.array(list(val_probes))
universe_tfs.sort()
comb = [x for x in itertools.combinations(universe_tfs, 2)]
universe_self_comb = pandas.DataFrame.from_records(comb, columns=['U', 'V'])
del universe_cart['key']
universe_complete = pandas.concat([universe_cart, universe_self_comb])

merged = pandas.merge(universe_complete, val, on=['U', 'V'], how='outer', indicator=True)
criteria_indices = merged[merged._merge=='left_only'].index
val_missing = merged.loc[criteria_indices, :]
criteria_indices = merged[merged._merge=='both'].index
val_present = merged.loc[criteria_indices, :]
del val_missing["_merge"]
del val_present["_merge"]
val_missing['WT'] = 0.0
val_missing['IND'] = 0
vn = pandas.concat([val_present, val_missing])

network = pandas.read_csv('ranking_knockdowns_700.txt', sep='\t', header=None)
network.columns=['U', 'V', 'WT']

network2  = network.loc[network['U'].isin(val_probes) | network['V'].isin(val_probes)]
network3U = network2.loc[network2['U'].isin(val_probes) & ~network2['V'].isin(val_probes)]
network3V = network2.loc[network2['V'].isin(val_probes) & ~network2['U'].isin(val_probes)]
network3UV = network2.loc[network2['U'].isin(val_probes) & network2['V'].isin(val_probes)]
network3UV_p1 = network3UV.loc[network3UV['U'] < network3UV['V'] ]
network3UV_p2 = network3UV.loc[network3UV['U'] > network3UV['V'] ]
network3V.columns = ["V", "U", "WT"]
network3UV_p2.columns = ["V", "U", "WT"]
net = pandas.concat([network3U, network3V, network3UV_p1, network3UV_p2])
net["IND"] = 0


class0 = (vn.shape[0] - numpy.count_nonzero(vn.IND.values))
class1 = (numpy.count_nonzero(vn.IND.values))
print "Number of instances in class 0 : " + str(class0)
print "Number of instances in class 1 : " + str(class1)

finaldf = pandas.concat([vn, net])
rocdf = finaldf.groupby(['U', 'V']).aggregate(numpy.sum)	
y_trues = numpy.array(rocdf.IND)
y_scores = numpy.array(rocdf.WT)
fpr, tpr, thres = metrics.roc_curve(y_trues, y_scores, pos_label=1)
print "AUC score : " + str(metrics.roc_auc_score(y_trues, y_scores))
print "AUPR score : " + str(metrics.average_precision_score(y_trues, y_scores))
plt.plot(fpr, tpr, '-')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('700 data points')
plt.show()
