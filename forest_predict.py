import pandas as pd
import numpy as np
import pickle
import glob
import os
import argparse
from forest_mp import hit_rate, enrichment, predict_tree

# x_fname = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_test/x_test_bin.txt'
# x_fname = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/x_test.txt'
# y_fname = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_test/y_test.txt'
#
# # x_fname = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_test/x_bin.txt'
# # y_fname = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_test/y.txt'
#
# y = pd.read_table(y_fname, sep="\t", index_col=0)
# x = pd.read_table(x_fname, sep="\t", index_col=0)
#
# # trees = pickle.load(open('/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_test/tree_x_bin_p10000_c10000_alg7.pkl', 'rb'))
# # trees = pickle.load(open('/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_test/tree.pkl', 'rb'))
# # trees = pickle.load(open('/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_test/forest_x_bin_t10_v3_p10000_c1000_alg1.pkl', 'rb'))
#
# # trees = pickle.load(open('/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_pbs/forest_0.pkl', 'rb'))
#
# # dir_name = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_pbs/'
# # dir_name = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_p3000_c1000_alg2_m3_t5_realx/'
# dir_name = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_p3000_c1000_alg3_m3_t5_realx/'
#
# flist = [file for file in glob.glob1(dir_name, "forest_*.pkl")]
#
# trees = []
# for fname in flist:
#     trees.extend(pickle.load(open(os.path.join(dir_name, fname), 'rb')))
#
# ref_hit_rate = np.apply_along_axis(hit_rate, 0, y)
#
# # estimate predictions of the forest
#
# pred = predict_forest(trees, x)
# for i in [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
#     ids = pred >= i
#     e = enrichment(y.loc[ids, :], ref_hit_rate, np.median)
#     print(i, sum(ids), round(sum(ids) / y.shape[0], 3), round(e, 3))
#
# # estimate predictions by trees
#
# pred = []
# for tree in trees:
#     pred.append(predict_tree(tree, x))
# pred = pd.concat(pred, axis=1)
# pred.columns = list(range(pred.shape[1]))
# pred = pred.cumsum(1).divide(pd.Series(list(range(1, pred.shape[1] + 1))))
#
# # for j in range(pred.shape[1]):
# #     print('ntree = %i' % (j + 1))
# #     for i in [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
# #         ids = pred.iloc[:, j] >= i
# #         e = enrichment(y.loc[ids, :], ref_hit_rate, np.median)
# #         print(i, sum(ids), round(sum(ids) / y.shape[0], 3), round(e, 3))
#
# with open(os.path.join(dir_name, 'stat.txt'), 'wt') as f:
#     f.write('tree\tcompounds\tcoverage\tmedian enrichment\tmean enrichment\n')
#     for j in range(pred.shape[1]):
#         # print('ntree = %i' % (j + 1))
#         ids = pred.iloc[:, j] >= 1
#         e_median = enrichment(y.loc[ids, :], ref_hit_rate, np.median)
#         e_mean = enrichment(y.loc[ids, :], ref_hit_rate, np.mean)
#         print(j + 1, sum(ids), round(sum(ids) / y.shape[0], 3), round(e_median, 3), round(e_mean, 3))
#         f.write('\t'.join(map(str, (j + 1, sum(ids), round(sum(ids) / y.shape[0], 3), round(e_median, 3), round(e_mean, 3)))) + '\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make prediction with Random Forest model.')
    parser.add_argument('-x', metavar='descriptors.txt', required=True,
                        help='text file with descriptors (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-y', metavar='activity.txt', required=False, default=None,
                        help='text file with activity values 0/1/NA (tab-separated).'
                             'Header is present. The first column contains compound names.'
                             'If not specified no statistics will be calculated.')
    parser.add_argument('-m', '--model', metavar='model.pkl', required=True,
                        help='file with pickled model.')
    parser.add_argument('-p', '--prediction', metavar='predictions.txt', required=False, default=None,
                        help='text file with predicted values. Default: None.')
    parser.add_argument('-s', '--stat', metavar='stat.txt', required=False, default=None,
                        help='text file with calculated statistics. A file with observed values should '
                             'be supplied to calculate statistics. Default: None.')
    parser.add_argument('-d', '--detailed', action='store_true', default=False,
                        help='calculate statistics after addition of each tree and '
                             'store predictions for ensembles with variable number of trees.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print progress.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = v
        if o == "y": y_fname = v
        if o == "model": model_fname = v
        if o == "prediction": pred_fname = v
        if o == "stat": stat_fname = v
        if o == "detailed": detailed = v
        if o == "verbose": verbose = v

    model = pickle.load(open(model_fname, 'rb'))

    x = pd.read_table(x_fname, sep="\t", index_col=0)
    if y_fname:
        y = pd.read_table(y_fname, sep="\t", index_col=0)
        ref_hit_rate = np.apply_along_axis(hit_rate, 0, y)

    # estimate predictions of the forest

    # pred = predict_forest(model, x)
    # for i in [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
    #     ids = pred >= i
    #     e = enrichment(y.loc[ids, :], ref_hit_rate, np.median)
    #     print(i, sum(ids), round(sum(ids) / y.shape[0], 3), round(e, 3))

    # estimate predictions by trees

    pred = []
    for tree in model:
        pred.append(predict_tree(tree, x))
    pred = pd.concat(pred, axis=1)
    pred.columns = list(range(pred.shape[1]))
    # matrix N mols x T trees with predicted values for different number of trees in a forest
    pred = pred.cumsum(1).divide(pd.Series(list(range(1, pred.shape[1] + 1))))
    pred.columns = list(range(1, pred.shape[1] + 1))

    if pred_fname:
        if detailed:
            pred.round(3).to_csv(pred_fname, sep='\t')
        else:
            pred.iloc[:, -1].round(3).to_csv(pred_fname, sep='\t')

    if stat_fname:
        f = open(stat_fname, 'wt')
        f.write('tree\tcompounds\tcoverage\tmedian enrichment\tmean enrichment\n')

        r = range(pred.shape[1])   # iterate over all trees
        if not detailed:
            r = [pred.shape[1] - 1]   # last tree

        for j in r:
            ids = pred.iloc[:, j] >= 1
            e_median = enrichment(y.loc[ids, :], ref_hit_rate, np.median)
            e_mean = enrichment(y.loc[ids, :], ref_hit_rate, np.mean)
            f.write('\t'.join(map(str, (j + 1, sum(ids), round(sum(ids) / y.shape[0], 3), round(e_median, 3), round(e_mean, 3)))) + '\n')
            if verbose:
                print(j + 1, sum(ids), round(sum(ids) / y.shape[0], 3), round(e_median, 3), round(e_mean, 3))
