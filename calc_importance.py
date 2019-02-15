#!/usr/bin/env python3
#==============================================================================
# author          : Pavel Polishchuk
# date            : 15-02-2019
# version         : 
# python_version  : 
# copyright       : Pavel Polishchuk 2019
# license         : 
#==============================================================================

import argparse
import numpy as np
import pickle
import pandas as pd
from forest import predict_oob, hit_rate, enrichment


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make prediction with Random Forest model.')
    parser.add_argument('-x', metavar='descriptors.txt', required=True,
                        help='text file with descriptors (tab-separated) for the training set.'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-y', metavar='activity.txt', required=True,
                        help='text file with activity values 0/1/NA (tab-separated) for the training set.'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-m', '--model', metavar='model.pkl', required=True,
                        help='file with a pickled model.')
    parser.add_argument('-o', '--output', metavar='importances.txt', required=True,
                        help='output text file.')
    parser.add_argument('-r', '--repeats', metavar='NUMBER', required=False, default=1,
                        help='number of randomization repeats. Default: 1')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = v
        if o == "y": y_fname = v
        if o == "model": model_fname = v
        if o == "output": output_fname = v
        if o == "repeats": n_repeats = int(v)

    model = pickle.load(open(model_fname, 'rb'))[:10]
    x = pd.read_table(x_fname, sep="\t", index_col=0)
    y = pd.read_table(y_fname, sep="\t", index_col=0)
    ref_hit_rate = np.apply_along_axis(hit_rate, 0, y)

    ref_pred = predict_oob(model, x)
    ids = ref_pred >= 1
    ref_e_median = enrichment(y.loc[ids, :], ref_hit_rate, np.median)

    imp = pd.DataFrame(index=x.columns, columns=range(n_repeats))
    for n in range(n_repeats):
        for i in x:
            xx = x.copy()
            xx[i] = xx[i].sample(frac=1).tolist()
            pred = predict_oob(model, xx)
            ids = pred >= 1
            imp.loc[i, n] = enrichment(y.loc[ids, :], ref_hit_rate, np.median)

    imp = ref_e_median - imp
    res = pd.concat([imp, imp.mean(axis=1), imp.std(axis=1)], axis=1).round(3)
    res.columns = list(range(1, n_repeats + 1)) + ['mean', 'sd']
    res.to_csv(output_fname, sep="\t")
