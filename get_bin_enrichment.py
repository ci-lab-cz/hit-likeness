#!/usr/bin/env python3

from binning import read_thresholds
from tree_mp import load_x, load_y, hit_rate
import numpy as np
from collections import defaultdict
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate statistics for binned physicochemical parameters.')
    parser.add_argument('-x', metavar='parameters_bin.txt', required=True,
                        help='input text file with binned physicochemical parameters.')
    parser.add_argument('-y', metavar='activity.txt', required=True,
                        help='input text file with activities.')
    parser.add_argument('-t', metavar='thresholds.txt', required=True,
                        help='input text file with thresholds.')
    parser.add_argument('-o', metavar='output.txt', required=True,
                        help='output text file with statistics.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = v
        if o == "y": y_fname = v
        if o == "t": tr_fname = v
        if o == "o": out_fname = v

    t = read_thresholds(tr_fname)

    bin_labels = defaultdict(list)
    for k, v in t.items():
        bin_labels[k].append('<%i' % v[0])
        for low, high in zip(v[:-1], v[1:]):
            bin_labels[k].append('[%i-%i)' % (low, high))
        bin_labels[k].append('%i+' % v[-1])

    x, var_names = load_x(x_fname)

    y, mol_names, assays = load_y(y_fname)

    ref_hit_rate = np.apply_along_axis(hit_rate, 0, y)

    with open(out_fname, 'wt') as f:
        f.write('Parameter\tbin\tbin_label\tnum_compounds\t' + '\t'.join(assays) + '\n')
        for col_id, col in enumerate(x.T):
            for i in sorted(np.unique(col)):
                e = np.apply_along_axis(hit_rate, 0, y[col == i, :]) / ref_hit_rate
                e = np.round(e, 3)
                f.write('%s\t%i\t%s\t%i\t' % (var_names[col_id], i, bin_labels[var_names[col_id]][i], sum(col == i)) + '\t'.join(map(str, e)) + '\n')
