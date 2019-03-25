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
import pandas as pd
import numpy as np
import warnings
from forest import hit_rate, enrichment


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculate statistics.')
    parser.add_argument('-y', metavar='activity.txt', required=True,
                        help='text file with activity values 0/1/NA (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-p', '--predictions', metavar='predictions.txt', required=True,
                        help='text file with predictions returned by forest_predict.py.')
    parser.add_argument('-s', '--overall_stat', metavar='overall_stat.txt', required=False, default=None,
                        help='text file with calculated statistics. Default: None.')
    parser.add_argument('-a', '--assay_stat', metavar='assay_stat.txt', required=False, default=None,
                        help='text file with calculated statistics for each assay. Default: None.')
    parser.add_argument('-t', '--threshold', metavar='NUMBER', required=False, default=1,
                        help='floating point number which defines the threshold of compounds to select. Default: 1.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "y": y_fname = v
        if o == "predictions": pred_fname = v
        if o == "overall_stat": stat_fname = v
        if o == "assay_stat": astat_fname = v
        if o == "threshold": thr = float(v)

    if stat_fname is None and astat_fname is None:
        raise ValueError('at least one of outputs should be specified: overall_stat or assay_stat.')

    y = pd.read_table(y_fname, sep="\t", index_col=0)
    ref_hit_rate = np.apply_along_axis(hit_rate, 0, y)

    pred = pd.read_table(pred_fname, sep="\t", index_col=0)

    if stat_fname:
        f_stat = open(stat_fname, 'wt')
    if astat_fname:
        f_astat = open(astat_fname, 'wt')

    for j in range(pred.shape[1]):

        ids = pred.iloc[:, j] >= thr

        if stat_fname:
            if j == 0:
                f_stat.write('tree\tcompounds\tcoverage\tmedian enrichment\tmean enrichment\n')
            e_median = enrichment(y.loc[ids, :], ref_hit_rate, np.median)
            e_mean = enrichment(y.loc[ids, :], ref_hit_rate, np.mean)
            f_stat.write('\t'.join(map(str, (pred.columns[j], sum(ids), round(sum(ids) / y.shape[0], 3), round(e_median, 3), round(e_mean, 3)))) + '\n')

        if astat_fname:
            if j == 0:
                f_astat.write('tree\t' + '\t'.join(y.columns) + '\n')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                e_assay = np.apply_along_axis(hit_rate, 0, y.loc[ids, :]) / ref_hit_rate
            f_astat.write(str(pred.columns[j]) + '\t' + '\t'.join(map(str, np.round(e_assay, 3))) + '\n')

    if stat_fname:
        f_stat.close()
    if astat_fname:
        f_astat.close()
