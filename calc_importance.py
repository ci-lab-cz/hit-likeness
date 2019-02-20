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
import random
from multiprocessing import Pool, cpu_count
from forest import predict_oob, hit_rate, enrichment


def get_permutation_pred(model, x, y, ref_hit_rate):
    enr = []
    cov = []
    np.random.seed(random.randint(1, 10000000))
    for i in x:
        xx = x.copy()
        xx[i] = xx[i].sample(frac=1).tolist()
        pred = predict_oob(model, xx)
        ids = pred >= 1
        enr.append(round(enrichment(y.loc[ids, :], ref_hit_rate, np.median), 4))
        cov.append(round(sum(ids) / len(ids), 4))
    return enr, cov


def get_permutation_pred_mp(items):
    return get_permutation_pred(*items)


def supply_data(model, x, y, ref_hit_rate, n):
    for _ in range(n):
        yield model, x, y, ref_hit_rate


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
    parser.add_argument('-c', '--ncpu', metavar='NUMBER', required=False, default=1,
                        help='number of CPU to use. Default: 1')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = v
        if o == "y": y_fname = v
        if o == "model": model_fname = v
        if o == "output": output_fname = v
        if o == "repeats": n_repeats = int(v)
        if o == "ncpu": ncpu = int(v)

    pool = Pool(min(ncpu, cpu_count())) if ncpu > 1 else None

    model = pickle.load(open(model_fname, 'rb'))
    x = pd.read_table(x_fname, sep="\t", index_col=0)
    y = pd.read_table(y_fname, sep="\t", index_col=0)
    ref_hit_rate = np.apply_along_axis(hit_rate, 0, y)

    ref_pred = predict_oob(model, x)
    ids = ref_pred >= 1
    ref_enrichment = round(enrichment(y.loc[ids, :], ref_hit_rate, np.median), 4)
    ref_coverage = round(sum(ids) / len(ids), 4)
    ref_imp = round(ref_enrichment * ref_coverage, 4)

    calc_enrichment = pd.DataFrame(index=x.columns, columns=range(n_repeats))
    calc_coverage = pd.DataFrame(index=x.columns, columns=range(n_repeats))

    if pool is not None:
        for n, (enr, cov) in enumerate(pool.imap(get_permutation_pred_mp, supply_data(model, x, y, ref_hit_rate, n_repeats))):
            calc_enrichment[n] = enr
            calc_coverage[n] = cov
    else:
        for n in range(n_repeats):
            for i in x:
                xx = x.copy()
                xx[i] = xx[i].sample(frac=1).tolist()
                pred = predict_oob(model, xx)
                ids = pred >= 1
                calc_enrichment.loc[i, n] = round(enrichment(y.loc[ids, :], ref_hit_rate, np.median), 4)
                calc_coverage.loc[i, n] = round(sum(ids) / len(ids), 4)

    imp = ref_imp - calc_enrichment * calc_coverage
    imp = imp.round(4)
    imp = pd.concat([imp, imp.mean(axis=1), imp.std(axis=1)], axis=1)
    imp.columns = ['overall_imp_' + str(v) for v in imp.columns[:-2]] + ['mean_overall_imp', 'sd_overall_imp']

    enrichment_score = ref_enrichment - calc_enrichment
    enrichment_score = enrichment_score.round(4)
    enrichment_score = pd.concat([enrichment_score, enrichment_score.mean(axis=1), enrichment_score.std(axis=1)], axis=1)
    enrichment_score.columns = ['enrichemnt_imp_' + str(v) for v in enrichment_score.columns[:-2]] + ['mean_enrichment_imp', 'sd_enrichment_imp']

    coverage_score = ref_coverage - calc_coverage
    coverage_score = coverage_score.round(4)
    coverage_score = pd.concat([coverage_score, coverage_score.mean(axis=1), coverage_score.std(axis=1)], axis=1)
    coverage_score.columns = ['coverage_imp_' + str(v) for v in coverage_score.columns[:-2]] + ['mean_coverage_imp', 'sd_coverage_imp']

    pd.concat([imp, enrichment_score, coverage_score], axis=1).to_csv(output_fname, sep="\t", float_format='%.4f')
