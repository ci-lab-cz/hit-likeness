#!/usr/bin/env python3

import numpy as np
import argparse
import os
from tree_mp import hit_rate, load_x, load_y


def read_rules(fname):
    """

    :param fname:
    :return: dict of dicts with lower and upper values for each variable
             {'rule1': {'HBD': (1,3), 'HBA': (3,7), ...}, ...}
    """
    rules = {}
    with open(fname) as f:
        for line in f:
            if line.strip():
                name, rule = line.strip().split('\t')
                d = {}
                for r in rule.split(' and '):
                    items = r.split(' ')
                    d[items[2]] = (float(items[0]), float(items[4]))
                rules[name] = d
    return rules


def select(x, var_names, rule):
    """

    :param x:
    :param var_names:
    :param rule:
    :return: boolean numpy array with compounds matching the given rule
    """
    ids = []
    for k, v in rule.items():
        ids.append((v[0] < x[:, var_names.index(k)]) & (x[:, var_names.index(k)] <= v[1]))
    ids = np.array(ids)
    ids = np.all(ids, axis=0)
    return ids


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make compounds selection based on rules.')
    parser.add_argument('-x', metavar='descriptors.txt', required=True,
                        help='text file with descriptors (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-y', metavar='activity.txt', required=True,
                        help='text file with activity values 0/1/NA (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-r', '--rules', metavar='rules.txt', required=True,
                        help='text file with rules to select compounds.'
                             'No header. The first column is rule names, the second one is rules.')
    parser.add_argument('-o', '--output', metavar='output.txt', required=True,
                        help='text file to store the report.')
    parser.add_argument('-s', '--save_ids', action='store_true', default=False,
                        help='save mol names of selected compounds to separate text files named '
                             'accordingly to rule names.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = v
        if o == "y": y_fname = v
        if o == "rules": rule_fname = v
        if o == "output": out_fname = v
        if o == "save_ids": save_ids = v

    y, mol_names, assay_names = load_y(y_fname)
    x, var_names = load_x(x_fname)
    var_names = var_names.tolist()
    rules = read_rules(rule_fname)

    with open(out_fname, 'wt') as f:
        f.write('rule\tcompounds_selected\tmedian_entichment\t' + '\t'.join(assay_names) + '\n')
        for name, rule in rules.items():
            ids = select(x, var_names, rule)
            ref_hit_rate = np.apply_along_axis(hit_rate, 0, y)
            e = np.apply_along_axis(hit_rate, 0, y[ids, :]) / ref_hit_rate  # array of enrichment for each assay
            e = np.round(e, 3)
            median_e = np.round(np.median(e[np.logical_not(np.isnan(e))]), 3)  # remove nan before calc median
            print('%s %f' % (name, median_e))
            f.write('%s\t%i\t%.3f\t' % (name, sum(ids), median_e) + '\t'.join(map(str, e)) + '\n')
            if save_ids:
                np.savetxt(os.path.join(os.path.dirname(out_fname), '%s_compounds.txt' % name),
                           mol_names[ids],
                           fmt='%s')
