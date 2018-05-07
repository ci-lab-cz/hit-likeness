#!/usr/bin/env python3

import argparse
import numpy as np
import networkx as nx
import pickle
import math
import os
from multiprocessing import Pool, cpu_count


def load_y(fname):
    y = []
    mol_names = []
    with open(fname) as f:
        assays = f.readline().strip().split('\t')[1:]
        for line in f:
            items = line.strip().split()
            mol_names.append(items[0])
            y.append([int(i) if i != 'NA' else None for i in items[1:]])
    return np.array(y, dtype=np.float), np.array(mol_names), assays


def load_x(fname):
    x = []
    with open(fname) as f:
        var_names = f.readline().strip().split()[1:]
        for line in f:
            items = line.strip().split()
            try:
                x.append(list(map(int, items[1:])))
            except ValueError:
                x.append(list(map(float, items[1:])))
    return np.array(x), np.array(var_names)


def hit_rate(b):
    b = list(b[np.logical_not(np.isnan(b))])
    if len(b) > 0:
        return sum(b) / len(b)
    else:
        return 0


def enrichment(y, ref, fun):
    """
    returns enrichment value across multiple assays
    :param y: numpy array N mols (rows) x M assays (cols) with 1/0/nan
    :param ref: numpy array M reference hit rates
    :param fun: function to summarise enrichment across assays (mean, median, etc)
    :return:
    """
    e = np.apply_along_axis(hit_rate, 0, y) / ref
    return fun(e)


def find_x_split(x1, y, ref, fun, min_num):
    # max_diff = 0
    max_diff = float('inf')
    opt_threshold = None
    e1 = None
    e2 = None
    u = np.sort(np.unique(x1))
    for i in u[np.logical_not(np.isnan(u))][:-1]:
        if sum(x1 <= i) >= min_num and sum(x1 > i) >= min_num:
            h1 = enrichment(y[x1 <= i, :], ref, fun)
            h2 = enrichment(y[x1 > i, :], ref, fun)
            # score = abs(h1 * sum(x1 <= i) - h2 * sum(x1 > i))
            # score = max(h1 * sum(x1 <= i), h2 * sum(x1 > i))
            # score = max(sum(x1 <= i) ** h1, sum(x1 > i) ** h2)
            # score = max(h1 * math.log10(sum(x1 <= i)), h2 * math.log10(sum(x1 > i)))
            # score = max(h1, h2)
            # score = min((h1 + 0.000001) * sum(x1 <= i), (h2 + 0.000001) * sum(x1 > i))
            score = min(h1, h2)
            if score < max_diff:
                max_diff = score
                opt_threshold = i
                e1 = h1
                e2 = h2
    return max_diff, opt_threshold, e1, e2


def grow(x, y, ref, fun, parent_id, min_parent_num, min_child_num, mol_names, x_var_names, ncpu, verbose, tree):
    """

    :param x: numpy array N mols (rows) x P descriptors (cols)
    :param y: numpy array N mols (rows) x M assays (cols) with 1/0/nan
    :param ref: numpy array M reference hit rates
    :param fun: function to summarise enrichment across assays (mean, median, etc)
    :param parent_id: id of the parent node
    :param min_parent_num: minimum number of mols in a node to split
    :param tree: the tree built
    :return:
    """
    if x.shape[0] >= min_parent_num:
        res = []
        with Pool(ncpu) as p:
            for j, output in enumerate(p.starmap(find_x_split, [(x[:,i], y, ref, fun, min_child_num) for i in range(x.shape[1])])):
                res.append(output + (j,))
        max_diff, opt_threshold, e1, e2, i = sorted(res)[0]
        if opt_threshold is not None:
            ids = x[:, i] <= opt_threshold
            left_id = len(tree)
            right_id = left_id + 1
            tree.add_node(left_id, parent=parent_id, score=max_diff, treshold=opt_threshold, mol_names=mol_names[ids], nmols=sum(ids), rule='%s <= %f' % (x_var_names[i], opt_threshold), enrichment=e1)
            tree.add_node(right_id, parent=parent_id, score=max_diff, treshold=opt_threshold, mol_names=mol_names[np.logical_not(ids)], nmols=sum(np.logical_not(ids)), rule='%s > %f' % (x_var_names[i], opt_threshold), enrichment=e2)
            tree.add_edge(parent_id, left_id)
            tree.add_edge(parent_id, right_id)
            if verbose:
                print(parent_id, left_id, tree.node[left_id]['rule'], tree.node[left_id]['nmols'], tree.node[left_id]['enrichment'])
            grow(x[ids], y[ids], ref, fun, left_id, min_parent_num, min_child_num, mol_names[ids], x_var_names, ncpu, verbose, tree)
            if verbose:
                print(parent_id, right_id, tree.node[right_id]['rule'], tree.node[right_id]['nmols'], tree.node[right_id]['enrichment'])
            grow(x[np.logical_not(ids)], y[np.logical_not(ids)], ref, fun, right_id, min_parent_num, min_child_num, mol_names[np.logical_not(ids)], x_var_names, ncpu, verbose, tree)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create a decision tree.')
    parser.add_argument('-x', metavar='descriptors.txt', required=True,
                        help='text file with descriptors (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-y', metavar='activity.txt', required=True,
                        help='text file with activity values 0/1/NA (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-o', '--output', metavar='output.pkl', required=False, default=None,
                        help='pickled tree object (networkx). If missing the file will be stored with automatically '
                             'generated name in the dir with the descriptor input file. Default: None.')
    parser.add_argument('-p', '--min_parent', metavar='INTEGER', required=False, default=10000,
                        help='minimum number of items in parent node to split. Default: 10000.')
    parser.add_argument('-c', '--min_child', metavar='INTEGER', required=False, default=100,
                        help='minimum number of items in child node to create. Default: 100.')
    parser.add_argument('-n', '--ncpu', metavar='INTEGER', required=False, default=1,
                        help='number of CPUs used to built a tree. Default: 1.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print progress.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = v
        if o == "y": y_fname = v
        if o == "min_child": min_child_num = int(v)
        if o == "min_parent": min_parent_num = int(v)
        if o == "output": out_fname = v
        if o == "verbose": verbose = v
        if o == "ncpu": ncpu = min(int(v), cpu_count())

    # y_fname = "/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/mlsmr_act_ordered_nopains_freqhit.txt"
    y, mol_names, assay_names = load_y(y_fname)

    # x_fname = "/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/mlsmr_physchemprop_rdkit_bin_ordered_nopains_freqhit.txt"
    x, var_names = load_x(x_fname)

    ref_hit_rate = np.apply_along_axis(hit_rate, 0, y)

    tree = nx.DiGraph()

    # min_parent_num = 10000
    # min_child_num = 100

    grow(x, y, ref_hit_rate, np.median, -1, min_parent_num, min_child_num, mol_names, var_names, ncpu, verbose, tree)

    if out_fname is None:
        out_fname = os.path.join(os.path.dirname(x_fname),
                                 "tree_%s_p%i_c%i_alg7.pkl" %
                                 (os.path.basename(x_fname).rsplit('.', 1)[0],
                                  min_parent_num,
                                  min_child_num))
    pickle.dump(tree, open(out_fname, 'wb'))
