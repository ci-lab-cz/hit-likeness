#!/usr/bin/env python3

import argparse
import numpy as np
import networkx as nx
import pickle
import os
import warnings
import pandas as pd
from multiprocessing import Pool, cpu_count


min_alg_list = [1]
max_alg_list = [2, 3]


def hit_rate(b):
    b = list(b[np.logical_not(np.isnan(b))])
    if len(b) > 0:
        return sum(b) / len(b)
    else:
        return np.nan


def enrichment(y, ref, fun):
    """
    returns enrichment value across multiple assays
    :param y: numpy array N mols (rows) x M assays (cols) with 1/0/nan
    :param ref: numpy array M reference hit rates
    :param fun: function to summarise enrichment across assays (mean, median, etc)
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        e = np.apply_along_axis(hit_rate, 0, y) / ref
    return fun(e[np.logical_not(np.isnan(e))])


def select(seq, num):
    if num > len(seq):
        num = len(seq)
    f = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
    res = [seq[i] for i in f(num, len(seq))]
    return res


def find_x_split(x1, y, ref, fun, min_num, algorithm):
    """

    :param x1: 1D numpy array with N descriptor values
    :param y: numpy array N mols (rows) x M assays (cols) with 1/0/nan
    :param ref: numpy array M reference hit rates
    :param fun: function to summarise enrichment across assays (mean, median, etc)
    :param min_num: minimum number of compounds in a child node
    :return:
    """
    if algorithm in min_alg_list:
        opt_score = float('inf')  # minimization
    elif algorithm in max_alg_list:
        opt_score = 0  # maximization
    else:
        opt_score = None

    opt_threshold = None
    e1 = None
    e2 = None
    u = np.sort(np.unique(x1))
    u = u[np.logical_not(np.isnan(u))]
    u = (u[1:] + u[:-1]) / 2  # possible splits
    if u.shape[0] > 100:
        u = select(u.tolist(), 100)  # select maximum 100 splits
    for i in u:
        if sum(x1 < i) >= min_num and sum(x1 >= i) >= min_num:
            h1 = enrichment(y.loc[x1 < i, :], ref, fun)
            h2 = enrichment(y.loc[x1 >= i, :], ref, fun)
            if algorithm == 1:
                score = min(h1, h2)
            elif algorithm == 2:
                score = max(h1, h2)
            elif algorithm == 3:
                score = abs(h1 - h2)
            # score = abs(h1 * sum(x1 <= i) - h2 * sum(x1 > i))
            # score = max(h1 * sum(x1 <= i), h2 * sum(x1 > i))
            # score = max(sum(x1 <= i) ** h1, sum(x1 > i) ** h2)
            # score = max(h1 * math.log10(sum(x1 <= i)), h2 * math.log10(sum(x1 > i)))
            # score = max(h1, h2)
            # score = min((h1 + 0.000001) * sum(x1 <= i), (h2 + 0.000001) * sum(x1 > i))
            if (algorithm in min_alg_list and score < opt_score) or \
                    (algorithm in max_alg_list and score > opt_score):  # minimization or maximization
                opt_score = score
                opt_threshold = i
                e1 = h1
                e2 = h2
    return opt_score, opt_threshold, e1, e2


def find_optimal_split(x, y, ref, fun, min_child_num, algorithm):
    res = []
    for var_name in x.columns:
        output = find_x_split(x.loc[:, var_name], y, ref, fun, min_child_num, algorithm)
        res.append(tuple((*output, var_name)))
    if algorithm in min_alg_list:
        score, threshold, e1, e2, var_name = sorted(res)[0]  # minimization
    elif algorithm in max_alg_list:
        score, threshold, e1, e2, var_name = sorted(res)[-1]  # maximization
    return score, threshold, e1, e2, var_name


def grow_tree(x, y, nvar, ref, fun, parent_id, min_parent_num, min_child_num, algorithm,
              verbose, tree):
    """

    :param x: numpy array N mols (rows) x P descriptors (cols)
    :param y: numpy array N mols (rows) x M assays (cols) with 1/0/nan
    :param nvar: number of randomly chosen variables
    :param ref: numpy array M reference hit rates
    :param fun: function to summarise enrichment across assays (mean, median, etc)
    :param parent_id: id of the parent node
    :param min_parent_num: minimum number of mols in a node to split
    :param min_child_num: minimum number of mols in a child node
    :param ncpu: number of cpus to use
    :param verbose: print progress
    :param tree: the tree built
    :return:
    """
    if x.shape[0] >= min_parent_num:
        score, threshold, e1, e2, var_name = find_optimal_split(x=x.sample(n=nvar, axis=1),
                                                                y=y,
                                                                ref=ref,
                                                                fun=fun,
                                                                min_child_num=min_child_num,
                                                                algorithm=algorithm)
        if threshold is not None:
            ids = x.loc[:, var_name] <= threshold
            left_id = len(tree)
            right_id = left_id + 1
            tree.add_node(left_id,
                          parent=parent_id,
                          score=score,
                          treshold=threshold,
                          # mol_names=mol_names[ids],
                          nmols=sum(ids),
                          rule=(var_name, '<', threshold),
                          enrichment=e1)
            tree.add_node(right_id,
                          parent=parent_id,
                          score=score,
                          treshold=threshold,
                          # mol_names=mol_names[np.logical_not(ids)],
                          nmols=sum(np.logical_not(ids)),
                          rule=(var_name, '>=', threshold),
                          enrichment=e2)
            tree.add_edge(parent_id, left_id)
            tree.add_edge(parent_id, right_id)
            if verbose:
                print(parent_id, left_id, tree.nodes[left_id]['rule'], tree.nodes[left_id]['nmols'], tree.nodes[left_id]['enrichment'])
            grow_tree(x=x[ids],
                      y=y[ids],
                      nvar=nvar,
                      ref=ref,
                      fun=fun,
                      parent_id=left_id,
                      min_parent_num=min_parent_num,
                      min_child_num=min_child_num,
                      algorithm=algorithm,
                      verbose=verbose,
                      tree=tree)
            if verbose:
                print(parent_id, right_id, tree.nodes[right_id]['rule'], tree.nodes[right_id]['nmols'], tree.nodes[right_id]['enrichment'])
            grow_tree(x=x[np.logical_not(ids)],
                      y=y[np.logical_not(ids)],
                      nvar=nvar,
                      ref=ref,
                      fun=fun,
                      parent_id=right_id,
                      min_parent_num=min_parent_num,
                      min_child_num=min_child_num,
                      algorithm=algorithm,
                      verbose=verbose,
                      tree=tree)


def predict_tree(tree, x):

    def __predict(tree, node_id, x, prediction):
        s = list(tree.successors(node_id))
        if s:
            if tree.nodes[s[0]]['rule'][1] == '<':
                case_ids = x.loc[:, tree.nodes[s[0]]['rule'][0]] < tree.nodes[s[0]]['rule'][2]
            elif tree.nodes[s[0]]['rule'][1] == '>=':
                case_ids = x.loc[:, tree.nodes[s[0]]['rule'][0]] >= tree.nodes[s[0]]['rule'][2]
            elif tree.nodes[s[0]]['rule'][1] == '>':
                case_ids = x.loc[:, tree.nodes[s[0]]['rule'][0]] > tree.nodes[s[0]]['rule'][2]
            elif tree.nodes[s[0]]['rule'][1] == '<=':
                case_ids = x.loc[:, tree.nodes[s[0]]['rule'][0]] <= tree.nodes[s[0]]['rule'][2]
            else:
                raise ValueError('Value of the inequality sign in the tree rule is not correct')
            __predict(tree, s[0], x.loc[case_ids, :], prediction)
            __predict(tree, s[1], x.loc[~case_ids, :], prediction)
        else:
            prediction.append(pd.DataFrame([tree.nodes[node_id]['enrichment']] * x.shape[0],
                                           index=x.index))

    pred = []
    __predict(tree, -1, x, pred)
    pred = pred[0].append(pred[1:])
    return pred.loc[x.index, :]   # TODO: if x.index contains duplicates result would contain more rows then expected


def predict_forest(forest, x):
    pred = []
    for tree in forest:
        pred.append(predict_tree(tree, x))
    pred = pd.concat(pred, axis=1)
    return pred.mean(axis=1)


def predict_oob(forest, x):
    pred = []
    for tree in forest:
        pred.append(predict_tree(tree, x.loc[~x.index.isin(tree.nodes[-1]['mol_names']), :]))
    return pd.concat(pred, axis=1).mean(axis=1)


def create_tree(x, y, ref_hit_rate, nvar, nsamples, min_parent_num, min_child_num, algorithm, verbose):
    tree = nx.DiGraph()
    np.random.seed()
    case_ids = np.random.choice(a=[False, True], size=x.shape[0], p=[1 - nsamples, nsamples])
    tree.add_node(-1, mol_names=x.index[case_ids])
    grow_tree(x=x.iloc[case_ids, :],
              y=y.iloc[case_ids, :],
              nvar=nvar,
              ref=ref_hit_rate,
              fun=np.nanmedian,
              parent_id=-1,
              min_parent_num=min_parent_num,
              min_child_num=min_child_num,
              algorithm=algorithm,
              verbose=verbose,
              tree=tree)
    return tree


def create_tree_mp(args):
    return create_tree(*args)


def grow_forest(x, y, ntree, nvar, nsamples, min_parent_num, min_child_num, pool, algorithm, verbose):

    ref_hit_rate = np.apply_along_axis(hit_rate, 0, y)
    forest = []
    if pool:
        for tree in pool.imap_unordered(create_tree_mp, ((x, y, ref_hit_rate, nvar, nsamples, min_parent_num, min_child_num, algorithm, verbose) for _ in range(ntree))):
            forest.append(tree)
    else:
        for _ in range(ntree):
            forest.append(create_tree(x, y, ref_hit_rate, nvar, nsamples, min_parent_num, min_child_num, algorithm, verbose))
    return forest


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create a random forest model.')
    parser.add_argument('-x', metavar='descriptors.txt', required=True,
                        help='text file with descriptors (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-y', metavar='activity.txt', required=True,
                        help='text file with activity values 0/1/NA (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-o', '--output', metavar='output.pkl', required=False, default=None,
                        help='pickled model (networkx object). If omitted the file will be stored with automatically '
                             'generated name in the dir with the descriptor input file. Default: None.')
    parser.add_argument('-t', '--ntree', metavar='INTEGER', required=False, default=200,
                        help='number of trees to build. Default: 200.')
    parser.add_argument('-m', '--nvar', metavar='INTEGER', required=False, default=3,
                        help='number of randomly chosen variables used to split nodes. '
                             'Values 0 and less indicate to use all variables. Default: 3.')
    parser.add_argument('-s', '--nsamples', metavar='INTEGER', required=False, default=0.67,
                        help='percentage of randomly chosen compounds to train each tree. Should be greater than 0 and '
                             'less or equal to 1. Default: 0.67.')
    parser.add_argument('-p', '--min_parent', metavar='INTEGER', required=False, default=3000,
                        help='minimum number of items in a parent node to split. Default: 3000.')
    parser.add_argument('-n', '--min_child', metavar='INTEGER', required=False, default=1000,
                        help='minimum number of items in a child node to create. Default: 1000.')
    parser.add_argument('-a', '--algorithm', metavar='INTEGER', required=False, default=2,
                        help='the number indicating the cost function optimized during model building. '
                             '1: minimization of min(H1, H2). 2: maximization of max(H1, H2). 3: maximization of '
                             'abs(H1 - H2). H is median hit rate enrichment in child nodes. '
                             'Default: 2.')
    parser.add_argument('-c', '--ncpu', metavar='INTEGER', required=False, default=1,
                        help='number of CPUs used to built a model. Default: 1.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='print progress.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = v
        if o == "y": y_fname = v
        if o == "ntree": ntree = int(v)
        if o == "nvar": nvar = int(v)
        if o == "nsamples": nsamples = float(v)
        if o == "min_child": min_child_num = int(v)
        if o == "min_parent": min_parent_num = int(v)
        if o == "output": out_fname = v
        if o == "algorithm": algorithm = int(v)
        if o == "verbose": verbose = v
        if o == "ncpu": ncpu = min(int(v), cpu_count())

    if nsamples <= 0 or nsamples > 1:
        print('nsamples argument should be within (0, 1] range')
        exit()

    if ncpu == 1:
        pool = None
    else:
        pool = Pool(ncpu)

    y = pd.read_table(y_fname, sep="\t", index_col=0)
    x = pd.read_table(x_fname, sep="\t", index_col=0)

    if not all(x == y for x, y in zip(sorted(x.index), sorted(y.index))):
        raise ValueError('compound names in X and Y files do not correspond.')

    x = x.reindex(y.index)

    if nvar <= 0:
        nvar = x.shape[1]

    forest = grow_forest(x=x,
                         y=y,
                         nvar=nvar,
                         nsamples=nsamples,
                         ntree=ntree,
                         min_parent_num=min_parent_num,
                         min_child_num=min_child_num,
                         pool=pool,
                         algorithm=algorithm,
                         verbose=verbose)

    if out_fname is None:
        out_fname = os.path.join(os.path.dirname(x_fname),
                                 "forest_%s_%s_t%i_v%i_p%i_c%i_alg%i.pkl" %
                                 (os.path.basename(x_fname).rsplit('.', 1)[0],
                                  os.path.basename(y_fname).rsplit('.', 1)[0],
                                  ntree,
                                  nvar,
                                  min_parent_num,
                                  min_child_num,
                                  algorithm))
    pickle.dump(forest, open(out_fname, 'wb'))
