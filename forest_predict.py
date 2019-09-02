import pandas as pd
import pickle
import argparse
import os
from multiprocessing import Pool, cpu_count
from forest import predict_tree


def predict_tree_mp(items):
    return predict_tree(*items)


def supply_data(model, x):
    for tree in model:
        yield tree, x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make prediction with Random Forest model.')
    parser.add_argument('-x', metavar='descriptors.txt', required=True,
                        help='text file with descriptors (tab-separated).'
                             'Header is present. The first column contains compound names.')
    parser.add_argument('-m', '--model', metavar='model.pkl', required=True,
                        help='file with a pickled model.')
    parser.add_argument('-p', '--prediction', metavar='predictions.txt', required=False, default=None,
                        help='text file with predicted values. Default: None.')
    parser.add_argument('-o', '--oob', metavar='oob_predictions.txt', required=False, default=None,
                        help='text file with predicted values. X values for the training set of the model '
                             'should be supplied to get correct results. Default: None.')
    parser.add_argument('-u', '--cumulative', action='store_true', default=False,
                        help='to make cumulative predictions: the first column is predictions for the first tree, '
                             'the second one - the first two tress, and so on. This option is only needed if one wants '
                             'to track changes in accuracy predictions with increasing number of trees in the model.')
    parser.add_argument('-s', '--sd', action='store_true', default=False,
                        help='set this argument to calculate standard deviation among predictions of individual trees '
                             'which can be used as a measure of applicability domain. If -u agrument was set the '
                             'sd argument will be ignored.')
    parser.add_argument('-c', '--ncpu', metavar='NUMBER', required=False, default=1,
                        help='number of CPU to use. Default: 1.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = v
        if o == "model": model_fname = v
        if o == "prediction": pred_fname = v
        if o == "oob": oob_fname = v
        if o == "cumulative": cumulative = v
        if o == "ncpu": ncpu = int(v)
        if o == "sd": sd = v

    if pred_fname is None and oob_fname is None:
        raise ValueError('at least one of outputs should be specified: prediction for the whole set or for oob.')

    if pred_fname is not None and os.path.isfile(pred_fname):
        os.remove(pred_fname)

    if oob_fname is not None and os.path.isfile(oob_fname):
        os.remove(oob_fname)

    if cumulative:
        sd = False

    pool = Pool(min(ncpu, cpu_count())) if ncpu > 1 else None

    model = pickle.load(open(model_fname, 'rb'))

    for x in pd.read_table(x_fname, sep="\t", index_col=0, chunksize=100000):

        if pool is not None:
            pred = list(pool.imap(predict_tree_mp, supply_data(model, x)))
        else:
            pred = []
            for tree in model:
                pred.append(predict_tree(tree, x))
        pred = pd.concat(pred, axis=1)
        pred.columns = list(range(pred.shape[1]))

        if pred_fname:
            if cumulative:
                pred_cum = pred.cumsum(1).divide(pd.Series(list(range(1, pred.shape[1] + 1))))
                pred_cum.columns = list(range(1, pred_cum.shape[1] + 1))
                pred_result = pred_cum.round(3)
            else:
                tmp = pred.mean(axis=1).round(3).to_frame(name=len(model))
                if not sd:
                    pred_result = tmp
                else:
                    tmp_sd = pred.std(axis=1).round(3).to_frame(name='sd')
                    pred_result = pd.concat([tmp, tmp_sd], axis=1)

            if os.path.isfile(pred_fname):
                pred_result.to_csv(pred_fname, sep='\t', mode='a', header=False)
            else:
                pred_result.to_csv(pred_fname, sep='\t')

        if oob_fname:
            for i in range(len(model)):
                pred.loc[pred.index.isin(model[i].node[-1]['mol_names']), i] = None
            if cumulative:
                pred_cum = pred.cumsum(1).divide(pd.Series(list(range(1, pred.shape[1] + 1))))
                pred_cum.columns = list(range(1, pred_cum.shape[1] + 1))
                oob_result = pred_cum.round(3).fillna(axis=1, method='ffill')
            else:
                tmp = pred.mean(axis=1).round(3).to_frame(name=len(model))
                if not sd:
                    oob_result = tmp
                else:
                    tmp_sd = pred.std(axis=1).round(3).to_frame(name='sd')
                    oob_result = pd.concat([tmp, tmp_sd], axis=1)

            if os.path.isfile(oob_fname):
                oob_result.to_csv(oob_fname, sep='\t', mode='a', header=False)
            else:
                oob_result.to_csv(oob_fname, sep='\t')
