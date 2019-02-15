import pandas as pd
import pickle
import argparse
from forest import predict_tree


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
    parser.add_argument('-c', '--cumulative', action='store_true', default=False,
                        help='to make cumulative predictions: the first column is predictions for the first tree, '
                             'the second one - the first two tress, and so on. This option is only needed if one wants '
                             'to track changes in accuracy predictions with increasing number of trees in the model.')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "x": x_fname = v
        if o == "model": model_fname = v
        if o == "prediction": pred_fname = v
        if o == "oob": oob_fname = v
        if o == "cumulative": cumulative = v

    if pred_fname is None and oob_fname is None:
        raise ValueError('at least one of outputs should be specified: prediction for the whole set or for oob.')

    model = pickle.load(open(model_fname, 'rb'))

    x = pd.read_table(x_fname, sep="\t", index_col=0)

    pred = []
    for tree in model:
        pred.append(predict_tree(tree, x))
    pred = pd.concat(pred, axis=1)
    pred.columns = list(range(pred.shape[1]))

    if pred_fname:
        if cumulative:
            pred_cum = pred.cumsum(1).divide(pd.Series(list(range(1, pred.shape[1] + 1))))
            pred_cum.columns = list(range(1, pred_cum.shape[1] + 1))
            pred_cum.round(3).to_csv(pred_fname, sep='\t')
        else:
            tmp = pred.mean(axis=1).round(3).to_frame(name=len(model))
            tmp.to_csv(pred_fname, sep='\t')

    if oob_fname:
        for i in range(len(model)):
            pred.loc[pred.index.isin(model[i].node[-1]['mol_names']), i] = None
        if cumulative:
            pred_cum = pred.cumsum(1).divide(pd.Series(list(range(1, pred.shape[1] + 1))))
            pred_cum.columns = list(range(1, pred_cum.shape[1] + 1))
            pred_cum.round(3).fillna(axis=1, method='ffill').to_csv(oob_fname, sep='\t')
        else:
            tmp = pred.mean(axis=1).round(3).to_frame(name=len(model))
            tmp.to_csv(oob_fname, sep='\t')
