#!/usr/bin/env python3
# author          : Pavel Polishchuk
# license         : BSD-3
#==============================================================================

__author__ = 'Pavel Polishchuk'

import os
import argparse
import pickle
import gzip
import pandas as pd
from multiprocessing import Pool, cpu_count
from physchem_calc import calc_mp, calc, descriptor_names
from forest_predict import supply_data, predict_tree_mp, predict_tree


class PredictHTS:

    def __init__(self, model_fname, ncpu, chunk_size=100000):
        if model_fname.endswith('.gz'):
            self.model = pickle.load(gzip.open(model_fname))
        else:
            self.model = pickle.load(open(model_fname, 'rb'))
        if ncpu > 1:
            self.pool = Pool(max(1, min(cpu_count(), ncpu)))
        else:
            self.pool = None
        self.chunk_size = chunk_size

    def __get_chunk(self, smiles):
        for i in range(0, len(smiles), self.chunk_size):
            yield smiles[i:i + self.chunk_size]

    def __get_chunk_from_file(self, smi_fname):
        output = []
        with open(smi_fname) as f:
            for line in f:
                items = line.strip().split()
                if len(items) == 1:
                    output.append((items[0], items[0]))
                elif len(items) > 1:
                    output.append((items[0], items[1]))
                if len(output) == self.chunk_size:
                    yield output
                    output = []
            yield output

    def __calc_desciptors(self, smiles):
        """

        :param smiles: can be a list of tuples (SMILES, NAME) or list of SMILES
        :return: pandas dataframe with calculated descriptors
        """
        output = []
        if isinstance(smiles[0], str):
            smiles = [(s, s) for s in smiles]
        if self.pool is not None:
            for res in self.pool.imap(calc_mp, smiles, chunksize=100):
                if res:
                    output.append(res)
        else:
            for smi, name in smiles:
                res = calc(smi, name)
                if res:
                    output.append(res)
        df = pd.DataFrame.from_records(output).set_index(0)
        df.columns = descriptor_names
        return df

    def __predict(self, x):
        if self.pool is not None:
            pred = list(self.pool.imap(predict_tree_mp, supply_data(self.model, x)))
        else:
            pred = []
            for tree in self.model:
                pred.append(predict_tree(tree, x))
        pred = pd.concat(pred, axis=1)
        return pred.mean(axis=1).round(3).to_frame()

    def predict(self, smiles):
        pred = []
        for smiles_chunk in self.__get_chunk(smiles):
            x = self.__calc_desciptors(smiles_chunk)
            pred.append(self.__predict(x))
        pred = pd.concat(pred)
        pred.columns = ['HTS-likeness']
        return pred

    def predict_from_file(self, smi_fname):
        pred = []
        for smiles_chunk in self.__get_chunk_from_file(smi_fname):
            x = self.__calc_desciptors(smiles_chunk)
            pred.append(self.__predict(x))
        pred = pd.concat(pred)
        pred.columns = ['HTS-likeness']
        return pred

    def predict_from_file2(self, smi_fname):
        for smiles_chunk in self.__get_chunk_from_file(smi_fname):
            x = self.__calc_desciptors(smiles_chunk)
            pred = pd.concat([self.__predict(x)])
            pred.columns = ['HTS-likeness']
            yield pred


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make prediction with Random Forest model.')
    parser.add_argument('-i', '--input', metavar='input.smi', required=True,
                        help='text file with SMILES. Single column or two columns with SMILES and compound name '
                             '(whitespace-separated). No header.')
    parser.add_argument('-m', '--model', metavar='model.pkl', required=True,
                        help='file with a pickled (optionally gzipped) model.')
    parser.add_argument('-o', '--output', metavar='predictions.txt', required=True,
                        help='text file with predicted values.')
    parser.add_argument('-n', '--chunk_size', metavar='NUMBER', required=False, default=100000, type=int,
                        help='number of simultaneously processing molecules. It will make prediction of large sets '
                             'more efficient and feasible. Default: 100000.')
    parser.add_argument('-c', '--ncpu', metavar='NUMBER', required=False, default=1, type=int,
                        help='number of CPU to use. Default: 1.')

    args = parser.parse_args()

    predict_hts = PredictHTS(model_fname=args.model, ncpu=args.ncpu, chunk_size=args.chunk_size)

    if os.path.isfile(args.output):
        os.remove(args.output)

    for i, pred in enumerate(predict_hts.predict_from_file2(args.input)):
        if i > 0:
            pred.to_csv(args.output, sep='\t', mode='a', header=False)
        else:
            pred.to_csv(args.output, sep='\t', mode='a')
