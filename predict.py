#!/usr/bin/env python3
#==============================================================================
# author          : Pavel Polishchuk
# date            : 29-06-2019
# version         : 
# python_version  : 
# copyright       : Pavel Polishchuk 2019
# license         : 
#==============================================================================

import pickle
import pandas as pd
from multiprocessing import Pool, cpu_count
from physchem_calc import calc_mp, calc, descriptor_names
from forest_predict import supply_data, predict_tree_mp, predict_tree


class PredictHTS:

    def __init__(self, model_fname, ncpu, chunk_size=100000):
        self.model = pickle.load(open(model_fname, 'rb'))
        if ncpu > 1:
            self.pool = Pool(max(1, min(cpu_count(), ncpu)))
        else:
            self.pool = None
        self.chunk_size = chunk_size

    def __get_chunk(self, smiles):
        for i in range(0, len(smiles), self.chunk_size):
            yield smiles[i:i + self.chunk_size]

    def __calc_desciptors(self, smiles):
        output = []
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
