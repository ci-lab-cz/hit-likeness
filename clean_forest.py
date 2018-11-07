#!/usr/bin/env python3
#==============================================================================
# author          : Pavel Polishchuk
# date            : 07-11-2018
# version         : 
# python_version  : 
# copyright       : Pavel Polishchuk 2018
# license         : 
#==============================================================================

import argparse
import pickle
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean forest model to remove unnessesary information and get files '
                                                 'of a small size.')
    parser.add_argument('-i', metavar='MODEL_LIST', nargs='*', required=True,
                        help='list of model names to clean. Output models will be stored at the same location '
                             'with a suffix _clean')

    args = vars(parser.parse_args())
    for o, v in args.items():
        if o == "i": fnames = v

    for fname in fnames:
        forest = pickle.load(open(fname, 'rb'))
        for tree in forest:
            del tree.node[-1]['mol_names']
        pickle.dump(forest, open(os.path.splitext(fname)[0] + '_clean.pkl', 'wb'))
