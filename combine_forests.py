import glob
import pickle
import os

# dir_name = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_p3000_c1000_alg2_m3_t5_realx/'
# dir_name = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_p3000_c1000_alg3_m3_t5_realx/'
# dir_name = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_p3000_c1000_alg2_m3_t5_realx_cellbased'
dir_name = '/home/pavel/QSAR/pmapper/nconf/tree/rdkit-desc/49assays_no_pains_fh/forest_p3000_c1000_alg2_m3_t5_realx_biochemprotein'

flist = [file for file in glob.glob1(dir_name, "forest_*.pkl")]

trees = []
for fname in flist:
    trees.extend(pickle.load(open(os.path.join(dir_name, fname), 'rb')))

fname = 'forest.pkl'
pickle.dump(trees, open(os.path.join(dir_name, fname), 'wb'))
