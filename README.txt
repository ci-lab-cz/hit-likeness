How to use

## generate descriptors

physchem_calc.py -i input.smi -o descriptors.txt -c (number of cores)

imput.smi - SMILES file, no header, tab-separated, second field is mol titles


## calc enrichment over descriptor bins:

# descriptor binning
binning.py -i descriptors.txt -o descriptors_bin.txt -t bin_thresholds.txt

bin_thresholds.txt - to setup thresholds to split on bins (maually created)

## gather statistics
get_bin_enrichment.py -x descriptors_bin.txt -y y.txt -t bin_thresholds.txt -o bins_enrichment.txt

y.txt - text files with 0(inactive)/1(active)/NA(not tested) for assays, tab-separated, first column is compound names, header contains names of assays


## To make prediction with existed rules:

tree_predict.py -x descriptors.txt -y y.txt -r rules.txt -o selection_stat.txt

rules.txt - list of rules (manually created)

optionally to retrieve compound ids for each assay add argument -s


## Build random forest model

forest_mp.py -x descriptors.txt -y y.txt -o model.pkl -t 100 -m 3 -p 3000 -n 1000 -a 2 -c 2

the script consumes a lot of memory, the number of cores should be specified carefully. However, even 1 cpu will not guarantee to avoid memory issues.


## To make prediction with random forest model

forest_predict.py -x descriptors_test.txt -y y_test.txt -m model.pkl -p predictions.txt -s stat.txt

y argument is optional, it is needed if statistics should be calculated
