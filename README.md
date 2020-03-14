# Hit-likeness prediction

This software is for development of models to predict likelihood of compounds to be hits in in vitro assays.  

It includes three models trained on a large PubChem data set consisting of 270k compounds tested in 46 biochemical and cell-based confirmatory assays and common physicochemcial descriptors. These models predict overall hit-likeness as well as biochemical and cell-based ones. The latter two were trained on subsets of assays of the corresponding format.  

The main idea behind is to identify regions of chemical space where hits were found more frequently across multiple assays. Frequent hitters were removed from the training set to avoid a bias. The models were built using the RandomForest algorithm with a custom cost function. These models were successfully validated on three external test sets (the set of 88k compounds tested in 49 confirmatory PubChem assays, the set of 159k compounds tested in 272 PubChem primary assays and the set of 45k compounds tested in 70 NCI cytotoxicity assays) and demonstrated their ability to increase hit rates across multiple assays.  

The predicted hit-likeness is within the range from 0 to a large number which indicates average hit rate enrichment of training sets compounds having similar physicochemical properties to your test set molecules.  

Hit-likeness cen be used to rank compounds for their selection for in vitro testing and will be especially useful for design of diverse libraries to excluded compounds with low likelihood to be hits in in vitro assays.  

More details will be provided in the upcoming publication which link will be inserted here as it will be available.  

## Dependency

networkx (version 2) should be installed

## How to use:

### Prediction with available models:
Three pre-trained models can be found in the directory models to predict overall, biochemcial and cell-based hit-likeness. 

- using pre-calculated descriptors  
it is useful to make prediction by multiple models to avoid repeated calculation of descriptors for a large data set; the supplied model can be gzipped (model.pkl.gz)

```
physchem_calc.py -i input.smi -o descriptors.txt -c 2
forest_predict.py -x descriptors.txt -m model.pkl.gz -p predictions.txt -c 2
```

- using SMILES file  
it will compute descriptors on the fly and make prediction; optimized for very large data sets due to processing of the input file by chunks

```
predict.py -i input.smi -m model.pkl.gz -o predictions.txt -c 2
```

- using custom scripts  
one may import PredictHTS class from predict.py and use it for prediction within own Python scripts


### Model building:

- generate descriptors  
```
physchem_calc.py -i input.smi -o descriptors.txt -c 2
```

- build model  
```
forest.py -x descriptors.txt -y y.txt -o model.pkl -c 2
```

y.txt - is a tab-separated text file where the first column contains molecule names, the first row contains name of assays and cell are filled with 0 (inactive) and 1 (active). Missing values can be labeled as NA, but it is not recommended to train model on data sets with a lot of missing data because this will create substantial bias.  
The script consumes a lot of memory depending on the number of compounds and assays, the number of cores should be specified wisely.  

- clean model (optional)  
original model file stores additional information which is not required for making predictions, therefore one may remove this information and get a much smaller file size; script can take multiple models at once and will return stripped models with the added suffix _clean to file names

```
clean_forest.py -i model1.pkl model2.pkl
```

- out-of-bag set prediction (robustness of models)  
only works with not stripped models

```
forest_predict.py -x descriptors_test.txt -m model.pkl -o oob_predictions.txt -c 2
```

- test set prediction  
see above

- calculation of statistics  
it uses y.txt file with actual values and predictions.txt with predicted hit-likeness; statistics can be calculated for data sets with missing values (where some compounds were not tested in some assays)  

this will calculated overall statistics using hit-likeness threshold 1.2
```
calc_stat.py -y y.txt -p predictions.txt -s stat.txt -t 1.2
```

this will calculated statistics per assay using hit-likeness threshold 1.2  
```
calc_stat.py -y y.txt -p predictions.txt -a assay_stat.txt -t 1.2
```

- estimate variable importance  
use train set data and a not stripped model, one may specify a number of permutation rounds and a number of CPUs  
the scripts saves many statistics, you mainly need mean_enrichment_imp and sd_enrichment_imp columns  

```
calc_importance.py -x descriptors.txt -y y.txt -m model.pkl -o importance.txt -r 10 -c 2
```
