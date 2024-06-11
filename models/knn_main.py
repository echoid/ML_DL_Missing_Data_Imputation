import numpy as np
from sklearn.impute import KNNImputer
import sys
import argparse
sys.path.append("..")
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import missing_process.missing_method as missing_method
from missing_process.block_rules import *
from utils import load_train_test,make_plot,RMSE,mask_check
import json
import matplotlib.pyplot as plt
import seaborn as sns


real_datalist = ["banknote","concrete_compression",
            "wine_quality_white","wine_quality_red",
            "california","climate_model_crashes",
            "connectionist_bench_sonar","qsar_biodegradation",
            "yeast","yacht_hydrodynamics","syn1"
            ]

missingtypelist = ["quantile","diffuse","logistic"]


def imputer_model(args):
    seed = 1
    nfold = 5
    dataname = args.data_name
    missingtype = args.miss_type
    model_name = "knn"

    if missingtype == "logistic":
        missing_rule = load_json_file("missing_rate.json")
    elif missingtype == "diffuse":
        missing_rule = load_json_file("diffuse_ratio.json")
    elif missingtype == "quantile":
        missing_rule = load_json_file("quantile_full.json")
    elif missingtype == "mcar" or missingtype == "mar":
        missing_rule = load_json_file("mcar.json")

    print(dataname)
    path = f"../impute/{missingtype}/{dataname}/{model_name}"
    if not os.path.exists(path):
        # If the path does not exist, create it
        os.makedirs(path)

    for rule_name in missing_rule:

            directory_path = f"../datasets/{dataname}"  
            # Opening JSON file
            f = open(f'{directory_path}/split_index_cv_seed-{seed}_nfold-{nfold}.json')
            index_file = json.load(f)

            for fold in index_file:

                train_values,train_masks,test_values,test_masks = load_train_test(index_file[fold],missingtype,rule_name,directory_path,dataname)

                train_values_na = np.where(train_masks == 0, np.nan, train_values)
                test_values_na = np.where(test_masks == 0, np.nan, test_values)
                
                imp = KNNImputer(n_neighbors=2)
                imp.fit(train_values_na)
                test_imp = imp.transform(test_values_na)
                train_imp = imp.transform(train_values_na)


                np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_train.npy', train_imp.astype("float32"))
                np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_test.npy', test_imp.astype("float32"))

                if  (test_imp.shape != test_values_na.shape):
                    print("test shape")
                if  (train_imp.shape != train_values_na.shape):
                    print("train shape")



parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='Reproducibility')
parser.add_argument('--data_name',choices=[
            "banknote",
            "concrete_compression",
            "wine_quality_white",
            "wine_quality_red",
            "california",
            "climate_model_crashes",
            "connectionist_bench_sonar",
            "qsar_biodegradation", 
            "yeast", 
            "yacht_hydrodynamics","syn1"
            ],
    default="syn1",
    type=str)
parser.add_argument(
    '--miss_type',
    help='missing data type',
    # choices=["quantile",
    #             "diffuse",
    #             "logistic"
    #             ],
    default="logistic",
    type=str)
args = parser.parse_args()

imputer_model(args)