"""
Hyper Imputer
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
from tqdm import tqdm
import argparse
sys.path.append(os.getcwd())
from ot import OTimputer
import torch 

from missing_process.block_rules import *
from utils import load_train_test,na_data_loader
import json



parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
#from missing_process.block_rules import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"





def main(args, seed = 1, nfold = 5):
    data_name = args.data_name
    miss_type = args.miss_type

    imputer = OTimputer(batchsize=32, niter=1000)

    # if miss_type == "logistic":
    #     missing_rule = load_json_file("missing_rate.json")
    # elif miss_type == "diffuse":
    #     missing_rule = load_json_file("diffuse_ratio.json")
    # elif miss_type == "quantile":
    #     missing_rule = load_json_file("quantile_full.json")


    if miss_type == "logistic":
        missing_rule = load_json_file("missing_rate.json")
    elif miss_type == "diffuse":
        missing_rule = load_json_file("diffuse_ratio.json")
    elif miss_type == "quantile":
        missing_rule = load_json_file("quantile_full.json")
    elif miss_type == "test_MNAR_1":
        missing_rule = load_json_file(f"{miss_type}.json")
        miss_type = "logistic"
    elif miss_type == "test_MNAR_2":
        missing_rule = load_json_file(f"{miss_type}.json")
        miss_type = "quantile"

    elif miss_type == "mcar" or miss_type == "mar":
        missing_rule = load_json_file("mcar.json")

    path = f"../impute/{miss_type}/{data_name}/ot"
    if not os.path.exists(path):
        # If the path does not exist, create it
        os.makedirs(path)

    for rule_name in tqdm(missing_rule):
        print("Rule name:",rule_name)

        directory_path = f"../datasets/{data_name}"  
        # Opening JSON file
        f = open(f'{directory_path}/split_index_cv_seed-{seed}_nfold-{nfold}.json')
        index_file = json.load(f)
        for fold in index_file:
        
            train_values,train_masks,test_values,test_masks = load_train_test(index_file[fold],miss_type,rule_name,directory_path,data_name)

            train_x, train_na, train_mask = na_data_loader(train_values,train_masks)
            test_x, test_na, test_mask = na_data_loader(test_values,test_masks)

            # ------------------- #
            # ----ot ---- #
            # ------------------- #
            imputed_train_x =  imputer.fit_transform(torch.from_numpy(train_na)).detach().numpy()
            imputed_test_x = imputer.fit_transform(torch.from_numpy(test_na)).detach().numpy()


            np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_train.npy', imputed_train_x.astype("float32"))
            np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_test.npy', imputed_test_x.astype("float32"))
        
    







parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument(
      '--data_name',
    #   choices=[
    #         "banknote",
    #          "concrete_compression",
    #          "wine_quality_white",
    #         "wine_quality_red",
    #         "california",
    #         "climate_model_crashes",
    #         "connectionist_bench_sonar",
    #         "qsar_biodegradation", 
    #         "yeast", 
    #         "yacht_hydrodynamics","syn1"
    #         ],
      default="syn1",
      type=str)
parser.add_argument(
      '--miss_type',
      help='missing data type',
    #   choices=["quantile",
    #                "diffuse",
    #                "logistic"
    #                ],
      default="logistic",
      type=str)
args = parser.parse_args()



# Calls main function  
main(args)

    