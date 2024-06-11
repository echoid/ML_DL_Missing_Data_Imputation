import numpy as np
import pandas as pd
import os
import sys
import argparse
import torch
import json
#import yaml
sys.path.append("..")
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from missing_process.block_rules import *
from utils import tabcsdi_get_dataloader,tabular_dataset
import json
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
from tqdm import tqdm
import argparse
from tabcsdi import TabCSDI
from tabcsdi_utils import train, evaluate



parser = argparse.ArgumentParser(description="TabCSDI")
#parser.add_argument("--config", type=str, default="tabcsdi_config.yaml")
parser.add_argument("--config", type=str, default="tabcsdi_config.json")
parser.add_argument("--device", default="cpu", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=3)
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
    #                "logistic",
    #                "quantile_1",
    #                "quantile_2",
    #                "quantile_3",
    #                "quantile_4"
    #                ],
      default="logistic",
      type=str)

args = parser.parse_args()
print(args)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

path =  args.config
with open(path, "r") as f:
    #config = yaml.safe_load(f)
    config = json.load(f)

print(config)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio



def main(args,config):

    data_name = args.data_name
    miss_type = args.miss_type
    
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
    else:
        missing_rule = load_json_file(f"{miss_type}.json")
        miss_type = "quantile"

    path = f"../impute_hpc/{miss_type}/{data_name}/tabcsdi"
    if not os.path.exists(path):
        # If the path does not exist, create it
        os.makedirs(path)

    for rule_name in tqdm(missing_rule):
        print("Rule name:",rule_name)

        directory_path = f"../datasets/{data_name}"  
        # Opening JSON file
        f = open(f'{directory_path}/split_index_cv_seed-{args.seed}_nfold-{args.nfold}.json')
        index_file = json.load(f)
        for fold in tqdm(index_file):

            # Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
            train_loader, valid_loader, test_loader = tabcsdi_get_dataloader(
                data_name=data_name,
                directory_path=directory_path,
                miss_type = miss_type,
                rule_name = rule_name,
                batch_size=config["train"]["batch_size"],
                index_file = index_file[fold],
            )

            model = TabCSDI(config, args.device).to(args.device)

            
            train(
                model,
                config["train"],
                train_loader,
                #valid_loader=valid_loader,
                )
            
            # print("---------------Start testing---------------")
            imputed_train_x = evaluate(model, train_loader, nsample=args.nsample)
            imputed_test_x = evaluate(model, test_loader, nsample=args.nsample)
            np.save(f'{path}/{rule_name}_seed-{args.seed}_{fold}_train.npy', imputed_train_x.astype("float32"))
            np.save(f'{path}/{rule_name}_seed-{args.seed}_{fold}_test.npy', imputed_test_x.astype("float32"))
        

main(args,config)