# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

#from data_loader import data_loader
from gain import gain
from utils import rmse_loss
import os
from tqdm import tqdm
from utils import load_train_test,na_data_loader
import sys
sys.path.append("..")
from missing_process.block_rules import *

#GAIN 0->Missing, 1-> Presenting, input mask and data with nan

real_datalist = [
            "banknote",
             "concrete_compression",
             "wine_quality_white",
            "wine_quality_red",
            "california",
            "climate_model_crashes",
            "connectionist_bench_sonar",
            "qsar_biodegradation", 
            "yeast", 
            "yacht_hydrodynamics"
            ]
syn_datalist = ["syn1"]
missingtypelist = ["quantile",
                   "diffuse",
                   "logistic"
                   ]

seed = 1
nfold = 5

def main (args):
  '''Main function for  datasets.
  
  Args:
    - data_name: from real_datalist or syn_datalist
    - miss_type: missingdata type
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_type = args.miss_type
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness

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

  path = f"../impute/{miss_type}/{data_name}/GAIN"
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

      # Impute missing data
      imputed_train_x,imputed_test_x = gain(train_na, gain_parameters,test_na)
      np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_train.npy', imputed_train_x.astype("float32"))
      np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_test.npy', imputed_test_x.astype("float32"))
  
  return imputed_train_x,imputed_test_x

  

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      # choices=[
      #       "banknote",
      #        "concrete_compression",
      #        "wine_quality_white",
      #       "wine_quality_red",
      #       "california",
      #       "climate_model_crashes",
      #       "connectionist_bench_sonar",
      #       "qsar_biodegradation", 
      #       "yeast", 
      #       "yacht_hydrodynamics","syn1"
      #       ],
      default="syn1",
      type=str)
  parser.add_argument(
      '--miss_type',
      help='missing data type',
      # choices=["quantile",
      #              "diffuse",
      #              "logistic"
      #              ],
      default="logistic",
      type=str)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=32,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=2000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_train_x,imputed_test_x = main(args)



