import numpy as np
import sys
sys.path.append("..")

import numpy as np

import sys
import missing_process.missing_method as missing_method
from missing_process.block_rules import *
import json
import matplotlib.pyplot as plt

import torch

def load_train_test(index_file,missingtype,rule_name,directory_path,dataname):  
    
    train_index = index_file["train_index"]
    test_index = index_file["test_index"]

    norm_values = np.load(f'{directory_path}/{dataname}_norm.npy')
    observed_masks = np.load(f'{directory_path}/{missingtype}/{rule_name}.npy')

    train_values = norm_values[train_index,:]

    train_masks = observed_masks[train_index,:]

    test_values = norm_values[test_index,:]

    test_masks = observed_masks[test_index,:]

    return train_values,mask_check(train_masks),test_values,mask_check(test_masks)




def load_train_test_val(index_file,missingtype,rule_name,directory_path,dataname):  
    
    train_index = index_file["train_index"]
    test_index = index_file["test_index"]
    val_index = index_file["valid_index"]

    norm_values = np.load(f'{directory_path}/{dataname}_norm.npy')
    observed_masks = np.load(f'{directory_path}/{missingtype}/{rule_name}.npy')

    train_values = norm_values[train_index,:]

    train_masks = observed_masks[train_index,:]

    test_values = norm_values[test_index,:]

    test_masks = observed_masks[test_index,:]

    val_values = norm_values[val_index,:]

    val_masks = observed_masks[val_index,:]


    return train_values,mask_check(train_masks),test_values,mask_check(test_masks),val_values,mask_check(val_masks)


def make_plot(norm_value,mask, title = None):
    import seaborn as sns
        # Create a 4x2 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes = axes.ravel()  # Flatten the 4x2 grid for easy iteration
    masked_value = mask

    for i in range(norm_value.shape[1]):
        norm_column = norm_value[:, i]
        mask_column = masked_value[:, i]

        bins = np.histogram_bin_edges([norm_column, mask_column], bins='auto')
        #data_column2 = array_list[1][i]
        # Histogram
        #axes[i].hist(data_column, bins=30, alpha=0.7, edgecolor='black',kde = True)
        sns.histplot(data=norm_column, bins=bins, color='blue',alpha = 0.4, ax=axes[i], kde=True, label='Complete Data',
                     hatch="/", 
                     fill=False
                     )
        sns.histplot(data=mask_column, bins=bins, color='green',alpha = 0.6, ax=axes[i], kde=True, label='Imputed Data',
                     hatch='...', 
                     fill=False
                     )
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Histogram for Column {i+1}')
        axes[i].legend()

    if title is not None:
        fig.suptitle(title, fontsize=16)
        # Adjust the layout
    plt.tight_layout()
    # plt.savefig(f"../plot/{title}.png")
    plt.show()


def RMSE(mask,complete,imputed):

    missing_data_points = mask == 0
    
    # Extracting the originally missing data from both complete and imputed data
    original_values = complete[missing_data_points]
    imputed_values = imputed[missing_data_points]

    # Calculating RMSE
    rmse = np.sqrt(np.mean((original_values - imputed_values) ** 2))

    return rmse

def mask_check(array):
# Iterate over each column
    for i in range(array.shape[1]):
        if np.all(array[:, i] == 0):
            array[0, i] = 1
    return array

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

'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''
 
# Necessary packages
import numpy as np
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities



def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data


def rounding (imputed_data, data_x):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data


def rmse_loss (ori_data, imputed_data, data_m):
  '''Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness
    
  Returns:
    - rmse: Root Mean Squared Error
  '''
  
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
    
  # Only for missing values
  nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
  denominator = np.sum(1-data_m)
  
  rmse = np.sqrt(nominator/float(denominator))
  
  return rmse


def xavier_init(size):
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev)
      

def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])       


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx
  

  

def na_data_loader(values,masks):
  miss_data_x = values.copy()
  miss_data_x[masks == 0] = np.nan
  return values, miss_data_x, masks


def ot_nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)



#### Accuracy Metrics ####
def ot_MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        MAE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def ot_RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        RMSE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())


import numpy as np

from torch.utils.data import DataLoader, Dataset


class tabular_dataset(Dataset):
    # eval_length should be equal to attributes number.
    def __init__(
        self,index_list,data_name="syn1",directory_path=5,
        miss_type = "logistic",rule_name = "0.1"):

        self.index = index_list
        
        norm_values = np.load(f'{directory_path}/{data_name}_norm.npy')
        observed_masks = np.load(f'{directory_path}/{miss_type}/{rule_name}.npy')
        self.eval_length = norm_values.shape[1]

        self.observed_values = norm_values
        self.gt_masks = observed_masks
        self.observed_masks = np.ones_like(norm_values)


    def __getitem__(self, org_index):
        index = self.index[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.index)



#def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.1):
def tabcsdi_get_dataloader(index_file,directory_path,data_name="syn1",miss_type = "logistic",rule_name = "0.1",
                   batch_size=32):
    
    train_index = index_file["train_index"]
    test_index = index_file["test_index"]
    val_index = index_file["valid_index"]

    # Create datasets and corresponding data loaders objects.
    train_dataset = tabular_dataset(
        train_index,data_name,directory_path=directory_path,
        miss_type = miss_type,rule_name = rule_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    valid_dataset = tabular_dataset(
        val_index,data_name,directory_path=directory_path,
        miss_type = miss_type,rule_name = rule_name)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = tabular_dataset(
        test_index,data_name,directory_path=directory_path,
        miss_type = miss_type,rule_name = rule_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(valid_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader

