"""
Use the MIWAE and not-MIWAE on UCI data
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())
from MIWAE import MIWAE
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from missing_process.block_rules import *
from utils import load_train_test_val,na_data_loader
import json
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_directory)
import sys
sys.path.append("..")
from tqdm import tqdm
import argparse



# ---- choose the missing model
# mprocess = 'linear'
# mprocess = 'selfmasking'
mprocess = 'selfmasking_known'

def train(model, batch_size, max_iter=2000, name=None):

    if name is not None:
        model.save(name)

    #start = time.time()
    best = float("inf")

    for i in range(max_iter):
        loss = model.train_batch(batch_size=batch_size)

        if i % 100 == 0:
            #took = time.time() - start
            #start = time.time()

            val_loss = model.val_batch()

            if val_loss < best and name is not None:
                best = val_loss
                model.save(name)
            # print("{0}/{1} updates, {2:.2f} s, {3:.2f} train_loss, {4:.2f} val_loss"
            #       .format(i, max_iter, took, loss, val_loss))
            sys.stdout.flush()


def imputationRMSE(model, Xorg, Xz, X, S, L):
    """
    #(miwae, Xtrain(full data), Xz(X_0), Xnan(X_na), S(X_mask), L)[0])
    Imputation error of missing data
    """
    N = len(X)

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]

    def imp(model, xz, s, L):
        l_out, log_p_x_given_z, log_p_z, log_q_z_given_x = model.sess.run(
            [model.l_out_mu, model.log_p_x_given_z, model.log_p_z, model.log_q_z_given_x],
            {model.x_pl: xz, model.s_pl: s, model.n_pl: L})

        wl = softmax(log_p_x_given_z + log_p_z - log_q_z_given_x)

        xm = np.sum((l_out.T * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)

        return l_out, wl, xm, xmix

    XM = np.zeros_like(Xorg)

    for i in range(N):

        xz = Xz[i, :][None, :]
        s = S[i, :][None, :]

        l_out, wl, xm, xmix = imp(model, xz, s, L)

        XM[i, :] = xm

        # if i % 100 == 0:
        #     print('{0} / {1}'.format(i, N))

    return XM


def main(args, seed = 1, nfold = 5):


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


    path = f"../impute/{miss_type}/{data_name}/MIWAE"
    if not os.path.exists(path):
        # If the path does not exist, create it
        os.makedirs(path)

    for rule_name in tqdm(missing_rule):
        print("Rule name:",rule_name)

        directory_path = f"../datasets/{data_name}"  
        # Opening JSON file
        f = open(f'{directory_path}/split_index_cv_seed-{seed}_nfold-{nfold}.json')
        index_file = json.load(f)
        for fold in tqdm(index_file):
        
            train_values,train_masks,test_values,test_masks,val_values,val_masks = load_train_test_val(index_file[fold],miss_type,rule_name,directory_path,data_name)

            train_x, train_na, train_mask = na_data_loader(train_values,train_masks)
            test_x, test_na, test_mask = na_data_loader(test_values,test_masks)
            val_x, val_na, val_mask = na_data_loader(test_values,test_masks)

            train_0 = np.nan_to_num(train_na, nan=0)
            test_0 = np.nan_to_num(test_na, nan=0)

            N, D = train_x.shape

            dl = D - 1

            for _ in range(args.n_run):
                
                imputed_list_train = []
                imputed_list_test = []

                # ------------------- #
                # ---- fit MIWAE ---- #
                # ------------------- #
                miwae = MIWAE(train_na, val_na, n_latent=dl, n_samples=args.n_samples, n_hidden=args.n_hidden, name=args.data_name)

                # ---- do the training
                train(miwae, batch_size=args.batch_size, max_iter=args.max_iter, name=args.data_name + 'miwae')

                # ---- find imputation RMSE
                imputed_list_train.append(imputationRMSE(miwae, train_x, train_0, train_na, train_mask, args.L))
                imputed_list_test.append(imputationRMSE(miwae, test_x, test_0, test_na, test_mask, args.L))

            impute_train = np.mean(imputed_list_train, axis=0)
            impute_test = np.mean(imputed_list_test, axis=0)

            np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_train.npy', impute_train.astype("float32"))
            np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_test.npy', impute_test.astype("float32"))




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
parser.add_argument(
      '--n_run',
      help='number of runtime',
      default=1,
      type=int)
parser.add_argument(
      '--n_hidden',
      default=128,
      type=int)
parser.add_argument(
      '--n_samples',
      default=20,
      type=int)

parser.add_argument(
      '--max_iter',
      default=2000,
      type=int)

parser.add_argument(
      '--batch_size',
      default=32,
      type=int)

parser.add_argument(
      '--L',
      default=10000,
      type=int)
args = parser.parse_args()





main(args)
