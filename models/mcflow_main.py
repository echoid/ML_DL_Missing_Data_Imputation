"""
Official implementation of MCFlow -
"""
import numpy as np
import torch
import argparse
import os
from mcflow_models import InterpRealNVP
from mcflow_utils import MCFLOW_DataLoader,load_data
from mcflow_models import LatentToLatentApprox
import sys
from tqdm import tqdm
sys.path.append("..")
from missing_process.block_rules import *

seed = 1
nfold = 5
def main(args):

    data_name = args.data_name
    miss_type = args.miss_type

    # Load data and introduce missingness

    if miss_type == "logistic":
        missing_rule = load_json_file("missing_rate.json")
    elif miss_type == "diffuse":
        missing_rule = load_json_file("diffuse_ratio.json")
    elif miss_type == "quantile":
        missing_rule = load_json_file("quantile_full.json")

    path = f"../impute/{miss_type}/{data_name}/mcflow"
    if not os.path.exists(path):
        # If the path does not exist, create it
        os.makedirs(path)

    for rule_name in tqdm(missing_rule):
        print("Rule name:",rule_name)

        directory_path = f"../datasets/{data_name}"  
        # Opening JSON file
        f = open(f'{directory_path}/split_index_cv_seed-{args.seed}_nfold-{nfold}.json')
        index_file = json.load(f)
        for fold in index_file:
        
            norm_values,observed_masks = load_data(miss_type,rule_name,directory_path,data_name)
            #initialize dataset class
                
            ldr = MCFLOW_DataLoader(mode=0, seed=args.seed, norm_values = norm_values,observed_masks = observed_masks,index = index_file[fold])

            data_loader = torch.utils.data.DataLoader(ldr, batch_size=args.batch_size, shuffle=False, drop_last=False)
            num_neurons = int(ldr.train[0].shape[0])

            #Initialize normalizing flow model neural network and its optimizer
            flow = mcflow_utils.init_flow_model(num_neurons, args.num_nf_layers, InterpRealNVP, ldr.train[0].shape[0], args)
            nf_optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=args.lr)

            #Initialize latent space neural network and its optimizer
            num_hidden_neurons = [int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]), int(ldr.train[0].shape[0]),  int(ldr.train[0].shape[0])]
            nn_model = LatentToLatentApprox(int(ldr.train[0].shape[0]), num_hidden_neurons).float()
            if args.use_cuda:
                nn_model.cuda()
            nn_optimizer = torch.optim.Adam([p for p in nn_model.parameters() if p.requires_grad==True], lr=args.lr)

            reset_scheduler = 2

            # #Train and test MCFlow
            for epoch in tqdm(range(args.n_epochs)):
                
                mcflow_utils.endtoend_train(flow, nn_model, nf_optimizer, nn_optimizer, data_loader, args) #Train the MCFlow model

                with torch.no_grad():
                    ldr.mode=0 #Use training data
                    tr_mse, _ ,imputed_data= mcflow_utils.endtoend_test(flow, nn_model, data_loader, args) #Test MCFlow model
                    np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_train.npy', imputed_data.astype("float32"))
                    

                    ldr.mode=1 #Use testing data
                    te_mse, _ ,imputed_data= mcflow_utils.endtoend_test(flow, nn_model, data_loader, args) 
                    np.save(f'{path}/{rule_name}_seed-{seed}_{fold}_test.npy', imputed_data.astype("float32"))
                    #print(te_mse)

                if (epoch+1) % reset_scheduler == 0:
                    #Reset unknown values in the dataset using predicted estimates
                    ldr.reset_imputed_values(nn_model, flow, args.seed, args)
                    flow = mcflow_utils.init_flow_model(num_neurons, args.num_nf_layers, InterpRealNVP, ldr.train[0].shape[0], args) #Initialize brand new flow model to train on new dataset
                    nf_optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=args.lr)
                    reset_scheduler = reset_scheduler*2
            print(tr_mse)



''' Run MCFlow experiment '''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Reproducibility')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-nf-layers', type=int, default=3)
    parser.add_argument('--n-epochs', type=int, default=1)
    parser.add_argument('--drp-impt', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use-cuda', type=mcflow_utils.str2bool, default=True)
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
        choices=["quantile",
                    "diffuse",
                    "logistic"
                    ],
        default="logistic",
        type=str)
    args = parser.parse_args()

    ''' Reproducibility '''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    ''' Cuda enabled experimentation check '''
    if not torch.cuda.is_available() or args.use_cuda==False:
        print("CUDA Unavailable. Using cpu. Check torch.cuda.is_available()")
        args.use_cuda = False

    main(args)