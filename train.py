import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import argparse
import numpy as np
from random import SystemRandom

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import model.utils as utils
from model.latent_ode import create_LatentODE_model
from model.preprocessing import parse_datasets
from model.utils import compute_loss_all_batches

parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=4)

parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/random-even-batch-0.h5', help="Path for dataset")
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=414, help="Random_seed")

parser.add_argument('-l', '--latents', type=int, default=16, help="Dim of the latent state")
parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random()*100000)
    print(f'ExperimentID: {experimentID}')
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

    start = time.time()
    
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    utils.makedirs("results/")

    ##################################################################
    data_obj = parse_datasets(args, device)

    input_dim = data_obj["input_dim"]
    output_dim = data_obj['output_dim']
    n_labels = data_obj["n_labels"]
    ##################################################################
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)

    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    model = create_LatentODE_model(args, input_dim, output_dim, z0_prior, obsrv_std, device, 
            n_labels = n_labels)
    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        utils.get_ckpt_model(ckpt_path, model, device)
        exit()
    ##################################################################
    # Training
    log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
    utils.makedirs("logs/")
    
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    num_batches = data_obj["n_train_batches"]

    for itr in range(1, num_batches * (args.niters + 1)):
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

        wait_until_kl_inc = 10
        if itr // num_batches < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

        batch_dict = utils.get_next_batch(data_obj["train_random_dataloader"],
                                        data_obj["train_label_dataloader"],
                                        data_obj["train_even_dataloader"],
                                        device=device)
        
        train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
        train_res["loss"].backward()
        optimizer.step()

        print(train_res["loss"].item())

        if itr % 5 == 0:
            with torch.no_grad():
                test_res = compute_loss_all_batches(model, 
                    data_obj["test_random_dataloader"], data_obj["test_label_dataloader"], data_obj["test_even_dataloader"],
                    args,
                    n_batches = data_obj["n_test_batches"],
                    experimentID = experimentID,
                    device = device,
                    n_traj_samples = 3, kl_coef = kl_coef)

                message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
                    itr//num_batches, 
                    test_res["loss"].detach(), test_res["likelihood"].detach(), 
                    test_res["kl_first_p"], test_res["std_first_p"])
             
                logger.info("Experiment " + str(experimentID))
                logger.info(message)
                logger.info("KL coef: {}".format(kl_coef))
                logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
                logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))
                
                if "auc" in test_res:
                    logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

                if "mse" in test_res:
                    logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

                if "accuracy" in train_res:
                    logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

                if "accuracy" in test_res:
                    logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

                if "pois_likelihood" in test_res:
                    logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

                if "ce_loss" in test_res:
                    logger.info("CE loss: {}".format(test_res["ce_loss"]))

            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
            }, ckpt_path)

    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)

