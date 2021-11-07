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
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import model.utils as utils
from model.gen_ode import GenODE
from model.preprocessing import parse_datasets
from model.utils import compute_loss_all_batches

parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=4e-3, help="Starting learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=64)

parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/random-even-batch-0.h5', help="Path for dataset")
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=414, help="Random_seed")

parser.add_argument('-l', '--latents', type=int, default=16, help="Dim of the latent state")
parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=128, help="Number of units per layer in ODE func")
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
    print('Loading Data')
    with h5py.File(args.dataset, mode='r') as dataset_file:
        Y = dataset_file['Y'][...]
        X_even = dataset_file['X_even'][...]

    train_test_split = Y.shape[0] - 1024
    # normalize
    Y[:, 3] = np.log(Y[:, 3]) # rho
    Y[:, 4] = np.log(Y[:, 4]) # q
    Y[:, 5] = np.log(Y[:, 5]) # s
    Y[:, 6] = Y[:, 6]/360 # alpha
    
    train_label_dataloader = DataLoader(Y[:train_test_split], batch_size=args.batch_size, shuffle=False)
    train_even_dataloader = DataLoader(X_even[:train_test_split, :, 1:], batch_size=args.batch_size, shuffle=False)
    test_label = Y[train_test_split:]
    test_even = X_even[train_test_split:, :, 1:]


    input_dim = Y.shape[-1]
    output_dim = X_even.shape[-1] - 1
    ##################################################################
    # Create the model
    model = GenODE(args, input_dim, output_dim, device).to(device)
    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        utils.get_ckpt_model(ckpt_path, model, device)
        exit()
    ##################################################################
    # Training
    print('Start Training')
    log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
    utils.makedirs("logs/")
    
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)

    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    num_batches = len(train_label_dataloader)

    loss_func = nn.MSELoss()

    time_steps_to_predict = torch.tensor(X_even[0, :, 0]).to(device)

    for epoch in range(args.niters):
        for i, (y_batch, x_even_batch) in enumerate(zip(train_label_dataloader, train_even_dataloader)):

            y_batch = y_batch.float().to(device)
            x_even_batch = x_even_batch.float().to(device)

            optimizer.zero_grad()
            utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

            x_even_pred = model(y_batch, time_steps_to_predict)

            loss = loss_func(x_even_pred, x_even_batch)
            loss.backward()
            optimizer.step()

            print(f'batch {i}/{num_batches}, loss {loss.item()}')

        with torch.no_grad():
            y_batch = test_label
            x_even_batch = test_even
            x_even_pred = model(y_batch)
            loss = loss_func(x_even_pred, x_even_batch)

            message = f'Epoch {epoch}, Test Loss {loss.item()}'
            logger.info("Experiment " + str(experimentID))
            logger.info(message)

            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
            }, ckpt_path)

    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)

