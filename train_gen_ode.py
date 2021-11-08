import os
import sys

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

parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr',  type=float, default=4e-3, help="Starting learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=128)

parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/random-even-batch-0.h5', help="Path for dataset")
parser.add_argument('--save', type=str, default='/work/hmzhao/experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=414, help="Random_seed")

parser.add_argument('-l', '--latents', type=int, default=128, help="Dim of the latent state")
parser.add_argument('--gen-layers', type=int, default=7, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=1024, help="Number of units per layer in ODE func")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ['JOBLIB_TEMP_FOLDER'] = '/work/hmzhao/tmp'

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random()*100000)
    print(f'ExperimentID: {experimentID}')
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
    
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    ##################################################################
    print(f'Loading Data: {args.dataset}')
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
    test_label = torch.tensor(Y[train_test_split:])
    test_even = torch.tensor(X_even[train_test_split:, :, 1:])


    input_dim = Y.shape[-1]
    output_dim = X_even.shape[-1] - 1
    ##################################################################
    # Create the model
    model = GenODE(args, input_dim, output_dim, device).to(device)
    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        # Load checkpoint.
        checkpt = torch.load(ckpt_path)
        ckpt_args = checkpt['args']
        state_dict = checkpt['state_dict']
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict) 
        # 3. load the new state dict
        model.load_state_dict(state_dict)
        model.to(device)
    ##################################################################
    # Training
    print('Start Training')
    log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
    utils.makedirs("logs/")
    
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

            if i % 100 == 0:
                torch.save({
                'args': args,
                'state_dict': model.state_dict(),
                }, ckpt_path)
                print(f'Model saved to {ckpt_path}')

        with torch.no_grad():
            y_batch = test_label
            x_even_batch = test_even

            y_batch = y_batch.float().to(device)
            x_even_batch = x_even_batch.float().to(device)

            x_even_pred = model(y_batch, time_steps_to_predict)
            loss = loss_func(x_even_pred, x_even_batch)

            message = f'Epoch {epoch}, Test Loss {loss.item()}'
            logger.info("Experiment " + str(experimentID))
            logger.info(message)

            torch.save({
                'args': args,
                'state_dict': model.state_dict(),
            }, ckpt_path)
            print(f'Model saved to {ckpt_path}')

    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)
    print(f'Model saved to {ckpt_path}')

