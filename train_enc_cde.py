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
from model.encoder_cde import CDEEncoder

import torchcde

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr',  type=float, default=4e-6, help="Starting learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=128)

parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/random-even-batch-0.h5', help="Path for dataset")
parser.add_argument('--save', type=str, default='/work/hmzhao/experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=42, help="Random_seed")

parser.add_argument('-l', '--latents', type=int, default=64, help="Dim of the latent state")
parser.add_argument('--gen-layers', type=int, default=5, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=1024, help="Number of units per layer in ODE func")

args = parser.parse_args()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ['JOBLIB_TEMP_FOLDER'] = '/work/hmzhao/tmp'

    print(f'Num of GPUs available: {torch.cuda.device_count()}')

    writer = SummaryWriter(log_dir='/work/hmzhao/tbxdata/')

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
        Y = torch.tensor(dataset_file['Y'][...])
        X_even = torch.tensor(dataset_file['X_even'][...])

    test_size = 1024
    train_size = len(Y) - test_size

    # normalize
    Y[:, 3:6] = torch.log(Y[:, 3:6])
    mean_y = torch.mean(Y, axis=0)
    std_y = torch.std(Y, axis=0)
    std_mask = (std_y==0)
    std_y[std_mask] = 1
    # print(f'Y mean: {mean_y}\nY std: {std_y}')
    Y = (Y - mean_y) / std_y
    print(f'normalized Y mean: {torch.mean(Y)}\nY std: {torch.mean(torch.std(Y, axis=0)[~std_mask])}')
    
    mean_x_even = torch.mean(X_even[:, :, 1], axis=0)
    std_x_even = torch.std(X_even[:, :, 1], axis=0)
    # print(f'X mean: {mean_x_even}\nX std: {std_x_even}')
    X_even[:, :, 1] = (X_even[:, :, 1] - mean_x_even) / std_x_even
    print(f'normalized X mean: {torch.mean(X_even[:, :, 1])}\nX std: {torch.mean(torch.std(X_even[:, :, 1], axis=0))}')
    
    # CDE interpolation
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_even[:train_size, :, :])
    train_dataset = torch.utils.data.TensorDataset(train_coeffs, Y[:train_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_even[(-test_size):, :, :]).float().to(device)
    test_Y = Y[(-test_size):].float().to(device)

    output_dim = Y.shape[-1]
    input_dim = X_even.shape[-1]
    latent_dim = args.latents
    ##################################################################
    # Create the model
    model = CDEEncoder(input_dim, latent_dim, output_dim).to(device)
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
        model.load_state_dict(model_dict)
        model.to(device)
    ##################################################################
    # Training
    print('Start Training')
    log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
    utils.makedirs("logs/")
    
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info("Experiment " + str(experimentID))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)

    num_batches = len(train_dataloader)

    loss_func = nn.MSELoss()

    for epoch in range(args.niters):
        utils.update_learning_rate(optimizer, decay_rate = 0.99, lowest = args.lr / 10)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'Epoch {epoch}, Learning Rate {lr}')
        writer.add_scalar('learning_rate', lr, epoch)
        for i, (batch_coeffs, batch_y) in enumerate(train_dataloader):

            batch_y = batch_y.float().to(device)
            batch_coeffs = batch_coeffs.float().to(device)

            optimizer.zero_grad()

            pred_y = model(batch_coeffs)
            
            loss = loss_func(batch_y, pred_y)
            loss.backward()
            optimizer.step()

            print(f'batch {i}/{num_batches}, loss {loss.item()}')
            writer.add_scalar('loss/batch_loss', loss.item(), (i + epoch * num_batches))

            if i == 0:
                torch.save({
                'args': args,
                'state_dict': model.state_dict(),
                }, ckpt_path)
                print(f'Model saved to {ckpt_path}')

                with torch.no_grad():
                    pred_y = model(test_coeffs)
                    loss = loss_func(test_Y, pred_y)

                    message = f'Epoch {epoch}, Test Loss {loss.item()}'
                    writer.add_scalar('loss/test_loss', loss.item(), epoch)
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                    # logger.info("Experiment " + str(experimentID))
                    logger.info(message)

    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)
    print(f'Model saved to {ckpt_path}')
    writer.close()
