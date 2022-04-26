import os
import sys
import gc

import argparse
import numpy as np
from random import SystemRandom
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import model.utils as utils
from model.locator import Locator

import torchcde

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser('Locator')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr',  type=float, default=4e-3, help="Starting learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=128)

parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/KMT-loc-0.h5', help="Path for dataset")
# parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/roman-0-8dof.h5', help="Path for dataset")
parser.add_argument('--save', type=str, default='/work/hmzhao/experiments/locator/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--resume', type=int, default=0, help="Epoch to resume.")
parser.add_argument('-r', '--random-seed', type=int, default=42, help="Random_seed")

parser.add_argument('-l', '--latents', type=int, default=32, help="Dim of the latent state")

args = parser.parse_args()

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ['JOBLIB_TEMP_FOLDER'] = '/work/hmzhao/tmp'

    print(f'Num of GPUs available: {torch.cuda.device_count()}')

    experimentID = args.load
    if experimentID is None:
        # Make a new experiment ID
        experimentID = int(SystemRandom().random() * 100000)
    print(f'ExperimentID: {experimentID}')
    ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
    # ckpt_path_load = os.path.join(args.save, "experiment_" + '89670' + '.ckpt')
    ckpt_path_load = ckpt_path
    
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    writer = SummaryWriter(log_dir=f'/work/hmzhao/tbxdata/locator-{experimentID}')

    ##################################################################
    print(f'Loading Data: {args.dataset}')
    with h5py.File(args.dataset, mode='r') as dataset_file:
        Y = torch.tensor(dataset_file['Y'][...])
        X = torch.tensor(dataset_file['X'][...])

    # filter nan
    nanind = torch.where(~torch.isnan(X[:, 0, 1]))[0]
    Y = Y[nanind]
    X = X[nanind]

    test_size = 1024
    train_size = len(Y) - test_size
    # train_size = 128
    print(f'Training Set Size: {train_size}')

    # # normalize
    # Y: t_0, t_E, u_0, rho, q, s, alpha, f_s
    Y = Y[:, [0, 1, -1]]
    mean_y = torch.mean(Y, axis=0)
    std_y = torch.std(Y, axis=0)
    print(f'Y mean: {mean_y}\nY std: {std_y}')
    # Y = (Y - mean_y) / std_y
    # print(f'normalized Y mean: {torch.mean(Y)}\nY std: {torch.mean(torch.std(Y, axis=0)[~std_mask])}')

    X = X[:, :, :2]
    X[:, :, 1] = (X[:, :, 1] - 14.5 - 2.5 * torch.log10(Y[:, [-1]])) / 0.2
    print(f'normalized X mean: {torch.mean(X[:, :, 1])}\nX std: {torch.mean(torch.std(X[:, :, 1], axis=0))}')
    Y = Y[:, [0, 1]]
        
    # interpolation
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X[:train_size, :, :])
    train_dataset = torch.utils.data.TensorDataset(train_coeffs, Y[:train_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X[(-test_size):, :, :]).float().to(device)
    test_Y = Y[(-test_size):].float().to(device)
    
    output_dim = Y.shape[-1]
    input_dim = X.shape[-1]
    latent_dim = args.latents

    del Y
    del X
    gc.collect()
    ##################################################################
    # Create the model
    model = Locator(device, threshold=0.5).to(device)
    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        # Load checkpoint.
        checkpt = torch.load(ckpt_path_load, map_location='cpu')
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
    print('Start Training Locator')
    log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
    utils.makedirs("logs/")
    
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info("Experiment Locator " + str(experimentID))

    optimizer = optim.Adam(
        [
            {"params": model.parameters(), "lr": args.lr*1e0},
            # {"params": model.cde_func.parameters(), "lr": args.lr*1e0},
            # {"params": model.cde_func_r.parameters(), "lr": args.lr*1e0},
            # {"params": model.readout.parameters(), "lr": args.lr*1e0}
        ],
        lr=args.lr, weight_decay=0,
        )
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)

    num_batches = len(train_dataloader)

    loss_func = nn.MSELoss()

    for epoch in range(args.resume, args.resume + args.niters):
        
        e_dataloader = train_dataloader
        num_batches = len(e_dataloader)

        utils.update_learning_rate(optimizer, decay_rate = 0.9, lowest = args.lr / 10)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'Epoch {epoch}, Learning Rate {lr}')
        writer.add_scalar('learning_rate', lr, epoch)
            
        for i, (batch_coeffs, batch_y) in enumerate(e_dataloader):

            model.train()
            batch_y = batch_y.float().to(device)
            batch_coeffs = batch_coeffs.float().to(device)

            # rescaley = (batch_y / 72 * 4000).int()
            # left = rescaley[:, [0]] - 2*rescaley[:, [1]]
            # right = rescaley[:, [0]] + 2*rescaley[:, [1]]
            # z = torch.tile(torch.arange(0, 4000).unsqueeze(0), (len(batch_y), 1)).to(device)
            # z = ((z > left) * (z < right)).int()

            optimizer.zero_grad()

            pred_y, mse_z = model(batch_coeffs, batch_y)

            mse = torch.mean((batch_y - pred_y)**2, dim=0).detach().cpu() # * (std_Y**2)
            
            # loss = loss_func(pred_y, batch_y)
            loss = mse_z + loss_func(pred_y*torch.tensor([1, 10]).to(device), batch_y*torch.tensor([1, 10]).to(device))/10
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            writer.add_scalar('gradient_norm', total_norm, (i + epoch * num_batches))

            optimizer.step()

            print(f'batch {i}/{num_batches}, loss {loss.item()}, mse_t0, tE {mse}')
            writer.add_scalar('loss/batch_loss', loss.item(), (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_t0', mse[0], (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_tE', mse[1], (i + epoch * num_batches))

            if (i + epoch * num_batches) % 20 == 0:
                model.eval()
                torch.save({
                'args': args,
                'state_dict': model.state_dict(),
                }, ckpt_path)
                print(f'Model saved to {ckpt_path}')

                with torch.no_grad():
                    pred_y, mse_z = model(test_coeffs, test_Y)
                    # loss = loss_func(pred_y, test_Y)
                    loss = mse_z + loss_func(pred_y*torch.tensor([1, 10]).to(device), test_Y*torch.tensor([1, 10]).to(device))/10

                    mse = torch.mean((test_Y - pred_y)**2, dim=0).detach().cpu() # * (std_Y**2)

                    message = f'Epoch {(i + epoch * num_batches)/num_batches}, Test Loss {loss.item()}, mse_t0, tE {mse}'
                    writer.add_scalar('loss/test_loss', loss.item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_t0', mse[0], (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_tE', mse[1], (i + epoch * num_batches)/20)

                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), (i + epoch * num_batches)/20)
                    # logger.info("Experiment " + str(experimentID))
                    logger.info(message)

                model.train()

        # change dataset
        if os.path.exists(args.dataset[:-4] + str((int(args.dataset[-4])+1)) + '.h5'):
            args.dataset = args.dataset[:-4] + str((int(args.dataset[-4])+1)) + '.h5'
        else:
            args.dataset = args.dataset[:-4] + '0.h5'
        print(f'Loading Data: {args.dataset}')
        with h5py.File(args.dataset, mode='r') as dataset_file:
            Y = torch.tensor(dataset_file['Y'][...])
            X = torch.tensor(dataset_file['X'][...])

        # filter nan
        nanind = torch.where(~torch.isnan(X[:, 0, 1]))[0]
        Y = Y[nanind]
        X = X[nanind]

        test_size = 1024
        train_size = len(Y) - test_size
        # train_size = 128
        print(f'Training Set Size: {train_size}')

        # # normalize
        # Y: t_0, t_E, u_0, rho, q, s, alpha, f_s
        Y = Y[:, [0, 1, -1]]
        mean_y = torch.mean(Y, axis=0)
        std_y = torch.std(Y, axis=0)
        print(f'Y mean: {mean_y}\nY std: {std_y}')
        # Y = (Y - mean_y) / std_y
        # print(f'normalized Y mean: {torch.mean(Y)}\nY std: {torch.mean(torch.std(Y, axis=0)[~std_mask])}')

        X = X[:, :, :2]
        X[:, :, 1] = (X[:, :, 1] - 14.5 - 2.5 * torch.log10(Y[:, [-1]])) / 0.2
        print(f'normalized X mean: {torch.mean(X[:, :, 1])}\nX std: {torch.mean(torch.std(X[:, :, 1], axis=0))}')
        Y = Y[:, [0, 1]]
            
        # interpolation
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X[:train_size, :, :])
        train_dataset = torch.utils.data.TensorDataset(train_coeffs, Y[:train_size])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

        test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X[(-test_size):, :, :]).float().to(device)
        test_Y = Y[(-test_size):].float().to(device)
        
        output_dim = Y.shape[-1]
        input_dim = X.shape[-1]
        latent_dim = args.latents

        del Y
        del X
        gc.collect()

    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)
    print(f'Model saved to {ckpt_path}')
    writer.close()
