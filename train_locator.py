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
from model.locator import Locator

import torchcde

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
parser = argparse.ArgumentParser('Locator')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr',  type=float, default=4e-3, help="Starting learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=128)

parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/roman-0.h5', help="Path for dataset")
parser.add_argument('--save', type=str, default='/work/hmzhao/experiments/locator/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--resume', type=int, default=0, help="Epoch to resume.")
parser.add_argument('-r', '--random-seed', type=int, default=42, help="Random_seed")

parser.add_argument('-l', '--latents', type=int, default=32, help="Dim of the latent state")

args = parser.parse_args()

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
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
        X_even = torch.tensor(dataset_file['X_even'][...])
        X_rand = torch.tensor(dataset_file['X_random'][...])

    test_size = 1024
    train_size = len(Y) - test_size
    # train_size = 128

    # preprocess
    nanind = torch.where(~torch.isnan(X_even[:, 0, 1]))[0]
    # Y: t_0, t_E, u_0, rho, q, s, alpha, f_s
    # Y = Y[:, [0, 1, -1]] # locator predicts t_0, t_E and f_s
    Y[:, -1] = torch.log10(Y[:, -1])
    Y[:, -3] = torch.log10(Y[:, -3])
    Y[:, -4] = torch.log10(Y[:, -4])
    Y[:, -5] = torch.log10(Y[:, -5])
    std_Y = torch.tensor([1, 1, 1, 1, 1, 0.1, 360, 1])
    Y = Y / std_Y
    Y = Y[nanind]
    Y = Y[:, [0, 1]]
    std_Y = std_Y[[0, 1]]
    
    # discard uncertainty bar
    X_even = X_even[nanind]
    X_rand = X_rand[nanind]

    # mean_x_even = 13.0
    # std_x_even = 0.7
    # X_even[:, :, 1] = (X_even[:, :, 1] - mean_x_even) / std_x_even
    # print(f'normalized X mean: {torch.mean(X_even[:, :, 1])}\nX std: {torch.mean(torch.std(X_even[:, :, 1], axis=0))}')

    X_rand = X_rand[:, :, :2]
    # X_rand[:, :, 1] = (X_rand[:, :, 1] - mean_x_even) / std_x_even
 
    # CDE interpolation with log_sig
    depth = 2; window_length = 1; window_length_rand = 1
    train_logsig = torchcde.logsig_windows(X_even[:train_size, :, :], depth, window_length=window_length)
    # train_logsig = X_even[:train_size, :, :]
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_logsig)

    train_logsig_rand = torchcde.logsig_windows(X_rand[:train_size, :, :], depth, window_length=window_length_rand)
    # train_logsig_rand = X_rand[:train_size, :, :]
    train_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_logsig_rand)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, Y[:train_size])
    train_rand_dataset = torch.utils.data.TensorDataset(train_rand_coeffs, Y[:train_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    train_rand_dataloader = DataLoader(train_rand_dataset, batch_size=args.batch_size, shuffle=False)

    train_mix_dataset = torch.utils.data.TensorDataset(torch.cat([train_coeffs, train_rand_coeffs], dim=0), Y[:train_size].repeat(2, 1))
    train_mix_dataloader = DataLoader(train_mix_dataset, batch_size=args.batch_size, shuffle=True)

    test_logsig = torchcde.logsig_windows(X_even[(-test_size):, :, :].float().to(device), depth, window_length=window_length)
    # test_logsig = X_even[(-test_size):, :, :].float().to(device)
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_logsig)
    test_Y = Y[(-test_size):].float().to(device)
    
    test_logsig_rand = torchcde.logsig_windows(X_rand[(-test_size):, :, :].float().to(device), depth, window_length=window_length_rand)
    # test_logsig_rand = X_rand[(-test_size):, :, :].float().to(device)
    test_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_logsig_rand).float().to(device)

    rescaley = (test_Y / 72 * 4000).int()
    left = rescaley[:, [0]] - 2*rescaley[:, [1]]
    right = rescaley[:, [0]] + 2*rescaley[:, [1]]
    ztest = torch.tile(torch.arange(0, 4000).unsqueeze(0), (len(test_Y), 1)).to(device)
    ztest = ((ztest > left)*(ztest<right)).int()

    output_dim = Y.shape[-1]
    input_dim = train_logsig.shape[-1]
    # input_dim = 1
    latent_dim = args.latents

    del Y
    del X_even
    del X_rand
    ##################################################################
    # Create the model
    model = Locator(input_dim, latent_dim, output_dim, device).to(device)
    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        # Load checkpoint.
        checkpt = torch.load(ckpt_path, map_location='cpu')
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
        utils.update_learning_rate(optimizer, decay_rate = 0.99, lowest = args.lr / 10)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'Epoch {epoch}, Learning Rate {lr}')
        writer.add_scalar('learning_rate', lr, epoch)
        
        # if epoch % 2 == 0:
        #     e_dataloader = train_dataloader
        #     print('Using regular data')
        # else:
        #     e_dataloader = train_rand_dataloader
        #     print('Using irregular data')
        # e_dataloader = train_mix_dataloader
        e_dataloader = train_dataloader
        num_batches = len(e_dataloader)
            
        for i, (batch_coeffs, batch_y) in enumerate(e_dataloader):

            batch_y = batch_y.float().to(device)
            batch_coeffs = batch_coeffs.float().to(device)

            rescaley = (batch_y / 72 * 4000).int()
            left = rescaley[:, [0]] - 2*rescaley[:, [1]]
            right = rescaley[:, [0]] + 2*rescaley[:, [1]]
            z = torch.tile(torch.arange(0, 4000).unsqueeze(0), (len(batch_y), 1)).to(device)
            z = ((z > left) * (z < right)).int()

            optimizer.zero_grad()

            pred_y, mse_z = model(batch_coeffs, z)

            mse = torch.mean((batch_y - pred_y)**2, dim=0).detach().cpu() * (std_Y**2)
            
            # loss = loss_func(pred_y, batch_y)
            loss = mse_z + loss_func(pred_y, batch_y)/10000
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

            print(f'batch {i}/{num_batches}, loss {loss.item()}, mse_t0, tE, log10fs {mse}')
            writer.add_scalar('loss/batch_loss', loss.item(), (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_t0', mse[0], (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_tE', mse[1], (i + epoch * num_batches))
            # writer.add_scalar('mse_batch/batch_mse_log10fs', mse[2], (i + epoch * num_batches))

            if (i + epoch * num_batches) % 20 == 0:
                model.eval()
                torch.save({
                'args': args,
                'state_dict': model.state_dict(),
                }, ckpt_path)
                print(f'Model saved to {ckpt_path}')

                with torch.no_grad():
                    pred_y, mse_z = model(test_coeffs, ztest)
                    # loss = loss_func(pred_y, test_Y)
                    loss = mse_z + loss_func(pred_y, test_Y)/10000

                    mse = torch.mean((test_Y - pred_y)**2, dim=0).detach().cpu() * (std_Y**2)

                    pred_y_rand, mse_z_rand = model(test_rand_coeffs, ztest)
                    # loss_rand = loss_func(pred_y_rand, test_Y)
                    loss_rand = mse_z_rand + loss_func(pred_y_rand, test_Y)/10000

                    mse_rand = torch.mean((test_Y - pred_y_rand)**2, dim=0).detach().cpu() * (std_Y**2)

                    message = f'Epoch {(i + epoch * num_batches)/20}, Test Loss {loss.item()}, mse_t0, tE, log10fs {mse}, loss_rand {loss_rand.item()}, mse_t0, tE, log10fs_rand {mse_rand}'
                    writer.add_scalar('loss/test_loss', loss.item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('loss/test_loss_rand', loss_rand.item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_t0', mse[0], (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_tE', mse[1], (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse/test_mse_log10fs', mse[2], (i + epoch * num_batches)/20)
                    writer.add_scalar('mse_rand/test_mse_t0', mse_rand[0], (i + epoch * num_batches)/20)
                    writer.add_scalar('mse_rand/test_mse_tE', mse_rand[1], (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse_rand/test_mse_log10fs', mse_rand[2], (i + epoch * num_batches)/20)
                    for name, param in model.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), (i + epoch * num_batches)/20)
                    # logger.info("Experiment " + str(experimentID))
                    logger.info(message)

                model.train()

    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)
    print(f'Model saved to {ckpt_path}')
    writer.close()
