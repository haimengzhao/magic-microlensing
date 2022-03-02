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

parser = argparse.ArgumentParser('CDE Encoder')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr',  type=float, default=4e-5, help="Starting learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=128)

parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/random-even-batch-0.h5', help="Path for dataset")
# parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/roman-0-8dof-located-logsig-gt.h5', help="Path for dataset")
parser.add_argument('--save', type=str, default='/work/hmzhao/experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--resume', type=int, default=0, help="Epoch to resume.")
parser.add_argument('-r', '--random-seed', type=int, default=42, help="Random_seed")

parser.add_argument('-l', '--latents', type=int, default=32, help="Dim of the latent state")
parser.add_argument('--gen-layers', type=int, default=5, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=1024, help="Number of units per layer in ODE func")

args = parser.parse_args()

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
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

    writer = SummaryWriter(log_dir=f'/work/hmzhao/tbxdata/{experimentID}')

    ##################################################################
    print(f'Loading Data: {args.dataset}')
    with h5py.File(args.dataset, mode='r') as dataset_file:
        Y = torch.tensor(dataset_file['Y'][...])
        X_even = torch.tensor(dataset_file['X_even'][...])
        X_rand = torch.tensor(dataset_file['X_random'][...])

    # filter nan
    nanind = torch.where(~torch.isnan(X_even[:, 0, 1]))[0]
    Y = Y[nanind]
    X_even = X_even[nanind]
    X_rand = X_rand[nanind]

    # nanind = torch.where(Y[:, 4]>1e-4)[0]
    # Y = Y[nanind]
    # X_even = X_even[nanind]
    # X_rand = X_rand[nanind]

    # nanind = torch.where((Y[:, 5]>0.3) * (Y[:, 5]<3))[0]
    # Y = Y[nanind]
    # X_even = X_even[nanind]
    # X_rand = X_rand[nanind]

    test_size = 1024
    train_size = len(Y) - test_size
    # train_size = 128

    print(f'Training Set Size: {train_size}')

    # # normalize
    Y[:, 3:6] = torch.log10(Y[:, 3:6])
    Y[:, 6] = torch.cos(Y[:, 6] / 180 * np.pi)
    Y = torch.hstack([Y, torch.sin(Y[:, [6]] / 180 * np.pi)])
    Y = Y[:, 2:]
    mean_y = torch.mean(Y, axis=0)
    std_y = torch.std(Y, axis=0)
    # std_mask = (std_y==0)
    # std_y[std_mask] = 1
    print(f'Y mean: {mean_y}\nY std: {std_y}')
    # Y = (Y - mean_y) / std_y
    # print(f'normalized Y mean: {torch.mean(Y)}\nY std: {torch.mean(torch.std(Y, axis=0)[~std_mask])}')

    # only target at q (4) and s (5)
    # Y = Y[:, 4:6]
    # mean_y = mean_y[4:6]
    # std_y = std_y[4:6]
    # print(f'Y mean: {mean_y}\nY std: {std_y}')
    # std_y = torch.tensor([1., 1.])
    
    #
    # adaptive normalize is not compatible with irregular data, ABANDONED
    # use static normalize instead
    #
    # mean_x_even = torch.mean(X_even[:, :, 1], axis=0)
    # std_x_even = torch.std(X_even[:, :, 1], axis=0)
    # print(f'X mean: {mean_x_even}\nX std: {std_x_even}')
    mean_x_even = 14.5
    std_x_even = 0.2
    # X_even[:, :, 1] = 10**((22-X_even[:, :, 1])/2.5)/1000
    # X_even[:, :, 1] = 22 - 2.5*torch.log10(1000*X_even[:, :, 1])
    X_even[:, :, 1] = (X_even[:, :, 1] - mean_x_even) / std_x_even
    print(f'normalized X mean: {torch.mean(X_even[:, :, 1])}\nX std: {torch.mean(torch.std(X_even[:, :, 1], axis=0))}')
    X_rand = X_rand[:, :, :2]
    # X_rand[:, :, 1] = 10**((22-X_rand[:, :, 1])/2.5)/1000
    # X_rand[:, :, 1] = 22 - 2.5*torch.log10(1000*X_rand[:, :, 1])
    X_rand[:, :, 1] = (X_rand[:, :, 1] - mean_x_even) / std_x_even

    # time rescale
    X_even[:, :, 0] = X_even[:, :, 0] / 200
    X_rand[:, :, 0] = X_rand[:, :, 0] / 200
    
    # CDE interpolation with log_sig
    depth = 3; window_length = 10; window_length_rand = 2
    train_logsig = torchcde.logsig_windows(X_even[:train_size, :, :], depth, window_length=window_length)
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_logsig)
    # train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_even[:train_size, :, :])

    train_logsig_rand = torchcde.logsig_windows(X_rand[:train_size, :, :], depth, window_length=window_length_rand)
    train_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_logsig_rand)
    # train_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_rand[:train_size, :, :])

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, Y[:train_size])
    train_rand_dataset = torch.utils.data.TensorDataset(train_rand_coeffs, Y[:train_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    train_rand_dataloader = DataLoader(train_rand_dataset, batch_size=args.batch_size, shuffle=False)

    train_mix_dataset = torch.utils.data.TensorDataset(torch.cat([train_coeffs, train_rand_coeffs[:, :train_coeffs.shape[1]]], dim=0), Y[:train_size].repeat(2, 1))
    train_mix_dataloader = DataLoader(train_mix_dataset, batch_size=args.batch_size, shuffle=True)

    test_logsig = torchcde.logsig_windows(X_even[(-test_size):, :, :].float().to(device), depth, window_length=window_length)
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_logsig)
    # test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_even[(-test_size):, :, :].float().to(device))
    test_Y = Y[(-test_size):].float().to(device)
    
    test_logsig_rand = torchcde.logsig_windows(X_rand[(-test_size):, :, :].float().to(device), depth, window_length=window_length_rand)
    test_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_logsig_rand).float().to(device)
    # test_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_rand[(-test_size):, :, :]).float().to(device)

    output_dim = Y.shape[-1]
    input_dim = train_logsig.shape[-1]
    # input_dim = X_even.shape[-1]
    latent_dim = args.latents
    del Y
    del X_even
    del X_rand
    ##################################################################
    # Create the model
    model = CDEEncoder(input_dim, latent_dim, output_dim).to(device)
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
    print('Start Training')
    log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
    utils.makedirs("logs/")
    
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info("Experiment " + str(experimentID))

    optimizer = optim.Adam(
        [
            {"params": model.initial.parameters(), "lr": args.lr*1e1},
            {"params": model.cde_func.parameters(), "lr": args.lr},
            {"params": model.readout.parameters(), "lr": args.lr*1e1},
        ],
        lr=args.lr
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
        e_dataloader = train_mix_dataloader
        # e_dataloader = train_dataloader
        num_batches = len(e_dataloader)
            
        for i, (batch_coeffs, batch_y) in enumerate(e_dataloader):
            # print(batch_y)
            batch_y = batch_y.float().to(device)
            batch_coeffs = batch_coeffs.float().to(device)

            optimizer.zero_grad()

            pred_y = model(batch_coeffs)

            # mse_log10q = torch.mean((batch_y[:, 0] / np.log(10) - pred_y[:, 0] / np.log(10))**2).detach().cpu() * std_y[0]
            # mse_log10s = torch.mean((batch_y[:, 1] / np.log(10) - pred_y[:, 1] / np.log(10))**2).detach().cpu() * std_y[1]
            mse = torch.mean((batch_y - pred_y)**2, dim=0).detach().cpu()
            loss = loss_func(pred_y, batch_y)
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

            # print(f'batch {i}/{num_batches}, loss {loss.item()}, mse_log10q {mse_log10q.item()}, mse_log10s {mse_log10s.item()}')
            # writer.add_scalar('loss/batch_loss', loss.item(), (i + epoch * num_batches))
            # writer.add_scalar('mse/batch_mse_log10q', mse_log10q.item(), (i + epoch * num_batches))
            # writer.add_scalar('mse/batch_mse_log10s', mse_log10s.item(), (i + epoch * num_batches))

            print(f'batch {i}/{num_batches}, loss {loss.item()}, mse {mse}')
            writer.add_scalar('loss/batch_loss', loss.item(), (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_log10q', mse[2], (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_log10s', mse[3], (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_u0', mse[0], (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_rho', mse[1], (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_cosa', mse[4], (i + epoch * num_batches))
            writer.add_scalar('mse_batch/batch_mse_sina', mse[5], (i + epoch * num_batches))

            # for param_group in optimizer.param_groups:
            #     lr = param_group['lr']
            #     if lr < args.lr * 1e2:
            #         lr = lr * 2
            #     param_group['lr'] = lr

            if (i + epoch * num_batches) % 20 == 0:
                model.eval()
                torch.save({
                'args': args,
                'state_dict': model.state_dict(),
                }, ckpt_path)
                print(f'Model saved to {ckpt_path}')

                with torch.no_grad():
                    pred_y = model(test_coeffs)
                    loss = loss_func(pred_y, test_Y)

                    # mse_log10q = torch.mean((test_Y[:, 0] / np.log(10) - pred_y[:, 0] / np.log(10))**2).detach().cpu() * std_y[0]
                    # mse_log10s = torch.mean((test_Y[:, 1] / np.log(10) - pred_y[:, 1] / np.log(10))**2).detach().cpu() * std_y[1]
                    mse = torch.mean((test_Y - pred_y)**2, dim=0).detach().cpu()
                    pred_y_rand = model(test_rand_coeffs)
                    loss_rand = loss_func(pred_y_rand, test_Y)

                    mse_log10q_rand = torch.mean((test_Y[:, 0] / np.log(10) - pred_y_rand[:, 0] / np.log(10))**2).detach().cpu() * std_y[0]
                    mse_log10s_rand = torch.mean((test_Y[:, 1] / np.log(10) - pred_y_rand[:, 1] / np.log(10))**2).detach().cpu() * std_y[1]
                    mse_rand = torch.mean((test_Y - pred_y_rand)**2, dim=0).detach().cpu()

                    message = f'Epoch {(i + epoch * num_batches)/20}, Test Loss {loss.item()}, mse {mse}, loss_rand {loss_rand.item()}, mse_rand {mse_rand}'
                    writer.add_scalar('loss/test_loss', loss.item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('loss/test_loss_rand', loss_rand.item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_log10q', mse[2], (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_log10s', mse[3], (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_u0', mse[0], (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_rho', mse[1], (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_cosa', mse[4], (i + epoch * num_batches)/20)
                    writer.add_scalar('mse/test_mse_sina', mse[5], (i + epoch * num_batches)/20)

                    writer.add_scalar('mse_rand/test_mse_log10q_rand', mse_rand[2].item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('mse_rand/test_mse_log10s_rand', mse_rand[3].item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('mse_rand/test_mse_u0', mse_rand[0].item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('mse_rand/test_mse_rho', mse_rand[1].item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('mse_rand/test_mse_cosa', mse_rand[4].item(), (i + epoch * num_batches)/20)
                    writer.add_scalar('mse_rand/test_mse_sina', mse_rand[5].item(), (i + epoch * num_batches)/20)
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
