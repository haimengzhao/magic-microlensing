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
from model.cde_mdn import CDE_MDN

import torchcde

from tensorboardX import SummaryWriter

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu_ids = [0, 1, 2, 3]
# n_gpus = len(gpu_ids)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_ids=[0]
n_gpus = 1

parser = argparse.ArgumentParser('CDE-MDN')
parser.add_argument('--niters', type=int, default=50)
parser.add_argument('--lr',  type=float, default=1e-4, help="Starting learning rate")
parser.add_argument('-b', '--batch-size', type=int, default=128 * n_gpus)
parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/KMT-fixrho-0.h5', help="Path for dataset")
# parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/KMT-0.h5', help="Path for dataset")
# parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/random-even-batch-0.h5', help="Path for dataset")
# parser.add_argument('--dataset', type=str, default='/work/hmzhao/irregular-lc/roman-0-8dof-located-logsig.h5', help="Path for dataset")
parser.add_argument('--save', type=str, default='/work/hmzhao/experiments/cde_mdn/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('--resume', type=int, default=0, help="Epoch to resume.")
parser.add_argument('-r', '--random-seed', type=int, default=42, help="Random_seed")
parser.add_argument('-ng', '--ngaussians', type=int, default=12, help="Number of Gaussians in mixture density network")
parser.add_argument('-l', '--latents', type=int, default=64, help="Dim of the latent state")

args = parser.parse_args()

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
    # ckpt_path_load = os.path.join(args.save, "experiment_" + '859' + '.ckpt')
    ckpt_path_load = ckpt_path
    
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    writer = SummaryWriter(log_dir=f'/work/hmzhao/tbxdata/CDE_MDN_{experimentID}')

    ##################################################################
    print(f'Loading Data: {args.dataset}')
    with h5py.File(args.dataset, mode='r') as dataset_file:
        Y = torch.tensor(dataset_file['Y'][...])
        X = torch.tensor(dataset_file['X'][...])

    # filter nan
    nanind = torch.where(~torch.isnan(X[:, 0, 1]))[0]
    Y = Y[nanind]
    X = X[nanind]

    # nanind = torch.where(Y[:, 4]>1e-4)[0]
    # Y = Y[nanind]
    # X_even = X_even[nanind]
    # X_rand = X_rand[nanind]

    n_chunks = 25
    gap_len = int(500 / n_chunks)
    gap_left = torch.randint(0, X.shape[1]-gap_len, (len(X),))
    X_gap = torch.zeros((X.shape[0], X.shape[1]-gap_len, X.shape[2]))
    for i in range(len(X)):
        left, gap, right = torch.split(X[i], [gap_left[i], gap_len, X.shape[1]-gap_left[i]-gap_len], dim=0)
        lc = torch.vstack([left, right])
        X_gap[i] = lc
    X = X_gap

    test_size = 1024
    train_size = len(Y) - test_size
    # train_size = 128
    print(f'Training Set Size: {train_size}')


    # # normalize
    # Y: t_0, t_E, u_0, rho, q, s, alpha, f_s
    Y[:, 3:6] = torch.log10(Y[:, 3:6])
    Y[:, 7] = torch.log10(Y[:, 7])
    # Y[:, 7] = torch.sin(Y[:, 6] / 180 * np.pi)
    # Y = torch.hstack([Y, torch.sin(Y[:, [6]] / 180 * np.pi)])
    # Y[:, 6] = torch.cos(Y[:, 6] / 180 * np.pi)
    Y[:, 6] = Y[:, 6] / 180 # * np.pi
    Y = Y[:, [2, 4, 5, 6, 7]]
    mean_y = torch.mean(Y, axis=0)
    std_y = torch.std(Y, axis=0)
    # std_mask = (std_y==0)
    # std_y[std_mask] = 1
    print(f'Y mean: {mean_y}\nY std: {std_y}')
    # Y = (Y - mean_y) / std_y
    # print(f'normalized Y mean: {torch.mean(Y)}\nY std: {torch.mean(torch.std(Y, axis=0)[~std_mask])}')

    mean_x_even = 14.5
    # std_x_even = 0.2
    # X_even[:, :, 1] = 10**((22-X_even[:, :, 1])/2.5)/1000
    # X_even[:, :, 1] = 22 - 2.5*torch.log10(1000*X_even[:, :, 1])
    X = X[:, :, :2]
    X[:, :, 1] = (X[:, :, 1] - mean_x_even - 2.5 * Y[:, [-1]]) / 0.2
    print(f'normalized X mean: {torch.mean(X[:, :, 1])}\nX std: {torch.mean(torch.std(X[:, :, 1], axis=0))}')
    # X_rand = X_rand[:, :, :2]
    # X_rand[:, :, 1] = 10**((22-X_rand[:, :, 1])/2.5)/1000
    # X_rand[:, :, 1] = 22 - 2.5*torch.log10(1000*X_rand[:, :, 1])
    # X_rand[:, :, 1] = (X_rand[:, :, 1] - mean_x_even) / std_x_even

    # Y = Y[:, :-1]

    # time rescale
    # X_even[:, :, 0] = X_even[:, :, 0] / 200
    # X_rand[:, :, 0] = X_rand[:, :, 0] / 200
        
    # CDE interpolation with log_sig
    depth = 3; window_length = 5; window_length_rand = 2
    train_logsig = torchcde.logsig_windows(X[:train_size, :, :], depth, window_length=window_length)
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_logsig)
    # train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_even[:train_size, :, :])

    # train_logsig_rand = torchcde.logsig_windows(X_rand[:train_size, :, :], depth, window_length=window_length_rand)
    # train_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_logsig_rand)
    # train_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_rand[:train_size, :, :])

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, Y[:train_size])
    # train_rand_dataset = torch.utils.data.TensorDataset(train_rand_coeffs, Y[:train_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    # train_rand_dataloader = DataLoader(train_rand_dataset, batch_size=args.batch_size, shuffle=False)

    # train_mix_dataset = torch.utils.data.TensorDataset(torch.cat([train_coeffs, train_rand_coeffs[:, :train_coeffs.shape[1]]], dim=0), Y[:train_size].repeat(2, 1))
    # train_mix_dataloader = DataLoader(train_mix_dataset, batch_size=args.batch_size, shuffle=True)

    test_logsig = torchcde.logsig_windows(X[(-test_size):, :, :].float().to(device), depth, window_length=window_length)
    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_logsig)
    # test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_even[(-test_size):, :, :].float().to(device))
    test_Y = Y[(-test_size):].float().to(device)
    
    # test_logsig_rand = torchcde.logsig_windows(X_rand[(-test_size):, :, :].float().to(device), depth, window_length=window_length_rand)
    # test_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_logsig_rand).float().to(device)
    # test_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_rand[(-test_size):, :, :]).float().to(device)

    output_dim = Y.shape[-1]
    input_dim = train_logsig.shape[-1]
    # input_dim = X_even.shape[-1]
    latent_dim = args.latents
    del Y
    del X
    # del X_even
    # del X_rand
    gc.collect()
    ##################################################################
    # Create the model
    if n_gpus > 1:
        model = CDE_MDN(input_dim, latent_dim, output_dim, args.ngaussians, dataparallel=True)
        model = nn.DataParallel(model, device_ids = gpu_ids)
    else:
        model = CDE_MDN(input_dim, latent_dim, output_dim, args.ngaussians)
    model = model.to(device)
    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        # Load checkpoint.
        checkpt = torch.load(ckpt_path_load, map_location='cpu')
        ckpt_args = checkpt['args']
        state_dict = checkpt['state_dict']
        model_dict = model.state_dict()

        # 1. filter out unnecessary keys
        # state_dict = {k: v for k, v in state_dict.items() if ((k in model_dict) and ('readout' not in k) and ('readout' not in k) and ('mdn' not in k))}
        state_dict = {k: v for k, v in state_dict.items() if ((k in model_dict))}
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
            {"params": model.parameters(), "lr": args.lr},
            # {"params": model.initial.parameters(), "lr": args.lr/10},
            # {"params": model.cde_func.parameters(), "lr": args.lr/10},
            # {"params": model.readout.parameters(), "lr": args.lr},
            # {"params": model.mdn.parameters(), "lr": args.lr},
        ],
        lr=args.lr
        )
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)

    num_batches = len(train_dataloader)

    for epoch in range(args.resume, args.resume + args.niters):
        utils.update_learning_rate(optimizer, decay_rate = 0.9, lowest = args.lr / 100)
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

            optimizer.zero_grad()

            if n_gpus > 1:
                probs, locs, scales = model(batch_coeffs)
                pi = torch.distributions.OneHotCategorical(probs=probs)
                normal = torch.distributions.Normal(locs, scales)
                loss = model.module.mdn_loss(pi, normal, batch_y)
                pred_y = model.module.sample(pi, normal)
            else:
                pi, normal = model(batch_coeffs)
                loss = model.mdn_loss(pi, normal, batch_y)
                pred_y = model.sample(pi, normal)

            mse = torch.mean((batch_y - pred_y)**2, dim=0).detach().cpu()
            
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

            print(f'batch {i}/{num_batches}, loss {loss.item()}, mse {mse}')
            writer.add_scalar('loss/batch_loss', loss.item(), (i + epoch * num_batches))
            # writer.add_scalar('mse_batch/batch_mse_log10q', mse[2], (i + epoch * num_batches))
            # writer.add_scalar('mse_batch/batch_mse_log10s', mse[3], (i + epoch * num_batches))
            # writer.add_scalar('mse_batch/batch_mse_u0', mse[0], (i + epoch * num_batches))
            # writer.add_scalar('mse_batch/batch_mse_rho', mse[1], (i + epoch * num_batches))
            # writer.add_scalar('mse_batch/batch_mse_a', mse[4], (i + epoch * num_batches))
            # writer.add_scalar('mse_batch/batch_mse_log10fs', mse[5], (i + epoch * num_batches))
            # writer.add_scalar('mse_batch/batch_mse_cosa', mse[4], (i + epoch * num_batches))
            # writer.add_scalar('mse_batch/batch_mse_sina', mse[5], (i + epoch * num_batches))

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
                    if n_gpus > 1:
                        probs, locs, scales = model(test_coeffs)
                        pi = torch.distributions.OneHotCategorical(probs=probs)
                        normal = torch.distributions.Normal(locs, scales)
                        loss = model.module.mdn_loss(pi, normal, test_Y)
                        pred_y = model.module.sample(pi, normal)
                    else:
                        pi, normal = model(test_coeffs)
                        loss = model.mdn_loss(pi, normal, test_Y)
                        pred_y = model.sample(pi, normal)

                    mse = torch.mean((test_Y - pred_y)**2, dim=0).detach().cpu()

                    # pi, normal = model(test_rand_coeffs)
                    # loss_rand = model.mdn_loss(pi, normal, test_Y)
                    # pred_y_rand = model.sample(pi, normal)

                    # mse_rand = torch.mean((test_Y - pred_y_rand)**2, dim=0).detach().cpu()

                    message = f'Epoch {(i + epoch * num_batches)/num_batches}, Test Loss {loss.item()}, mse {mse}'
                    writer.add_scalar('loss/test_loss', loss.item(), (i + epoch * num_batches)/20)
                    # writer.add_scalar('loss/test_loss_rand', loss_rand.item(), (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse/test_mse_log10q', mse[2], (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse/test_mse_log10s', mse[3], (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse/test_mse_u0', mse[0], (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse/test_mse_rho', mse[1], (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse/test_mse_a', mse[4], (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse/test_mse_log10fs', mse[5], (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse/test_mse_cosa', mse[4], (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse/test_mse_sina', mse[5], (i + epoch * num_batches)/20)

                    # writer.add_scalar('mse_rand/test_mse_log10q_rand', mse_rand[2].item(), (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse_rand/test_mse_log10s_rand', mse_rand[3].item(), (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse_rand/test_mse_u0', mse_rand[0].item(), (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse_rand/test_mse_rho', mse_rand[1].item(), (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse_rand/test_mse_a', mse_rand[4].item(), (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse_rand/test_mse_log10fs', mse_rand[4].item(), (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse_rand/test_mse_cosa', mse_rand[4].item(), (i + epoch * num_batches)/20)
                    # writer.add_scalar('mse_rand/test_mse_sina', mse_rand[5].item(), (i + epoch * num_batches)/20)

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

        n_chunks = 25
        gap_len = int(500 / n_chunks)
        gap_left = torch.randint(0, X.shape[1]-gap_len, (len(X),))
        X_gap = torch.zeros((X.shape[0], X.shape[1]-gap_len, X.shape[2]))
        for i in range(len(X)):
            left, gap, right = torch.split(X[i], [gap_left[i], gap_len, X.shape[1]-gap_left[i]-gap_len], dim=0)
            lc = torch.vstack([left, right])
            X_gap[i] = lc
        X = X_gap

        # nanind = torch.where(Y[:, 4]>1e-4)[0]
        # Y = Y[nanind]
        # X_even = X_even[nanind]
        # X_rand = X_rand[nanind]

        # test_size = 1024
        train_size = len(Y)
        # train_size = 128
        print(f'Training Set Size: {train_size}')


        # # normalize
        # Y: t_0, t_E, u_0, rho, q, s, alpha, f_s
        Y[:, 3:6] = torch.log10(Y[:, 3:6])
        Y[:, 7] = torch.log10(Y[:, 7])
        # Y[:, 7] = torch.sin(Y[:, 6] / 180 * np.pi)
        # Y = torch.hstack([Y, torch.sin(Y[:, [6]] / 180 * np.pi)])
        # Y[:, 6] = torch.cos(Y[:, 6] / 180 * np.pi)
        Y[:, 6] = Y[:, 6] / 180 # * np.pi
        Y = Y[:, [2, 4, 5, 6, 7]]
        mean_y = torch.mean(Y, axis=0)
        std_y = torch.std(Y, axis=0)
        # std_mask = (std_y==0)
        # std_y[std_mask] = 1
        print(f'Y mean: {mean_y}\nY std: {std_y}')
        # Y = (Y - mean_y) / std_y
        # print(f'normalized Y mean: {torch.mean(Y)}\nY std: {torch.mean(torch.std(Y, axis=0)[~std_mask])}')

        mean_x_even = 14.5
        # std_x_even = 0.2
        # X_even[:, :, 1] = 10**((22-X_even[:, :, 1])/2.5)/1000
        # X_even[:, :, 1] = 22 - 2.5*torch.log10(1000*X_even[:, :, 1])
        X = X[:, :, :2]
        X[:, :, 1] = (X[:, :, 1] - mean_x_even - 2.5 * Y[:, [-1]]) / 0.2
        print(f'normalized X mean: {torch.mean(X[:, :, 1])}\nX std: {torch.mean(torch.std(X[:, :, 1], axis=0))}')
        # X_rand = X_rand[:, :, :2]
        # X_rand[:, :, 1] = 10**((22-X_rand[:, :, 1])/2.5)/1000
        # X_rand[:, :, 1] = 22 - 2.5*torch.log10(1000*X_rand[:, :, 1])
        # X_rand[:, :, 1] = (X_rand[:, :, 1] - mean_x_even) / std_x_even

        # Y = Y[:, :-1]

        # time rescale
        # X_even[:, :, 0] = X_even[:, :, 0] / 200
        # X_rand[:, :, 0] = X_rand[:, :, 0] / 200
            
        # CDE interpolation with log_sig
        depth = 3; window_length = 5; window_length_rand = 2
        train_logsig = torchcde.logsig_windows(X, depth, window_length=window_length)
        train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_logsig)
        # train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_even[:train_size, :, :])

        # train_logsig_rand = torchcde.logsig_windows(X_rand[:train_size, :, :], depth, window_length=window_length_rand)
        # train_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_logsig_rand)
        # train_rand_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(X_rand[:train_size, :, :])

        train_dataset = torch.utils.data.TensorDataset(train_coeffs, Y)
        # train_rand_dataset = torch.utils.data.TensorDataset(train_rand_coeffs, Y[:train_size])

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        # train_rand_dataloader = DataLoader(train_rand_dataset, batch_size=args.batch_size, shuffle=False)

        # train_mix_dataset = torch.utils.data.TensorDataset(torch.cat([train_coeffs, train_rand_coeffs[:, :train_coeffs.shape[1]]], dim=0), Y[:train_size].repeat(2, 1))
        # train_mix_dataloader = DataLoader(train_mix_dataset, batch_size=args.batch_size, shuffle=True)

        del Y
        del X
        # del X_even
        # del X_rand
        gc.collect()

    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)
    print(f'Model saved to {ckpt_path}')
    writer.close()
