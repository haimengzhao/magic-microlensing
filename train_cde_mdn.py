import os
import sys
import gc

import argparse
import numpy as np
import h5py
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import model.utils as utils
from model.cde_mdn import CDE_MDN

import torchcde

import torchinfo
from accelerate import Accelerator

# arg parser
parser = utils.get_parser()
args = parser.parse_args()

utils.makedirs(args.save)
os.environ['JOBLIB_TEMP_FOLDER'] = '/userhome/tmp'

#####################################################################################################

if __name__ == '__main__':
    # fix random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    experimentID = args.name
    ckpt_path = os.path.join(args.save, "estimator_" + str(experimentID) + '.ckpt')
    ckpt_path_load = os.path.join(args.save, "estimator_" + str(args.load) + '.ckpt')

    accelerator = Accelerator(log_with="tensorboard", project_dir=f'/userhome/training_log/')
    device = accelerator.device
    accelerator.print(f'ExperimentID: {experimentID}')
    accelerator.print(f'Num of GPUs available: {torch.cuda.device_count()}')
    accelerator.init_trackers(str(experimentID))

    ##################################################################
    # Load data
    accelerator.print(f'Loading Data: {args.dataset}')
    X, Y = utils.get_data(args.dataset)
    
    mean_y = torch.mean(Y, axis=0)
    std_y = torch.std(Y, axis=0)
    accelerator.print(f'Y mean: {mean_y}\nY std: {std_y}')
    accelerator.print(f'X mean: {torch.mean(X[:, :, 1])}\nX std: {torch.mean(torch.std(X[:, :, 1], axis=0))}')
    
    test_size = 4096
    train_size = len(Y) - test_size
    accelerator.print(f'Training Set Size: {train_size}; Test Set Size: {test_size}')
        
    # CDE interpolation with log_sig
    depth = 4; window_length = 10; 
    train_logsig, train_coeffs = utils.get_CDE_logsig_coeffs(X[:train_size, :, :], depth, window_length)
    accelerator.print(f'logsig last dim std: {torch.std(train_logsig[:, :, -1])}')

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, Y[:train_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    test_logsig, test_coeffs = utils.get_CDE_logsig_coeffs(X[(-test_size):, :, :].float().to(device), depth, window_length)
    test_Y = Y[(-test_size):].float().to(device)

    output_dim = Y.shape[-1]
    input_dim = train_logsig.shape[-1]
    latent_dim = args.latents
    
    del Y
    del X
    gc.collect()
    
    ##################################################################
    # Create the model
    model = CDE_MDN(input_dim, latent_dim, output_dim, args.ngaussians)
    model = model.to(device)
    accelerator.print(torchinfo.summary(model, verbose=0))
    ##################################################################
    # Load checkpoint and evaluate the model
    if args.load is not None:
        model = utils.load_model(model, ckpt_path_load)
    ##################################################################
    # Training
    accelerator.print('Start Training')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    num_batches = len(train_dataloader)
    
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    for epoch in range(args.resume, args.resume + args.niters):
        utils.update_learning_rate(optimizer, decay_rate = 0.9, lowest = args.lr / 100)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        accelerator.print(f'Epoch {epoch}, Learning Rate {lr}')
        accelerator.log({'learning_rate': lr}, step=epoch)
            
        for i, (batch_coeffs, batch_y) in enumerate(train_dataloader):

            batch_y = batch_y.float().to(device)
            batch_coeffs = batch_coeffs.float().to(device)

            optimizer.zero_grad()
            
            with accelerator.autocast():
                loss, rmse = utils.get_loss_rmse(model, batch_coeffs, batch_y)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), 20)
            total_norm = utils.get_grad_norm(model)
            accelerator.log({'gradient_norm': total_norm}, step=(epoch + i / num_batches))

            optimizer.step()

            print(f'Epoch {(epoch + i / num_batches):.2f}, Training loss {loss.item()}, rmse {rmse.tolist()}')
            utils.log_loss_rmse(accelerator, 'training', loss, rmse, epoch + i / num_batches)

            if accelerator.is_main_process:
                if (i + epoch * num_batches) % 20 == 0:
                    model.eval()
                    accelerator.save({
                        'args': args,
                        'state_dict': accelerator.unwrap_model(model).state_dict(),
                        }, ckpt_path)
                    print(f'Model saved to {ckpt_path}')

                    with torch.no_grad():
                        loss, rmse = utils.get_loss_rmse(model, test_coeffs, test_Y)

                        print(f'Epoch {(epoch + i / num_batches):.2f}, Test Loss {loss.item()}, rmse {rmse.tolist()}'）
                        utils.log_loss_rmse(accelerator, 'test', loss, rmse, epoch + i / num_batches)

                    model.train()
        
        # change dataset
        args.dataset = utils.get_next_dataset(args.dataset)
        
        accelerator.print(f'Loading Data: {args.dataset}')
        X, Y = utils.get_data(args.dataset)
            
        # CDE interpolation with log_sig
        train_logsig, train_coeffs = utils.get_CDE_logsig_coeffs(X, depth, window_length)

        train_dataset = torch.utils.data.TensorDataset(train_coeffs, Y)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

        del Y
        del X
        gc.collect()
        
        train_dataloader = accelerator.prepare(
            train_dataloader
        )

    if accelerator.is_main_process:
        accelerator.save({
            'args': args,
            'state_dict': accelerator.unwrap_model(model).state_dict(),
            }, ckpt_path)
        print(f'Model saved to {ckpt_path}')
    accelerator.end_training()
