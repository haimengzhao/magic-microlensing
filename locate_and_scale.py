import numpy as np
import h5py
import torch
import torchcde
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.locator import Locator
from model.scaler import Scaler

dataset = '/work/hmzhao/irregular-lc/roman-0.h5'
device_1 = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")
device_2 = torch.device("cuda:8" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    with h5py.File(dataset, mode='r') as dataset_file:
        Y = torch.tensor(dataset_file['Y'][...])
        X_even = torch.tensor(dataset_file['X_even'][...])
        X_rand = torch.tensor(dataset_file['X_random'][...])
    
    # preprocess
    nanind = torch.where(~torch.isnan(X_even[:, 0, 1]))[0]
    Y = Y[nanind]
    X_even = X_even[nanind]
    X_rand = X_rand[nanind, :, :2]

    depth = 2; window_length = 1; 
    logsig_even = torchcde.logsig_windows(X_even, depth, window_length=window_length)
    logsig_rand = torchcde.logsig_windows(X_rand, depth, window_length=window_length)
    coeffs_even = torchcde.hermite_cubic_coefficients_with_backward_differences(logsig_even)
    coeffs_rand = torchcde.hermite_cubic_coefficients_with_backward_differences(logsig_rand)

    # load locator
    print('loading locator')
    checkpt = torch.load('/work/hmzhao/experiments/locator/best_rand_locator.ckpt', map_location='cpu')
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dict']

    output_dim = 2
    input_dim = logsig_even.shape[-1]
    latent_dim = ckpt_args.latents

    model_loc = Locator(input_dim, latent_dim, output_dim, device_1).to(device_1)
    model_dict = model_loc.state_dict()
    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict) 
    # 3. load the new state dict
    model_loc.load_state_dict(state_dict)
    model_loc.to(device_1)

    # load scaler
    print('loading scaler')
    checkpt = torch.load('/work/hmzhao/experiments/scaler/best_rand_scaler.ckpt', map_location='cpu')
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dict']

    output_dim = 1
    input_dim = logsig_even.shape[-1]
    latent_dim = ckpt_args.latents

    model_sca = Scaler(input_dim, latent_dim, output_dim, device_2).to(device_2)
    model_dict = model_sca.state_dict()
    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict) 
    # 3. load the new state dict
    model_sca.load_state_dict(state_dict)
    model_sca.to(device_2)

    # inference
    batchsize = 128
    pred = torch.zeros((len(Y), 2))
    pred_s = torch.zeros((len(Y), 1))
    pred_rand = torch.zeros((len(Y), 2))
    pred_rand_s = torch.zeros((len(Y), 1))
    z = torch.zeros((batchsize, 4000)).to(device_1)
    model_loc.eval()
    model_loc.threshold = 0.5
    model_sca.eval()
    for i in tqdm(range((len(Y) // batchsize) + 1)):
        batch = coeffs_even[i*batchsize:min(i*batchsize+batchsize, len(Y))].float().to(device_1)
        batch_rand = coeffs_rand[i*batchsize:min(i*batchsize+batchsize, len(Y))].float().to(device_1)
        pred[i*batchsize:min(i*batchsize+batchsize, len(Y))] = model_loc(batch, z[:len(batch)])[0].detach().cpu()
        pred_s[i*batchsize:min(i*batchsize+batchsize, len(Y))] = model_sca(batch).detach().cpu()
        pred_rand[i*batchsize:min(i*batchsize+batchsize, len(Y))] = model_loc(batch_rand, z[:len(batch_rand)])[0].detach().cpu()
        pred_rand_s[i*batchsize:min(i*batchsize+batchsize, len(Y))] = model_sca(batch_rand).detach().cpu()

    # transform
    X_even[:, :, 0] = (X_even[:, :, 0] - pred[:, [0]]) / pred[:, [1]]
    X_even[:, :, 1] = 10. ** ((22 - X_even[:, :, 1]) / 2.5)
    X_even[:, :, 1] = X_even[:, :, 1] / 1000 - (1 - (10. ** pred_s)) / (10. ** pred_s)
    X_rand[:, :, 0] = (X_rand[:, :, 0] - pred_rand[:, [0]]) / pred_rand[:, [1]]
    X_rand[:, :, 1] = 10. ** ((22 - X_rand[:, :, 1]) / 2.5)
    X_rand[:, :, 1] = X_rand[:, :, 1] / 1000 - (1 - (10. ** pred_rand_s)) / (10. ** pred_rand_s)

    for i in tqdm(range(len(Y))):
        lc = X_even[i]
        lc = lc[torch.where((lc[:, 0] <= 2) * (lc[:, 0] >= -2))]
        lc = torch.cat([lc, lc[-1].expand(X_even.shape[1] - len(lc), lc.shape[-1])])
        X_even[i] = lc

        try:
            lc = X_rand[i]
            lc = lc[torch.where((lc[:, 0] <= 2) * (lc[:, 0] >= -2))]
            lc = torch.cat([lc, lc[-1].expand(X_rand.shape[1] - len(lc), lc.shape[-1])])
            X_rand[i] = lc
        except:
            print(X_rand[i, :, 0])
            plt.plot(X_rand[i, :, 0], X_rand[i, :, 1])
            plt.show()
            X_rand[i] = np.nan

    # save
    with h5py.File(dataset[:-3]+'-located.h5', mode='w') as dataset_file:
        dataset_file['Y'] = Y
        dataset_file['X_even'] = X_even
        dataset_file['X_random'] = X_rand
