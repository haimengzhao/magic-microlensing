import os
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm.notebook import tqdm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import MulensModel as mm

import h5py
import torchcde
import argparse

def mdn_loss_fisher(pi, normal, y, fisher, n_sample=1024):
    """Calculate MDN loss function given the fisher matrix that induce a Gaussian distribution around y.

    Args:
        pi (nn.distributions.OneHotCategorical): mixture weights.
        normal (nn.distributions.Normal): Gaussians.
        y (tensor): target, i.e. the ground truth parameters.
        fisher (tensor): Fisher information matrix.
        n_sample (int, optional): number of samples to estimate the loss. Defaults to 1024.

    Returns:
        loss (tensor): loss averaged on a batch of data.
    """
    # sample from the Gaussian distribution induced by the fisher matrix
    # check positive definite
    # min_eig = torch.linalg.eigvalsh(fisher).min()
    # print('min_eig of fisher', min_eig)
    
    # WATCH OUT: manually calculating covariance matrix
    # directly using precision matrix with very large entries 
    # will result in negative eigenvalues when calculating covariance 
    # matrix due to numerical instability
    
    cov = torch.linalg.inv(fisher)
    cov += torch.eye(cov.shape[-1], device=cov.device)
    # min_eig = torch.linalg.eigvalsh(cov).min()
    # print('min_eig of cov', min_eig)
    dist_fisher = torch.distributions.MultivariateNormal(y, covariance_matrix=cov)
    y_sample = dist_fisher.sample((n_sample,)) # (n_sample, batch_size, n_parameters)
    # duplicate the mixture weights and Gaussians
    loglik = normal.log_prob(y_sample.unsqueeze(2).expand_as(torch.tile(normal.loc.unsqueeze(0), (n_sample, 1, 1, 1))))
    loglik = torch.sum(loglik, dim=-1)
    loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=-1)
    return loss.mean()

def mdn_loss(pi, normal, y):
    """Calculate MDN loss function.

    Args:
        pi (nn.distributions.OneHotCategorical): mixture weights.
        normal (nn.distributions.Normal): Gaussians.
        y (tensor): target, i.e. the ground truth parameters.

    Returns:
        loss (tensor): loss averaged on a batch of data.
    """
    loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
    loglik = torch.sum(loglik, dim=2)
    loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
    return loss.mean()

def sample(pi, normal):
    """Sample from MDN.

    Args:
        pi (nn.distributions.OneHotCategorical): mixture weights.
        normal (nn.distributions.Normal): Gaussians.

    Returns:
        samples (tensor): one sample for each light curve.
    """
    samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
    return samples

def get_parser():
    parser = argparse.ArgumentParser('Estimator')
    parser.add_argument('--niters', type=int, default=50)
    parser.add_argument('--lr',  type=float, default=1e-4, help="Starting learning rate")
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='/work/hmzhao/data/data-0.h5', help="Path for dataset")
    parser.add_argument('--save', type=str, default='/work/hmzhao/training_ckpt', help="Path for save checkpoints")
    parser.add_argument('--name', type=str, default='test', help="Name of the experiment")
    parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
    parser.add_argument('--resume', type=int, default=0, help="Epoch to resume.")
    parser.add_argument('-r', '--random-seed', type=int, default=42, help="Random_seed")
    parser.add_argument('-ng', '--ngaussians', type=int, default=12, help="Number of Gaussians in mixture density network")
    parser.add_argument('-l', '--latents', type=int, default=512, help="Dim of the latent state")

    return parser

def get_next_dataset(data_path):
    if os.path.exists(data_path[:-4] + str((int(data_path[-4])+1)) + '.h5'):
        data_path = data_path[:-4] + str((int(data_path[-4])+1)) + '.h5'
    else:
        data_path = data_path[:-4] + '0.h5'
    return data_path

def load_model(model, ckpt_path_load, device):
    # Load checkpoint.
    checkpt = torch.load(ckpt_path_load, map_location='cpu')
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dict']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if ((k in model_dict))}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    model.to(device)
    
    return model

def get_data(data_path, random_shift=False, inject_gap=False, fisher=False):
    with h5py.File(data_path, mode='r') as dataset_file:
        X = torch.tensor(dataset_file['X'][...])
        Y = torch.tensor(dataset_file['Y'][...])
        if fisher:
            F = torch.tensor(dataset_file['F'][...])

    # filter nan
    nanind = torch.where(~torch.isnan(X[:, 0, 1]))[0]
    Y = Y[nanind]
    X = X[nanind]
    F = F[nanind]
    
    # make F postive definite
    # min_eig = torch.linalg.eigvalsh(F).min()
    # print('min_eig', min_eig)
    # if min_eig <= 0:
    # F = F + (-min_eig + 1) * torch.eye(F.shape[-1])
    F += torch.eye(F.shape[-1]) * 10
    # print(F.shape, torch.linalg.eigvalsh(F).shape)
    min_eig = torch.linalg.eigvalsh(F).min()
    print('min_eig', min_eig)

    if inject_gap:
        n_chunks = 25 * 4
        gap_len = int(500 * 4 / n_chunks)
        gap_left = torch.randint(0, X.shape[1]-gap_len, (len(X),))
        X_gap = torch.zeros((X.shape[0], X.shape[1]-gap_len, X.shape[2]))
        for i in range(len(X)):
            left, gap, right = torch.split(X[i], [gap_left[i], gap_len, X.shape[1]-gap_left[i]-gap_len], dim=0)
            lc = torch.vstack([left, right])
            X_gap[i] = lc
        X = X_gap
    
    if random_shift:
        # random shift and rescale
        X[:, :, 0] = X[:, :, 0] + torch.randn(X.shape[0]).reshape(-1, 1) * 0.5
        X[:, :, 0] = X[:, :, 0] * (1 + torch.randn(X.shape[0]).reshape(-1, 1) * 0.2)

    # # normalize
    # Y: t_0, t_E, u_0, rho, q, s, alpha, f_s
    Y = Y[:, 2:] # drop t_0 t_E
    F = F[:, 2:, 2:] # drop t_0 t_E
    # 0: u_0, 1: rho, 2: q, 3:s, 4: alpha, 5: f_s
    Y[:, 1:4] = torch.log10(Y[:, 1:4])
    Y[:, 5] = torch.log10(Y[:, 5])
    Y[:, 4] = Y[:, 4] / 180
    # 0: u_0, 1: lg rho, 2: lg q, 3:lg s, 4: alpha/180, 5: lg f_s

    X = X[:, :, :2] # remove errorbar
    
    if fisher:
        return X, Y, F
    return X, Y
    
def get_CDE_logsig_coeffs(X, depth=3, window_length=5):
    logsig = torchcde.logsig_windows(X, depth, window_length=window_length)
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(logsig)
    return logsig, coeffs

def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_loss_rmse(model, coeffs, y, fisher=None, n_sample=1024):
    pi, normal = model(coeffs)
    if fisher is not None:
        loss = mdn_loss_fisher(pi, normal, y, fisher, n_sample)
    else:
        loss = mdn_loss(pi, normal, y)
    pred_y = sample(pi, normal)

    rmse = torch.sqrt(torch.mean((y - pred_y)**2, dim=0)).detach().cpu()

    return loss, rmse
    
def log_loss_rmse(accelerator, name, loss, rmse, step):
    accelerator.log({
        f'{name}/loss': loss.item(),
        f'{name}/rmse_u0': rmse[0],
        f'{name}/rmse_rho': rmse[1],
        f'{name}/rmse_lgq': rmse[2],
        f'{name}/rmse_lgs': rmse[3],
        f'{name}/rmse_alpha_180': rmse[4],
        f'{name}/rmse_lgfs': rmse[5],
    }, step=step)

def ecdf(x):
    """Compute the empirical cumulative distribution function of a dataset.

    Args:
        x (array): Dataset.

    Returns:
        xval (array): x values where data points are presented.
        cdf (array): cumulative distribution function values.
    """
    xnew = np.sort(x)
    xval, cdf = [], []
    for i in range(len(xnew)):
        cdf.append(i)
        xval.append(xnew[i])
        cdf.append(i+1)
        xval.append(xnew[i])
    cdf = np.array(cdf)/cdf[-1]
    xval = np.array(xval)
    return xval, cdf

def get_fsfb(amp, flux, ferr):
    """Compute the source flux and background flux from the computed magnification and the observed flux.

    Args:
        amp (array): computed magnification.
        flux (array): observed flux.
        ferr (array): observed flux uncertainties.

    Returns:
        chi2 (float): chi2 value.
        fs (float): source flux.
        fb (float): background flux.
        fserr (float): source flux uncertainty.
        fberr (float): background flux uncertainty.
    """
    sig2 = ferr**2
    wght = flux/sig2
    d = np.ones(2)
    d[0] = np.sum(wght*amp)
    d[1] = np.sum(wght)
    b = np.zeros((2,2))
    b[0,0] = np.sum(amp**2/sig2)
    b[0,1] = np.sum(amp/sig2)
    b[1,0] = b[0,1]
    b[1,1] = np.sum(1./sig2)
    c = np.linalg.inv(b)
    fs = np.sum(c[0]*d)
    fb = np.sum(c[1]*d)
    fserr = np.sqrt(c[0,0])
    fberr = np.sqrt(c[1,1])
    fmod = fs*amp+fb
    chi2 = np.sum((flux-fmod)**2/sig2)
    return chi2,fs,fb,fserr,fberr

def getfsfb(times, iflux, iferr, t_0, t_E, u_0, lgrho, lgq, lgs, alpha_180, **kwargs):
    """Compute the source flux and background flux from the binary microlensing parameters and the observed flux
    using MulensModel.

    Args:
        times (array): time stamps.
        iflux (array): observed flux.
        iferr (array): observed flux uncertainties.
        t_0 (float): t_0.
        t_E (float): t_E.
        u_0 (float): u_0.
        lgrho (float): lg of rho.
        lgq (float): lg of mass ratio q.
        lgs (float): lg of seperation s.
        alpha_180 (float): alpha divided by 180.

    Returns:
        chi2 (float): chi2 value.
        fs (float): source flux.
        fb (float): background flux.
        fserr (float): source flux uncertainty.
        fberr (float): background flux uncertainty.
    """
    parameters = {
            't_0': t_0,
            't_E': t_E,
            'u_0': u_0,
            'rho': 10**lgrho, 
            'q': 10**lgq, 
            's': 10**lgs, 
            'alpha': alpha_180*180,
        }
    modelmm = mm.Model(parameters, coords=None)
    modelmm.set_magnification_methods([parameters['t_0']-2*parameters['t_E'], 'VBBL', parameters['t_0']+2*parameters['t_E']])
    iamp = modelmm.get_magnification(times)
    sig2 = iferr**2
    wght = iflux/sig2
    d = np.ones(2)
    d[0] = np.sum(wght*iamp)
    d[1] = np.sum(wght)
    b = np.zeros((2,2))
    b[0,0] = np.sum(iamp**2/sig2)
    b[0,1] = np.sum(iamp/sig2)
    b[1,0] = b[0,1]
    b[1,1] = np.sum(1./sig2)
    c = np.linalg.inv(b)
    fs = np.sum(c[0]*d)
    fb = np.sum(c[1]*d)
    fserr = np.sqrt(c[0,0])
    fberr = np.sqrt(c[1,1])
    fmod = fs*iamp+fb
    chi2 = np.sum((iflux-fmod)**2/sig2)
    return chi2,fs,fb,fserr,fberr

def infer_lgfs(X, pred, relative_uncertainty=0.03):
    """Infer the logarithm of the source flux for each light curve in a dataset.

    Args:
        X (array): dataset with shape (n_light_curves, n_time_stamps, 2).
        pred (array): prediction of other parameters with shape (n_light_curves, :).
        relative_uncertainty (float, optional): relative uncertainty of the light curve computed in flux. Defaults to 0.03.

    Returns:
        pred (array): the input pred appended with the inferred logarithm of the source flux.
    """
    lgfs = np.zeros((pred.shape[0], 1))
    for i in tqdm(range(pred.shape[0])):
        times = X[i, :, 0]
        iflux = 10 ** (X[i, :, 1] / 5 / (-2.5))
        iferr = relative_uncertainty * iflux
        chi2, fs, fb, fserr, fberr = getfsfb(times, iflux, iferr, 0, 1, pred[i, 0], -3, pred[i, 1], pred[i, 2], pred[i, 3])
        lgfs[i, 0] = np.log10(fs / (fs + fb))
    pred = np.hstack([pred, lgfs])
    return pred


def inference(model, total_size, batch_size, coeffs, device='cpu', full_cov=False, **kwargs):
    """Infer the posterior distribution of the parameters given the preprocessed light curve dataset.

    Args:
        model (estimator): the estimator.
        total_size (int): total size of the dataset.
        batch_size (int): batch size.
        coeffs (tensor): preprocessed light curve data, shape (total_size, :).
        device (str, optional): torch device. Defaults to 'cpu'.
        full_cov (bool, optional): whether to use diagonal covariance of full covariance Gaussians. Defaults to False.

    Returns:
        pis (tensor): predicted weights of the Gaussian mixture, shape (total_size, n_components).
        locs (tensor): predicted means of the Gaussians, shape (total_size, n_components, n_parameters).
        scales (tensor): predicted variances of the Gaussians, shape (total_size, n_components, n_parameters) if full_cov==False, shape (total_size, n_components, n_parameters, n_parameters) if full_cov==True.
    """
    num = total_size
    batchsize = batch_size
    n_gaussian = model.n_gaussian
    output_dim = model.output_dim
    pis = torch.zeros((num, n_gaussian))
    locs = torch.zeros((num, n_gaussian, output_dim))
    if full_cov:
        scales = torch.zeros((num, n_gaussian, output_dim, output_dim))
    else:
        scales = torch.zeros((num, n_gaussian, output_dim))
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(int(np.ceil(num / batchsize)))):
            batch = coeffs[i*batchsize:min(i*batchsize+batchsize, num)].float().to(device)
            pi, normal = model(batch)
            pis[i*batchsize:min(i*batchsize+batchsize, num)] = pi.probs.detach().cpu()
            locs[i*batchsize:min(i*batchsize+batchsize, num)] = normal.loc.detach().cpu()
            if full_cov:
                scales[i*batchsize:min(i*batchsize+batchsize, num)] = normal.covariance_matrix.detach().cpu()
            else:
                scales[i*batchsize:min(i*batchsize+batchsize, num)] = normal.scale.detach().cpu()
    return pis, locs, scales

def get_loglik(pi, loc, scale, x, margin_dim, exp=False, individual_gaussian=False):
    shape = x.shape
    if len(scale.shape) > len(loc.shape):
        # for full covariance
        scale = scale[..., margin_dim, margin_dim]
    else:
        scale = scale[..., margin_dim]
    loc = loc[..., margin_dim]
    normal = torch.distributions.Normal(loc, scale)
    x = x.reshape(-1, loc.shape[0], 1).tile(1, loc.shape[-1])
    loglik = normal.log_prob(x).reshape(*shape[:-1], -1)
    if not individual_gaussian:
        loglik = torch.logsumexp(torch.log(pi) + loglik, dim=-1)
    if exp:
        return torch.exp(loglik)
    return loglik

def get_peak_pred(pis, locs, scales, Y, n_step=1000, verbose=False):
    """Get the global peak and combined marginal closest peak as the prediction of the MDN posterior.

    Args:
        pis (tensor): weights of the Gaussian mixture, shape (n_light_curves, n_components).
        locs (tensor): means of the Gaussians, shape (n_light_curves, n_components, n_parameters).
        scales (tensor): variances of the Gaussians, shape (n_light_curves, n_components, n_parameters) if full_cov==False, shape (n_light_curves, n_components, n_parameters, n_parameters) if full_cov==True.
        Y (tensor): ground truth, shape (n_light_curves, n_parameters).
        n_step (int, optional): number of steps when dividing the parameter interval. Defaults to 1000.
        verbose (bool, optional): whether to print the progress. Defaults to False.

    Returns:
        pred_global (tensor): global peak, shape (n_light_curves, n_parameters).
        pred_global_loglik (tensor): global peak log likelihood, shape (n_light_curves, n_parameters).
        pred_closest (tensor): closest peak, shape (n_light_curves, n_parameters).
        pred_closest_loglik (tensor): closest peak log likelihood, shape (n_light_curves, n_parameters).
    """
    num = len(pis); output_dim = locs.shape[-1]
    pred_global = torch.zeros((num, output_dim))
    pred_global_loglik = torch.zeros((num, output_dim))
    pred_close = torch.zeros((num, output_dim))
    pred_close_loglik = torch.zeros((num, output_dim))
    grid = [torch.linspace(0, 1, n_step),
        torch.linspace(-4, 0, n_step),
        torch.linspace(-0.6, 0.6, n_step),
        torch.linspace(0, 2, n_step),
        torch.linspace(-1, 0, n_step)]
    for dim in tqdm(range(output_dim)):
        param_list = grid[dim].reshape(-1, 1, 1).tile(1, num, 1) 
        loglik = get_loglik(pis, locs, scales, param_list, margin_dim=dim, exp=False).transpose(1, 0)
        for i in tqdm(range(num)):
            peaks = find_peaks(loglik[i])[0]
            if len(peaks) == 0:
                pred_global[i, dim] = grid[dim][torch.argmax(loglik[i])]
                pred_close[i, dim] = grid[dim][torch.argmax(loglik[i])]
                pred_global_loglik[i, dim] = torch.max(loglik[i])
                pred_close_loglik[i, dim] = torch.max(loglik[i])
                if verbose:
                    print('no peak found, use maximum instead')
                    plt.plot(grid[dim], loglik[i])
                    plt.vlines(Y[i, dim], 0, 10, color='red')
                    plt.vlines(grid[dim][torch.argmax(loglik[i])], 0, 10, color='blue')
                    print(Y[i, dim])
                    plt.show()
            else:
                order = torch.argsort(loglik[i, peaks], descending=True)
                global_peak = grid[dim][peaks[order[0]]]
                close_peak = grid[dim][peaks][torch.argmin((grid[dim][peaks] - Y[i, dim])**2)]
                pred_global[i, dim] = global_peak
                pred_close[i, dim] = close_peak
                pred_global_loglik[i, dim] = loglik[i][peaks[order[0]]]
                pred_close_loglik[i, dim] = loglik[i][peaks][torch.argmin((grid[dim][peaks] - Y[i, dim])**2)]
    return pred_global, pred_global_loglik, pred_close, pred_close_loglik

def plot_params(num, Y, pred_global, pred_global_loglik, pred_close, pred_close_loglik, 
                title=None, figsize=(16, 8), labelsize=14, alpha=0.1, save=None,
                example_groundtruth=np.ones(5)*np.inf, example_pred=np.ones(5)*np.inf):
    """Plot the predicted v.s. groundtruth parameters.
    """
    rmse = []

    fig = plt.figure(figsize=figsize)
    axq = plt.subplot2grid(shape=(2, 4), loc=(0, 0), rowspan=1, colspan=1)
    axq.axis('square')
    axq.set_xlim(-3, 0)
    axq.set_ylim(-3, 0)
    axq.set_xlabel(r'true $\lg q$', fontsize=labelsize)
    axq.set_ylabel(r'predicted $\lg q$', fontsize=labelsize)
    axq.scatter(Y[:num, 1], pred_global.numpy()[:num, 1], s=3, cmap='Blues', label='global', alpha=alpha, rasterized=True)
    axq.scatter(Y[:num, 1], pred_close.numpy()[:num, 1], s=3, cmap='Oranges', label='close', alpha=alpha, rasterized=True)
    axq.scatter(example_groundtruth[1], example_pred[1], s=100, color='black', marker='*')
    axq.plot(np.linspace(-3, 0), np.linspace(-3, 0), color='b', linestyle='dashed')
    # axq.legend(loc='lower right')
    print('mse of log10q global: ', torch.mean((Y[:num, 1] -  pred_global.numpy()[:num, 1])**2).detach().cpu().item())
    print('mse of log10q close: ', torch.mean((Y[:num, 1] -  pred_close.numpy()[:num, 1])**2).detach().cpu().item())
    constraint_ind = pred_global_loglik[:num, 1]>np.log(2*1/3)
    print('constraint', torch.sum(constraint_ind).item()/num)
    print('correct', torch.sum(pred_global[:num, 1][constraint_ind]==pred_close[:num, 1][constraint_ind]).item()/torch.sum(constraint_ind).item())
    at = AnchoredText(
        "RMSE=%.4f" % (np.sqrt(torch.mean((Y[:num, 1] -  pred_close.numpy()[:num, 1])**2).detach().cpu().item())), prop=dict(size=12), frameon=False, loc='upper left')
    axq.add_artist(at)
    rmse.append(np.sqrt(torch.mean((Y[:num, 1] -  pred_close.numpy()[:num, 1])**2).detach().cpu().item()))
    
    axs = plt.subplot2grid(shape=(2, 4), loc=(0, 1), rowspan=1, colspan=1)
    axs.axis('square')
    axs.set_xlim(np.log10(0.3), np.log10(3))
    axs.set_ylim(np.log10(0.3), np.log10(3))
    axs.set_xlabel(r'true $\lg s$', fontsize=labelsize)
    axs.set_ylabel(r'predicted $\lg s$', fontsize=labelsize)
    axs.scatter(Y[:num, 2], pred_global.numpy()[:num, 2], s=3, cmap='Blues', label='global', alpha=alpha, rasterized=True)
    axs.scatter(Y[:num, 2], pred_close.numpy()[:num, 2], s=3, cmap='Oranges', label='close', alpha=alpha, rasterized=True)
    axs.scatter(example_groundtruth[2], example_pred[2], s=100, color='black', marker='*')
    axs.plot(np.linspace(-0.6, 0.6), np.linspace(-0.6, 0.6), color='b', linestyle='dashed')
    # axs.legend(loc='lower right')
    print('mse of log10s global: ', torch.mean((Y[:num, 2] -  pred_global.numpy()[:num, 2])**2).detach().cpu().item())
    print('mse of log10s close: ', torch.mean((Y[:num, 2] -  pred_close.numpy()[:num, 2])**2).detach().cpu().item())
    constraint_ind = pred_global_loglik[:num, 2]>np.log(2*1/2/np.log10(3))
    print('constraint', torch.sum(constraint_ind).item()/num)
    print('correct', torch.sum(pred_global[:num, 2][constraint_ind]==pred_close[:num, 2][constraint_ind]).item()/torch.sum(constraint_ind).item())
    at = AnchoredText(
        "RMSE=%.4f" % (np.sqrt(torch.mean((Y[:num, 2] -  pred_close.numpy()[:num, 2])**2).detach().cpu().item())), prop=dict(size=12), frameon=False, loc='upper left')
    axs.add_artist(at)
    rmse.append(np.sqrt(torch.mean((Y[:num, 2] -  pred_close.numpy()[:num, 2])**2).detach().cpu().item()))

    axu = plt.subplot2grid(shape=(2, 4), loc=(0, 2), rowspan=1, colspan=1)
    axu.axis('square')
    axu.set_xlim(0, 1)
    axu.set_ylim(0, 1)
    axu.set_xlabel(r'true $u_0$', fontsize=labelsize)
    axu.set_ylabel(r'predicted $u_0$', fontsize=labelsize)
    axu.scatter(Y[:num, 0], pred_global.numpy()[:num, 0], s=3, cmap='Blues', label='global', alpha=alpha, rasterized=True)
    axu.scatter(Y[:num, 0], pred_close.numpy()[:num, 0], s=3, cmap='Oranges', label='close', alpha=alpha, rasterized=True)
    axu.scatter(example_groundtruth[0], example_pred[0], s=100, color='black', marker='*')
    axu.plot(np.linspace(0, 1), np.linspace(0, 1), color='b', linestyle='dashed')
    # axu.legend(loc='lower right')
    print('mse of u0: ', torch.mean((Y[:num, 0] -  pred_global.numpy()[:num, 0])**2).detach().cpu().item())
    print('mse of u0: ', torch.mean((Y[:num, 0] -  pred_close.numpy()[:num, 0])**2).detach().cpu().item())
    constraint_ind = pred_global_loglik[:num, 0]>np.log(2*1/1)
    print('constraint', torch.sum(constraint_ind).item()/num)
    print('correct', torch.sum(pred_global[:num, 0][constraint_ind]==pred_close[:num, 0][constraint_ind]).item()/torch.sum(constraint_ind).item())
    at = AnchoredText(
        "RMSE=%.4f" % (np.sqrt(torch.mean((Y[:num, 0] -  pred_close.numpy()[:num, 0])**2).detach().cpu().item())), prop=dict(size=12), frameon=False, loc='upper left')
    axu.add_artist(at)
    rmse.append(np.sqrt(torch.mean((Y[:num, 0] -  pred_close.numpy()[:num, 0])**2).detach().cpu().item()))

    axa = plt.subplot2grid(shape=(2, 4), loc=(0, 3), rowspan=1, colspan=1)
    axa.axis('square')
    axa.set_xlim(0, 360)
    axa.set_ylim(0, 360)
    axa.set_xlabel(r'true $\alpha$ (deg)', fontsize=labelsize)
    axa.set_ylabel(r'predicted $\alpha$ (deg)', fontsize=labelsize)
    axa.scatter(Y[:num, 3]*180, pred_global.numpy()[:num, 3]*180, s=3, cmap='Blues', label='global', alpha=alpha, rasterized=True)
    axa.scatter(Y[:num, 3]*180, pred_close.numpy()[:num, 3]*180, s=3, cmap='Oranges', label='close', alpha=alpha, rasterized=True)
    axa.scatter(example_groundtruth[3]*180, example_pred[3]*180, s=100, color='black', marker='*')
    axa.plot(np.linspace(0, 360), np.linspace(0, 360), color='b', linestyle='dashed')
    # axa.legend(loc='lower right')
    print('mse of alpha global: ', torch.mean((Y[:num, 3]*180 -  pred_global.numpy()[:num, 3]*180)**2).detach().cpu().item())
    print('mse of alpha close: ', torch.mean((Y[:num, 3]*180 -  pred_close.numpy()[:num, 3]*180)**2).detach().cpu().item())
    constraint_ind = pred_global_loglik[:num, 3]>np.log(2*1/2)
    print('constraint', torch.sum(constraint_ind).item()/num)
    print('correct', torch.sum(pred_global[:num, 3][constraint_ind]==pred_close[:num, 3][constraint_ind]).item()/torch.sum(constraint_ind).item())
    at = AnchoredText(
        "RMSE=%.3f" % np.sqrt((torch.mean((Y[:num, 3] -  pred_close.numpy()[:num, 3])**2).detach().cpu().item())*180), prop=dict(size=12), frameon=False, loc='upper left')
    axa.add_artist(at)
    rmse.append(np.sqrt(torch.mean((Y[:num, 3] -  pred_close.numpy()[:num, 3])**2).detach().cpu().item()))

    axf = plt.subplot2grid(shape=(2, 4), loc=(1, 0), rowspan=1, colspan=1)
    axf.axis('square')
    axf.set_xlim(-1, 0)
    axf.set_ylim(-1, 0)
    axf.set_xlabel(r'true $\lg f_s$', fontsize=labelsize)
    axf.set_ylabel(r'predicted $\lg f_s$', fontsize=labelsize)
    axf.scatter(Y[:num, 4], pred_global.numpy()[:num, 4], s=3, cmap='Blues', label='global', alpha=alpha, rasterized=True)
    axf.scatter(Y[:num, 4], pred_close.numpy()[:num, 4], s=3, cmap='Oranges', label='close', alpha=alpha, rasterized=True)
    axf.scatter(example_groundtruth[4], example_pred[4], s=100, color='black', marker='*')
    axf.plot(np.linspace(-1, 0), np.linspace(-1, 0), color='b', linestyle='dashed')
    # axf.legend(loc='lower right')
    print('mse of log10fs global: ', torch.mean((Y[:num, 4] -  pred_global.numpy()[:num, 4])**2).detach().cpu().item())
    print('mse of log10fs close: ', torch.mean((Y[:num, 4] -  pred_close.numpy()[:num, 4])**2).detach().cpu().item())
    # constraint_ind = pred_global_loglik[:num, 4]>np.log(2*1/1)
    # print('constraint', torch.sum(constraint_ind).item()/num)
    # print('correct', torch.sum(pred_global[:num, 4][constraint_ind]==pred_close[:num, 4][constraint_ind]).item()/torch.sum(constraint_ind).item())
    at = AnchoredText(
        "RMSE=%.4f" % np.sqrt((torch.mean((Y[:num, 4] -  pred_close.numpy()[:num, 4])**2).detach().cpu().item())), prop=dict(size=12), frameon=False, loc='upper left')
    axf.add_artist(at)
    rmse.append(np.sqrt(torch.mean((Y[:num, 4] -  pred_close.numpy()[:num, 4])**2).detach().cpu().item()))
    
    plt.tight_layout()
        
    if title != None:
        fig.suptitle(title)

    if save != None:
        plt.savefig(save)
    
    plt.show()
    return rmse
    

def simulate_lc(t_0, t_E, u_0, lgrho, lgq, lgs, alpha_180, lgfs, relative_uncertainty=0, n_points=1000, orig=False, orig_param=False, tmin=None, tmax=None):
    """Simulate a with MulensModel.

    Args:
        t_0 (float): t_0.
        t_E (float): t_E.
        u_0 (float): u_0.
        lgrho (float): lg of rho.
        lgq (float): lg of q.
        lgs (float): lg of s.
        alpha_180 (float): alpha divided by 180.
        lgfs (float): lg of fs.
        relative_uncertainty (float, optional): relative uncertainty in flux. Defaults to 0.
        n_points (int, optional): number of data points to plot. Defaults to 1000.
        orig (bool, optional): whether to return the original, unpreprocessed light curve. Defaults to False.
        orig_param (bool, optional): whether the parameters are given in original, unpreproceesed form. Defaults to False.
        tmin (float, optional): start time of light curve. Defaults to None.
        tmax (float, optional): end time light curve. Defaults to None.

    Returns:
        lc: the simulated light curve, shape (n_points, 2).
    """
    fs = 10**lgfs
    parameters = {
            't_0': t_0,
            't_E': t_E,
            'u_0': u_0,
            'rho': 10**lgrho, 
            'q': 10**lgq, 
            's': 10**lgs, 
            'alpha': alpha_180*180,
        }
    if orig_param:
        fs = lgfs
        parameters = {
                't_0': t_0,
                't_E': t_E,
                'u_0': u_0,
                'rho': lgrho, 
                'q': lgq, 
                's': lgs, 
                'alpha': alpha_180,
            }
    modelmm = mm.Model(parameters, coords=None)
    if tmin == None:
        tmin = parameters['t_0']-2*parameters['t_E']
    if tmax == None:
        tmax = parameters['t_0']+2*parameters['t_E']
    times = modelmm.set_times(t_start=tmin, t_stop=tmax, n_epochs=n_points)
    modelmm.set_magnification_methods([tmin, 'VBBL', tmax])
    magnification = modelmm.get_magnification(times)
    flux = 1000 * (magnification + (1-fs)/fs)
    flux *= 1 + relative_uncertainty * np.random.randn(len(flux))
    if orig:
        mag = (22 - 2.5 * np.log10(flux) - 14.5 - 2.5*np.log10(fs))
    else:
        mag = (22 - 2.5 * np.log10(flux) - 14.5 - 2.5*np.log10(fs)) / 0.2
    lc = np.stack([times, mag], axis=-1)
    return lc


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "mean",
    sigmoid = False,
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    if sigmoid:
        p = torch.sigmoid(inputs)
    else:
        p = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

class DiceLoss(nn.Module):
    """Dice loss.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.InstanceNorm1d(out_layer)
        self.relu = nn.PReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out       

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out          

class UNET_1D(nn.Module):
    '''The 1-dim version of U-Net.
    Ref: https://www.kaggle.com/super13579/u-net-1d-cnn-with-pytorch
    '''
    def __init__(self, input_dim, layer_n, kernel_size, depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,5, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,5, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')
        
        self.outcov = nn.Conv1d(self.layer_n, 1, kernel_size=self.kernel_size, stride=1,padding = 3)
        self.sig = nn.Sigmoid()
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)
        
        #############Decoder####################
        
        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        
        out = self.sig(out)
        
        return out


def makedirs(dirname):
    '''
    Make directory if not exist.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class ResBlock(nn.Module):
    def __init__(self, dim, hidden_dim, nonlinear=nn.PReLU, layernorm=False):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.nonlinear1 = nonlinear()
        self.linear2 = nn.Linear(hidden_dim, dim)

        self.layernorm = layernorm
        if layernorm:
            self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.nonlinear1(out)

        out = self.linear2(out)

        if self.layernorm:
            out = self.layernorm(out)

        out += residual
        
        return out

class CNNResBlock(nn.Module):
    def __init__(self, dim, hidden_dim, nonlinear=nn.PReLU, layernorm=False):
        super(CNNResBlock, self).__init__()
        self.linear1 = nn.Conv1d(dim, hidden_dim, kernel_size=1, stride=1, padding_mode='replicate')
        self.nonlinear1 = nonlinear()
        self.linear2 = nn.Conv1d(hidden_dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate')

        self.layernorm = layernorm
        if layernorm:
            self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.nonlinear1(out)

        out = self.linear2(out)

        if self.layernorm:
            out = self.layernorm(out)

        out += residual
        
        return out

def create_net(n_inputs, n_outputs, n_layers = 1, n_units = 100, nonlinear = nn.ReLU, normalize=False):
    '''
    Create a fully connected net:
    
    n_inputs --nonlinear-> (n_units --nonlinear-> ) * n_layers -> n_outputs

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        if normalize:
            layers.append(nn.LayerNorm(n_units))
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    if normalize:
        layers.append(nn.LayerNorm(n_units))
    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

def init_network_weights(net, method=nn.init.kaiming_normal_):
    '''
    Initialize network weights.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    for m in net.modules():
        if isinstance(m, nn.Linear):
            method(m.weight)
            nn.init.constant_(m.bias, val=0)

def get_device(tensor):
    '''
    Get device of tensor.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device

def sample_standard_gaussian(mu, sigma):
    '''
    Sample from a gaussian given mu and sigma.

    From https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    '''
    Get logger.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
    '''
    Update learning rate.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.

    From: https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out
