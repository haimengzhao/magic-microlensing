import torch
import torch.nn as nn

import model.utils as utils

import torchcde
import model.mdn as mdn
# Uncomment the following line and set full_cov to True to use full covariance MDN.
# import model.mdn_full as mdn

class CDEFunc(nn.Module):
    '''
    Neural CDE grad function.

    Ref: https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
    '''
    def __init__(self, input_dim, latent_dim):
        super(CDEFunc, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(latent_dim, 1024)
        self.relu1 = nn.PReLU()
        self.resblocks = nn.Sequential(*[utils.ResBlock(1024, 1024, nonlinear=nn.PReLU, layernorm=False) for i in range(8)])
        self.relu2 = nn.PReLU()
        self.linear2 = nn.Linear(1024, input_dim * latent_dim)
        self.relu3 = nn.PReLU()
        self.linear3 = nn.Linear(input_dim * latent_dim, 4096)
        self.tanh = nn.Tanh()
        self.linear4 = nn.Linear(4096, input_dim * latent_dim)
    
    def forward(self, t, z):
        z = self.linear1(z)
        z = self.relu1(z)
        z = self.resblocks(z)
        z = self.relu2(z)
        z = self.linear2(z)
        z = self.relu3(z)
        z = self.linear3(z)
        z = self.tanh(z) # important!
        z = self.linear4(z)

        z = z.view(z.shape[0], self.latent_dim, self.input_dim)

        return z



class CDE_MDN(nn.Module):
    '''
    A Neural CDE Mixture Density Network.

    Args:
            input_dim (int): dimension of the input.
            latent_dim (int): dimension of the latent space.
            output_dim (int): dimension of the output.
            n_gaussian (int, optional): number of Gaussians to use. Defaults to 12.
            dataparallel (bool, optional): whether to use dataparallel. Defaults to False.
            full_cov (bool, optional): whether to use full covariance MDN. Defaults to False.

    Returns:
            pi (tensor): predicted mixture weights.
            normal (tensor): predicted Gaussians. If dataparallel is True, this is split into loc, scale.
    '''
    def __init__(self, input_dim, latent_dim, output_dim, n_gaussian=12, full_cov=False):
        super(CDE_MDN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_gaussian = n_gaussian
        self.full_cov = full_cov
        self.output_feature = False # used to explore the embedding/feature space

        self.cde_func = torch.compile(CDEFunc(input_dim, latent_dim))
        self.initial = torch.compile(nn.Sequential(
            utils.create_net(input_dim, 1024, n_layers=0, n_units=1024, nonlinear=nn.ReLU),
            *[utils.ResBlock(1024, 1024, nonlinear=nn.ReLU, layernorm=False) for i in range(3)],
            utils.create_net(1024, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.ReLU),
        ), dynamic=False)
        self.readout = torch.compile(nn.Sequential(
            utils.create_net(latent_dim, 1024, n_layers=0, n_units=1024, nonlinear=nn.ReLU),
            *[utils.ResBlock(1024, 1024, nonlinear=nn.ReLU, layernorm=False) for i in range(3)],
            nn.Linear(1024, 1024)
        ), dynamic=False)
        self.mdn = torch.compile(mdn.MixtureDensityNetwork(1024, output_dim, self.n_gaussian, full_cov=full_cov), dynamic=False)
        # utils.init_network_weights(self.cde_func, nn.init.orthogonal_)
        
    def forward(self, coeffs):
        X = torchcde.CubicSpline(coeffs)

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.cde_func,
                              t=X.interval,
                              adjoint=False,
                              method="dopri5", rtol=1e-3, atol=1e-5)
        # changing (rtol, atol) from (1e-5, 1e-7) to (1e-3, 1e-5) can speed up 4x with indistinguishable performance

        z_T = z_T[:, -1]

        if self.output_feature:
            return z_T

        z_T = self.readout(z_T)
        pi, normal = self.mdn(z_T)
        
        return pi, normal