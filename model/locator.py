import torch
import torch.nn as nn
from torch.nn.modules.activation import Tanh

import model.utils as utils

import torchcde

class LocatorCDEFunc(nn.Module):
    '''
    Neural CDE grad function.

    Ref: https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
    '''
    def __init__(self, input_dim, latent_dim):
        super(LocatorCDEFunc, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(latent_dim, 1024)
        self.relu1 = nn.ReLU()
        self.resblocks = nn.Sequential(*[utils.ResBlock(1024, 1024, nonlinear=nn.ReLU, layernorm=False) for i in range(3)])
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(1024, input_dim * latent_dim)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(input_dim * latent_dim, input_dim * latent_dim)
    
    def forward(self, t, z):
        # z = torch.cat([t.repeat(z.shape[0], 1), z], dim=-1)
        z = self.linear1(z)
        z = self.relu1(z)
        z = self.resblocks(z)
        z = self.relu2(z)
        z = self.linear2(z)
        z = self.tanh(z) # important!
        z = self.linear3(z)

        z = z.view(z.shape[0], self.latent_dim, self.input_dim)

        return z

class Locator(nn.Module):
    '''
    A Neural CDE Locator.

    Ref: https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
    '''
    def __init__(self, input_dim, latent_dim, output_dim, device):
        super(Locator, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_intervals = 2048
        self.device = device
        self.cde_func = LocatorCDEFunc(input_dim, latent_dim)
        utils.init_network_weights(self.cde_func, method=nn.init.orthogonal_)
        self.cde_func_r = LocatorCDEFunc(input_dim, latent_dim)
        utils.init_network_weights(self.cde_func_r, method=nn.init.orthogonal_)

        self.initial = nn.Sequential(
            utils.create_net(input_dim, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
            *[utils.ResBlock(1024, 1024, nonlinear=nn.PReLU) for i in range(3)],
            utils.create_net(1024, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
        )
        
        self.gate = nn.PReLU()

        self.readout = nn.Sequential(
            utils.create_net(latent_dim, output_dim, n_layers=0, n_units=1024, nonlinear=nn.ReLU),
            nn.Sigmoid()
        )
        
       
    def forward(self, coeffs):
        X = torchcde.CubicSpline(coeffs)

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        X1 = X.evaluate(X.interval[-1])
        z1 = self.initial(X1)

        interval = torch.linspace(X.interval[0], X.interval[-1], self.n_intervals).to(self.device)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.cde_func,
                              t=interval,
                              adjoint=False,
                              method="dopri5", rtol=1e-5, atol=1e-7)

        z_T1 = torchcde.cdeint(X=X,
                              z0=z1,
                              func=self.cde_func_r,
                              t=torch.flip(interval, dims=[0]),
                              adjoint=False,
                              method="dopri5", rtol=1e-5, atol=1e-7)

        z = self.gate(z_T + torch.flip(z_T1, dims=[-2]))

        z = self.readout(z)

        return z