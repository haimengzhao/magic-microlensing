import torch
import torch.nn as nn
from torch.nn.modules.activation import Tanh

import model.utils as utils

import torchcde

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



class CDEEncoder(nn.Module):
    '''
    A Neural CDE Encoder.

    Ref: https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
    '''
    def __init__(self, input_dim, latent_dim, output_dim):
        super(CDEEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.cde_func = CDEFunc(input_dim, latent_dim)
        utils.init_network_weights(self.cde_func, method=nn.init.orthogonal_)
        self.initial = nn.Sequential(
            utils.create_net(input_dim, 1024, n_layers=0, n_units=1024, nonlinear=nn.ReLU),
            *[utils.ResBlock(1024, 1024, nonlinear=nn.ReLU) for i in range(3)],
            utils.create_net(1024, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.ReLU),
        )
        # self.initial = utils.create_net(input_dim, latent_dim, n_layers=5, n_units=1024, nonlinear=nn.PReLU)
        # self.initial = nn.Sequential(
        #     utils.create_net(input_dim, latent_dim, n_layers=5, n_units=1024, nonlinear=nn.PReLU),
        #     )
        self.readout = nn.Sequential(
            utils.create_net(latent_dim, 1024, n_layers=0, n_units=1024, nonlinear=nn.ReLU),
            *[utils.ResBlock(1024, 1024, nonlinear=nn.ReLU) for i in range(3)],
            nn.Linear(1024, output_dim)
        )
        # self.readout = utils.create_net(latent_dim, output_dim, n_layers=5, n_units=1024, nonlinear=nn.ReLU)
        # self.readout = utils.create_net(latent_dim, output_dim, n_layers=5, n_units=1024, nonlinear=nn.SiLU)
        
    def forward(self, coeffs):
        X = torchcde.CubicSpline(coeffs)

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.cde_func,
                              t=X.interval,
                              adjoint=False,
                              method="dopri5", rtol=1e-5, atol=1e-7)

        z_T = z_T[:, -1]
        pred = self.readout(z_T)
        return pred