import torch
import torch.nn as nn

import model.utils as utils

import torchcde

class LocatorCDEFunc(nn.Module):
    '''
    Locator's neural CDE grad function.

    Ref: https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
    '''
    def __init__(self, input_dim, latent_dim):
        super(LocatorCDEFunc, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.linear1 = nn.Linear(latent_dim, 1024)
        
        self.relu1 = nn.PReLU()
        self.resblocks = nn.Sequential(*[utils.ResBlock(1024, 1024, nonlinear=nn.PReLU, layernorm=False) for i in range(1)])
        self.relu2 = nn.PReLU()
        self.linear2 = nn.Linear(1024, input_dim * latent_dim)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(input_dim * latent_dim, input_dim * latent_dim)
    
    def forward(self, t, z):
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
    def __init__(self, input_dim, latent_dim, output_dim):
        super(Locator, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.cde_func = LocatorCDEFunc(input_dim, latent_dim)
        self.cde_func_r = LocatorCDEFunc(input_dim, latent_dim)
        # utils.init_network_weights(self.cde_func, method=nn.init.orthogonal_)
        self.initial = nn.Sequential(
            utils.create_net(input_dim, 1024, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
            *[utils.ResBlock(1024, 1024, nonlinear=nn.PReLU) for i in range(3)],
            utils.create_net(1024, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
        )
        self.initial_r = nn.Sequential(
            utils.create_net(input_dim, 1024, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
            *[utils.ResBlock(1024, 1024, nonlinear=nn.PReLU) for i in range(3)],
            utils.create_net(1024, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
        )
        self.initial = utils.create_net(input_dim, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU)
        # self.initial_r = utils.create_net(input_dim, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU)
        # utils.init_network_weights(self.initial, method=nn.init.orthogonal_)
        self.readout = nn.Sequential(
            utils.create_net(latent_dim * 2, 1024, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
            *[utils.ResBlock(1024, 1024, nonlinear=nn.PReLU) for i in range(3)],
            nn.Linear(1024, output_dim)
        )
        # self.readout = utils.create_net(latent_dim * 2, output_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU)
        # utils.init_network_weights(self.readout, method=nn.init.orthogonal_)
       
    def forward(self, coeffs):
        X = torchcde.CubicSpline(coeffs)

        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        X1 = X.evaluate(X.interval[-1])
        z1 = self.initial(X1)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.cde_func,
                              t=X.interval,
                              adjoint=False,
                              method="dopri5", rtol=1e-5, atol=1e-7)
        z_T = z_T[:, -1]
        z_T1 = torchcde.cdeint(X=X,
                              z0=z1,
                              func=self.cde_func_r,
                              t=torch.flip(X.interval, dims=[0]),
                              adjoint=False,
                              method="dopri5", rtol=1e-5, atol=1e-7)
        z_T1 = z_T1[:, -1]

        pred = self.readout(torch.hstack([z_T, z_T1]))
        return pred