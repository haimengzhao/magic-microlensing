import torch
import torch.nn as nn

import model.utils as utils

import torchcde
import model.mdn as mdn

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
        self.linear2 = nn.Linear(1024, input_dim * latent_dim)
        self.tanh = nn.Tanh()
        self.linear3 = nn.Linear(input_dim * latent_dim, input_dim * latent_dim)
    
    def forward(self, t, z):
        z = self.linear1(z)
        z = self.relu1(z)
        z = self.linear2(z)
        z = self.tanh(z) # important!
        z = self.linear3(z)

        z = z.view(z.shape[0], self.latent_dim, self.input_dim)

        return z



class CDE_MDN(nn.Module):
    '''
    A Neural CDE Mixture Density Network.
    '''
    def __init__(self, input_dim, latent_dim, output_dim):
        super(CDE_MDN, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_gaussian = 12

        # self.cde_func = CDEFunc(input_dim, latent_dim)
        # utils.init_network_weights(self.cde_func, nn.init.orthogonal_)
        # self.initial = nn.Sequential(
        #     utils.create_net(input_dim, latent_dim, n_layers=1, n_units=1024, nonlinear=nn.PReLU),
        # )
        # self.readout = nn.Sequential(
        #     utils.create_net(latent_dim, 1024, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
        #     *[utils.ResBlock(1024, 1024, nonlinear=nn.PReLU) for i in range(3)],
        #     mdn.MDN(in_features=1024, out_features=self.output_dim, num_gaussians=self.n_gaussian)
        # )
        self.n_cnn_intervals = 2048

        self.initial = nn.Sequential(
            utils.create_net(input_dim, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU),
        )
        
        self.gate = nn.PReLU()

        self.cnn_featurizer = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=15, stride=2, padding=7), nn.PReLU(),
            nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7), nn.PReLU(),
            *[utils.CNNResBlock(512, 128, nonlinear=nn.PReLU, layernorm=False) for i in range(15)],
            nn.Conv1d(512, 128, kernel_size=15, stride=2, padding=7), nn.PReLU(),
            nn.Conv1d(128, 64, kernel_size=15, stride=2, padding=7), nn.PReLU(),
            nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=0), nn.PReLU(),
        )
        utils.init_network_weights(self.cnn_featurizer, nn.init.kaiming_normal_)

        self.regressor = nn.Sequential(
            mdn.MDN(in_features=128*32, out_features=self.output_dim, num_gaussians=self.n_gaussian)
        )
        
    def forward(self, coeffs):
        # X = torchcde.CubicSpline(coeffs)

        # X0 = X.evaluate(X.interval[0])
        # z0 = self.initial(X0)

        # z_T = torchcde.cdeint(X=X,
        #                       z0=z0,
        #                       func=self.cde_func,
        #                       t=X.interval,
        #                       adjoint=False,
        #                       method="dopri5", rtol=1e-5, atol=1e-7)

        # z_T = z_T[:, -1]
        # pi, sigma, mu = self.readout(z0)
        X = torchcde.CubicSpline(coeffs)
        interval = torch.linspace(X.interval[0], X.interval[-1], self.n_cnn_intervals).to(coeffs.device)

        z = self.initial(X.evaluate(interval))
        z = z.transpose(-1, -2) # (batch, time, channel) -> (batch, channel, time)

        z = self.cnn_featurizer(z).flatten(start_dim=1)

        pi, sigma, mu = self.regressor(z)
        return pi, sigma, mu
    
    def mdn_loss(self, pi, sigma, mu, labels):
        return mdn.mdn_loss(pi, sigma, mu, labels)

    def sample(self, pi, sigma, mu):
        return mdn.sample(pi, sigma, mu)