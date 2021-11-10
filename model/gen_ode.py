import torch
import torch.nn as nn

import model.utils as utils
from model.neural_ode import ODEFunc
from torchdiffeq import odeint


class GenODE(nn.Module):
    '''
    Create a Generative ODE model

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/create_latent_ode_model.py
    '''
    def __init__(self, args, input_dim, output_dim, device):
        super(GenODE, self).__init__()
        latent_dim = args.latents
        self.device = device

        # self.aug_net = utils.create_net(input_dim, latent_dim, n_layers=0, n_units=latent_dim, nonlinear=nn.PReLU, normalize=False).to(device)
        # self.batchnorm = nn.BatchNorm1d(latent_dim).to(device)
        self.aug_net = nn.Sequential(utils.create_net(input_dim, latent_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU, normalize=False),
            *[utils.ResBlock(latent_dim) for i in range(3)],)
        utils.init_network_weights(self.aug_net)
        
        ode_func_net = utils.create_net(latent_dim + 1, latent_dim, 
            n_layers=args.gen_layers, n_units=args.units, nonlinear=nn.PReLU, normalize=True).to(device)
        self.ode_func = ODEFunc(ode_func_net = ode_func_net, device = device).to(device)
        utils.init_network_weights(self.ode_func)
        
        self.decoder = nn.Sequential(*[utils.ResBlock(latent_dim) for i in range(3)],
            utils.create_net(latent_dim, output_dim, n_layers=0, n_units=1024, nonlinear=nn.PReLU, normalize=False))
        utils.init_network_weights(self.decoder)
        

    def forward(self, x, time_steps_to_predict):
        x = self.aug_net(x)
        # x = self.batchnorm(x)
        sol = odeint(self.ode_func, x, time_steps_to_predict, method="dopri5",
            rtol=1e-7, atol=1e-9)
        sol = sol.permute(1, 0, 2)
        x = self.decoder(sol)
        return x

