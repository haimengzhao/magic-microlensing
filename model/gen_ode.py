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

        self.aug_net = utils.create_net(input_dim, latent_dim, n_layers=2, n_units=32, nonlinear=nn.ReLU).to(device)
        
        ode_func_net = utils.create_net(latent_dim, latent_dim, 
            n_layers=args.gen_layers, n_units=args.units, nonlinear=nn.Tanh).to(device)
        self.ode_func = ODEFunc(ode_func_net = ode_func_net, device = device).to(device)
        
        self.decoder = utils.create_net(latent_dim, output_dim, n_layers=2, n_units=100).to(device)

    def forward(self, x, time_steps_to_predict):
        x = self.aug_net(x)
        sol = odeint(self.ode_func, x, time_steps_to_predict, method="dopri8",
            rtol=1e-3, atol=1e-4)
        sol = sol.permute(1, 0, 2)
        x = self.decoder(sol)
        return x

