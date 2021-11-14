import torch
import torch.nn as nn
from torch.types import Device

import model.utils as utils
from model.neural_ode import ODEFunc
from torchdiffeq import odeint


class GenODE(nn.Module):
    '''
    Create a Generative ODE model

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/create_latent_ode_model.py
    '''
    def __init__(self, args, input_dim, output_dim, device, init_time_shift=0):
        super(GenODE, self).__init__()
        latent_dim = args.latents
        self.device = device
        self.init_time_shift = torch.tensor([init_time_shift]).to(device)

        self.aug_net = nn.Sequential(utils.create_net(input_dim, 256, n_layers=0, n_units=256, nonlinear=nn.Tanh, normalize=False),
            *[utils.ResBlock(256, 1024, nonlinear=nn.Tanh) for i in range(3)],
            nn.Linear(256, latent_dim), 
            nn.Tanh()
            )
        utils.init_network_weights(self.aug_net, nn.init.xavier_normal_)
        
        self.ode_func_net = nn.Sequential(utils.create_net(1 + latent_dim, args.units, n_layers=0, n_units=args.units, nonlinear=nn.Tanh, normalize=False),
            *[utils.ResBlock(args.units, hidden_dim=args.units, nonlinear=nn.Tanh) for i in range(args.gen_layers)],
            nn.Linear(args.units, latent_dim), 
            nn.Tanh())
        self.ode_func = ODEFunc(ode_func_net = self.ode_func_net, device = device).to(device)
        utils.init_network_weights(self.ode_func, nn.init.xavier_normal_)

        self.sol_normalize = nn.LayerNorm(latent_dim)
        
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.Tanh(),
            *[utils.ResBlock(256, hidden_dim=1024, nonlinear=nn.Tanh) for i in range(3)],
            nn.Linear(256, output_dim))
        utils.init_network_weights(self.decoder, nn.init.xavier_normal_)
        

    def forward(self, x, time_steps_to_predict):
        x = self.aug_net(x)
        # x = self.latent_normalize(x)
        sol = odeint(self.ode_func, x, time_steps_to_predict, method="dopri5",
            rtol=1e-3, atol=1e-5)
        sol = sol.permute(1, 0, 2)
        # sol = self.sol_normalize(sol)
        x = self.decoder(sol)
        return x

