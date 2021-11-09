import torch
import torch.nn as nn
import model.utils as utils

from torchdiffeq import odeint as odeint

class ODEFunc(nn.Module):
    '''
    Module for ode function f in dy/dt = f

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/ode_func.py
    '''
    def __init__(self, ode_func_net, device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.device = device
 
        utils.init_network_weights(ode_func_net)
        self.gradient_net = ode_func_net

    def forward(self, t_local, y, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        y: value at the current time point
        """
        grad = self.gradient_net(torch.cat([t_local.repeat(y.shape[0], 1), y], dim=-1))
        if backwards:
            grad = -grad
        return grad


class DiffeqSolver(nn.Module):
    '''
    Differential equation solver for neural-ODE

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/diffeq_solver.py
    '''
    def __init__(self, ode_func, method, 
            odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()
        self.device = device

        self.ode_func = ode_func
        self.ode_method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, initial_value, time_steps_to_predict, backwards = False):
        """
        Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = initial_value.size()[0], initial_value.size()[1]

        pred_y = odeint(self.ode_func, initial_value, time_steps_to_predict, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)

        assert(torch.mean(pred_y[:, :, 0, :]  - initial_value) < 0.001)
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)

        return pred_y

    # def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, 
    #     n_traj_samples = 1):
    #     """
    #     Decode the trajectory through ODE Solver using samples from the prior

    #     time_steps_to_predict: time steps at which we want to sample the new trajectory
    #     """
    #     func = self.ode_func.sample_next_point_from_prior

    #     pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
    #         rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
    #     # shape: [n_traj_samples, n_traj, n_tp, n_dim]
    #     pred_y = pred_y.permute(1,2,0,3)
    #     return pred_y


