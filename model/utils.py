import torch
import torch.nn as nn

def create_net(n_inputs, n_outputs, n_layers = 1, n_units = 100, nonlinear = nn.ReLU):
    '''
    Create a fully connected net:
    
    n_inputs --nonlinear-> (n_units --nonlinear-> ) * n_layers -> n_outputs

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)

def init_network_weights(net, std = 0.1):
    '''
    Initialize network weights.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
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

def split_last_dim(data):
    '''
    Split the last dim of data into halves.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    last_dim = data.size()[-1]
    last_dim = last_dim//2

    if len(data.size()) == 3:
        res = data[:,:,:last_dim], data[:,:,last_dim:]

    if len(data.size()) == 2:
        res = data[:,:last_dim], data[:,last_dim:]
    return res