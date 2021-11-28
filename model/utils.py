import os
import logging
import torch
import torch.nn as nn
import numpy as np

def get_dict_template():
    '''
    Template for data_dict

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    return {"observed_data": None,
            "observed_tp": None,
            "data_to_predict": None,
            "tp_to_predict": None,
            "observed_mask": None,
            "mask_predicted_data": None,
            "regression_to_predict": None
            }

def makedirs(dirname):
    '''
    Make directory if not exist.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)

class ResBlock(nn.Module):
    def __init__(self, dim, hidden_dim, nonlinear=nn.PReLU, layernorm=False):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.nonlinear1 = nonlinear()
        self.linear2 = nn.Linear(hidden_dim, dim)

        self.layernorm = layernorm
        if layernorm:
            self.layernorm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x

        out = self.linear1(x)
        out = self.nonlinear1(out)

        out = self.linear2(out)

        if self.layernorm:
            out = self.layernorm(out)

        out += residual
        
        return out

def create_net(n_inputs, n_outputs, n_layers = 1, n_units = 100, nonlinear = nn.ReLU, normalize=False):
    '''
    Create a fully connected net:
    
    n_inputs --nonlinear-> (n_units --nonlinear-> ) * n_layers -> n_outputs

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        if normalize:
            layers.append(nn.LayerNorm(n_units))
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    if normalize:
        layers.append(nn.LayerNorm(n_units))
    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

def init_network_weights(net, method=nn.init.kaiming_normal_):
    '''
    Initialize network weights.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    for m in net.modules():
        if isinstance(m, nn.Linear):
            method(m.weight)
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

def sample_standard_gaussian(mu, sigma):
    '''
    Sample from a gaussian given mu and sigma.

    From https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()

def get_next_batch(random_dataloader, label_dataloader, even_dataloader, device):
    '''
    Prepare batch by unioning all time points.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    # Make the union of all time points and perform normalization across the whole dataset
    X_random = random_dataloader.__next__()
    Y = label_dataloader.__next__()
    X_even = even_dataloader.__next__()

    batch_dict = get_dict_template()

    D = X_random.shape[-1] - 1
    combined_tt, inverse_indices = torch.unique(X_random[:, :, 0].flatten(), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(X_random), len(combined_tt), D]).to(device)
    
    for b in range(len(X_random)):
        tt = X_random[b, :, 0].to(device)
        vals = X_random[b, :, 1:].to(device).float()

        indices = inverse_indices[offset:(offset + len(tt))]
        offset += len(tt)

        combined_vals[b, indices] = vals

    batch_dict["observed_data"] = combined_vals
    batch_dict["observed_tp"] = combined_tt
    batch_dict["data_to_predict"] = X_even[:, :, 1].to(device).float()
    batch_dict["tp_to_predict"] = X_even[0, :, 0].to(device).float()
    batch_dict["regression_to_predict"] = Y.to(device)

    return batch_dict

def compute_loss_all_batches(model,
    test_random_dataloader, test_label_dataloader, test_even_dataloader, n_batches, device,
    n_traj_samples = 1, kl_coef = 1.):
    '''
    Compute loss.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''

    total = {}
    total["loss"] = 0
    total["likelihood"] = 0
    total["mse"] = 0
    total["kl_first_p"] = 0
    total["std_first_p"] = 0
    total["reg_loss"] = 0

    n_test_batches = 0
    
    reg_predictions = torch.Tensor([]).to(device)
    reg_to_predict =  torch.Tensor([]).to(device)

    for i in range(n_batches):
        print("Computing loss... " + str(i))
        
        batch_dict = get_next_batch(test_random_dataloader, test_label_dataloader, test_even_dataloader, device)

        results  = model.compute_all_losses(batch_dict,
            n_traj_samples = n_traj_samples, kl_coef = kl_coef)

        n_labels = model.n_labels # batch_dict["labels"].size(-1)
        n_traj_samples = results["regression_predicted"].size(0)

        classif_predictions = torch.cat((classif_predictions, 
            results["regression_predicted"].reshape(n_traj_samples, -1, n_labels)),1)
        reg_to_predict = torch.cat((reg_to_predict, 
            batch_dict["regression_to_predict"].reshape(-1, n_labels)),0)

        for key in total.keys(): 
            if key in results:
                var = results[key]
                if isinstance(var, torch.Tensor):
                    var = var.detach()
                total[key] += var

        n_test_batches += 1

    if n_test_batches > 0:
        for key, value in total.items():
            total[key] = total[key] / n_test_batches

    return total

def inf_generator(iterable):
    """
    Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    
    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    '''
    Get logger.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger

def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
    '''
    Update learning rate.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = max(lr * decay_rate, lowest)
        param_group['lr'] = lr

def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

