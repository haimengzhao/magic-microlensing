import os
import torch
import torch.nn as nn

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

def sample_standard_gaussian(mu, sigma):
    '''
    Sample from a gaussian given mu and sigma.

    From https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()

def get_next_batch(dataloader):
    '''
    Prepare batch by unioning all time points.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    # Make the union of all time points and perform normalization across the whole dataset
    data_dict = dataloader.__next__()

    batch_dict = get_dict_template()

    # remove the time points where there are no observations in this batch
    non_missing_tp = torch.sum(data_dict["observed_data"],(0,2)) != 0.
    batch_dict["observed_data"] = data_dict["observed_data"][:, non_missing_tp]
    batch_dict["observed_tp"] = data_dict["observed_tp"][non_missing_tp]

    # print("observed data")
    # print(batch_dict["observed_data"].size())

    if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
        batch_dict["observed_mask"] = data_dict["observed_mask"][:, non_missing_tp]

    batch_dict[ "data_to_predict"] = data_dict["data_to_predict"]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]

    non_missing_tp = torch.sum(data_dict["data_to_predict"],(0,2)) != 0.
    batch_dict["data_to_predict"] = data_dict["data_to_predict"][:, non_missing_tp]
    batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][non_missing_tp]

    # print("data_to_predict")
    # print(batch_dict["data_to_predict"].size())

    if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
        batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"][:, non_missing_tp]

    if ("regression_to_predict" in data_dict) and (data_dict["regression_to_predict"] is not None):
        batch_dict["regression_to_predict"] = data_dict["regression_to_predict"]

    batch_dict["mode"] = data_dict["mode"]
    return batch_dict

def compute_loss_all_batches(model,
    test_dataloader, n_batches, device,
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
        
        batch_dict = get_next_batch(test_dataloader)

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

def split_data_interp(data_dict):
    '''
    Split data into data_dict.
    
    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''

    split_dict = {"observed_data": data_dict["data"].clone(),
                "observed_tp": data_dict["time_steps"].clone(),
                "data_to_predict": data_dict["data"].clone(),
                "tp_to_predict": data_dict["time_steps"].clone()}

    split_dict["observed_mask"] = None 
    split_dict["mask_predicted_data"] = None 
    split_dict["regression_to_predict"] = None 

    if "mask" in data_dict and data_dict["mask"] is not None:
        split_dict["observed_mask"] = data_dict["mask"].clone()
        split_dict["mask_predicted_data"] = data_dict["mask"].clone()

    if ("regression_to_predict" in data_dict) and (data_dict["regression_to_predict"] is not None):
        split_dict["regression_to_predict"] = data_dict["regression_to_predict"].clone()

    split_dict["mode"] = "interp"
    return split_dict

def add_mask(data_dict):
    '''
    Add mask.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''
    data = data_dict["observed_data"]
    mask = data_dict["observed_mask"]

    if mask is None:
        mask = torch.ones_like(data).to(get_device(data))

    data_dict["observed_mask"] = mask
    return data_dict

def split_batch(data_dict):
    '''
    Split batch.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/utils.py
    '''

    processed_dict = split_data_interp(data_dict)

    # add mask
    processed_dict = add_mask(processed_dict)

    return processed_dict

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

