import os
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.

    From: https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1
        
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.InstanceNorm1d(out_layer)
        self.relu = nn.PReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out       

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):

        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out          

class UNET_1D(nn.Module):
    '''
    Ref: https://www.kaggle.com/super13579/u-net-1d-cnn-with-pytorch
    '''
    def __init__(self ,input_dim,layer_n,kernel_size,depth):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        self.layer_n = layer_n
        self.kernel_size = kernel_size
        self.depth = depth
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)
        
        self.layer1 = self.down_layer(self.input_dim, self.layer_n, self.kernel_size,1, 2)
        self.layer2 = self.down_layer(self.layer_n, int(self.layer_n*2), self.kernel_size,5, 2)
        self.layer3 = self.down_layer(int(self.layer_n*2)+int(self.input_dim), int(self.layer_n*3), self.kernel_size,5, 2)
        self.layer4 = self.down_layer(int(self.layer_n*3)+int(self.input_dim), int(self.layer_n*4), self.kernel_size,5, 2)
        self.layer5 = self.down_layer(int(self.layer_n*4)+int(self.input_dim), int(self.layer_n*5), self.kernel_size,4, 2)

        self.cbr_up1 = conbr_block(int(self.layer_n*7), int(self.layer_n*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.layer_n*5), int(self.layer_n*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.layer_n*3), self.layer_n, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')
        
        self.outcov = nn.Conv1d(self.layer_n, 1, kernel_size=self.kernel_size, stride=1,padding = 3)
        self.sig = nn.Sigmoid()
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)
        
        #############Decoder####################
        
        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        
        out = self.sig(out)
        
        return out

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

class CNNResBlock(nn.Module):
    def __init__(self, dim, hidden_dim, nonlinear=nn.PReLU, layernorm=False):
        super(CNNResBlock, self).__init__()
        self.linear1 = nn.Conv1d(dim, hidden_dim, kernel_size=1, stride=1, padding_mode='replicate')
        self.nonlinear1 = nonlinear()
        self.linear2 = nn.Conv1d(hidden_dim, dim, kernel_size=3, stride=1, padding=1, padding_mode='replicate')

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

