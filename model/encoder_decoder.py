import torch
import torch.nn as nn

import model.utils as utils

class GRU_Unit(nn.Module):
    '''
    GRU module in RNN.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/encoder_decoder.py
    '''
    def __init__(self, latent_dim, input_dim, 
            update_gate = None,
            reset_gate = None,
            new_state_net = None,
            n_units = 100,
            device = torch.device("cpu")):
        super(GRU_Unit, self).__init__()
        self.device = device

        if update_gate is None:
            self.update_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
            utils.init_network_weights(self.update_gate)
        else: 
            self.update_gate  = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
            utils.init_network_weights(self.reset_gate)
        else: 
            self.reset_gate  = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2))
            utils.init_network_weights(self.new_state_net)
        else: 
            self.new_state_net  = new_state_net


    def forward(self, y_mean, y_std, x, masked_update = False):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1-update_gate) * new_state + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

        assert(not torch.isnan(new_y).any())

        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            n_data_dims = x.size(-1)//2
            mask = x[:, :, n_data_dims:]
            utils.check_mask(x[:, :, :n_data_dims], mask)
            
            mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

            assert(not torch.isnan(mask).any())

            new_y = mask * new_y + (1-mask) * y_mean
            new_y_std = mask * new_y_std + (1-mask) * y_std

            if torch.isnan(new_y).any():
                print("new_y is nan!")
                print(mask)
                print(y_mean)
                exit()

        new_y_std = new_y_std.abs()
        return new_y, new_y_std

class ODE_RNN_Encoder(nn.Module):
    '''
    Run ODE backwards to encode the input time sequence into latent z_0.

    ODE_RNN:

    For every time step, we have two ways to compute the hidden state:
    
    1. from the data point, as a normal RNN does;
    2. from ODE, by running it backwards from the last data point t_{i+1} to t_i.
	
    The hiddens state is defined as a weighted sum of the two ways. 
    
    It's then used as the initial value for the ODE runing from t_i to t_{i-1}.
	
    Repeat this until we get to z0.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/encoder_decoder.py
    '''
    def __init__(self, latent_dim, input_dim, z0_dim, diffeq_solver = None, 
            GRU_update = None, n_gru_units = 100, 
            device = torch.device("cpu")):

        super(ODE_RNN_Encoder, self).__init__()
        self.device = device

        self.latent_dim = latent_dim
        self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_Unit(latent_dim, input_dim, n_units = n_gru_units, device=device).to(device)
        else:
            self.GRU_update = GRU_update

        self.diffeq_solver = diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim * 2, 100),
            nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2),)
        utils.init_network_weights(self.transform_z0)


    def forward(self, data, time_steps, run_backwards = True, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:
            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps, run_backwards = run_backwards,
                save_info = save_info)

        mean_latent = last_yi.reshape(1, n_traj, self.latent_dim)
        std_latent = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = utils.split_last_dim(self.transform_z0(torch.cat((mean_latent, std_latent), -1)))
        std_z0 = std_z0.abs()
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0


    def run_odernn(self, data, time_steps, 
        run_backwards = True, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it 
        n_traj, n_tp, n_dims = data.size()
        extra_info = []

        t0 = time_steps[-1]
        if run_backwards:
            t0 = time_steps[0]

        device = utils.get_device(data)

        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)

        prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 1000

        # print("minimum step: {}".format(minimum_step))

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        latent_ys = []
        # Run ODE backwards and combine the y(t) estimates using gating
        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)

        for i in time_points_iter:
            if (prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                inc = self.diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert(not torch.isnan(inc).any())

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

                assert(not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp).to(device)
                ode_sol = self.diffeq_solver(prev_y, time_points)

                assert(not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
                exit()
            # assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)

            yi_ode = ode_sol[:, :, -1, :]
            xi = data[:,i,:].unsqueeze(0)
            
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)

            prev_y, prev_std = yi, yi_std		
            if i >= 1:	
                prev_t, t_i = time_steps[i],  time_steps[i-1]
            else:
                break
            # print(f'i: {i}, prev_t: {prev_t}, t_i: {t_i}')

            latent_ys.append(yi)

            if save_info:
                d = {"yi_ode": yi_ode.detach(), # "yi_from_data": yi_from_data,
                        "yi": yi.detach(), "yi_std": yi_std.detach(), 
                        "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)

        latent_ys = torch.stack(latent_ys, 1)

        assert(not torch.isnan(yi).any())
        assert(not torch.isnan(yi_std).any())

        return yi, yi_std, latent_ys, extra_info

class Decoder(nn.Module):
    '''
    Decode latent state back into data points.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/encoder_decoder.py
    '''
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()

        # decoder = nn.Sequential(
        #     nn.Linear(latent_dim, input_dim),)
        decoder = utils.create_net(latent_dim, output_dim)
        utils.init_network_weights(decoder)	
        
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)
