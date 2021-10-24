import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

import utils
from likelihood_eval import masked_gaussian_log_density, compute_mse

from encoder_decoder import ODE_RNN_Encoder, Decoder
from neural_ode import DiffeqSolver, ODEFunc

def create_regression_model(z0_dim, n_labels):
    '''
    Create a network for regression task
    '''
    return nn.Sequential(
            nn.Linear(z0_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, n_labels),)

class VAE_Baseline(nn.Module):
    '''
    VAE Base Model

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/base_models.py
    '''
    def __init__(self, latent_dim,
        z0_prior, device,
        obsrv_std = 0.01, n_labels=7):

        super(VAE_Baseline, self).__init__()

        self.device = device

        self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

        self.z0_prior = z0_prior

        self.regression_model = create_regression_model(latent_dim, n_labels)
        utils.init_network_weights(self.regression_model)

    def get_gaussian_likelihood(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)
        log_density_data = masked_gaussian_log_density(pred_y, truth_repeated, 
            obsrv_std = self.obsrv_std, mask = mask)
        log_density_data = log_density_data.permute(1,0)
        log_density = torch.mean(log_density_data, 1)

        # shape: [n_traj_samples]
        return log_density


    def get_mse(self, truth, pred_y, mask = None):
        # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
        # truth shape  [n_traj, n_tp, n_dim]
        n_traj, n_tp, n_dim = truth.size()

        # Compute likelihood of the data under the predictions
        truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
        
        if mask is not None:
            mask = mask.repeat(pred_y.size(0), 1, 1, 1)

        # Compute likelihood of the data under the predictions
        log_density_data = compute_mse(pred_y, truth_repeated, mask = mask)
        # shape: [1]
        return torch.mean(log_density_data)


    def compute_all_losses(self, batch_dict, n_traj_samples = 1, kl_coef = 1.):
        # Condition on subsampled points
        # Make predictions for all the points
        pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
            batch_dict["observed_data"], batch_dict["observed_tp"], 
            mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
            mode = batch_dict["mode"])

        #print("get_reconstruction done -- computing likelihood")
        fp_mu, fp_std, fp_enc = info["first_point"]
        fp_std = fp_std.abs()
        fp_distr = Normal(fp_mu, fp_std)

        assert(torch.sum(fp_std < 0) == 0.)

        kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)

        if torch.isnan(kldiv_z0).any():
            print(fp_mu)
            print(fp_std)
            raise Exception("kldiv_z0 is Nan!")

        # Mean over number of latent dimensions
        # kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
        # kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
        # shape after: [n_traj_samples]
        kldiv_z0 = torch.mean(kldiv_z0,(1,2))

        # Compute likelihood of all the points
        rec_likelihood = self.get_gaussian_likelihood(
            batch_dict["data_to_predict"], pred_y,
            mask = batch_dict["mask_predicted_data"])

        mse = self.get_mse(
            batch_dict["data_to_predict"], pred_y,
            mask = batch_dict["mask_predicted_data"])

        reg_loss = self.get_mse(
            batch_dict['regression_to_predict'], info['regression_predicted']
        )


        # IWAE loss
        loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
        if torch.isnan(loss):
            loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)

        loss += reg_loss

        results = {}
        results["loss"] = torch.mean(loss)
        results["likelihood"] = torch.mean(rec_likelihood).detach()
        results["mse"] = torch.mean(mse).detach()
        results['reg_loss'] = torch.mean(reg_loss).detach()
        results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
        results["std_first_p"] = torch.mean(fp_std).detach()

        return results

class LatentODE(VAE_Baseline):
    '''
    The Latent-ODE Model

    VAE with ODE-RNN as encoder, and another ODE + Decoder as decoder.

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/latent_ode.py
    '''
    def __init__(self, latent_dim, encoder, decoder, diffeq_solver, 
        z0_prior, device, obsrv_std = None, 
        n_labels = 7):

        super(LatentODE, self).__init__(
            latent_dim=latent_dim,
            z0_prior = z0_prior, 
            device = device, obsrv_std = obsrv_std, 
            n_labels = n_labels)

        self.encoder = encoder
        self.diffeq_solver = diffeq_solver
        self.decoder = decoder

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, 
        mask = None, n_traj_samples = 1, run_backwards = True):

        if isinstance(self.encoder, ODE_RNN_Encoder):
            truth_w_mask = truth
            if mask is not None:
                truth_w_mask = torch.cat((truth, mask), -1)
            initial_mu, initial_sigma = self.encoder(
                truth_w_mask, truth_time_steps, run_backwards = run_backwards)

            mean_z0 = initial_mu.repeat(n_traj_samples, 1, 1)
            sigma_z0 = initial_sigma.repeat(n_traj_samples, 1, 1)
            initial_value = utils.sample_standard_gaussian(mean_z0, sigma_z0)

        else:
            raise Exception("Unknown encoder type {}".format(type(self.encoder).__name__))
        
        initial_sigma = initial_sigma.abs()
        assert(torch.sum(initial_sigma < 0) == 0.)
            
        assert(not torch.isnan(time_steps_to_predict).any())
        assert(not torch.isnan(initial_sigma).any())
        assert(not torch.isnan(initial_mu).any())

        # Shape of sol_y [n_traj_samples, n_samples, n_timepoints, n_latents]
        sol_y = self.diffeq_solver(initial_value, time_steps_to_predict)

        pred_x = self.decoder(sol_y)

        all_extra_info = {
            "first_point": (initial_mu, initial_sigma, initial_value),
            "latent_traj": sol_y.detach()
        }

        all_extra_info["regression_predicted"] = self.regression_model(initial_value).squeeze(-1)

        return pred_x, all_extra_info


    def sample_traj_from_prior(self, time_steps_to_predict, n_traj_samples = 1):
        # Sample z0 from prior
        starting_point_enc = self.z0_prior.sample([n_traj_samples, 1, self.latent_dim]).squeeze(-1)

        starting_point_enc_aug = starting_point_enc

        sol_y = self.diffeq_solver.sample_traj_from_prior(starting_point_enc_aug, time_steps_to_predict, 
            n_traj_samples = 3)

        return self.decoder(sol_y)

def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, n_labels=7):
    '''
    Create a Latent ODE model

    Modified from https://github.com/YuliaRubanova/latent_ode/blob/master/lib/create_latent_ode_model.py
    '''
    latent_dim = args.latents
    n_rec_dim = args.rec_dims
    enc_input_dim = int(input_dim) * 2 # we concatenate the mask
    gen_data_dim = input_dim

    enc_ode_func_net = utils.create_net(latent_dim, latent_dim, 
        n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.ReLU)
    enc_ode_func = ODEFunc(ode_func_net = enc_ode_func_net, device = device).to(device)
    enc_diffeq_solver = DiffeqSolver(enc_ode_func, "dopri5",
        odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

    encoder = ODE_RNN_Encoder(latent_dim, enc_input_dim, z0_dim=latent_dim, diffeq_solver=enc_diffeq_solver, 
        n_gru_units = args.gru_units, device = device).to(device)


    dec_ode_func_net = utils.create_net(n_rec_dim, n_rec_dim, 
        n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.ReLU)
    dec_ode_func = ODEFunc(ode_func_net = dec_ode_func_net, device = device).to(device)
    
    decoder = Decoder(args.latents, gen_data_dim).to(device)

    diffeq_solver = DiffeqSolver(dec_ode_func, 'dopri5', 
        odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

    model = LatentODE(
        latent_dim = latent_dim,
        encoder = encoder, 
        decoder = decoder, 
        diffeq_solver = diffeq_solver, 
        z0_prior = z0_prior, 
        device = device,
        obsrv_std = obsrv_std,
        n_labels = n_labels,
        ).to(device)

    return model



