import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

import utils
from likelihood_eval import masked_gaussian_log_density, compute_mse

def create_regression_model(z0_dim, n_labels):
    return nn.Sequential(
            nn.Linear(z0_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, n_labels),)

class VAE_Baseline(nn.Module):
    def __init__(self, input_dim, latent_dim, 
        z0_prior, device,
        obsrv_std = 0.01, n_labels=7):

        super(VAE_Baseline, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
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
            batch_dict['Y'], info['Y']
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



