import torch
import torch.nn as nn

import model.utils as utils

import torchcde
import matplotlib.pyplot as plt

class Locator(nn.Module):
    '''
    A Neural CDE Locator.

    Ref: https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
    '''
    def __init__(self, input_dim, latent_dim, output_dim, device):
        super(Locator, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_intervals = 4000
        self.device = device
        self.unet = utils.UNET_1D(1, 128, 7, 3)
        self.loss = utils.SoftDiceLoss()
        self.threshold = 0.4
       
    def forward(self, coeffs, zt):
        X = torchcde.CubicSpline(coeffs)

        # X0 = X.evaluate(X.interval[0])
        # z0 = self.initial(X0)

        # X1 = X.evaluate(X.interval[-1])
        # z1 = self.initial(X1)

        interval = torch.linspace(X.interval[0], X.interval[-1], self.n_intervals).to(self.device)

        # z_T = torchcde.cdeint(X=X,
        #                       z0=z0,
        #                       func=self.cde_func,
        #                       t=interval,
        #                       adjoint=False,
        #                       method="dopri5", rtol=1e-3, atol=1e-5)

        # z_T1 = torchcde.cdeint(X=X,
        #                       z0=z1,
        #                       func=self.cde_func_r,
        #                       t=torch.flip(interval, dims=[0]),
        #                       adjoint=False,
        #                       method="dopri5", rtol=1e-3, atol=1e-5)

        # z = self.gate(z_T + torch.flip(z_T1, dims=[-2]))

        # z = self.readout(z).squeeze(-1)

        z = X.evaluate(interval)[:, :, [1]]
        # plt.plot(interval.cpu(), z[0, :, 0].cpu())
        # plt.plot(interval.cpu(), z[0, :, 1].cpu())
        # plt.show()
        z = z.transpose(-1, -2) # (batch, time, channel) -> (batch, channel, time)

        z = self.unet(z)
        z = z.squeeze(-2)

        # mse_z = -torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10))
        mse_z = (self.loss(z, zt)-torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10)))/2

        z = (z > self.threshold).int()
        diffz = torch.diff(z, append=z[:, [-1]])
        timelist = X.evaluate(interval)[:, :, 0]
        plus = torch.sum(torch.abs(diffz) * timelist, dim=-1, keepdim=True)
        minus = torch.sum(diffz * timelist, dim=-1, keepdim=True)
        reg = torch.hstack([plus / 2, -minus / 4])

        # plt.plot(timelist[0].cpu(), X.evaluate(interval)[0, :, 1].cpu())
        # plt.plot(timelist[0].cpu(), zt[0].cpu()+14)
        # plt.plot(timelist[0].cpu(), z[0].cpu().detach().numpy()+14)
        # plt.plot(timelist[0].cpu(), diffz[0].cpu().detach().numpy()+14)
        # plt.show()

        # return torch.hstack([plus / 2, -minus / 4]), mse_z
        return reg, mse_z