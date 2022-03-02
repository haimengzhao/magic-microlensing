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
        self.threshold = 0.5
        self.animate = False
        self.crop = False
       
    def forward(self, coeffs, y):
        X = torchcde.CubicSpline(coeffs)

        interval = torch.linspace(X.interval[0], X.interval[-1], self.n_intervals).to(self.device)

        z = X.evaluate(interval)[:, :, [1]]
        z = z.transpose(-1, -2) # (batch, time, channel) -> (batch, channel, time)
        z = self.unet(z)
        z = z.squeeze(-2)

        left = y[:, [0]] - 2*y[:, [1]]
        right = y[:, [0]] + 2*y[:, [1]]
        zt = X.evaluate(interval)[:, :, 0]
        zt = ((zt > left) * (zt < right)).int()

        # mse_z = -torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10))
        mse_z = (self.loss(z, zt)-torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10)))/2

        # z = (z > self.threshold).int()
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

        if self.crop and self.training:
            length = (torch.rand((1)).to(self.device) + 1) / 2 * (X.interval[-1] - X.interval[0])
            left = torch.rand((1)).to(self.device) * (X.interval[-1] - length - X.interval[0]) + X.interval[0]
            interval = torch.linspace(left.item(), left.item() + length.item(), self.n_intervals).to(self.device)
            z = X.evaluate(interval)[:, :, [1]]
            z = z.transpose(-1, -2) # (batch, time, channel) -> (batch, channel, time)
            z = self.unet(z)
            z = z.squeeze(-2)

            left = y[:, [0]] - 2*y[:, [1]]
            right = y[:, [0]] + 2*y[:, [1]]
            zt = X.evaluate(interval)[:, :, 0]
            zt = ((zt > left) * (zt < right)).int()

            # mse_z = -torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10))
            mse_z = (self.loss(z, zt)-torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10)))/2

        if self.animate:
            return reg, mse_z, z
        return reg, mse_z