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
    def __init__(self, device, n_intervals=4000, threshold=0.5, soft_threshold=False, crop=False, animate=False, plot=False):
        super(Locator, self).__init__()
        
        self.n_intervals = n_intervals
        self.device = device
        self.prefilter = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.PReLU(),
            nn.Conv1d(16, 16, kernel_size=15, stride=1, padding=7, padding_mode='reflect'),
            nn.PReLU(),
            nn.Conv1d(16, 1, kernel_size=1, stride=1, padding=0, padding_mode='reflect'),
            nn.PReLU(),
        )
        self.unet = utils.UNET_1D(1, 128, 7, 3)
        self.loss = utils.SoftDiceLoss()
        self.threshold = threshold
        self.animate = animate
        self.crop = crop
        self.soft_threshold = soft_threshold
        self.plot = plot
       
    def forward(self, coeffs, y):
        X = torchcde.CubicSpline(coeffs)

        interval = torch.linspace(X.interval[0], X.interval[-1], self.n_intervals).to(self.device)

        z = X.evaluate(interval)[:, :, [1]]
        z = z.transpose(-1, -2) # (batch, time, channel) -> (batch, channel, time)
        z = self.prefilter(z)
        z = self.unet(z)
        z = z.squeeze(-2)

        left = y[:, [0]] - y[:, [1]] / 10
        right = y[:, [0]] + y[:, [1]] / 10
        zt = X.evaluate(interval)[:, :, 0]
        zt = ((zt > left) * (zt < right)).int().float()

        cross_entropy = -torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10))
        # mse_z = (self.loss(z, zt)-torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10)))/2

        if not self.soft_threshold:
            z = (z > self.threshold).int()
        diffz = torch.diff(z, append=z[:, [-1]])
        timelist = X.evaluate(interval)[:, :, 0]
        plus = torch.sum(torch.abs(diffz) * timelist, dim=-1, keepdim=True)
        minus = torch.sum(diffz * timelist, dim=-1, keepdim=True)
        reg = torch.hstack([plus / 2, -minus * 5])

        mse_z = torch.log(torch.mean((z-zt)**2) + 1e-10)
        length_penalty = torch.log(torch.mean((torch.sum(z, dim=-1) - torch.sum(zt, dim=-1))**2) + 1e-10)
        diffz_abssum_penalty = torch.log(torch.mean((torch.sum(torch.abs(diffz), dim=-1) - 2.)**2) + 1e-10)
        diffz_sum_penalty = torch.log(torch.mean((torch.sum(diffz.float(), dim=-1))**2) + 1e-10)

        loss_z = cross_entropy + mse_z + length_penalty + diffz_abssum_penalty + diffz_sum_penalty

        if self.plot:
            avg = torch.mean(X.evaluate(interval)[0, :, 1].cpu()).item()
            plt.plot(timelist[0].cpu(), X.evaluate(interval)[0, :, 1].cpu())
            plt.plot(timelist[0].cpu(), zt[0].cpu()+avg)
            plt.plot(timelist[0].cpu(), z[0].cpu().detach().numpy()+avg)
            plt.plot(timelist[0].cpu(), diffz[0].cpu().detach().numpy()+avg)
            plt.show()

        # if self.crop and self.training:
        #     length = (torch.rand((1)).to(self.device) + 1) / 2 * (X.interval[-1] - X.interval[0])
        #     left = torch.rand((1)).to(self.device) * (X.interval[-1] - length - X.interval[0]) + X.interval[0]
        #     interval = torch.linspace(left.item(), left.item() + length.item(), self.n_intervals).to(self.device)
        #     z = X.evaluate(interval)[:, :, [1]]
        #     z = z.transpose(-1, -2) # (batch, time, channel) -> (batch, channel, time)
        #     z = self.unet(z)
        #     z = z.squeeze(-2)

        #     left = y[:, [0]] - 2*y[:, [1]]
        #     right = y[:, [0]] + 2*y[:, [1]]
        #     zt = X.evaluate(interval)[:, :, 0]
        #     zt = ((zt > left) * (zt < right)).int()

        #     # mse_z = -torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10))
        #     mse_z = (self.loss(z, zt)-torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10)))/2

        if self.animate:
            return reg, loss_z, z
        return reg, loss_z