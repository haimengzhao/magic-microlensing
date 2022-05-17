import torch
import torch.nn as nn
from torchvision import ops

import model.utils as utils

import torchcde
import matplotlib.pyplot as plt

class Locator(nn.Module):
    '''
    A Neural CDE Locator.

    Ref: https://github.com/patrick-kidger/torchcde/blob/master/example/time_series_classification.py
    '''
    def __init__(self, device, k=1/3, n_intervals=4000, threshold=0.5, soft_threshold=True, crop=False, animate=False, plot=False, method='diff'):
        super(Locator, self).__init__()
        self.k = k
        self.n_intervals = n_intervals
        self.method = method
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
        # self.loss = utils.focal_loss
        self.loss = utils.DiceLoss()
        self.threshold = threshold
        self.animate = animate
        self.crop = crop
        self.soft_threshold = soft_threshold
        self.plot = plot
       
    def forward(self, coeffs, y, interval=None):
        X = torchcde.CubicSpline(coeffs)

        if interval is None:
            interval = torch.linspace(X.interval[0], X.interval[-1], self.n_intervals).to(self.device)
        else:
            interval = interval.to(self.device)

        z = X.evaluate(interval)
        if len(z.shape) > 3:
            z = torch.diagonal(z, dim1=0, dim2=1).permute(2, 0, 1)
        z0 = z[:, :, [1]]
        z = z0.transpose(-1, -2) # (batch, time, channel) -> (batch, channel, time)
        z = self.prefilter(z) + z
        z = self.unet(z)
        z = z.squeeze(-2)

        left = y[:, [0]] - y[:, [1]] * self.k
        right = y[:, [0]] + y[:, [1]] * self.k
        zt = X.evaluate(interval)
        if len(zt.shape) > 3:
            zt = torch.diagonal(zt, dim1=0, dim2=1).permute(2, 0, 1)
        zt = zt[:, :, 0]
        zt = ((zt > left) * (zt < right)).int().float()

        cross_entropy = -torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10))
        dice_loss = self.loss(z, zt)
        # mse_z = (self.loss(z, zt)-torch.mean(zt*torch.log(z+1e-10)+(1-zt)*torch.log(1-z+1e-10)))/2


        timelist = X.evaluate(interval)
        if len(timelist.shape) > 3:
            timelist = torch.diagonal(timelist, dim1=0, dim2=1).permute(2, 0, 1)
        timelist = timelist[:, :, 0]

        if not self.soft_threshold:
            z = (z > self.threshold).int()
        diffz = torch.diff(z, append=z[:, [-1]])

        if self.method == 'diff':
            plus = torch.trapz(torch.abs(diffz) * timelist, dim=-1).unsqueeze(-1)
            minus = torch.trapz(diffz * timelist, dim=-1).unsqueeze(-1)
            reg = torch.hstack([plus / 2, -minus / (2 * self.k)])
        elif self.method == 'avg':
            area = torch.trapz(z, timelist, dim=-1).unsqueeze(-1)
            avg = torch.trapz(z * timelist, timelist, dim=-1).unsqueeze(-1) / area
            reg = torch.hstack([avg, area / (2 * self.k)])
        else:
            print('method can only be diff or avg')
            

        # length_penalty = torch.log(torch.mean((torch.sum(z, dim=-1) - torch.sum(zt, dim=-1))**2) + 1e-10)
        diffz_abssum_penalty = torch.mean((torch.sum(torch.abs(diffz), dim=-1) - 2.)**2)
        # diffz_sum_penalty = torch.log(torch.mean((torch.sum(diffz.float(), dim=-1))**2) + 1e-10)

        # loss_z = cross_entropy # + length_penalty + diffz_abssum_penalty + diffz_sum_penalty
        loss_z = cross_entropy + dice_loss + diffz_abssum_penalty

        if self.plot:
            avg = torch.mean(z0[0, :, 0].cpu()).item()
            plt.plot(timelist[0].cpu(), z0[0, :, 0].cpu())
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

        mask = torch.stack([timelist.detach(), z.detach()], dim=-1)
        if self.animate:
            return reg, loss_z, mask
        return reg, loss_z