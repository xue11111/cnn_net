import torch
import torch.nn as nn


class Batch_N(nn.Module):
    def __init__(self, gamma, beta, moving_mean, moving_val, eps, moment):
        super(Batch_N, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.moving_mean = moving_mean
        self.moving_val = moving_val
        self.eps = eps
        self.moment = moment

    def forward(self,x):
        if not torch.is_grad_enabled():
            x_hat = (x - self.moving_mean) / torch.sqrt(self.moving_val + self.eps)
        else:
            assert len(x.shape) in (2, 4)
            if len(x.shape) == 2:

                mean = x.mean(dim=0)
                var = ((x - mean) ** 2).mean(dim=0)
            else:
                mean = x.mean(dim=(0,2,3))
                var = ((x - mean)**2).mean(dim=(0,2,3))
            x_hat = (x -mean) / torch.sqrt(var + self.eps)

            moving_mean = self.moving_mean * self.moment + (1 - self.moment) * mean
            moving_var = self.moving_val * self.moment + (1 - self.moment) * var

        y = self.gamma * x_hat + self.beta
        return y, moving_mean.data, moving_var.data

