import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, OneHotCategorical

class MixtureDensityNetwork(nn.Module):
    """
    Mixture density network compatible with full covariance.

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    full_cov: bool; whether to use full or diagonal covariance matrix
    """
    def __init__(self, dim_in, dim_out, n_components, full_cov=True):
        super().__init__()
        self.pi_network = CategoricalNetwork(dim_in, n_components)
        self.normal_network = NormalNetwork(dim_in, dim_out, n_components, full_cov)

    def forward(self, x):
        return self.pi_network(x), self.normal_network(x)

    def loss(self, x, y):
        pi, normal = self.forward(x)
        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss

    def sample(self, x):
        pi, normal = self.forward(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)
        return samples

class NormalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, n_components, full_cov=True):
        super().__init__()
        self.n_components = n_components
        self.out_dim = out_dim
        self.full_cov = full_cov
        self.tril_indices = torch.tril_indices(row=out_dim, col=out_dim, offset=0)
        self.elu = nn.ELU()
        self.mean_net = nn.Sequential(
                nn.Linear(in_dim, out_dim * n_components),
            )
        if full_cov:
            # Cholesky decomposition of the covariance matrix
            self.tril_net = nn.Sequential(
                nn.Linear(in_dim, int(out_dim * (out_dim + 1) / 2 * n_components)),
            )
        else:
            self.tril_net = nn.Sequential(
                nn.Linear(in_dim, out_dim * n_components),
            )

    def forward(self, x):
        mean = self.mean_net(x).reshape(-1, self.n_components, self.out_dim)
        if self.full_cov:
            tril_values = self.tril_net(x).reshape(mean.shape[0], self.n_components, -1)
            tril = torch.zeros(mean.shape[0], mean.shape[1], mean.shape[2], mean.shape[2]).to(x.device)
            tril[:, :, self.tril_indices[0], self.tril_indices[1]] = tril_values
            # diagonal element must be strictly positive
            # use diag = elu(diag) + 1 to ensure positivity
            tril = tril - torch.diag_embed(torch.diagonal(tril, dim1=-2, dim2=-1)) + torch.diag_embed(self.elu(torch.diagonal(tril, dim1=-2, dim2=-1)) + 1)
        else:
            tril = self.tril_net(x).reshape(mean.shape[0], self.n_components, -1)
            tril = torch.diag_embed(self.elu(tril) + 1)
        return MultivariateNormal(mean, scale_tril=tril)

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        params = self.network(x)
        return OneHotCategorical(logits=params)