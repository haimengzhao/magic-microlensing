'''
Upper bound KL divergence between GMMs using Eq. (20) in https://ieeexplore.ieee.org/abstract/document/6289001/

D(f||g) where f is unknown but we has sample access, and g is parameterized and we want to learn it.
So we will omit terms that do not depend on g!

The covariance matrices are given in terms of precision matrices, the inverse of the covariance matrices.
'''

import torch

def kl_between_gaussians(mean_f, precision_f, mean_g, precision_g):
    '''
    Compute KL divergence D(f||g) between two Gaussians. Everything batched.

    Args:
            mean_f (torch.Tensor): mean of the first Gaussian.
            precision_f (torch.Tensor): precision of the first Gaussian.
            mean_g (torch.Tensor): mean of the second Gaussian.
            precision_g (torch.Tensor): precision of the second Gaussian.

    Returns:
            torch.Tensor: KL divergence.
    '''

    logdet_precision_g = torch.logdet(precision_g)
    logdet_precision_f = torch.logdet(precision_f)
    kl = - 0.5 * logdet_precision_g + 0.5 * logdet_precision_f
    
    cov_f = torch.linalg.inv(precision_f)
    kl += 0.5 * torch.einsum('...ij, ...ji->...', precision_g, cov_f)
    
    kl += 0.5 * torch.einsum('...i, ...ij, ...j->...', mean_f - mean_g, precision_g, mean_f - mean_g)
    
    kl -= 0.5 * mean_f.shape[-1]
    
    # ensure positivity
    kl = torch.clamp(kl, min=0)
    
    return kl

def kl_upper_bound_GMM(mean_f, precision_f, weight_g, mean_g, precision_g):
    '''
        Compute the upper bound KL divergence D(f||g) between one Gaussian and one GMM. Everything batched.

        Args:
                mean_f (torch.Tensor), shape (batch_size, dim): mean of the Gaussian.
                precision_f (torch.Tensor), shape (batch_size, dim, dim): precision of the Gaussian.
                weight_g (torch.Tensor), shape (batch_size, n_gaussians): weights of the GMM.
                mean_g (torch.Tensor), shape (batch_size, n_gaussians, dim): means of the GMM.
                precision_g (torch.Tensor), shape (batch_size, n_gaussians, dim, dim): precisions of the GMM.

        Returns:
                torch.Tensor: KL divergence.
    '''
    kl_individual = kl_between_gaussians(mean_f.unsqueeze(-2), precision_f.unsqueeze(-3), mean_g, precision_g)
    kl = -torch.logsumexp(torch.log(weight_g) - kl_individual, dim=-1)
    
    # ensure positivity
    kl = torch.clamp(kl, min=0)
    
    return kl 
        
if __name__ == '__main__':
    mean_f = torch.randn(10, 2)
    precision_f = torch.randn(10, 2, 2)
    precision_f = torch.matmul(precision_f, precision_f.transpose(1, 2))
    
    mean_g = torch.randn(10, 2)
    precision_g = torch.randn(10, 2, 2)
    precision_g = torch.matmul(precision_g, precision_g.transpose(1, 2))
    
    kl = kl_between_gaussians(mean_f, precision_f, mean_g, precision_g)
    print(kl)
    
    kl = kl_between_gaussians(mean_f, precision_f, mean_f, precision_f)
    print(kl)
    
    weight_g = torch.randn(10, 3)
    weight_g = torch.nn.functional.softmax(weight_g, dim=-1)
    mean_g = torch.randn(10, 3, 2)
    precision_g = torch.randn(10, 3, 2, 2)
    precision_g = torch.matmul(precision_g, precision_g.transpose(2, 3))

    kl = kl_upper_bound_GMM(mean_f, precision_f, weight_g, mean_g, precision_g)
    print(kl)
    
    # with f itself
    kl = kl_upper_bound_GMM(mean_f, precision_f, torch.ones(10, 1), mean_f.unsqueeze(1), precision_f.unsqueeze(1))
    print(kl)
    