import numpy as np
import torch

import utils.models as models
import utils.dataloaders as dl


def find_lam(gmm, percentile, dataloader):
    X = []
    for i, (x, _) in enumerate(dataloader):
        if i>600:
            break;
        X.append(torch.logsumexp(gmm(x.view(-1,gmm.D)),0).detach() )
    X = torch.cat(X, 0)
    lam = np.percentile(X.numpy(), percentile)
    return lam


def rescale(gmm, percentile, dataloader):
    m2 = gmm.logvar.exp().mean().detach()
    m = []
    for i, (data, _) in enumerate(dataloader):
        if i>50:
            break;
        M = ((data.view(data.shape[0], -1)[:,None,:]-gmm.mu[None,:,:])**2).sum(-1)
        m.append(M.min(-1)[0].detach())
    m = torch.cat(m, 0)
    target_m = np.percentile(m.numpy(), 100 - percentile)
    gmm.logvar.data += (target_m/m2).log()
    return gmm


def get_b(r, x, gmm, b=1.):
    x = x.view(-1)
    d = gmm.metric(gmm.mu[None,:,:], x.view(-1)[None, None,:]).squeeze()

    var = gmm.logvar.exp()
    norm_const = .5*gmm.D*gmm.logvar + gmm.norm_const

    exponent = torch.stack([(d - r), torch.zeros_like(var)], 0).max(0)[0]
    
    exponent = exponent**2 / (2*var)
    
    bound = torch.logsumexp(gmm.alpha - norm_const - exponent, 0).detach().cpu().item() - np.log(b)
    return bound


def get_b_out(r, x, gmm, gmm_out, b=1.):
    x = x.view(-1)
    d = gmm.metric(gmm.mu[None,:,:], x.view(-1)[None, None,:]).squeeze()
    d_out = gmm.metric(gmm_out.mu[None,:,:], x.view(-1)[None, None,:]).squeeze()

    var = gmm.logvar.exp()
    var_out = gmm_out.logvar.exp()
    
    norm_const = .5 * gmm.D * gmm.logvar + gmm.norm_const
    norm_const_out = .5 * gmm_out.D * gmm_out.logvar + gmm_out.norm_const

    exponent = torch.stack([(d - r), torch.zeros_like(var)], 0).max(0)[0]
    exponent_out = d_out + r
    
    exponent = exponent**2 / (2*var)
    exponent_out = exponent_out**2 / (2*var_out)
    
    
    bound = (torch.logsumexp(gmm.alpha - norm_const - exponent, 0).detach().cpu().item()
             - torch.logsumexp(gmm_out.alpha - norm_const_out - exponent_out, 0).detach().cpu().item()
             - np.log(b))
    
    #bound2 = torch.logsumexp(gmm.alpha - norm_const - exponent, 0).detach().cpu().item()
    return bound
