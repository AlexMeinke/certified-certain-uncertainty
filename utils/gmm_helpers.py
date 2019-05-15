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