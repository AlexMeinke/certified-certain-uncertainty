import numpy as np
import torch

import utils.models as models
import utils.dataloaders as dl

def find_lam(gmm, percentile, dataloader):
    X = []
    for i, (x, _) in enumerate(dataloader):
        if i>50:
            break;
        X.append(torch.logsumexp(gmm(x.view(-1,gmm.D)),0).detach() )
    X = torch.cat(X, 0)
    lam = np.percentile(X.numpy(), percentile)
    return lam

def rescale_gmm(gmm, loader):
    m2 = gmm.logvar.exp().mean().detach()
    m = []
    for i, (data, _) in enumerate(loader):
        if i>50:
            break;
        M = ((data.view(data.shape[0], -1)[:,None,:]-gmm.mu[None,:,:])**2).sum(-1)
        m.append(M.min(-1)[0])
    m = torch.cat(m, 0)
    gmm.logvar.data += (m.mean()/m2).log()
    return gmm