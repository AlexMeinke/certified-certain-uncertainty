import torch
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans

import numpy as np
from sklearn.decomposition import PCA


class LpMetric(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
        
    def forward(self, x, y, dim=None):
        return (x-y).norm(p=self.p, dim=dim)


class GMM(nn.Module):
    def __init__(self, K, D, mu=None, logvar=None, alpha=None, metric=LpMetric()):
        """
        Initializes means, variances and weights randomly
        :param K: number of centroids
        :param D: number of features
        """
        super().__init__()
        
        self.D = D
        self.K = K
        self.metric = metric
        if mu is None:
            self.mu = nn.Parameter(torch.rand(K, D))
        else:
            self.mu = nn.Parameter(mu)
            
        if logvar is None:
            self.logvar = nn.Parameter(torch.rand(K))
        else:
            self.logvar = nn.Parameter(logvar)
            
        if alpha is None:
            self.alpha = nn.Parameter(torch.empty(K).fill_(1. / K)).log()
        else:
            self.alpha = nn.Parameter(alpha)

    def forward(self, X):
        """
        Compute the likelihood of each data point under each gaussians.
        :param X: design matrix (examples, features) (N,D)
        :return likelihoods: (K, examples) (K, N)
        """
        a = self.metric(X[None,:,:], self.mu[:,None,:], dim=2)**2
        b = self.logvar[:,None].exp()
        return (self.alpha[:,None] - .5*self.D*self.logvar[:,None]
                - .5*( a/b ) )
                    