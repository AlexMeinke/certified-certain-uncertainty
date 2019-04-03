import torch
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans

class MixtureModel(nn.Module):
    
    def __init__(self, K, D, mu=None, logvar=None, alpha=None):
        """
        Initializes means, variances and weights randomly
        :param K: number of centroids
        :param D: number of features
        """
        super().__init__()

        self.D = D
        self.K = K
        if mu is None:
            self.mu = nn.Parameter(torch.rand(K, D))
        else:
            self.mu = nn.Parameter(mu.copy())
            
        if logvar is None:
            self.logvar = nn.Parameter(torch.rand(K))
        else:
            self.logvar = nn.Parameter(logvar.copy())
            
        if alpha is None:
            self.alpha = nn.Parameter(torch.empty(K).fill_(1. / K))
        else:
            self.alpha = nn.Parameter(alpha.copy())

        self.logvarbound = 0
        
    def forward(self, x):
        pass
    
    def calculate_bound(self, L):
        pass
        
    def find_solution(self, X, initialize=True, iterate=True, use_kmeans=True):
        assert X.device==self.mu.device, 'Data stored on ' + str(X.device) + ' but model on ' + str(self.mu.device)
        
        with torch.no_grad():
            if initialize:
                m = X.size(0)

                if (use_kmeans):
                    kmeans = KMeans(n_clusters=self.K, random_state=0, max_iter=300).fit(X.cpu())
                    self.mu.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=self.mu.device)
                else:
                    idxs = torch.from_numpy(np.random.choice(m, self.K, replace=False)).long()
                    self.mu.data = X[idxs]
                    
                index = (X[:,None,:]-self.mu.clone().detach()[None,:,:]).norm(dim=2).min(dim=1)[1]
                for i in range(self.K):
                    assert (index==i).sum()>0, 'Empty cluster'
                    self.alpha.data[i] = (index==i).float().sum() / (3*self.K)
                    temp = (X[index==i,:] -self.mu.data[i,:]).norm(dim=1).mean()
                    if temp < 0.00001:
                        temp = torch.tensor(1.)
                    self.logvar.data[i] = temp.log() * 2

                self.logvarbound = (X.var() / m).log()

    def prune(self):
        """
        prunes away centroids with negative weigths
        """
        with torch.no_grad():
            index = torch.nonzero(self.alpha > 0)

            self.mu = nn.Parameter(self.mu[index].squeeze(1))
            self.logvar = nn.Parameter(self.logvar[index].squeeze(1))
            self.alpha = nn.Parameter(self.alpha[index].squeeze(1))
            self.K = len(index)

            
class GMM(MixtureModel):
    def __init__(self, K, D, mu=None, logvar=None, alpha=None):
        """
        Initializes means, variances and weights randomly
        :param K: number of centroids
        :param D: number of features
        """
        super().__init__(K, D)

    def forward(self, X):
        """
        Compute the likelihood of each data point under each gaussians.
        :param X: design matrix (examples, features) (N,D)
        :return likelihoods: (K, examples) (K, N)
        """
        if self.alpha.min() < 0:
            self.prune()
        a = ((X[None,:,:]-self.mu[:,None,:])**2).sum(-1)
        b = self.logvar[:,None].exp()
        return (self.alpha.log()[:,None] 
                - ( a/b ) )
        
    def calculate_bound(self, L):
        var = self.logvar[:,None].exp()
        bound = (self.alpha.log()[:,None] 
                - ( L**2/var ) )
        return torch.logsumexp(bound.squeeze(),dim=0)
    
    def EM_step(self, X):
        log_post = self.get_posteriors(X)
        post = log_post.exp()
        Nk = post.sum(dim=1)
        
        self.mu.data = (post[:,:,None]*X[None,:,:]).sum(dim=1) / Nk[:,None]
        temp = log_post + ((X[None,:,:]-self.mu[:,None,:])**2).sum(dim=-1).log()
        self.logvar.data = (- Nk.log() 
                       + torch.logsumexp(temp, dim=1, keepdim=False))
        
        self.alpha.data = Nk/Nk.sum()
        
    def get_posteriors(self, X):
        log_like = self.forward(X)
        log_post = log_like - torch.logsumexp(log_like, dim=0, keepdim=True)
        return log_post
    
    def find_solution(self, X, initialize=True, iterate=True, use_kmeans=True):
        super().find_solution(X, initialize=True, iterate=True, use_kmeans=True)
        if iterate:
            for _ in range(500):
                mu_prev = self.mu
                logvar_prev = self.logvar
                alpha_prev = self.alpha
                self.EM_step(X)
                self.logvar.data[self.logvar < self.logvarbound] = self.logvarbound

                delta = torch.stack( ((mu_prev-self.mu).abs().max(),
                            (logvar_prev-self.logvar).abs().max(),
                            (alpha_prev-self.alpha).abs().max()) ).max()

                if delta<10e-6:
                    break

class QuadraticMixtureModel(MixtureModel):
    def __init__(self, K, D, mu=None, logvar=None, alpha=None):
        """
        Initializes means, variances and weights randomly
        :param K: number of centroids
        :param D: number of features
        """
        super().__init__(K, D)

    def forward(self, X):
        """
        Compute the likelihood of each data point under each centroid
        :param X: design matrix (examples, features) (N,D)
        :return likelihoods: (K, examples) (K, N)
        """
        if self.alpha.min() < 0:
            self.prune()
        a = ((X[None,:,:]-self.mu[:,None,:])**2).sum(-1)
        b = self.logvar[:,None].exp()
        return (self.alpha.log()[:,None] - ( 1+ b/a ).log() )
        
    def calculate_bound(self, L):
        return 1.
                        
class LeNet(nn.Module):
    def __init__(self, preproc=torch.zeros(28, 28)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4*4*64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.preproc = nn.Parameter(preproc[0], requires_grad=False)
        self.preproc_std = nn.Parameter(preproc[1], requires_grad=False)

    def forward(self, x):
        x = (x-self.preproc)/self.preproc_std
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


class RobustModel(nn.Module):
    def __init__(self, base_model, mixture_model, loglam, dim=784, classes=10):
        super().__init__()
        self.base_model = base_model
        
        self.dim = dim
        self.mm = mixture_model

        self.loglam = nn.Parameter(torch.tensor(loglam, dtype=torch.float))
        self.log_K = -torch.tensor(classes, dtype=torch.float).log()
        
        
    def forward(self, x):
        batch_size = x.shape[0]
        likelihood_per_peak = self.mm(x.view(batch_size, self.dim))
        like = torch.logsumexp(likelihood_per_peak, dim=0)

        x = self.base_model(x)
        
        a1 = torch.stack( (x + like[:,None], (self.loglam + self.log_K)*torch.ones_like(x) ), 0)
        b1 = torch.logsumexp(a1, 0).squeeze()

        a2 = torch.stack( (like , (self.loglam)*torch.ones_like(like) ), 0)
        b2 = torch.logsumexp(a2, 0).squeeze()[:,None]

        return b1-b2