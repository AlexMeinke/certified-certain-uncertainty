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
        self.norm_const = 0.
        
    def forward(self, x, y, dim=None):
        return (x-y).norm(p=self.p, dim=dim)

    
class PCAMetric(nn.Module):
    def __init__(self, X, p=2, min_sv_factor=100.):
        super().__init__()
        self.p = p
        X = np.array(X)
        pca = PCA()
        pca.fit(X)

        self.comp_vecs = nn.Parameter(torch.tensor(pca.components_), requires_grad=False)
        self.singular_values = torch.tensor(pca.singular_values_)
        self.min_sv = self.singular_values[0] / min_sv_factor
        self.singular_values[self.singular_values<self.min_sv] = self.min_sv
        self.singular_values = nn.Parameter(self.singular_values, requires_grad=False)
        
        self.norm_const = self.singular_values.log().sum()
        
    def forward(self, x, y, dim=None):
        rotated_dist = torch.einsum("ijk,lk->ijl", (x-y, self.comp_vecs))
        rescaled_dist = rotated_dist / self.singular_values[None,None,:]
        return rescaled_dist.norm(dim=2, p=self.p)

    
class LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        
    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2).view(-1,14*14*32)
        return x

    
class PerceptualMetric(nn.Module):
    def __init__(self, model, p=2):
        super().__init__()
        self.model = model
        self.p = p
        
    def forward(self, x, y, dim=None):
        return (self.model(x)[None,:,:]-self.model(y)[:,None,:]).norm(p=self.p, dim=dim)

    
class PerceptualPCA(nn.Module):
    def __init__(self, model, pca, p=2):
        super().__init__()
        self.model = model
        self.pca = pca
        
    def forward(self, x, y, dim=None):
        return self.pca(self.model(x)[None,:,:], self.model(y)[:,None,:])
    
    
class MixtureModel(nn.Module):
    
    def __init__(self, K, D, mu=None, logvar=None, alpha=None, metric=LpMetric()):
        """
        Initializes means, variances and weights randomly
        :param K: number of centroids
        :param D: number of features
        :param mu: centers of centroids (K,D)
        :param logvar: logarithm of the variances of the centroids (K)
        :param alpha: logarithm of the weights of the centroids (K)
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
            self.alpha = nn.Parameter(torch.empty(K).fill_(1. / K).log())
        else:
            self.alpha = nn.Parameter(alpha)

        self.logvarbound = 0
        
    def forward(self, x):
        pass
    
    def calculate_bound(self, L):
        pass
    
    def get_posteriors(self, X):
        log_like = self.forward(X)
        log_post = log_like - torch.logsumexp(log_like, dim=0, keepdim=True)
        return log_post
    
    def EM_step(self, X):
        log_post = self.get_posteriors(X)
        
        log_Nk = torch.logsumexp(log_post, 1)

        self.mu.data = ((log_post[:,:,None] - log_Nk[:,None,None]).exp() * X[None,:,:]).sum(1)
        temp = log_post + (((X[None,:,:]-self.mu[:,None,:])**2).sum(dim=-1)/self.D).log()
        self.logvar.data = (- log_Nk 
                            + torch.logsumexp(temp, dim=1, keepdim=False))
        self.alpha = ( log_Nk - torch.logsumexp(log_Nk, 0) ).clone().detach()
        
    def find_solution(self, X, initialize=True, iterate=True, use_kmeans=True, verbose=False):
        assert X.device==self.mu.device, 'Data stored on ' + str(X.device) + ' but model on ' + str(self.mu.device)
        
        with torch.no_grad():
            if initialize:
                m = X.size(0)

                if (use_kmeans):
                    kmeans = KMeans(n_clusters=self.K, random_state=0, max_iter=300).fit(X.cpu())
                    self.mu.data = torch.tensor(kmeans.cluster_centers_, 
                                                dtype=torch.float, 
                                                device=self.mu.device)
                else:
                    idxs = torch.from_numpy(np.random.choice(m, self.K, replace=False)).long()
                    self.mu.data = X[idxs]
                    
                index = (X[:,None,:]-self.mu.clone().detach()[None,:,:]).norm(dim=2).min(dim=1)[1]
                for i in range(self.K):
                    assert (index==i).sum()>0, 'Empty cluster'
                    self.alpha.data[i] = ((index==i).float().sum() / (3*self.K)).log()
                    temp = (X[index==i,:] - self.mu.data[i,:]).norm(dim=1).mean()
                    if temp < 0.00001:
                        temp = torch.tensor(1.)
                    self.logvar.data[i] = temp.log() * 2
                
                self.alpha.data = self.alpha.data.exp()
                self.alpha.data /= self.alpha.data.sum()
                self.alpha.data = self.alpha.data.log()

                self.logvarbound = (X.var() / m).log()

            if iterate:
                for i in range(50):
                    mu_prev = self.mu.clone().detach()
                    logvar_prev = self.logvar.clone().detach()
                    alpha_prev = self.alpha.clone().detach()
                    self.EM_step(X)
                      
                    self.logvar.data[self.logvar < self.logvarbound] = self.logvarbound

                    delta = torch.stack( ((mu_prev-self.mu).abs().max(),
                                (logvar_prev-self.logvar).abs().max(),
                                (alpha_prev-self.alpha).abs().max()) ).max()
                    if verbose:
                        print('Iteration: '+ str(i)+'\t delta: '+str(delta.item()))
                        print((mu_prev-self.mu).abs().max())
                        print((logvar_prev-self.logvar).abs().max())
                        print((alpha_prev-self.alpha).abs().max())
                    if delta<10e-6:
                        break

            
class GMM(MixtureModel):
    def __init__(self, K, D, mu=None, logvar=None, alpha=None, metric=LpMetric()):
        """
        Initializes means, variances and weights randomly
        :param K: number of centroids
        :param D: number of features
        """
        super().__init__(K, D, mu, logvar, alpha, metric)
        self.norm_const = torch.tensor(2*np.pi).log() * self.D + metric.norm_const

    def forward(self, X):
        """
        Compute the likelihood of each data point under each gaussians.
        :param X: design matrix (examples, features) (N,D)
        :return likelihoods: (K, examples) (K, N)
        """
        a = self.metric(X[None,:,:], self.mu[:,None,:], dim=2)**2
        b = self.logvar[:,None].exp()
        return (self.alpha[:,None] - .5*self.D*self.logvar[:,None]
                - .5*( a/b ) - self.norm_const)
    
    def calculate_bound(self, L):
        var = self.logvar[:,None].exp()
        bound = (self.alpha[:,None] - .5*self.D*self.logvar[:,None]
                - .5* ( L**2/(2*var) ) - self.norm_const )
        return torch.logsumexp(bound.squeeze(),dim=0)
                    

class QuadraticMixtureModel(MixtureModel):
    def __init__(self, K, D, mu=None, logvar=None, alpha=None, metric=LpMetric()):
        """
        Initializes means, variances and weights randomly
        :param K: number of centroids
        :param D: number of features
        """
        super().__init__(K, D, mu, logvar, alpha, metric)

    def forward(self, X):
        """
        Compute the likelihood of each data point under each centroid
        :param X: design matrix (examples, features) (N,D)
        :return likelihoods: (K, examples) (K, N)
        """
        a = self.metric(X[None,:,:], self.mu[:,None,:], dim=2)**2
        var = self.logvar[:,None].exp()
        return (self.alpha[:,None] - ( 1+ a/var ).log() )
        
    def calculate_bound(self, L):
        var = self.logvar[:,None].exp()
        bound = ( self.alpha[:,None] - ( 1 + L**2/var ).log() ).squeeze()
        return torch.logsumexp(bound, dim=0)

    
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

    
class LeNetMadry(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*64)
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
