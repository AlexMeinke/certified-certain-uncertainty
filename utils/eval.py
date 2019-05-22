import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import scipy

import utils.adversarial as adv
import utils.dataloaders as dl
import utils.models as models
import utils.gmm_helpers as gmm_helpers


def test_metrics(model, device, in_loader, out_loader):
    with torch.no_grad():
        model.eval()
        conf_in = []
        conf_out = []

        for data_in, _ in in_loader:
            data_in = data_in.to(device)
            output_in = model(data_in).max(1)[0].exp()
            conf_in.append(output_in)
            
        for data_out, _ in out_loader:    
            data_out = data_out.to(device)
            output_out = model(data_out).max(1)[0].exp()
            conf_out.append(output_out)
            
        conf_in = torch.cat(conf_in)
        conf_out = torch.cat(conf_out)
        
        y_true = torch.cat([torch.ones_like(conf_in), 
                            torch.zeros_like(conf_out)]).cpu().numpy()
        y_scores = torch.cat([conf_in, 
                              conf_out]).cpu().numpy()
        
        mmc = conf_out.mean().item()
        auroc = roc_auc_score(y_true, y_scores)
        fp95 = ((conf_out > 0.95).float().mean().item())
        return mmc, auroc, fp95

    
def evaluate_model(model, device, base_loader, loaders):
    metrics = []
    mmc, _, _ = test_metrics(model, device, base_loader, base_loader)
    metrics.append(['orig', mmc, 0.])
    for (name, data_loader) in loaders:
        mmc, auroc, fp95 = test_metrics(model, device, base_loader, data_loader)
        metrics.append([name, mmc, auroc])

    df = pd.DataFrame(metrics, columns = ['DataSet', 'MMC', 'AUROC'])
    return df.set_index('DataSet')


def write_log(df, writer, epoch=0):
    for i in df.index:
        if i!='orig':
            writer.add_scalar('AUROC/'+i, df.loc[i]['AUROC'], epoch)
            writer.add_scalar('MMC/'+i, df.loc[i]['MMC'], epoch)
    
    
def evaluate(model, device, dataset, loaders, load_adversaries=False, writer=None, epoch=0):

    if load_adversaries:
        NoiseLoader = loaders[-1][1]
        print('[INFO] Loading Adversaries...')
        AdversarialNoiseLoader = adv.create_adv_noise_loader(model, NoiseLoader, device, batches=5)
        AdversarialSampleLoader = adv.create_adv_sample_loader(model, 
                                                               dl.datasets_dict[dataset](train=False),
                                                               device, batches=5)
        temp = loaders + (
            [
             ('Adv. Noise', AdversarialNoiseLoader ),
             ('Adv. Sample', AdversarialSampleLoader)]
            )
    else:
        temp = loaders
    df = evaluate_model(model, device, dl.datasets_dict[dataset](train=False), temp)
                
    if writer is not None:
        write_log(df, writer, epoch)
    return df


def aggregate_adv_stats(model_list, gmm, device, shape, classes=10, 
                        batches=10, batch_size=100, steps=200, 
                        restarts=10, alpha=1., lam=1.):
    
    pca = models.MyPCA(gmm.metric.comp_vecs.t(), gmm.metric.singular_values, shape)
    
    f = 1.1
    b = lam * (f-1.) / (classes-f)

    bounds = []
    stats = []
    samples = []
    seeds = []

    for _ in range(batches):
        seed = torch.rand((batch_size,)+tuple(shape), device=device)
        batch_bounds = []
        batch_samples = []

        for x in seed:
            batch_bounds.append( scipy.optimize.brentq(gmm_helpers.get_b, 0, 10000., args = (x, gmm, b)) )
        batch_bounds = torch.tensor(batch_bounds, device=device)
        bounds.append(batch_bounds.clone().cpu())

        batch_stats = []
        for i, model in enumerate(model_list):
            adv_noise, _ = adv.gen_pca_noise(model, device, seed, pca, 
                                             epsilon=batch_bounds, perturb=True, 
                                             restarts=restarts, steps=steps, alpha=alpha)
            batch_stats.append(model(adv_noise).max(1)[0].exp().detach().cpu().clone())
            batch_samples.append(adv_noise[0].detach().cpu())
            
        seeds.append(seed[0].cpu())
        
        batch_samples = torch.stack(batch_samples, 0)
        batch_stats = torch.stack(batch_stats, 0)
        stats.append(batch_stats.clone())
        samples.append(batch_samples.clone())

    seeds = torch.cat(seeds, 0)
    samples = torch.cat(samples, 0)
    stats = torch.cat(stats, -1)
    bounds = torch.cat(bounds, 0)
    
    return stats, bounds, seeds, samples

class StatsContainer(torch.nn.Module):
    def __init__(self, stats, bounds, seeds, samples):
        self.stats = stats
        self.bounds = bounds
        self.seeds = seeds
        self.samples = samples
    def forward(self, x):
        return x
    