import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import utils.adversarial as adv
import utils.dataloaders as dl

def test_metrics(model, device, in_loader, out_loader):
    with torch.no_grad():
        model.eval()
        conf_in = []
        conf_out = []
        count = 0.
        for ((batch_idx, (data_in, _)), (_, (data_out, _))) in zip(enumerate(in_loader),enumerate(out_loader)):
            count += 1
            data_in = data_in.to(device)
            data_out = data_out.to(device)

            output_in = model(data_in).max(1)[0].exp()
            output_out = model(data_out).max(1)[0].exp()

            conf_in.append(output_in)
            conf_out.append(output_out)
        conf_in = torch.cat(conf_in)
        conf_out = torch.cat(conf_out)
        
        y_true = torch.cat([torch.ones_like(conf_in), 
                            torch.zeros_like(conf_out)]).cpu().numpy()
        y_scores = torch.cat([conf_in, 
                              conf_out]).cpu().numpy()
        
        mmc = conf_out.mean().item()
        auroc = roc_auc_score(y_true, y_scores)
        fp95 = ((conf_out > 0.95).sum().float()/(count*out_loader.batch_size)).item()
        return mmc, auroc, fp95
    
def evaluate_model(model, device, base_loader, loaders):
    metrics = []
    mmc, _, _ = test_metrics(model, device, base_loader, base_loader)
    metrics.append(['orig', mmc, '-', '-'])
    for (name, data_loader) in loaders:
        mmc, auroc, fp95 = test_metrics(model, device, base_loader, data_loader)
        metrics.append([name, mmc, auroc, fp95])
    df = pd.DataFrame(metrics, columns = ['DataSet', 'MMC', 'AUROC', 'FPR@95'])
    return df.set_index('DataSet')

def write_log(df, writer, epoch=0):
    for i in df.index:
        writer.add_scalar('AUROC/'+i, df.loc[i]['AUROC'], epoch)
        writer.add_scalar('MMC/'+i, df.loc[i]['MMC'], epoch)
    

def evaluate(model, device, dataset, loaders, load_adversaries=False, writer=None, epoch=0):
    NoiseLoader = loaders[-1][1]
    if load_adversaries:
        AdversarialNoiseLoader = adv.create_adv_noise_loader(model, NoiseLoader, device, batches=5)
        AdversarialSampleLoader = adv.create_adv_sample_loader(model, dl.datasets_dict[dataset](train=False), device, batches=5)
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