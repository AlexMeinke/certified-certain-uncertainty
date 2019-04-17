import torch
import numpy as np
import pandas as pd
import utils.adversarial as adv
import utils.dataloaders as dl

def test_metrics(model, device, in_loader, out_loader, thresholds=torch.linspace(.1, 1., 20000)):
    thresholds=thresholds.to(device)
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
        
        tp = (conf_in[:,None] > thresholds[None,:]).sum(0).float()/(count*in_loader.batch_size)
        fp = (conf_out[:,None] > thresholds[None,:]).sum(0).float()/(count*out_loader.batch_size)
        
        mmc = conf_out.mean().item()
        auroc = -np.trapz(tp.cpu().numpy(), x=fp.cpu().numpy())
        fp95 = ((conf_out > 0.95).sum().float()/(count*out_loader.batch_size)).item()
        return mmc, auroc, fp95
    
def evaluate_model(model, device, base_loader, loaders):
    metrics = []
    mmc, _, _ = test_metrics(model, device, base_loader, base_loader)
    metrics.append(['MNIST', mmc, '-', '-'])
    for (name, data_loader) in loaders:
        mmc, auroc, fp95 = test_metrics(model, device, base_loader, data_loader)
        metrics.append([name, mmc, auroc, fp95])
    df = pd.DataFrame(metrics, columns = ['DataSet', 'MMC', 'AUROC', 'FPR@95'])
    return df.set_index('DataSet')

def evaluate_MNIST(model, device):
    AdversarialNoiseLoader = adv.create_adv_noise_loader(model, dl.Noise_test_loader_MNIST, device)
    AdversarialSampleLoader = adv.create_adv_sample_loader(model, dl.MNIST_test_loader, device)

    loaders = (
    [('FMNIST', dl.FMNIST_test_loader), 
     ('EMNIST', dl.EMNIST_test_loader),
     ('GrayCIFAR10', dl.GrayCIFAR10_test_loader),
     ('Noise', dl.Noise_test_loader_MNIST),
     ('Adv. Noise', AdversarialNoiseLoader ),
     ('Adv. Sample', AdversarialSampleLoader)]
    )

    df = evaluate_model(model, device, dl.MNIST_test_loader, loaders)
    return df