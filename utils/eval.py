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

def write_log(df, writer, dataset, epoch=0):
    if dataset=='MNIST':
        writer.add_scalar('AUROC/FMNIST', df['AUROC'].iloc[1], epoch)
        writer.add_scalar('AUROC/EMNIST', df['AUROC'].iloc[2], epoch)
        writer.add_scalar('AUROC/GrayCIFAR10', df['AUROC'].iloc[3], epoch)
        writer.add_scalar('AUROC/Noise', df['AUROC'].iloc[4], epoch)
        writer.add_scalar('AUROC/AdvNoise', df['AUROC'].iloc[5], epoch)
        writer.add_scalar('AUROC/AdvSample', df['AUROC'].iloc[6], epoch)

        writer.add_scalar('MMC/FMNIST', df['MMC'].iloc[1], epoch)
        writer.add_scalar('MMC/EMNIST', df['MMC'].iloc[2], epoch)
        writer.add_scalar('MMC/GrayCIFAR10', df['MMC'].iloc[3], epoch)
        writer.add_scalar('MMC/Noise', df['MMC'].iloc[4], epoch)
        writer.add_scalar('MMC/AdvNoise', df['MMC'].iloc[5], epoch)
        writer.add_scalar('MMC/AdvSample', df['MMC'].iloc[6], epoch)
        return
    
    if dataset=='SVHN':
        writer.add_scalar('AUROC/CIFAR10', df['AUROC'].iloc[1], epoch)
        writer.add_scalar('AUROC/CIFAR100', df['AUROC'].iloc[2], epoch)
        writer.add_scalar('AUROC/LSUN CR', df['AUROC'].iloc[3], epoch)
        writer.add_scalar('AUROC/ImageNet-', df['AUROC'].iloc[4], epoch)
        writer.add_scalar('AUROC/Noise', df['AUROC'].iloc[5], epoch)
        writer.add_scalar('AUROC/AdvNoise', df['AUROC'].iloc[6], epoch)
        writer.add_scalar('AUROC/AdvSample', df['AUROC'].iloc[7], epoch)

        writer.add_scalar('MMC/CIFAR10', df['MMC'].iloc[1], epoch)
        writer.add_scalar('MMC/CIFAR100', df['MMC'].iloc[2], epoch)
        writer.add_scalar('MMC/LSUN CR', df['MMC'].iloc[3], epoch)
        writer.add_scalar('MMC/ImageNet-', df['MMC'].iloc[4], epoch)
        writer.add_scalar('MMC/Noise', df['MMC'].iloc[5], epoch)
        writer.add_scalar('MMC/AdvNoise', df['MMC'].iloc[6], epoch)
        writer.add_scalar('MMC/AdvSample', df['MMC'].iloc[7], epoch)
        return
    
    if dataset=='CIFAR10':
        writer.add_scalar('AUROC/SVHN', df['AUROC'].iloc[1], epoch)
        writer.add_scalar('AUROC/CIFAR100', df['AUROC'].iloc[2], epoch)
        writer.add_scalar('AUROC/LSUN CR', df['AUROC'].iloc[3], epoch)
        writer.add_scalar('AUROC/ImageNet-', df['AUROC'].iloc[4], epoch)
        writer.add_scalar('AUROC/Noise', df['AUROC'].iloc[5], epoch)
        writer.add_scalar('AUROC/AdvNoise', df['AUROC'].iloc[6], epoch)
        writer.add_scalar('AUROC/AdvSample', df['AUROC'].iloc[7], epoch)

        writer.add_scalar('MMC/SVHN', df['MMC'].iloc[1], epoch)
        writer.add_scalar('MMC/CIFAR100', df['MMC'].iloc[2], epoch)
        writer.add_scalar('MMC/LSUN CR', df['MMC'].iloc[3], epoch)
        writer.add_scalar('MMC/ImageNet-', df['MMC'].iloc[4], epoch)
        writer.add_scalar('MMC/Noise', df['MMC'].iloc[5], epoch)
        writer.add_scalar('MMC/AdvNoise', df['MMC'].iloc[6], epoch)
        writer.add_scalar('MMC/AdvSample', df['MMC'].iloc[7], epoch)
        return
    

def evaluate(model, device, dataset, loaders, writer=None, epoch=0):
    NoiseLoader = loaders[-1][1]
    AdversarialNoiseLoader = adv.create_adv_noise_loader(model, NoiseLoader, device)
    AdversarialSampleLoader = adv.create_adv_sample_loader(model, dl.datasets_dict[dataset](train=False), device)
    temp = loaders + (
        [
         ('Adv. Noise', AdversarialNoiseLoader ),
         ('Adv. Sample', AdversarialSampleLoader)]
        )
    df = evaluate_model(model, device, dl.datasets_dict[dataset](train=False), temp)
                
    if writer is not None:
        write_log(df, writer, dataset, epoch)
    return df