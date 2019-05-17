import torch
import torch.nn.functional as F
import torch.utils.data as data_utils


def gen_adv_noise(model, device, seed, epsilon=0.1, steps=40, step_size=0.01):
    
    with torch.no_grad():
        batch_size = seed.shape[0]
        alpha = step_size * torch.ones(batch_size,1,1,1, device=device)

        orig_data = seed.to(device)
        prev_data = seed.to(device)
        data = seed.to(device).requires_grad_()

        prev_losses = -100000.*torch.ones(batch_size, device=device)
        prev_grad = torch.zeros_like(seed, device=device)
        
    for _ in range(steps):
        with torch.enable_grad():
            y = model(data)
            losses = y.max(1)[0]
            losses.sum().backward()
            
        with torch.no_grad():
            grad = data.grad.sign()
            regret_index = losses<prev_losses
            alpha[regret_index] /= 2.
            data[regret_index] = prev_data[regret_index]
            grad[regret_index] = prev_grad[regret_index]
            
            prev_losses=losses
            prev_data = data
            prev_grad = grad
            
            data += alpha*grad
            delta = torch.clamp(data-orig_data, -epsilon, epsilon)
            data = torch.clamp(orig_data + delta, 0, 1).requires_grad_()
            
    return data.detach()


def gen_pca_noise(model, device, seed, pca, epsilon=0.1, steps=40, alpha=0.01):

    with torch.no_grad():
        batch_size = seed.shape[0]


        orig_data_pca = pca.trans(seed)
        data_pca = pca.trans(seed).requires_grad_()

        prev_losses = -100000.*torch.ones(batch_size, device=device)
        prev_grad = torch.zeros_like(seed, device=device)

    for _ in range(steps):
        with torch.enable_grad():
            y = model(pca.inv_trans(data_pca))
            losses = y.max(1)[0]
            losses.sum().backward()

        with torch.no_grad():
            delta = data_pca + alpha*data_pca.grad - orig_data_pca
            N = delta.norm(dim=-1)

            index = N>epsilon

            delta[index] *= (epsilon / N[index])[:, None]
            

            data_pca = orig_data_pca + delta
            data = pca.inv_trans(data_pca)
            data = torch.clamp(data, 0, 1)
            data_pca = pca.trans(data).requires_grad_()
    return data


def gen_adv_sample(model, device, seed, label, epsilon=0.1, steps=40, step_size=0.001):
    correct_index = label[:,None]!=torch.arange(10)[None,:]
    with torch.no_grad():
        batch_size = seed.shape[0]
        alpha = step_size * torch.ones(batch_size,1,1,1, device=device)

        orig_data = seed.to(device)
        prev_data = seed.to(device)
        data = seed.to(device).requires_grad_()

        prev_losses = -100000.*torch.ones(batch_size, device=device)
        prev_grad = torch.zeros_like(seed, device=device)
    for _ in range(steps):
        with torch.enable_grad():
            y = model(data)
            losses = y[correct_index].view(batch_size, 9).max(1)[0]
            losses.sum().backward()
            
        with torch.no_grad():
            grad = data.grad.sign()
            regret_index = losses<prev_losses
            alpha[regret_index] /= 2.
            data[regret_index] = prev_data[regret_index]
            grad[regret_index] = prev_grad[regret_index]
            
            prev_losses=losses
            prev_data = data
            prev_grad = grad
            
            data += alpha*grad
            delta = torch.clamp(data-orig_data, -epsilon, epsilon)
            data = torch.clamp(orig_data + delta, 0, 1).requires_grad_()
    return data.detach()

def create_adv_noise_loader(model, dataloader, device, batches=50):
    new_data = []
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx > batches:
            break
        new_data.append(gen_adv_noise(model, device, 
                                      data, epsilon=0.3,
                                      steps=200).detach().cpu()
                       )
    new_data = torch.cat(new_data, 0)

    adv_noise_set = data_utils.TensorDataset(new_data, torch.zeros(len(new_data),10))
    return data_utils.DataLoader(adv_noise_set, batch_size=100, shuffle=False)

def create_adv_sample_loader(model, dataloader, device, batches=50):
    new_data = []
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx > batches:
            break
        new_data.append(gen_adv_sample(model, device, 
                                       data, target,
                                       epsilon=0.3, steps=200).detach().cpu()
                       )
    new_data = torch.cat(new_data, 0)

    adv_sample_set = data_utils.TensorDataset(new_data, torch.zeros(len(new_data),10))
    return data_utils.DataLoader(adv_sample_set, batch_size=100, shuffle=False)