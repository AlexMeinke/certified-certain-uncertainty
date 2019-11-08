import torch
import torch.nn.functional as F
import torch.utils.data as data_utils


def gen_adv_noise(model, device, seed, epsilon=0.1, restarts=1, perturb=False, 
                  steps=40, step_size=0.01, norm='inf'):
    '''
        Runs an adversarial noise attack in l_inf norm
        Maximizes the confidence in some class (different from adversarial
        attack which maximizes confidence in some wrong class)
    '''
    model.eval()
    batch_size = seed.shape[0]
    orig_data = seed.clone()
    
    if restarts>1:
        data = seed.clone()
        losses = 1e5*torch.ones(batch_size, device=device)
        for _ in range(restarts):
            current_data, current_losses = gen_adv_noise(model, device, seed, epsilon=epsilon,
                                                         restarts=1, perturb=True, 
                                                         steps=steps)
            with torch.no_grad():
                index = losses > current_losses
                data[index] = current_data[index]

                losses[index] = current_losses[index]
        return data, losses
    
    else:
        with torch.no_grad():
            alpha = step_size * torch.ones(batch_size,1,1,1, device=device)

            orig_data = seed.to(device)
            prev_data = seed.to(device)
            data = seed.to(device).requires_grad_()

            if perturb:
                    perturbation = epsilon * (torch.rand_like(prev_data) - .5)
                    prev_data += perturbation
                    data += perturbation

            prev_losses = 1e5 * torch.ones(batch_size, device=device)
            prev_grad = torch.zeros_like(seed, device=device)

        for _ in range(steps):
            with torch.enable_grad():
                y = model(data)
                losses = -y.max(1)[0]
                #losses = y.sum(1)
                grad = -torch.autograd.grad(losses.sum(), data)[0]
                    

            with torch.no_grad():
                if norm=='inf':
                    grad = grad.sign()
                else:
                    grad = grad / grad.norm(p=norm)
                
                regret_index = losses > prev_losses

                alpha[regret_index] /= 2.
                alpha[1-regret_index] *= 1.1
                data[regret_index] = prev_data[regret_index]
                grad[regret_index] = prev_grad[regret_index]

                prev_losses = losses
                prev_data = data
                prev_grad = grad
                
                data += alpha * grad
                
                delta = data - orig_data
                
                if norm=='inf':
                    delta = torch.clamp(delta, -epsilon, epsilon)
                    
                else:                
                    N = delta.norm(dim=-1, p=norm)

                    index = N > epsilon

                    delta[index] *= (epsilon / N[index])[:, None]
                
                    
                data = torch.clamp(orig_data + delta, 0, 1).requires_grad_()
         
        with torch.no_grad():
            y = model(data)
            losses = -y.max(1)[0]
            
            orig_losses = -model(orig_data).max(1)[0]
            index = orig_losses < losses
            data[index] = orig_data[index]
            losses[index] = losses[index]
        return data, losses    


def gen_pca_noise(model, device, seed, pca, epsilon, restarts=1, perturb=False, steps=40, alpha=0.01):
    '''
        Runs an adversarial noise attack in Mahalanobis space
        takes a models.MyPCA object to get the orientation of the ellipsoid
        Since simultaneous projection onto a rotated ellipsoid and the [0,1]^D box
        is challenging, I instead rotate the coordinate system back and forth and 
        project once per gradient step.
        This doesn't guarantee a solution in the intersection so in the end I do
        10 more alternating projection steps.
    '''
    model.eval()
    batch_size = seed.shape[0]
    orig_data = seed.clone()
    
    if restarts>1:
        data = seed.clone()
        losses = 1e5*torch.ones(batch_size, device=device)
        for _ in range(restarts):
            current_data, current_losses = gen_pca_noise(model, device, seed, pca, epsilon,
                                                         restarts=1, perturb=True, steps=steps, alpha=alpha)
            with torch.no_grad():
                index = losses > current_losses
                data[index] = current_data[index]

                losses[index] = current_losses[index]
        return data, losses
        
    else:
        with torch.no_grad():
            alpha = alpha * torch.ones(batch_size,1, device=device)

            orig_data_pca = pca.trans(seed.clone())
            prev_data_pca = pca.trans(seed.clone()).to(device)
            data_pca = pca.trans(seed.clone()).requires_grad_()

            if perturb:
                perturbation = epsilon[:,None]*(torch.rand_like(prev_data_pca) - .5)
                prev_data_pca += perturbation
                data_pca += perturbation

            prev_losses = 1e5*torch.ones(batch_size, device=device)
            prev_grad = torch.zeros_like(data_pca, device=device)

        for _ in range(steps):
            with torch.enable_grad():
                y = model(pca.inv_trans(data_pca))
                # losses = y[correct_index].view(batch_size, 9).max(1)[0]
                
                losses = -y.max(1)[0]

                grad = -torch.autograd.grad (losses.sum(), data_pca)[0]

            with torch.no_grad():
                regret_index = losses > prev_losses

                alpha[regret_index] /= 2.
                alpha[1-regret_index] *= 1.1
                data_pca[regret_index] = prev_data_pca[regret_index]
                grad[regret_index] = prev_grad[regret_index]
                
                prev_losses=losses
                prev_data_pca = data_pca
                prev_grad = grad
                
                data_pca += alpha*grad
                
                delta = data_pca - orig_data_pca
                N = delta.norm(dim=-1)
                
                index = N > epsilon

                delta[index] *= (epsilon[index] / N[index])[:, None]
                
                
                data_pca = orig_data_pca + delta
                
                data = pca.inv_trans(data_pca)
                data = torch.clamp(data, 0, 1)
                data_pca = pca.trans(data).requires_grad_()
                
        for _ in range(10):
            with torch.no_grad():
                    delta = data_pca - orig_data_pca
                    N = delta.norm(dim=-1)
                    
                    index = N > epsilon
                    
                    delta[index] *= (epsilon[index] / N[index])[:, None]
                    
                    
                    data_pca = orig_data_pca + delta
                    
                    data = pca.inv_trans(data_pca)
                    data = torch.clamp(data, 0, 1).detach()
                    data_pca = pca.trans(data)

        with torch.no_grad():
            y = model(data)
            losses = -y.max(1)[0]
            
            orig_losses = -model(orig_data).max(1)[0]

            index = orig_losses < losses
            data[index] = orig_data[index]
            losses[index] = losses[index]
        return data, losses
    

def gen_adv_sample(model, device, seed, label, epsilon=0.1, steps=40, step_size=0.001):
    '''
        Runs adversarial attack in l_2 norm
        ot used for the results in https://arxiv.org/abs/1909.12180
    '''
    model.eval()
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


# deprecated functions that help evaluate OOD detection on adversaries 
# in the same pipeline as regular datasets
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
