import torch
import torch.nn.functional as F

def generate_adv_noise(model, epsilon, device=torch.device('cpu'), batch_size=10, norm=2, num_of_it=40, alpha=0.01, seed_images=None):
    if seed_images is None:
        image = (.22*torch.rand((batch_size,1,28,28)))
        #image = (torch.rand((batch_size,1,28,28)))
    else:
        image = seed_images
    image = image.to(device).requires_grad_()
    perturbed_image = image
    for _ in range(num_of_it):
        with torch.enable_grad():
            y = model(perturbed_image)
            loss = -y.sum()

        loss.backward()

        with torch.no_grad():
            perturbed_image += alpha*image.grad

            delta = perturbed_image-image
            delta /= delta.view((batch_size,784)).norm(p=norm, dim=1)[:,None,None,None]
            delta *= epsilon
            perturbed_image = image + delta
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

            perturbed_image = perturbed_image.detach()
    return perturbed_image

def generate_adv_sample(model, epsilon, seed_images, seed_labels,
                        device=torch.device('cpu'), 
                        batch_size=10, norm=2, 
                        num_of_it=40, alpha=0.01):
    
    image = seed_images.to(device).requires_grad_()
    perturbed_image = image
    for _ in range(num_of_it):
        with torch.enable_grad():
            y = model(perturbed_image)
            loss = F.nll_loss(y, seed_labels)
        
        loss.backward()

        with torch.no_grad():
            perturbed_image += alpha*image.grad

            delta = perturbed_image-image
            #delta /= delta.view((batch_size,784)).norm(p=norm, dim=1)[:,None,None,None]
            #delta *= epsilon
            perturbed_image = image + delta
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

            perturbed_image = perturbed_image.detach()
    return perturbed_image