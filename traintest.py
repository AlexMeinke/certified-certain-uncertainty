import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_adv(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        
        #noise = generate_adv_noise(model, 0.1, batch_size=train_loader.batch_size)
        noise = generate_adv_noise(model, 0.1, batch_size=10, device=device)
        output_adv = model(noise)
        
        loss = F.nll_loss(output, target) - output_adv.sum()/(10*train_loader.batch_size)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
def train_adv2(model, device, train_loader, optimizer, epoch, adv_loader, verbose=True):
    model.train()
    for ((batch_idx, (data, target)), (_, (data_adv, _))) in zip(enumerate(train_loader),enumerate(adv_loader)):
        data, target = data.to(device), target.to(device)
        data_adv = data_adv.to(device)

        optimizer.zero_grad()
        output = model(data)
        
        #noise = generate_adv_noise(model, 0.1, batch_size=train_loader.batch_size)
        #noise = generate_adv_noise(model, 0.1, batch_size=10)
        output_adv = model(data_adv)
        
        loss = F.nll_loss(output, target) - output_adv.sum()/(10*train_loader.batch_size)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, min_conf=.1):
    model.eval()
    test_loss = 0
    correct = 0
    av_conf = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            c, pred = output.max(1, keepdim=True) # get the index of the max log-probability
            correct += (pred.eq(target.view_as(pred))*(c.exp()>min_conf)).sum().item()
            av_conf += c.exp().sum().item()
            
    test_loss /= len(test_loader.dataset)
    av_conf /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Ave. Confidence: {:.0f}%\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), 100. * av_conf))
    return correct, av_conf

def test_adv(model, device, adv_test_loader, min_conf=.1):
    model.eval()
    av_conf = 0
    predicted = 0
    with torch.no_grad():
        for data, _ in adv_test_loader:
            data = data.to(device)
            output = model(data)

            c, pred = output.max(1, keepdim=True) # get the index of the max log-probability
            av_conf += c.exp().sum().item()
            predicted += (c.exp()>min_conf).float().sum().item()
            
    av_conf /= len(adv_test_loader.dataset)
    predicted /= len(adv_test_loader.dataset)

    print('\nAve. Confidence: {:.0f}% Predicted: {:.0f}%\n'.format(100.*av_conf, 100.*predicted))
    return av_conf

def generate_adv_noise(model, epsilon, device=torch.device('cpu'), batch_size=10, norm=2, num_of_it=10, alpha=0.01, seed_images=None):
    if seed_images is None:
        image = (.22*torch.rand((batch_size,1,28,28)))
        #image = (torch.rand((batch_size,1,28,28)))
    else:
        image = seed_images
    image = image.to(device).requires_grad_()

    perturbed_image = image
    for _ in range(num_of_it):
        y = model(perturbed_image)
        #loss = -y.max(dim=1)[0].sum()
        loss = -y.sum()
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