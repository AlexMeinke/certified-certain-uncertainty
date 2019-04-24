import torch
import torch.nn.functional as F
import utils.adversarial as adv

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
            
def train_CEDA(model, device, train_loader, noise_loader, optimizer, epoch, verbose=True):
    model.train()
    for ((batch_idx, (data, target)), (_, (noise, _))) in zip(enumerate(train_loader),enumerate(noise_loader)):
        data, target = data.to(device), target.to(device)
        noise = noise.to(device)

        optimizer.zero_grad()
        output = model(data)
        
        output_adv = model(noise)
        
        loss = F.nll_loss(output, target) - output_adv.sum()/(10*train_loader.batch_size)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        return loss
            

def train_ACET(model, device, train_loader, noise_loader, optimizer, epoch, steps=40, epsilon=0.3, verbose=True):
    model.train()
    for ((batch_idx, (data, target)), (_, (noise, _))) in zip(enumerate(train_loader),enumerate(noise_loader)):
        data, target = data.to(device), target.to(device)
        noise = noise.to(device)

        optimizer.zero_grad()
        output = model(data)

        adv_noise = adv.gen_adv_noise(model, device, noise, epsilon=epsilon, steps=steps)
        output_adv = model(adv_noise)
        
        loss = F.nll_loss(output, target) - output_adv.sum()/(10*train_loader.batch_size)
        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss
            
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

