import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.adversarial as adv
import numpy as np


def train_plain(model, device, train_loader, noise_loader, optimizer, epoch, lam=1., steps=40, epsilon=0.3, verbose=100):
    # noise_loader is useless but this way all training functions have the same format
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        if (batch_idx % verbose == 0) and verbose>0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return train_loss/len(train_loader.dataset), correct/len(train_loader.dataset)
  
    
def train_CEDA(model, device, train_loader, noise_loader, optimizer, epoch, lam=1., steps=40, epsilon=0.3, verbose=100):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    p_in = 1. / (1. + lam)
    p_out = lam * p_in
    
    enum = zip(enumerate(train_loader),enumerate(noise_loader))
    for ((batch_idx, (data, target)), (_, (noise, _))) in enum:
        data, target = data.to(device), target.to(device)
        
        noise = torch.rand_like(data)        

        optimizer.zero_grad()
        output = model(data)
        
        output_adv = model(noise)
        
        loss1 = F.nll_loss(output, target)
        loss2 = - output_adv.mean()
        
        loss = p_in*loss1 + p_out*loss2
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        if (batch_idx % verbose == 0) and verbose>0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return train_loss/len(train_loader.dataset), correct/len(train_loader.dataset)


def train_CEDA_gmm(model, device, train_loader, noise_loader, optimizer, epoch, lam=1., steps=40, epsilon=0.3, verbose=100):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    p_in = 1. / (1. + lam)
    p_out = lam * p_in
    
    log_p_in = torch.tensor(p_in, device=device).log()
    log_p_out = torch.tensor(p_out, device=device).log()
    
    enum = zip(enumerate(train_loader),enumerate(noise_loader))
    for ((batch_idx, (data, target)), (_, (noise, _))) in enum:
        data, target = data.to(device), target.to(device)
        
        noise = torch.rand_like(data)    

        optimizer.zero_grad()
        output = model(data)
        
        output_adv = model(noise)
        like_in = torch.logsumexp( model.mm(data.view(data.shape[0], -1)), 0 )
        like_out =  torch.logsumexp( model.mm(noise.view(noise.shape[0], -1)), 0 )
        
        loss1 = F.nll_loss(output, target)
        loss2 = - output_adv.mean()
        loss3 = - torch.logsumexp(torch.stack([log_p_in + like_in, 
                                               log_p_out*torch.ones_like(like_in)], 0), 0).mean()
        loss4 = - torch.logsumexp(torch.stack([log_p_in + like_out, 
                                               log_p_out*torch.ones_like(like_out)], 0), 0).mean()
        
        loss =  p_in*(loss1 + loss3) + p_out*(loss2 + loss4)

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        if (batch_idx % verbose == 0) and verbose>0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return train_loss/len(train_loader.dataset), correct/len(train_loader.dataset)


def train_ACET(model, device, train_loader, noise_loader, optimizer, epoch, lam=1., steps=40, epsilon=0.3, verbose=100):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    enum = zip(enumerate(train_loader),enumerate(noise_loader))
    for ((batch_idx, (data, target)), (_, (noise, _))) in enum:
        data, target = data.to(device), target.to(device)
        noise = noise.to(device)

        
        output = model(data)

        adv_noise = adv.gen_adv_noise(model, device, noise, epsilon=epsilon, steps=steps)
        optimizer.zero_grad()
        model.train()
        output_adv = model(adv_noise)
        

        loss = criterion(output, target) - output_adv.sum()/(noise_loader.batch_size)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        if (batch_idx % verbose == 0) and verbose>0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return train_loss/len(train_loader.dataset), correct/len(train_loader.dataset)
            

def test(model, device, test_loader, min_conf=.1):
    model.eval()
    test_loss = 0
    correct = 0.
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
    correct /= len(test_loader.dataset)
    
    return correct, av_conf, test_loss


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


training_dict = {'plain': train_plain,
                  'CEDA': train_CEDA,
                  'ACET': train_ACET,
                  'CEDA_GMM' : train_CEDA_gmm
                }