import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.adversarial as adv
import numpy as np


def train_plain(model, device, train_loader, optimizer, epoch, 
                lam=1., verbose=100, noise_loader=None, epsilon=.3):
    # lam not necessarily needed but there to ensure that the 
    # learning rates on the base and the CEDA model are comparable
    
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    p_in = 1. / (1. + lam)
    

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        
        loss = p_in*criterion(output, target)
        
        optimizer.zero_grad()
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
  
    
def train_CEDA(model, device, train_loader, optimizer, epoch, 
               lam=1., verbose=100, noise_loader=None, epsilon=.3):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    p_in = 1. / (1. + lam)
    p_out = lam * p_in
    
    if noise_loader is not None:
        enum = enumerate(noise_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        noise = torch.rand_like(data)
        
        if noise_loader is not None:
            noise = enum.__next__()[1][0].to(device)

        full_data = torch.cat([data, noise], 0)
        full_out = model(full_data)
        
        output = full_out[:data.shape[0]]
        output_adv = full_out[data.shape[0]+1:]
        
        loss1 = criterion(output, target)
        loss2 = - output_adv.mean()

        loss = p_in*loss1 + p_out*loss2

        optimizer.zero_grad()
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


def train_CEDA_gmm(model, device, train_loader, optimizer, epoch, 
                   lam=1., verbose=100, noise_loader=None, epsilon=.3):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    p_in = 1. / (1. + lam)
    p_out = lam * p_in
    
    log_p_in = torch.tensor(p_in, device=device).log()
    log_p_out = torch.tensor(p_out, device=device).log()
    
    if noise_loader is not None:
        enum2 = enumerate(noise_loader)
    
    enum = enumerate(train_loader)
    for batch_idx, (data, target) in enum:
        data, target = data.to(device), target.to(device)
        
        noise = torch.rand_like(data)
        
        if noise_loader is not None:
            noise = enum2.__next__()[1][0].to(device)

        optimizer.zero_grad()
        
        full_data = torch.cat([data, noise], 0)
        full_out = model(full_data)
        
        output = full_out[:data.shape[0]]
        output_adv = full_out[data.shape[0]+1:]
        
        like_in = torch.logsumexp( model.mm(data.view(data.shape[0], -1)), 0 )
        like_out =  torch.logsumexp( model.mm(noise.view(noise.shape[0], -1)), 0 )
        
        loss1 = criterion(output, target)
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


def train_ACET_gmm(model, device, train_loader, optimizer, epoch, 
                   lam=1., verbose=100, noise_loader=None, epsilon=.3):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    p_in = 1. / (1. + lam)
    p_out = lam * p_in
    
    log_p_in = torch.tensor(p_in, device=device).log()
    log_p_out = torch.tensor(p_out, device=device).log()
    
    if noise_loader is not None:
        enum2 = enumerate(noise_loader)
    
    enum = enumerate(train_loader)
    for batch_idx, (data, target) in enum:
        data, target = data.to(device), target.to(device)
        
        noise = torch.rand_like(data)
        
        if noise_loader is not None:
            noise = enum2.__next__()[1][0].to(device)
            
        noise, _ = adv.gen_adv_noise(model, device, noise, epsilon=epsilon, steps=40, step_size=0.01)
        model.train()

        optimizer.zero_grad()
        
        full_data = torch.cat([data, noise], 0)
        full_out = model(full_data)
        
        output = full_out[:data.shape[0]]
        output_adv = full_out[data.shape[0]+1:]
        
        like_in = torch.logsumexp( model.mm(data.view(data.shape[0], -1)), 0 )
        like_out =  torch.logsumexp( model.mm(noise.view(noise.shape[0], -1)), 0 )
        
        loss1 = criterion(output, target)
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
   

def train_ACET(model, device, train_loader, optimizer, epoch, 
               lam=1., verbose=-1, noise_loader=None, epsilon=.3):
    criterion = nn.NLLLoss()
    model.train()
    
    train_loss = 0
    correct = 0
    
    p_in = 1. / (1. + lam)
    p_out = lam * p_in
    
    if noise_loader is not None:
        enum2 = enumerate(noise_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        noise = torch.rand_like(data)
        if noise_loader is not None:
            noise = enum2.__next__()[1][0].to(device)
            
        noise, _ = adv.gen_adv_noise(model, device, noise, epsilon=epsilon, steps=40, step_size=0.01)
        model.train()
        
        full_data = torch.cat([data, noise], 0)
        full_out = model(full_data)
        
        output = full_out[:data.shape[0]]
        output_adv = full_out[data.shape[0]+1:]
        
        
        loss1 = criterion(output, target)
        loss2 = - output_adv.mean()
        
        #print(str(loss1) + '   ' + str(loss2) + '\n')
        loss = p_in*loss1 + p_out*loss2

        optimizer.zero_grad()
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


training_dict = {'plain': train_plain,
                  'CEDA': train_CEDA,
                  'CEDA_GMM' : train_CEDA_gmm,
                  'ACET_GMM' : train_ACET_gmm,
                  'ACET' : train_ACET
                }