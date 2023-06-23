import argparse
import os
import shutil
import time

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.nn as nn
import math
import resnet
import resnet_type_cifar
from utils import*

#net=resnet.resnet164().cuda()
net = resnet_type_cifar.resnet56(num_classes=100, bit = 4, ffun ='inplt', bfun = 'quant', gd_alpha = True).cuda()
#net = resnet.resnet110().cuda()
if __name__ == '__main__':
    use_cuda = torch.cuda.is_available
    global best_acc
    best_acc = 0
    start_epoch = 0
    best_count = 0
    #--------------------------------------------------------------------------
    # Load Cifar data
    #--------------------------------------------------------------------------
    print('==> Preparing data...')
    root = './data'
    download = True
    
    #Cifar 100
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    #Cifar10
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_set = torchvision.datasets.CIFAR100(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
        ]))
    
    test_set = torchvision.datasets.CIFAR100(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            #normalize,
        ]))
    
    
    kwargs = {'num_workers':1, 'pin_memory':True}
    batchsize_test = 100 #100
    print('Batch size of the test set: ', batchsize_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batchsize_test,
                                              shuffle=False, **kwargs
                                             )
    batchsize_train = 100
    print('Batch size of the train set: ', batchsize_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=batchsize_train,
                                               shuffle=True, **kwargs
                                              )
    
    #net, criterion, optimizer = get_model2(net, learning_rate=0.1, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion =  nn.CrossEntropyLoss().cuda()
    #all_G_kernels=ckpt['G_kernels']
    #m=ckpt['epoch']
    '''
    all_G_kernels = [
        Variable(kernel.data.clone(), requires_grad=True)
        for kernel in optimizer.param_groups[1]['params']
    ]


    all_W_kernels = [kernel for kernel in optimizer.param_groups[1]['params']]
    kernels = [{'params': all_G_kernels}]
    optimizer_quant = optim.SGD(kernels, lr=0)
    eta_rate = 1.05
    eta = 1
    '''
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120,160], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)

    
    nepoch = 200
    for epoch in range(nepoch):
        print('Epoch ID', epoch)
        #----------------------------------------------------------------------
        # Training
        #----------------------------------------------------------------------
        correct = 0; total = 0; train_loss = 0
        net.train()
        for batch_idx, (x, target) in enumerate(train_loader):
          #if batch_idx < 1:
            optimizer.zero_grad()
            x, target = Variable(x.cuda()), Variable(target.cuda())
            
         
                
            #score= net(x)[0]
            score, _, _, _= net(x)
            loss = criterion(score, target)
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.data
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        scheduler.step()   
        #----------------------------------------------------------------------
        # Testing
        #----------------------------------------------------------------------
        test_loss = 0; correct = 0; total = 0
        net.eval()
        
        
            
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
            #score= net(x)[0]
            score, _, _, _= net(x)
            loss = criterion(score, target)
            test_loss += loss.data
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
    
        acc = 100.*correct/total
        #if acc > best_acc:
        if correct > best_count:
            print('Saving model...')
            state = {
                'state': net,#net.state_dict(), 
                'acc': acc,
                'epoch': epoch,
            }
            
            torch.save(state, './models/resnet56_4A.t7')
            #torch.save(state, './models/resnet164_distill_cifar100.t7')
            best_acc = acc
            best_count = correct

        
    
    print('The best acc: ', best_acc)
