#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:07:01 2021

@author: pro
"""


import numpy as np
import pandas as pd
import torch
from torchvision.datasets import mnist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from resnet_cifar import*
from utils import *
import copy
import pickle
import argparse


parser = argparse.ArgumentParser(description='FAQD')
parser.add_argument('--num_bit', help='Weight Precision', type=int, default=4, choices = [1, 2, 4]) 
parser.add_argument('--num_epoch', help='number of epoch', type=int, default=200) 
parser.add_argument('--fa_coef', default=0.1, type=float, help='coefficient for FA loss.')
parser.add_argument('--label_coef', default=0.3, type=float, help='coefficient for ground truth.')
parser.add_argument('--fine_tuning', type = str, default = 'False', help = 'fine-tuning a pertained model.')
parser.add_argument('--teacher_root', type=str, default='./quant_activation/models/resnet164_distill_cifar100.t7', help='root of teacher checkpoint.') 
parser.add_argument('--student_root', type=str, default='./resnet110_distill_cifar100.t7', help='root of student checkpoint if fine-tuning.')
parser.add_argument('--data_root', type=str, default='./data', help='root of dataset.')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate.')
parser.add_argument('--KD_loss', default='MSE', type=str, help='type of KD Loss.', choices = ['MSE', 'KL'])
parser.add_argument('--KD_coef', default=1., type=float, help='weight for KD loss')
args = parser.parse_args()

def str2bool(string):
    if string == 'True':
        return True
    else:
        return False

def quantize_bw(kernel):
    delta = kernel.abs().mean()
    sign = kernel.sign().float()

    return sign*delta

def quantize_tnn(kernel):

    data=kernel.abs()
    delta=0.7*data.mean()
    delta=min(delta,100.0)
    index=data.ge(delta).float()
    sign=kernel.sign().float()
    scale=(data*index).mean()
    return scale*index*sign

def quantize_fbit(kernel):

    data=kernel.abs()
    delta=data.max()/15
    delta=min(delta,10.0)
    sign=kernel.sign().float()
    q=0.0*data
    for i in range(3,17,2):
        if i<15:
            index=data.gt((i-2)*delta).float()*data.le(i*delta).float()
        else:
            index=data.gt(13*delta).float()
        q+=(i-1)/2*index
    scale=(data*q).sum()/(q*q).sum()
    return scale*q*sign 

projection_ref = {1:quantize_bw, 2: quantize_tnn, 4:quantize_fbit}

def quantize_8bit(kernel):

    data=kernel.abs()
    delta=data.max()/255
    delta=min(delta,10.0)
    sign=kernel.sign().float()
    q=0.0*data
    for i in range(3,257,2):
        if i<255:
            index=data.gt((i-2)*delta).float()*data.le(i*delta).float()
        else:
            index=data.gt(253*delta).float()
        q+=(i-1)/2*index
    scale=(data*q).sum()/(q*q).sum()
    return scale*q*sign 


class additional_conv110(nn.Module):
    def __init__(self):
        super(additional_conv110, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True)
    def forward(self, x1, x2, x3):
        x1 = self.relu(self.conv1(x1))
        x2 = self.relu(self.conv2(x2))
        x3 = self.relu(self.conv3(x3))
        return x1, x2, x3


def fa_loss(feat, ref_feat):
    batch_size, ch, h, w = feat.size(0), feat.size(1), feat.size(2), feat.size(3)
        # print(feat.size(), ref_feat.size())
    feat = feat.view(batch_size, ch, -1)
    norm_feat = feat.norm(p=2,dim=1).unsqueeze(1)
    feat = torch.div(feat, norm_feat+1e-5)
    tran_feat = feat.permute(0,2,1)
    ft_map = torch.matmul(tran_feat,feat)

    ref_feat = F.interpolate(ref_feat, size=(h,w),mode='bilinear', align_corners=True)
        # print(feat.size(), ref_feat.size(1))
    ref_feat = ref_feat.view(batch_size, ref_feat.size(1), -1)
    norm_ref_feat = ref_feat.norm(p=2,dim=1).unsqueeze(1)
    ref_feat = torch.div(ref_feat, norm_ref_feat+1e-5)
    tran_ref_feat = ref_feat.permute(0,2,1)
    refft_map = torch.matmul(tran_ref_feat,ref_feat)

    loss = (ft_map-refft_map).norm(p=1)/(h*h*w*w)

    return  loss




def get_optimizer(model, learning_rate=1e-3, weight_decay=1e-4, additional = None):

    # set the first layer not trainable
    # model.features.conv0.weight.requires_grad = False

    # all fc layers
    weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and 'conv' not in n
    ]

    # all conv layers
    weights_to_be_quantized = [
        p for n, p in model.named_parameters()
        # if 'conv' in n and 'conv0' not in n
        if 'conv' in n and 'weight' in n
    ]

    biases = [
        p for n, p in model.named_parameters()
        if 'bias' in n
    ]    

    params = [
        {'params': weights, 'weight_decay': weight_decay},
        {'params': weights_to_be_quantized, 'weight_decay': weight_decay},
        {'params': biases,  'weight_decay': weight_decay}
    ]
    if additional is not None:
        additional_weights = [p for p in additional.parameters()]
        params += [{'params' : additional_weights, 'weight_decay': weight_decay}]
    optimizer = optim.SGD(params, lr= learning_rate, momentum = 0.9, nesterov = True)
    return optimizer


####### Load Cifar10 Data ##########
####################################
root = args.data_root
w_bit = args.num_bit
projection = projection_ref[w_bit]
lr_initial = args.lr
label_coef = args.label_coef
fa_coef = args.fa_coef
fine_tuning = str2bool(args.fine_tuning)
teacher_root = args.teacher_root
student_root = args.student_root
loss_type = args.KD_loss
KD_coef = args.KD_coef

download = True   
normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    
train_set = torchvision.datasets.CIFAR100(
        root=root,
        train=True,
        download=download,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
test_set = torchvision.datasets.CIFAR100(
        root=root,
        train=False,
        download=download,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    
kwargs = {'num_workers':1, 'pin_memory':True}
batchsize_test = 100
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



checkpoint = torch.load(teacher_root)
model = checkpoint['state']

additional = additional_conv110().cuda()

print('Model Accuracy: ', checkpoint['acc'])
del checkpoint

if fine_tuning:
    ckpt = torch.load(student_root)
    model_quant = ckpt['state']
    del ckpt
else:
    model_quant = resnet110(num_classes = 100).cuda()

optimizer = get_optimizer(model_quant, learning_rate=lr_initial, additional = additional)

all_G_kernels = [
        Variable(kernel.data.clone(), requires_grad=True)
        for kernel in optimizer.param_groups[1]['params']]
all_W_kernels = [kernel for kernel in optimizer.param_groups[1]['params']]
kernels = [{'params': all_G_kernels}]
optimizer_quant = optim.SGD(kernels, lr=0)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
if loss_type == 'MSE':
    criterion = nn.MSELoss().cuda()
elif loss_type == 'KL':
    criterion = nn.KLDivLoss(size_average=False).cuda()
criterion_nnl = nn.NLLLoss().cuda()
criterion_ce = nn.CrossEntropyLoss().cuda()

test_loss, total, correct = 0.0, 0.0, 0.0
correct1 = 0.0
best_acc = 0.0
with torch.no_grad():
    model.eval()
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
        score, _, _, _ = model(x)   
        score_t, _, _, _ = model_quant(x) 
        _, predicted = torch.max(score.data, 1)
        _, predicted1 = torch.max(score_t.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        correct1 += predicted1.eq(target.data).cpu().sum()
print('Re-tested model : ', 100.*correct.float()/total)
print('Re-tested model_quant : ', 100.*correct1.float()/total)
#epochs = 70
epochs =100
loss_kl_epoch = []
for epoch in range(epochs):
    print('Epoch: ', epoch)
    total, correct, test_loss = 0.0, 0.0, 0.0
    loss_kl_total = 0.0
    model_quant.train()
    #for x, target in Sampleloader:
    for batch_idx, (x, target) in enumerate(train_loader):
        x, target = Variable(x.cuda()), Variable(target.cuda())
        all_W_kernels = optimizer.param_groups[1]['params']
        all_G_kernels = optimizer_quant.param_groups[0]['params']

        for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_G = all_G_kernels[i]
            V = k_W.data
            k_G.data = projection(V)
            
            k_W.data, k_G.data = k_G.data, k_W.data

        with torch.no_grad():
            score, ref_feat1, ref_feat2, ref_feat3 = model(x)  # float trained model
        score_t, feat1, feat2, feat3 = model_quant(x)  # quantized model
        feat1, feat2, feat3 = additional(feat1, feat2, feat3)
        
    


        loss_KD = criterion(score_t, score)
        loss_nnl = criterion_nnl(F.log_softmax(score_t, dim=1), target)
        loss_f1, loss_f2, loss_f3 = fa_loss(feat1, ref_feat1), fa_loss(feat2, ref_feat2), fa_loss(feat3, ref_feat3)
        loss = KD_coef*loss_KD + fa_coef*loss_f1 + fa_coef*loss_f2 + fa_coef*loss_f3 + label_coef*loss_nnl

        loss.backward()
        total += target.size(0)
        
        
        
        for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_G = all_G_kernels[i]
            k_W.data, k_G.data = k_G.data, k_W.data

        torch.nn.utils.clip_grad_norm_(model_quant.parameters(), 20)
        optimizer.step()
        optimizer.zero_grad()
    
    scheduler.step()              
    total, correct, test_loss = 0.0, 0.0, 0.0
    
    for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_quant = all_G_kernels[i]    
            k_W.data, k_quant.data = k_quant.data, k_W.data
    
    
    with torch.no_grad():
    
        model_quant.eval()
        for x, target in test_loader:
            x, target = Variable(x.cuda()), Variable(target.cuda())
            #score,_ = model_quant(x)
            score,_,_,_ = model_quant(x)
            loss = criterion_ce(score, target)
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).sum()
            test_loss += loss
        #model_quant = model_quant.cuda()
        #model = model.cuda()
        acc = correct.float()/total
        print('test accuracy: ', acc)
        print('test loss: ', test_loss/total)
        state = {
                'net': model_quant, #net,
                'acc': acc,
                'epoch': epoch,
                'G_kernels':all_G_kernels,
            }
        if acc > best_acc:    
            print('svaing........')
            #torch.save(state, './models_distill/TNN/ckpt_quant_110_cifar100.t7')
            best_acc = acc


    #model_quant = model_quant.cuda()
    for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_quant = all_G_kernels[i]    
            k_W.data, k_quant.data = k_quant.data, k_W.data      
#with open('./models_distill/4bit/ResNet110_KL.txt', 'wb') as f:
#    pickle.dump(loss_kl_epoch, f)
        #    pickle.dump(ch_sp_epoch, f)
    #print('test accuracy: ', correct.float()/total)
    #print('test loss: ', test_loss/total)
    
