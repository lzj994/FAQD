import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
from resnet_tiny import*
from utils import *
import os
import argparse
import pickle

parser = argparse.ArgumentParser(description='FAQD')
parser.add_argument('--num_bit', help='Weight Precision', type=int, default=4, choices = [1, 2, 4]) 
parser.add_argument('--num_epoch', help='number of epoch', type=int, default=200) 
parser.add_argument('--fa_coef', default=0.1, type=float, help='coefficient for FA loss.')
parser.add_argument('--label_coef', default=0.3, type=float, help='coefficient for ground truth.')
parser.add_argument('--KD_loss', default= 'MSE', type=str, help='type of Knowledge distillation loss.', choices = ['MSE', 'KL'])
parser.add_argument('--KD_coef', default=1., type=float, help='weight of KD loss.')
parser.add_argument('--fine_tuning', type = str, default = 'False', help = 'fine-tuning a pertained model.')
parser.add_argument('--teacher_root', type=str, default='./models/resnet34_tiny.t7', help='directory of teacher checkpoint.') 
parser.add_argument('--student_root', type=str, default='./models/resnet18_pretrain.t7', help='directory of student checkpoint if fine-tuning.')
parser.add_argument('--data_root', type=str, default='./data', help='directory of dataset.')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate.')
parser.add_argument('--FA_type', default= 'FA', type=str, help='type of Knowledge distillation loss.', choices = ['FA', 'FFA'])
parser.add_argument('--num_ensemble', default=15, type=int, help='Number of ensemble for fast fa loss.')
args = parser.parse_args()

def str2bool(string):
    return True if string == 'True' else False

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

proj_ref = {1:quantize_bw, 2: quantize_tnn, 4:quantize_fbit}

class additional_conv20(nn.Module):
    def __init__(self):
        super(additional_conv20, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=True)
    def forward(self, x1, x2, x3, x4):
        x1 = self.relu(self.conv1(x1))
        x2 = self.relu(self.conv2(x2))
        x3 = self.relu(self.conv3(x3))
        x4 = self.relu(self.conv4(x4))
        return x1, x2, x3, x4

def fa_loss(feat, ref_feat, order = 1):
    batch_size, ch, h, w = feat.size(0), feat.size(1), feat.size(2), feat.size(3)
    feat = feat.view(batch_size, ch, -1)
    norm_feat = feat.norm(p=2,dim=1).unsqueeze(1)
    feat = torch.div(feat, norm_feat+1e-5)
    tran_feat = feat.permute(0,2,1)

    ft_map = torch.matmul(tran_feat,feat)

    ref_feat = F.interpolate(ref_feat, size=(h,w),mode='bilinear', align_corners=True)
    ref_feat = ref_feat.view(batch_size, ref_feat.size(1), -1)
    norm_ref_feat = ref_feat.norm(p=2,dim=1).unsqueeze(1)
    ref_feat = torch.div(ref_feat, norm_ref_feat+1e-5)
    tran_ref_feat = ref_feat.permute(0,2,1)
    refft_map = torch.matmul(tran_ref_feat,ref_feat)
    
    if order == 2:
        loss = (ft_map-refft_map).norm(p=2)/(h*w)
    elif order == 1:
        loss = (ft_map - refft_map).norm(p=1)/(h*h*w*w)

    return  loss

def fast_fa_loss(feat, ref_feat, dim = 15, order = 1):
    torch.manual_seed(1)
    batch_size, ch, h, w = feat.size(0), feat.size(1), feat.size(2), feat.size(3)

    # generating random vector  (HW) x dim
    vec = torch.randn(h*w, dim).detach().unsqueeze(0).repeat(batch_size,1,1).cuda()
        
    feat = feat.view(batch_size, ch, -1)  # [batch, ch, HW]
    norm_feat = feat.norm(p=2,dim=1).unsqueeze(1)
    feat = torch.div(feat, norm_feat)
    tran_feat = feat.permute(0,2,1)    # [batch, HW, ch]

    ft_map = torch.matmul(tran_feat, torch.matmul(feat, vec) )
        

    ref_feat = F.interpolate(ref_feat, size=(h,w),mode='bilinear', align_corners=True)
    ref_feat = ref_feat.view(batch_size, ref_feat.size(1), -1)
    norm_ref_feat = ref_feat.norm(p=2,dim=1).unsqueeze(1)
    ref_feat = torch.div(ref_feat, norm_ref_feat)
    tran_ref_feat = ref_feat.permute(0,2,1)

    refft_map = torch.matmul(tran_ref_feat, torch.matmul(ref_feat, vec))

    if order == 1:
        loss = (ft_map-refft_map).abs().sum(1).mean(-1).sum()/(h*h*w*w)
    elif order == 2:
        loss = torch.sqrt((ft_map-refft_map).pow(2).sum(1).mean(-1).sum())/(h*w)
        
    return  loss


def get_optimizer(model, learning_rate=1e-3, weight_decay=1e-4, additional = None):

    # set the first layer not trainable
    # model.features.conv0.weight.requires_grad = False

    
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
        {'params': weights_to_be_quantized, 'weight_decay': weight_decay},
        {'params': weights, 'weight_decay': weight_decay},
        {'params': biases,  'weight_decay': weight_decay}
    ]

    if additional is not None:
        additional_weights = [p for p in additional.parameters()]

    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, nesterov= True)
    
    return optimizer

####### Load Cifar10 Data ##########
####################################
root = args.data_root
w_bit =args.num_bit
lr_initial = args.lr
label_coef = args.label_coef
KD_coef = args.KD_coef
fa_coef = args.fa_coef
fine_tuning = str2bool(args.fine_tuning)
teacher_root = args.teacher_root
student_root = args.student_root
loss_type = args.KD_loss
epochs = args.num_epoch
FA_type = args.FA_type
num_ensemble = args.num_ensemble

root = './tiny-imagenet-200'

#if args.distributed:
#        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#    else:
train_sampler = None

traindir = os.path.join(root, 'train')
valdir = os.path.join(root, 'val')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=(train_sampler is None),
    num_workers=1, pin_memory=True, sampler=train_sampler)


val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
     batch_size=128, shuffle=False,
    num_workers=1, pin_memory=True)




checkpoint = torch.load(teacher_root)
model = checkpoint['state']

additional = additional_conv20().cuda()

print('Model Accuracy: ', checkpoint['acc'])
del checkpoint

if fine_tuning:
    ckpt = torch.load(student_root)
    model_quant = ckpt['state']
    print('Quant Model Accuracy: ', ckpt['acc'])
    del ckpt
else:
    model_quant = resnet18().cuda()


optimizer = get_optimizer(model_quant, learning_rate= lr_initial, additional = additional)
projection = proj_ref[w_bit]

all_G_kernels = [
        Variable(kernel.data.clone(), requires_grad=True)
        for kernel in optimizer.param_groups[0]['params']]
all_W_kernels = [kernel for kernel in optimizer.param_groups[0]['params']]
kernels = [{'params': all_G_kernels}]
optimizer_quant = optim.SGD(kernels, lr=0)

#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70, 90], gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

temperature = 3
criterion = nn.MSELoss().cuda()
criterion_ce = nn.CrossEntropyLoss().cuda()
criterion_kl = nn.KLDivLoss(size_average=False).cuda()
criterion_nnl = nn.NLLLoss().cuda()
if loss_type == 'MSE':
    criterion = nn.MSELoss().cuda()
elif loss_type == 'KL':
    criterion = lambda score_t,score: criterion_kl(F.log_softmax(score_t/temperature, dim=1), F.softmax(score/temperature, dim=1))*temperature**2
if FA_type == 'FA':
    FA_loss = lambda feat, ref_feat: fa_loss(feat, ref_feat)
elif FA_type == 'FFA':
    FA_loss = lambda feat, ref_feat: fast_fa_loss(feat, ref_feat, dim = num_ensemble)

test_loss, total = 0.0, 0.0
correct_t, correct = 0.0, 0.0
best_acc = 0.0
eta = 1.
eta_rate = 1.05
with torch.no_grad():
    model.eval()
    for batch_idx, (x, target) in enumerate(val_loader):
        
        x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
        with torch.no_grad():
            score = model(x)[0]
            score_t = model_quant(x)[0]
        loss = criterion_ce(score, target)
        test_loss += loss.data
        _, predicted = torch.max(score.data, 1)
        _, predicted_t = torch.max(score_t.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        correct_t += predicted_t.eq(target.data).cpu().sum()
print('Re-tested model: ', 100.*correct.float()/total)  
print('Re-tested model_quant: ', 100.*correct_t.float()/total)

for epoch in range(epochs):
    print('Epoch: ', epoch)
    total, correct, train_loss = 0.0, 0.0, 0.0
    model_quant.train()
    loss_mse_total = 0.0
    for batch_idx, (x, target) in enumerate(train_loader):
        x, target = Variable(x.cuda()), Variable(target.cuda())
        all_W_kernels = optimizer.param_groups[0]['params']
        all_G_kernels = optimizer_quant.param_groups[0]['params']


        for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_G = all_G_kernels[i]
            V = k_W.data

            k_G.data = projection(V)


            k_W.data, k_G.data = k_G.data, k_W.data

        with torch.no_grad():
            score, ref_feat1, ref_feat2, ref_feat3, ref_feat4 = model(x)  # float trained model
        score_t, feat1, feat2, feat3, feat4 = model_quant(x)  # quantized model
        feat1, feat2, feat3, feat4 = additional(feat1, feat2, feat3, feat4)
        
        loss_KD = criterion(score_t, score)
        loss_nnl = criterion_nnl(F.log_softmax(score_t, dim=1), target)
        loss_f1, loss_f2, loss_f3, loss_f4 = FA_loss(feat1, ref_feat1), FA_loss(feat2, ref_feat2), FA_loss(feat3, ref_feat3), FA_loss(feat4, ref_feat4)
        loss = KD_coef*loss_KD + fa_coef*loss_f1 + fa_coef*loss_f2 + fa_coef*loss_f3 + fa_coef*loss_f4 + label_coef*loss_nnl
        loss.backward()
        
        total += target.size(0)
        
        
        for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_G = all_G_kernels[i]
            k_W.data, k_G.data = k_G.data, k_W.data
        
        optimizer.step()
        optimizer.zero_grad()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    scheduler.step()            
    total, correct, test_loss = 0.0, 0.0, 0.0
    
    for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_quant = all_G_kernels[i]    
            k_W.data, k_quant.data = k_quant.data, k_W.data
     

    with torch.no_grad():
        model_quant.eval()
        for x, target in val_loader:
            x, target = Variable(x.cuda()), Variable(target.cuda())
            score = model_quant(x)[0]
            loss = criterion_ce(score, target)
            _, predicted = torch.max(score.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).sum()
            test_loss += loss
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
            #torch.save(state, './models_distill/T/ckpt_quant_56.t7')
            #torch.save(state, './models_distill/ckpt_quant_20.t7')
            best_acc = acc

    for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_quant = all_G_kernels[i]    
            k_W.data, k_quant.data = k_quant.data, k_W.data