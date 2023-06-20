import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os 
import resnet_tiny
import resnet_type_tiny
from utils import*

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

#model = resnet_tiny.resnet18().cuda()
model = resnet_type_tiny.resnet18_quant(bit = 1, ffun ='inplt', bfun = 'quant').cuda()
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
criterion =  nn.CrossEntropyLoss().cuda()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200)

nepoch = 200
best_acc = 0.
for epoch in range(nepoch):
    print('Epoch ID', epoch)
    #----------------------------------------------------------------------
    # Training
    #----------------------------------------------------------------------
    correct = 0; total = 0; train_loss = 0
    model.train()
    for batch_idx, (x, target) in enumerate(train_loader):
          #if batch_idx < 1:
        optimizer.zero_grad()
        x, target = Variable(x.cuda()), Variable(target.cuda())
            
        score, _, _, _,_ = model(x)
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
    model.eval()
        
        
            
    for batch_idx, (x, target) in enumerate(val_loader):
        x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
        score, _, _,_,_ = model(x)
        loss = criterion(score, target)
        test_loss += loss.data
        _, predicted = torch.max(score.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
        progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        #----------------------------------------------------------------------
        # Save the checkpoint
        #----------------------------------------------------------------------
    
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving model...')
        state = {
            'state': model.state_dict(), 
            'acc': acc,
            'epoch': epoch,
        }
            
        #torch.save(state, './models/enresnet20.t7')
        torch.save(state, './models/resnet18_pretrain_full_1A.t7')
        best_acc = acc
        