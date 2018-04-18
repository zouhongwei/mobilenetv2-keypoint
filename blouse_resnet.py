import torch
import torch.nn as nn
import numpy as np
import torchvision
import visdom
import time
import argparse
import torch.backends.cudnn as cudnn
#cudnn.benchmark = True

from ResNet  import ResNet
from dataset import dataload

#save models
def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


#parameters define:
parser = argparse.ArgumentParser(description='pytorch cifar10-resnet18')
parser.add_argument('--lr',default=0.002, type = float,help = 'learning rate')
parser.add_argument('--batchsize',default = 64, type = int, help = 'batch aize')
parser.add_argument('--max_iter',default = 40, type=int,help='max iteration of training set')
#parser.add_argument('--gpu_status',default = True, type=bool, help='whether to use gpu')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()

#model define
resnet18 = ResNet(num_classes = 26, depth=50, pre_trained=True)
#resnet18.load_state_dict(torch.load('checkpoints/blouse/a-0.02-16-resnet50.pth'))
if use_cuda:
    resnet18 = resnet18.cuda()
#print resnet18
print resnet18

#transform
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = torchvision.transforms.Compose([
       torchvision.transforms.Resize([224,224]),
       torchvision.transforms.ToTensor(),
       normalize
       ])

test_transform = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor(),
       normalize
       ])

#load data
train_dataset=dataload(datalist='train_blouse.txt',transform =train_transform)


#see some image information
im,target = train_dataset[3]
print 'train dataset length:',len(train_dataset)
print 'image size',im.size()
print 'label:',target

#load data
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = args.batchsize,  #set parameters here
                                           shuffle = True)

#loss function & optimizer
loss_fun = torch.nn.MSELoss()  #torch.nn.CrossEntropyLoss()

#ringloss_fun = RingLossFunc(lamda=0.001, param_R=1)

optimizer = torch.optim.SGD(resnet18.parameters(),lr = args.lr ,momentum = 0.95)

#start time
start_time = time.time()
#recored some states
tr_loss,step=[],[]

lr_steps = [0,10,20,30]
####################### train ##############################################################
for epoch in range(args.max_iter):
    if epoch in lr_steps:
        if epoch!=0: args.lr *= 0.1
        optimizer = torch.optim.SGD(resnet18.parameters(),lr = args.lr ,momentum = 0.95, weight_decay=5e-4)
    running_loss, running_accu = 0.0,0.0
    
    if epoch % 2 == 0:
         save_model(resnet18,'checkpoints/224-0.02-16-resnet50-{}.pth'.format(epoch))
    print 'the current epoch is:', epoch,'.................................'
    for i, (img,label) in enumerate(train_loader):
        if use_cuda:
            img,label = img.cuda(),label.cuda()
        img,label = torch.autograd.Variable(img),torch.autograd.Variable(label.float())*224.0/512.0
        out = resnet18(img)
        #r_loss = ringloss_fun(feat)
        loss= loss_fun(out*(label.ge(0).float()),label) #loss= loss_fun(out,label)
        #loss = r_loss + loss_ce
        running_loss += loss.data[0]*len(label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print resnet18.classifier.weight[1][1]
        
        if (i-1)  % 2 == 0: #tr_num == 0:
            print('Train:[{}/{}] | loss: {:.4f}'.format(epoch+1,args.max_iter, running_loss/((i+1)*args.batchsize)))
      
        tr_loss.append(running_loss/((i+1)*args.batchsize))
        
    
save_model(resnet18,'checkpoints/224-0.02-16-resnet50-{}.pth'.format(args.max_iter-1))
print('finish......')

