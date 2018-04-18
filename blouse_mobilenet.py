import torch
import torch.nn as nn
import numpy as np
import torchvision
import visdom
import time
import argparse
import torch.backends.cudnn as cudnn
#cudnn.benchmark = True

from MobileNetV2  import MobileNetV2
from dataset import dataload

class RingLossFunc(nn.Module):
    def __init__(self, lamda, param_R):
        super(RingLossFunc,self).__init__()
        self.lamda =  torch.autograd.Variable(torch.FloatTensor([lamda])) #.cuda()
        self.param_R = torch.autograd.Variable(torch.FloatTensor([param_R]),requires_grad=True) #.cuda()
        return 
        
    def forward(self, feature):
        batchsize = feature.size(0)
        loss = self.lamda/(2.0 *batchsize) * ((torch.norm(feature,2,1) - self.param_R).pow(2)).sum()
        #grad = -1.0*self.lamda/batchsize *(torch.norm(feature,2,1) - self.param_R).sum()
        #self.param_R.data += grad.data
        #print 'ccc:',grad
        return loss

#save models
def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)


#parameters define:
parser = argparse.ArgumentParser(description='pytorch cifar10-resnet18')
parser.add_argument('--lr',default=0.00001, type = float,help = 'learning rate')
parser.add_argument('--batchsize',default = 20, type = int, help = 'batch aize')
parser.add_argument('--max_iter',default = 25, type=int,help='max iteration of training set')
#parser.add_argument('--gpu_status',default = True, type=bool, help='whether to use gpu')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()

#model define
#resnet18 = torchvision.models.resnet18(pretrained=True)
#num_features = resnet18.fc.in_features
#resnet18.fc = torch.nn.Linear(num_features,10)
resnet18 = MobileNetV2(classes_num=26, feature_num=1024,input_size = 512)
#resnet18.load_state_dict(torch.load('checkpoints/gg-blouse-mobilenetv219.pth'))
if use_cuda:
    resnet18 = resnet18.cuda()
#print resnet18
print resnet18

#transform
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_transform = torchvision.transforms.Compose([
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

############visualize some data#############################################################
#classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#dataiter = iter(test_loader)
#image,label = dataiter.next()
#print image.size(),label
#image = viz.images(image[:10]*255, nrow = 10, padding=3,env='cifar10')
#text  = viz.text('||'.join('%6s' % classes[label[j]] for j in range(10)),env='cifar10')
############################################################################################

#loss function & optimizer
loss_fun = torch.nn.MSELoss()  #torch.nn.CrossEntropyLoss()

#ringloss_fun = RingLossFunc(lamda=0.001, param_R=1)

optimizer = torch.optim.SGD(resnet18.parameters(),lr = args.lr ,momentum = 0.95)
#lr scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 5, gamma=0.1)


#how long to print info
tr_num = len(train_dataset)/args.batchsize/5        #set parameter here
#start time
start_time = time.time()
#recored some states
tr_loss,step=[],[]
lr_steps = [0,5,15]
####################### train ##############################################################
for epoch in range(args.max_iter):
    running_loss, running_accu = 0.0,0.0
    if epoch in lr_steps:
        if epoch!=0: args.lr *= 0.1
        optimizer = torch.optim.SGD(resnet18.parameters(),lr = args.lr ,momentum = 0.90, weight_decay=5e-4)
    
    if epoch % 2 == 0:
         save_model(resnet18,'checkpoints/1e-5-blouse-mobilenetv2{}.pth'.format(epoch))
    print 'the current epoch is:', epoch,'.................................'
    for i, (img,label) in enumerate(train_loader):
        if use_cuda:
            img,label = img.cuda(),label.cuda()
        img,label = torch.autograd.Variable(img),torch.autograd.Variable(label.float())
        out = resnet18(img)
        #r_loss = ringloss_fun(feat)
        loss= loss_fun(out*(label.ge(0).float()),label*(label.ge(0).float()))
        #loss = torch.sqrt((out*(label.ge(0).float())-label*(label.ge(0).float())).pow(2)).sum()/20.0
        #loss = r_loss + loss_ce
        running_loss += loss.data[0]*len(label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #if (i-1)  % 5 == 0: #tr_num == 0:
        print('Train:[{}/{}] | loss: {:.4f}'.format(epoch+1,args.max_iter, running_loss/((i+1)*args.batchsize)))

        tr_loss.append(running_loss/((i+1)*args.batchsize))
        
    
save_model(resnet18,'checkpoints/1e-5-blouse-mobilenetv2{}.pth'.format(args.max_iter-1))
print('finish......')

