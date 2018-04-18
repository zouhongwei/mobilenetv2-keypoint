import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, expand_ratio, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_residual = (self.stride == 1)and(inp == oup)
        
        self.convblock = nn.Sequential(
                #pointwise
                nn.Conv2d(inp, inp*expand_ratio, 1, stride = 1, padding = 0, bias=False),
                nn.BatchNorm2d(inp*expand_ratio),
                nn.ReLU6(inplace = True),
                #depth-wise
                nn.Conv2d(inp*expand_ratio, inp*expand_ratio, 3, stride = self.stride, padding=1, groups = inp*expand_ratio, bias = False),
                nn.BatchNorm2d(inp*expand_ratio),
                nn.ReLU6(inplace = True),
                #point-wise linear
                nn.Conv2d(inp*expand_ratio, oup, 1, stride = 1, padding = 0, bias=False),
                nn.BatchNorm2d(oup),
                )
        
    def forward(self, x):
        if self.use_residual:
            return x + self.convblock(x)
        else:
            return self.convblock(x)
            
class MobileNetV2(nn.Module):
    def __init__(self, classes_num, feature_num,input_size=224):
        super(MobileNetV2,self).__init__()
        self.classes_num = classes_num
        self.feature_num = feature_num
        #setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.feature = [nn.Sequential(nn.Conv2d(3,32,3, stride=2, padding=1, bias=False),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU6(inplace = True))]
        
        input_channel = 32
        for t,c,n,s in self.interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.feature.append(InvertedResidual(input_channel, output_channel, t,  s))
                else:
                    self.feature.append(InvertedResidual(input_channel, output_channel, t,  1))
                input_channel = output_channel
        
        #build last several layers
        self.feature.append(nn.Sequential(nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
                                          nn.BatchNorm2d(1280),
                                          nn.ReLU6(inplace = True)))  
        self.feature.append(nn.Sequential(nn.AvgPool2d(input_size/32)))
        self.feature.append(nn.Sequential(nn.Conv2d(1280, self.feature_num, 1, 1, 0, bias=False)))
        
        self.faeture = nn.Sequential(*self.feature)     
        
        self.classfier = nn.Sequential(nn.Dropout(),
                                       nn.Linear(self.feature_num, self.classes_num))
        
    def forward(self,x):
        x = self.faeture(x)
        x = x.view(-1, self.feature_num)
        feat = x
        x = self.classfier(x)
        return x  #,feat
        
#mobilenet = MobileNetV2(10,1280)
#mobilenet.cuda()
#mobilenet.eval()
#print mobilenet
#x = torch.autograd.Variable(torch.randn(1,3,224,224)).cuda()
#print mobilenet(x).size()

#import time
#start = time.time()
#for i in range(1000):
#   out = mobilenet(x)
    
#end = time.time()

#print (end -start)/1000.0
