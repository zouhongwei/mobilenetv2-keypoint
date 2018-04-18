import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ResNet(nn.Module):
    __factory = {
          18: torchvision.models.resnet18,
          34: torchvision.models.resnet34,
          50: torchvision.models.resnet50,
          101: torchvision.models.resnet101,
          152: torchvision.models.resnet152,
      }
    def __init__(self, num_classes = 10, depth = 18, pre_trained = True, dropout=0, train = True):
          super(ResNet, self).__init__()
          self.depth = depth
          self.num_classes = num_classes
          self.pre_trained = pre_trained
          self.dropout = dropout
          self.train = train
          
          if depth not in ResNet.__factory:
              raise KeyError("Unsupported depth:", depth)
          self.base = ResNet.__factory[depth](pretrained = self.pre_trained)
          self.num_features = self.base.fc.in_features
          
          self.classifier = nn.Linear(self.num_features, self.num_classes)
          
          if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
          
          if not self.pre_trained:
              self.reset_params()
              
    def forward(self,x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        
        if self.dropout:
            x = self.drop(x)
            
        if self.num_classes > 0:
            x = self.classifier(x)
            
        return x
        
      
    def reset_params(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              init.kaiming_normal(m.weight, mode='fan_out')
              if m.bias is not None:
                  init.constant(m.bias, 0)
          elif isinstance(m, nn.BatchNorm2d):
              init.constant(m.weight, 1)
              init.constant(m.bias, 0)
          elif isinstance(m, nn.Linear):
              init.normal(m.weight, std=0.001)
              if m.bias is not None:
                  init.constant(m.bias, 0)    
                    
                    
def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

