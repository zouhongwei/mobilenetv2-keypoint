from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, utils

def default_loader(path):
    imgTemp = Image.open(path).convert('RGB')     # W*H*C
    if (imgTemp.size[0] < 512 or imgTemp.size[1] < 512):  # pad image to 512*512
      	imgPad = Image.new('RGB',(512,512),(0,0,0))  # create a black background image
      	imgPad.paste(imgTemp,(0,0))                 # then paste image to this black image 
      	imgFixSize = imgPad
    else:
      	imgFixSize = imgTemp
    return imgFixSize

class dataload(Dataset):
    def __init__(self, datalist, transform=None, loader=default_loader):
      	self.datalist = datalist
      	self.transform = transform
      	self.loader = loader
      
      	fh = open(datalist, 'r')
        imgs = []
        for line in fh:
            line = line.split(',')
            imgs.append((line[0],np.array(map(eval,line[1].split()))))
            #print type(map(eval,line[1].split()))
        self.imgs = imgs
    
    def __getitem__(self,index):
        fn,label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
          
        return img,label
    
    def __len__(self):
        return len(self.imgs)



 ################################
#transform1 = transforms.Compose([transforms.ToTensor()])
#train_data=dataload(datalist='train_blouse.txt',transform =transform1 )
#data_loader = DataLoader(train_data, batch_size=10,shuffle=True)
#
#print(len(data_loader))
#
#def show_batch(imgs):
#    grid = utils.make_grid(imgs)
#    plt.imshow(grid.numpy().transpose((1, 2, 0)))
#    plt.title('Batch from dataloader')
#
#for i, (batch_x, batch_y) in enumerate(data_loader):
#    if(i<4):
#        print(i, batch_x.size(),batch_y.size())
#        show_batch(batch_x)
#        print(batch_y)
#        plt.axis('off')
#        plt.show()