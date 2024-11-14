import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19,VGG19_Weights
import math

class ResBlock(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=in_features,out_channels=in_features,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_channels=in_features,out_channels=in_features,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(in_features)
        ) 
    def forward(self,x):
        return x + self.conv(x)
    
class SRGenerator(nn.Module):
    def __init__(self,in_channel=3,out_channel=3,res_block=16):
        super().__init__()

        self.conv1= nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=9,stride=1,padding=4),
            nn.PReLU()
        )

        res_block_list = []
        for n in range(res_block):
            res_block_list.append(ResBlock(64))
        self.res_deep_block = nn.Sequential(*res_block_list)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64)
        )

        self.sub_pixel1=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=256,stride=1,kernel_size=3,padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
        self.sub_pixel2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=256,stride=1,kernel_size=3,padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )
        self.conv3= nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=out_channel,kernel_size=9,stride=1,padding=4),
            nn.Tanh()
        )
        
    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.res_deep_block(out1)
        out3 = out1 + self.conv2(out2)
        out = self.sub_pixel1(out3)
        out = self.sub_pixel2(out)
        out = self.conv3(out)
        # out = self.activate(out)
        return (out+1)/2

class Discriminator(nn.Module):

    def __init__(self,):
        super().__init__()

        self.feature_extract= nn.Sequential(
            #64
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2),

            #128
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2),

            #256
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2),

            #512
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2)            
        )
        # self.fc1 = None
        self.classifier = nn.Sequential(
            nn.LazyLinear(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = self.feature_extract(x)
        # flat_size = x.view(x.size(0), -1).size(1) 
        # if self.fc1 is None:
        #     self.fc1 = nn.Linear(flat_size,1024).cuda()
        x = x.view(x.size(0),-1)

        return self.classifier(x)
    
class FeatureExtractVGG(nn.Module):
    def __init__(self):
        super().__init__()
        weights = VGG19_Weights.DEFAULT
        model = vgg19(weights=weights)
        model.eval()
        self.feature_extractor = model.features

    def forward(self,img):
        return self.feature_extractor(img)

 


