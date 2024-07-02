#discriminator is NOT conditioned!!! 
#It doesn't take input feature from generator!!! 

import torch
import torch.nn as nn
import torchinfo
from torchinfo import summary

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            #"reflect": reflect the input by padding with the input mirrored along the edges
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self,x):
        return self.conv(x)



class Discriminator(nn.Module):
    #send in 256 inputs and get size 30 output after 4 conv block 
    #(calculated by Conv2d equation with feature size)
    def __init__(self, in_channels=3,features=[64,128,256,512]):
        super().__init__()
        #initial block is different than conv! a special start
        self.initial = nn.Sequential(
            #why in_channels*2? because we are going to concatenate the input image (the "condition") with the output of the generator! We are actually sending 2 imgs here
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        
        #now construct rest 4 conv2 block
        layers=[]
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2), #do stride of 1 for last layer
            )
            in_channels = feature
            
        #ensure the output is a single value per patch
        #just add one additional conv2d 
        layers.append(
            #output a single channel 
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
        )
            
        #unpack layers block to model 
        self.model = nn.Sequential(*layers)
        
    
    #send in generator input x and a "condition" image y!
    def forward(self, x, label):
        #concatenate the input image with the output of the generator
        x = torch.cat([x,label], dim=1)
        x = self.initial(x)
        #return a probability of real or fake
        return self.model(x)

#test if descriminator work 
def test():
    x = torch.randn((1,3,256,256)) #[B, C, H, W]
    y = torch.randn((1,3,256,256))
    model = Discriminator()
    preds = model(x,y)
    #output shape should be [1,1,X,X] 
    #for each patch we have to output a single vaule of 0-1 (X is number of patch)
    print("Final output shape", preds.shape)
    
if __name__ == "__main__":
    test()
    
    model = Discriminator()
    # Print size of each layer of the model
    summary(model, input_size=((1, 3, 256, 256), (1, 3, 256, 256)))
        
        