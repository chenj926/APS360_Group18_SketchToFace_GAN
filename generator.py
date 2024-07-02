import torch
import torch.nn as nn
from torchinfo import summary
##########################################################
#may want to implement double conv block in generator 
###########################################################

class Block(nn.Module):
    #down: True means in encoder part of generator, False means in decoder part
    #act: default activation function is relu
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride = 2, padding=1, bias=False, padding_mode="reflect")
            
            #down sampling after a single conv if encoding 
            if down
            #if in decoder part perform upsampling 
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride = 2, padding=1, bias=False),
            
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2), 
        )
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        #use dropout only in first 3 layers of UNet
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        
        
        #simple encoder-decoder architecture
        #encoder part, as layer become deeper, feature increase 
        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=False) #output img size 64
        self.down2 = Block(features*2, features*4, down=True, act="leaky", use_dropout=False) #32
        self.down3 = Block(features*4, features*8, down=True, act="leaky",use_dropout=False) #16
        self.down4 = Block(features*8, features*8, down=True, act="leaky",use_dropout=False) #8
        self.down5 = Block(features*8, features*8, down=True, act="leaky",use_dropout=False) #4
        self.down6 = Block(features*8, features*8, down=True, act="leaky",use_dropout=False) #2
        
        #bottleneck 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1, padding_mode="reflect"), #1x1 size
            nn.ReLU(),
        ) 
        
        
        #decoder part 
        #as layer coming out of bottleneck, feature size decrease
        self.up1 = Block(features*8, features*8, down=False,act="relu", use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False,act="relu", use_dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False, act="relu",use_dropout=True)
        self.up4 = Block(features*8*2, features*8, down=False, act="relu",use_dropout=False)
        self.up5 = Block(features*8*2, features*4, down=False, act="relu",use_dropout=False)
        self.up6 = Block(features*4*2, features*2, down=False, act="relu",use_dropout=False)
        self.up7 = Block(features*2*2, features, down=False, act="relu", use_dropout=False)
        
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(), #output pixel value between -1 to 1
        )
    
    def forward(self,x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        
        u1 = self.up1(bottleneck)
        #use skip connection to concatenate encoder output with decoder input
        #inherit d7 from encoder and apply to u1 section
        #look at UNET it totally make sense in terms of skip connection 
        #only difference is using single layer instead of double conv2 layer
        u2 = self.up2(torch.cat((u1,d7), dim=1))
        u3 = self.up3(torch.cat((u2,d6), dim=1))
        u4 = self.up4(torch.cat((u3,d5), dim=1))
        u5 = self.up5(torch.cat((u4,d4), dim=1))
        u6 = self.up6(torch.cat((u5,d3), dim=1))
        u7 = self.up7(torch.cat((u6,d2), dim=1))
        
        return self.final_up(torch.cat((u7,d1), dim=1))

#I maybe incorrect in the GAN architecture flowchart. Probably noise factor is not necessary but only the label image is required to provide (since we don't have multiclass but img itself is the label)
def test():
    x = torch.randn((1,3,256,256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()
    model = Generator(in_channels=3, features=64)
    # Print size of each layer of the model
    summary(model, input_size=(1, 3, 256, 256))

