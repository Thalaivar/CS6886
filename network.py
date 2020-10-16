import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, in_channels):
        super(Network, self).__init__()
        
        self.input_conv = nn.Sequential(
                            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(num_features=16),
                            Hswish()
                        )
        in_channels = 16

        kernel_sz = [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5]
        exp = [16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576]
        out = [16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96]
        stride = [2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1]

        is_SE = [True for _ in exp]
        is_SE[1] = is_SE[2] = False

        nonlinearity = [Hswish() for _ in exp]
        nonlinearity[0] = nonlinearity[1] = nonlinearity[2] = nn.ReLU()

        fuse_blocks = []
        for i in range(len(exp)):
            fuse_blocks.append(FuseBlock(K=kernel_sz[i],
                                exp=exp[i],
                                stride=stride[i],
                                in_channels=in_channels,
                                out_channels=out[i],
                                nonlinearity=nonlinearity[i],
                                squeeze_and_excite=is_SE[i]
                            ))
            in_channels = out[i]
        self.fuse_blocks = nn.Sequential(*fuse_blocks)

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=in_channels, out_channels=576, stride=1, kernel_size=1, bias=False),
                        nn.BatchNorm2d(num_features=576),
                        SEModule(channel=576),
                        Hswish()
                    )
        in_channels = 576

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv2 = nn.Sequential(
                        nn.Dropout(p=0.2),
                        nn.Conv2d(in_channels=in_channels, out_channels=1024, stride=1, kernel_size=1, bias=False),
                        Hswish()
                    )
        
        self.output = nn.Linear(in_features=1024, out_features=100, bias=False)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.fuse_blocks(x)
        x = self.conv1(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = self.output(x.squeeze())
        return x
    
    def _initialize_weights(self):                                                                               
        # weight initialization                                                                                                  
        for m in self.modules():                                                                                                 
            if isinstance(m, nn.Conv2d):                                                                                         
                nn.init.kaiming_normal_(m.weight, mode='fan_out')                                                                
                if m.bias is not None:                                                                                           
                    nn.init.zeros_(m.bias)                                                                                       
            elif isinstance(m, nn.BatchNorm2d):                                                                                  
                nn.init.ones_(m.weight)                                                                                          
                nn.init.zeros_(m.bias)                                                                                           
            elif isinstance(m, nn.Linear):                                                                                       
                nn.init.normal_(m.weight, 0, 0.01)                                                                               
                if m.bias is not None:                                                                                           
                    nn.init.zeros_(m.bias)   


class FuseBlock(nn.Module):
    def __init__(self, K, exp, stride, in_channels, out_channels, nonlinearity, squeeze_and_excite=False):
        super(FuseBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(2 * exp, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        
        padding = math.ceil((K - 1) / 2)
        self.depth_conv1 = nn.Conv2d(exp, exp, kernel_size=(1, K), stride=stride, padding=(0, padding), groups=exp, bias=False)
        self.depth_conv2 = nn.Conv2d(exp, exp, kernel_size=(K, 1), stride=stride, padding=(padding, 0), groups=exp, bias=False)
        
        self.squeeze_and_excite = squeeze_and_excite
        if squeeze_and_excite:
            self.se_layer = SEModule(channel=2 * exp)
            self.hsigmoid = Hsigmoid(inplace=False)

        self.conv_bn1 = nn.BatchNorm2d(num_features=exp)
        self.conv_bn2 = nn.BatchNorm2d(num_features= out_channels)
        self.dw_conv_bn1 = nn.BatchNorm2d(num_features=exp)
        self.dw_conv_bn2 = nn.BatchNorm2d(num_features=exp)

        self.nonlinearity = nonlinearity

    def forward(self, x):
        out1 = self.conv_bn1(self.nonlinearity(self.conv1(x)))
        
        out2 = self.dw_conv_bn1(self.depth_conv1(out1))
        out3 = self.dw_conv_bn2(self.depth_conv2(out1))

        out = torch.cat([out2, out3], 1)

        if self.squeeze_and_excite:
            out = self.hsigmoid(self.se_layer(out))
        out = self.nonlinearity(out)

        out = self.conv_bn2(self.conv2(out))    

        return out      

class Hsigmoid(nn.Module):                                                                                                                                                                  
    def __init__(self, inplace=True):                                                                                                                                                       
        super(Hsigmoid, self).__init__()                                                                                                                                                    
        self.inplace = inplace                                                                                                                                                              
                                                                                                                                                                                            
    def forward(self, x):                                                                                                                                                                   
        return F.relu6(x + 3., inplace=self.inplace) / 6.                                                                                                                                   
                                                                                                                                                                                             
                                                                                                                                                                                             
class SEModule(nn.Module):                                                                                                                                                                  
    def __init__(self, channel, reduction=4):                                                                                                                                               
        super(SEModule, self).__init__()                                                                                                                                                    
        self.avg_pool = nn.AdaptiveAvgPool2d(1)                                                                                                                                             
        self.fc = nn.Sequential(                                                                                                                                                            
            nn.Linear(channel, channel // reduction, bias=False),                                                                                                                           
            nn.ReLU(inplace=True),                                                                                                                                                          
            nn.Linear(channel // reduction, channel, bias=False),                                                                                                                           
            Hsigmoid()                                                                                                                                                                      
        )                                                                                                                                                                                   
                                                                                                                                                                                            
    def forward(self, x):                                                                                                                                                                   
        b, c, _, _ = x.size()                                                                                                                                                               
        y = self.avg_pool(x).view(b, c)                                                                                                                                                     
        y = self.fc(y).view(b, c, 1, 1)                                                                                                                                                     
        return x * y.expand_as(x)  

class Hswish(nn.Module):                                                                                                                                                                    
    def __init__(self, inplace=True):                                                                                                                                                       
        super(Hswish, self).__init__()                                                                                                                                                      
        self.inplace = inplace

    def forward(self, x):                                                                                                                                                                   
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

if __name__ == "__main__":
    model = Network(in_channels=3)

    x = torch.rand(4, 224, 224, 3)
    y = model(x)
    print(y.shape)