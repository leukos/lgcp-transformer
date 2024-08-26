import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels , out_channels , kernel_size , stride , padding , bias=False):
        super(ConvBlock,self).__init__()

        # 2d convolution
        self.conv2d = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding , bias=False )

        # batchnorm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        # relu layer
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.batchnorm2d(self.conv2d(x)))



class SPPBlock(nn.Module):
    def __init__(self , in_channels , out_1 , out_2, out_3, out_4, out_5):
        super(SPPBlock,self).__init__()

        self.branch1 = nn.Sequential(ConvBlock(in_channels, out_1, 1, 1 ,0), nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.branch2 = nn.Sequential(ConvBlock(in_channels, out_2, 3, 1, 1), nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.branch3 = nn.Sequential(ConvBlock(in_channels, out_3, 5, 1, 2), nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.branch4 = nn.Sequential(ConvBlock(in_channels, out_4, 7, 1, 3), nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.branch5 = nn.Sequential(ConvBlock(in_channels, out_4, 9, 1, 4), nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

    def forward(self,x):

        # concatenation from dim=1 as dim=0 represents batchsize
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x), self.branch5(x)],dim=1)

class FF(nn.Module):

    def __init__(self):
        super(FF,self).__init__()
        self.fw = nn.Sequential(
          ConvBlock(1, 8, 3, 1, 1),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          ConvBlock(8, 16, 3, 1, 1),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          ConvBlock(16, 32, 3, 1, 1),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          ConvBlock(32, 64, 3, 1, 1),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          ConvBlock(64, 128, 3, 1, 1),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          ConvBlock(128, 256, 3, 1, 1),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          nn.Flatten(start_dim=1),
          nn.Linear(2304, 256),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(256, 3)
        )
      

    def forward(self,x):
        x = self.fw(x)
        return x
