import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from .noise_layer import noise_Conv2d, noise_Linear

# class DownsampleA(nn.Module):

#   def __init__(self, nIn, nOut, stride):
#     super(DownsampleA, self).__init__()
#     assert stride == 2
#     self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

#   def forward(self, x):
#     x = self.avg(x)
#     return torch.cat((x, x.mul(0)), 1)

# class ResNetBasicblock(nn.Module):
#   expansion = 1
#   """
#   RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
#   """
#   def __init__(self, inplanes, planes, stride=1, downsample=None):
#     super(ResNetBasicblock, self).__init__()

#     self.conv_a = noise_Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#     self.bn_a = nn.BatchNorm2d(planes)

#     self.conv_b = noise_Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#     self.bn_b = nn.BatchNorm2d(planes)

#     self.downsample = downsample

#   def forward(self, x):
#     residual = x

#     basicblock = self.conv_a(x)
#     basicblock = self.bn_a(basicblock)
#     basicblock = F.relu(basicblock, inplace=True)

#     basicblock = self.conv_b(basicblock)
#     basicblock = self.bn_b(basicblock)

#     if self.downsample is not None:
#       residual = self.downsample(x)
    
#     return F.relu(residual + basicblock, inplace=True)

# class CifarResNet(nn.Module):
#   """
#   ResNet optimized for the Cifar dataset, as specified in
#   https://arxiv.org/abs/1512.03385.pdf
#   """
#   def __init__(self, block, depth, num_classes):
#     """ Constructor
#     Args:
#       depth: number of layers.
#       num_classes: number of classes
#       base_width: base width
#     """
#     super(CifarResNet, self).__init__()

#     #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
#     assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
#     layer_blocks = (depth - 2) // 6
#     print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

#     self.num_classes = num_classes

#     self.conv_1_3x3 = noise_Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#     self.bn_1 = nn.BatchNorm2d(16)

#     self.inplanes = 16
#     self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
#     self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
#     self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
#     self.avgpool = nn.AvgPool2d(8)
#     self.classifier = noise_Linear(64*block.expansion, num_classes)

#     for m in self.modules():
#       if isinstance(m, noise_Conv2d):
#         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#         m.weight.data.normal_(0, math.sqrt(2. / n))
#         #m.bias.data.zero_()
#       elif isinstance(m, nn.BatchNorm2d):
#         m.weight.data.fill_(1)
#         m.bias.data.zero_()
#       elif isinstance(m, nn.Linear):
#         init.kaiming_normal(m.weight)
#         m.bias.data.zero_()

#   def _make_layer(self, block, planes, blocks, stride=1):
#     downsample = None
#     if stride != 1 or self.inplanes != planes * block.expansion:
#       downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

#     layers = []
#     layers.append(block(self.inplanes, planes, stride, downsample))
#     self.inplanes = planes * block.expansion
#     for i in range(1, blocks):
#       layers.append(block(self.inplanes, planes))

#     return nn.Sequential(*layers)

#   def forward(self, x):
#     x = self.conv_1_3x3(x)
#     x = F.relu(self.bn_1(x), inplace=True)
#     x = self.stage_1(x)
#     x = self.stage_2(x)
#     x = self.stage_3(x)
#     x = self.avgpool(x)
#     x = x.view(x.size(0), -1)
#     return self.classifier(x)

# def noise_resnet20(num_classes=10):
#   """Constructs a ResNet-20 model for CIFAR-10 (by default)
#   Args:
#     num_classes (uint): number of classes
#   """
#   model = CifarResNet(ResNetBasicblock, 20, num_classes)
#   return model

# def noise_resnet32(num_classes=10):
#   """Constructs a ResNet-32 model for CIFAR-10 (by default)
#   Args:
#     num_classes (uint): number of classes
#   """
#   model = CifarResNet(ResNetBasicblock, 32, num_classes)
#   return model

# def noise_resnet44(num_classes=10):
#   """Constructs a ResNet-44 model for CIFAR-10 (by default)
#   Args:
#     num_classes (uint): number of classes
#   """
#   model = CifarResNet(ResNetBasicblock, 44, num_classes)
#   return model

# def noise_resnet56(num_classes=10):
#   """Constructs a ResNet-56 model for CIFAR-10 (by default)
#   Args:
#     num_classes (uint): number of classes
#   """
#   model = CifarResNet(ResNetBasicblock, 56, num_classes)
#   return model

# def vanilla_resnet110(num_classes=10):
#   """Constructs a ResNet-110 model for CIFAR-10 (by default)
#   Args:
#     num_classes (uint): number of classes
#   """
#   model = CifarResNet(ResNetBasicblock, 110, num_classes)
#   return model


import torch
import torch.nn as  nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = noise_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = noise_Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = noise_Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = noise_Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = noise_Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class Noisy_ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(Noisy_ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = noise_Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = noise_Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                noise_Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)        
        
def noisy_ResNet50(num_classes, channels=3):
    return Noisy_ResNet(Bottleneck, [3,4,6,3], num_classes, channels)