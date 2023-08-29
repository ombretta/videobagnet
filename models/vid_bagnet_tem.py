import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1, kernel_size=3, padding=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, kernel_size=1, downsample=None):
        super().__init__()
        
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride, kernel_size, padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        #print("-- input conv 1", x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #print("-- input conv 2", out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        #print("-- input conv 3", out.shape)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :, :-diff, :-diff]
        if residual.size(2) != out.size(2):
            diff_t = residual.size(2) - out.size(2)
            residual = residual[:, :, :-diff_t, :, :]

        out += residual
        out = self.relu(out)

        return out


class VidBegNet_OVERLEAF_VERSION(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 strides=[1, 2, 2, 2],
                 kernel3=[0, 0, 0, 0],
                 n_input_channels=3,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(3, 7, 7),
                               stride=(1, 2, 2),
                               padding=(0, 3, 3),
                               bias=False)

        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0,1,1),dilation=1, ceil_mode=False)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, stride=(strides[0], 1, 1), kernel3=kernel3[0]) # Modified spatial stride 
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type, stride=(strides[1], 2, 2), kernel3=kernel3[1])
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type, stride=(strides[2], 2, 2), kernel3=kernel3[2])
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type, stride=(strides[3], 2, 2), kernel3=kernel3[3])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, kernel3=0):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        kernel = (1, 3, 3) if kernel3 == 0 else 3
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  kernel_size=kernel))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            kernel = (1, 3, 3) if kernel3 <= i else 3
            layers.append(block(self.in_planes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print("input x", x.size())
        x = self.conv1(x)
        #print("output conv1", x.size())
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        #print("output max pool", x.size())

        x = self.layer1(x)
        #print("output conv2_x", x.size())
        x = self.layer2(x)
        #print("output conv3_x", x.size())
        x = self.layer3(x)
        #print("output conv4_x", x.size())
        x = self.layer4(x)
        #print("output conv5_x", x.size())
        # print('output size', x.size())

        x = self.avgpool(x)
        
        #print("output avg pool", x.size())

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
class VidBegNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 strides=[1, 2, 2, 2],
                 kernel3=[0, 0, 0, 0],
                 n_input_channels=3,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        
        kernel_conv1 = (3, 7, 7) if kernel3[0] > 0 else (1, 7, 7)
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=kernel_conv1,
                               stride=(1, 2, 2),
                               padding=(0, 3, 3),
                               bias=False)

        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0,1,1),dilation=1, ceil_mode=False)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, stride=(strides[0], 1, 1), kernel3=kernel3[0]) # Modified spatial stride 
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type, stride=(strides[1], 2, 2), kernel3=kernel3[1])
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type, stride=(strides[2], 2, 2), kernel3=kernel3[2])
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type, stride=(strides[3], 2, 2), kernel3=kernel3[3])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool3d((1, 1, 1))

        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, kernel3=0):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        kernel = (1, 3, 3) if kernel3 == 0 else 3
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  kernel_size=kernel))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            kernel = (1, 3, 3) if kernel3 <= i else 3
            layers.append(block(self.in_planes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print("input x", x.size())
        x = self.conv1(x)
        #print("output conv1", x.size())
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        #print("output max pool", x.size())

        x = self.layer1(x)
        #print("output conv1_x", x.size())
        x = self.layer2(x)
        #print("output conv2_x", x.size())
        x = self.layer3(x)
        #print("output conv3_x", x.size())
        x = self.layer4(x)
        #print("output conv4_x", x.size())
        # print('output size', x.size())

        # Modification for the experiments with max pooling
        #x = self.avgpool(x)
        x = self.global_maxpool(x)
        
        #print("output avg pool", x.size())

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    

def generate_model(model_depth, receptive_size, strides=[1, 2, 2, 2], **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    assert receptive_size in [1, 9, 17, 33]
    kernel3_map = {1: [0, 0, 0, 0],
                   9: [1, 1, 0, 0],
                   17: [1, 1, 1, 0],
                   33: [1, 1, 1, 1]}

    if model_depth == 10:
        model = VidBegNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = VidBegNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = VidBegNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = VidBegNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), strides, kernel3=kernel3_map[receptive_size], **kwargs)
    elif model_depth == 101:
        model = VidBegNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = VidBegNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = VidBegNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
