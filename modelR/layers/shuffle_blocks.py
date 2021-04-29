from .activations import *
from ..layers.convolutions import Convolutional, Cond_Convolutional
import math
import numpy as np
class Shuffle_new(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3 ,c_tag=0.5, groups=3, dila=1):
        super(Shuffle_new, self).__init__()
        self.left_part = round(c_tag * filters_in)
        self.right_part = filters_out - self.left_part
        self.__dw = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=kernel_size, stride=1, pad=(kernel_size-1)//2, groups=self.right_part, dila=dila, norm="bn")
        self.__pw1 = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.groups = groups

    def channel_shuffle(self, features):
        batchsize, num_channels, height, width = features.data.size()
        assert (num_channels % self.groups == 0)
        channels_per_group = num_channels // self.groups
        features = features.view(batchsize, self.groups, channels_per_group, height, width)# reshape
        features = torch.transpose(features, 1, 2).contiguous()
        features = features.view(batchsize, -1, height, width)# flatten
        return features

    def forward(self, x):
        left = x[:, :self.left_part, :, :].contiguous()
        right = x[:, self.left_part:, :, :].contiguous()
        right = self.__dw(right)
        right = self.__pw1(right)
        cat = torch.cat((left, right), 1)
        out = self.channel_shuffle(cat)
        return out

class Shuffle_Cond_RFA(nn.Module):
    def __init__(self, filters_in, filters_out, c_tag=0.5, groups=3, dila_r=4, dila_l=6):
        super(Shuffle_Cond_RFA, self).__init__()
        self.left_part = round(c_tag * filters_in)
        self.right_part = filters_out - self.left_part
        self.__dw_right = Cond_Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=3,
                                       stride=1, pad=dila_r, groups=self.right_part, dila=dila_r,  bias=True,  norm="bn")
        self.__pw_right = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1,
                                        stride=1, pad=0, norm="bn", activate="leaky")

        self.__dw_left = Cond_Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=3,
                                       stride=1, pad=dila_l, groups=self.right_part, dila=dila_l,  bias=True,  norm="bn")
        self.__pw1_left = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1,
                                        stride=1, pad=0, norm="bn", activate="leaky")
        #self.groups = groups

    def forward(self, x):
        left = x[:, :self.left_part, :, :].contiguous()
        right = x[:, self.left_part:, :, :].contiguous()
        left = self.__dw_left(left)
        left = self.__pw1_left(left)
        right = self.__dw_right(right)
        right = self.__pw_right(right)
        #cat = torch.cat((left, right), 1)
        #out = self.channel_shuffle(cat)
        return left+right

class Shuffle_new_s(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3 ,c_tag=0.5, groups=3, dila=1):
        super(Shuffle_new_s, self).__init__()
        self.__dw = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=kernel_size, stride=1, pad=(kernel_size-1)//2, groups=filters_in, dila=dila, norm="bn")
        self.__pw1 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.groups = groups

    def channel_shuffle(self, features):
        batchsize, num_channels, height, width = features.data.size()
        assert (num_channels % self.groups == 0)
        channels_per_group = num_channels // self.groups
        features = features.view(batchsize, self.groups, channels_per_group, height, width)# reshape
        features = torch.transpose(features, 1, 2).contiguous()
        features = features.view(batchsize, -1, height, width)# flatten
        return features
    def forward(self, x):
        right = self.__dw(x)
        right = self.__pw1(right)
        cat = torch.cat((x, right), 1)
        out = self.channel_shuffle(cat)
        return out

class Shuffle_RFA(nn.Module):
    def __init__(self, filters_in, filters_out, c_tag=0.5, groups=3, dila_r=4, dila_l=6):
        super(Shuffle_RFA, self).__init__()
        self.left_part = round(c_tag * filters_in)
        self.right_part = filters_out - self.left_part
        self.__dw_right = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=3,
                                       stride=1, pad=dila_r, groups=self.right_part, dila=dila_r,  norm="bn")
        self.__pw_right = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1,
                                        stride=1, pad=0,  norm="bn", activate="relu")

        self.__dw_left = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=3,
                                       stride=1, pad=dila_l, groups=self.right_part, dila=dila_l, norm="bn")
        self.__pw1_left = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1,
                                        stride=1, pad=0, norm="bn", activate="relu")

        self.groups = groups

    def channel_shuffle(self, features):
        batchsize, num_channels, height, width = features.data.size()
        assert (num_channels % self.groups == 0)
        channels_per_group = num_channels // self.groups
        features = features.view(batchsize, self.groups, channels_per_group, height, width)# reshape
        features = torch.transpose(features, 1, 2).contiguous()
        features = features.view(batchsize, -1, height, width)# flatten
        return features
    def forward(self, x):
        left = x[:, :self.left_part, :, :].contiguous()
        right = x[:, self.left_part:, :, :].contiguous()
        left = self.__dw_left(left)
        left = self.__pw1_left(left)
        right = self.__dw_right(right)
        right = self.__pw_right(right)
        cat = torch.cat((left, right), 1)
        out = self.channel_shuffle(cat)
        return out

class DRF3(nn.Module):
    def __init__(self, filters_in, filters_out, c_tag=0.5, groups=3):
        super(DRF3, self).__init__()
        self.left_part = round(c_tag * filters_in)
        self.right_part = filters_out - self.left_part
        #self.__dw_right = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=5, stride=1, pad=dila_r*2, groups=self.right_part, dila=dila_r,  norm="bn")
        self.__right_weight = nn.Parameter(torch.Tensor(self.right_part,1,3,3), requires_grad=True)#torch.rand(self.right_part,self.right_part,5,5)
        self.__bn = nn.BatchNorm2d(self.right_part,affine=True)
        self.__pw_right = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1,
                                        stride=1, pad=0, norm="bn", activate="leaky")
        self.__globpool = nn.AdaptiveAvgPool2d(1)
        self.__fc = Convolutional(1,1,1,1,0,norm='bn',activate="leaky")
        self.groups = groups

    def channel_shuffle(self, features):
        batchsize, num_channels, height, width = features.data.size()
        assert (num_channels % self.groups == 0)
        channels_per_group = num_channels // self.groups
        features = features.view(batchsize, self.groups, channels_per_group, height, width)# reshape
        features = torch.transpose(features, 1, 2).contiguous()
        features = features.view(batchsize, -1, height, width)# flatten
        return features
    def forward(self, x):
        left = x[:, :self.left_part, :, :].contiguous()
        right = x[:, self.left_part:, :, :].contiguous()
        fc = self.__fc(self.__globpool(right[:, 0:1, :, :]))
        #print(fc.shape)
        fcc = fc.detach().cpu()
        #print(fcc.shape)
        rfa = round(torch.sigmoid(torch.sum(fcc)).item() * 2 + 1)
        right = self.__bn(F.conv2d(right, self.__right_weight, stride=1, padding=rfa, dilation=rfa, groups=self.right_part)) #self.__dw_right(right)
        right = self.__pw_right(right)
        cat = torch.cat((left, right), 1)
        out = self.channel_shuffle(cat)
        return out

class DRF5(nn.Module):
    def __init__(self, filters_in, filters_out, c_tag=0.5, groups=3):
        super(DRF5, self).__init__()
        self.left_part = round(c_tag * filters_in)
        self.right_part = filters_out - self.left_part
        #self.__dw_right = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=5, stride=1, pad=dila_r*2, groups=self.right_part, dila=dila_r,  norm="bn")
        self.__right_weight = nn.Parameter(torch.Tensor(self.right_part,1,5,5), requires_grad=True)#torch.rand(self.right_part,self.right_part,5,5)
        self.__bn = nn.BatchNorm2d(self.right_part,affine=True)
        self.__pw_right = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1,
                                        stride=1, pad=0, norm="bn", activate="leaky")
        self.__globpool = nn.AdaptiveAvgPool2d(1)
        self.__fc = Convolutional(1,1,1,1,0,norm='bn',activate="leaky")
        self.groups = groups

    def channel_shuffle(self, features):
        batchsize, num_channels, height, width = features.data.size()
        assert (num_channels % self.groups == 0)
        channels_per_group = num_channels // self.groups
        features = features.view(batchsize, self.groups, channels_per_group, height, width)# reshape
        features = torch.transpose(features, 1, 2).contiguous()
        features = features.view(batchsize, -1, height, width)# flatten
        return features
    def forward(self, x):
        left = x[:, :self.left_part, :, :].contiguous()
        right = x[:, self.left_part:, :, :].contiguous()
        fc = self.__fc(self.__globpool(right[:, 0:1, :, :]))
        #print(fc.shape)
        fcc = fc.detach().cpu()
        #print(fcc.shape)
        rfa = round(torch.sigmoid(torch.sum(fcc)).item() * 2 + 1)
        right = self.__bn(F.conv2d(right, self.__right_weight, stride=1, padding=2*rfa, dilation=rfa, groups=self.right_part)) #self.__dw_right(right)
        right = self.__pw_right(right)
        cat = torch.cat((left, right), 1)
        out = self.channel_shuffle(cat)
        return out
