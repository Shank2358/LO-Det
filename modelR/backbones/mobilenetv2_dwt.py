# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.

# ==========================================================================================

# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

# ==========================================================================================

# BSD-3 License

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import torch
import torch.nn as nn
from model.layers.conv_blocks import Downsample_DWT_tiny

__all__ = ['mobilenetv2']

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups = groups, bias = False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace = True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, wavename = 'haar'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size = 1))
        if (stride == 1):
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride = stride, groups = hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias = False),
                nn.BatchNorm2d(oup),
            ])
        else:
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride = 1, groups = hidden_dim),
                #Downsample(filt_size = filter_size, stride = stride, channels = hidden_dim),
                ########################
                Downsample_DWT_tiny(wavename=wavename),
                ########################
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias = False),
                nn.BatchNorm2d(oup),
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class _MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, wavename = 'haar'):
        super(_MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        #last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, 4 if width_mult == 0.1 else 8)
        #self.last_channel = _make_divisible(last_channel * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else last_channel
        features = [ConvBNReLU(3, input_channel, stride = 2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                ###############################
                    block(input_channel, output_channel, stride, expand_ratio = t, wavename = wavename))
                ###############################
                input_channel = output_channel
        # building last several layers
        #features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size = 1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        #self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
            #nn.Linear(self.last_channel, num_classes),
        #)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "features":
                for f_name, f_module in module._modules.items():
                    x = f_module(x)
                    if f_name in self.extracted_layers:
                        outputs.append(x)
            #if name is "conv":
                #x = module(x)
                #if name in self.extracted_layers:
                    #outputs.append(x)
        return outputs


class MobilenetV2(nn.Module):
    def __init__(self, extract_list, weight_path=None, wavename = 'haar', width_mult=1.):
        super(MobilenetV2, self).__init__()

        self.__submodule = _MobileNetV2(width_mult=width_mult, wavename=wavename)
        if weight_path:
            print("*"*40, "\nLoading weight of MobilenetV2 : {}".format(weight_path))
            pretrained_dict = torch.load(weight_path)
            model_dict = self.__submodule.state_dict()
            pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.__submodule.load_state_dict(model_dict)
            del pretrained_dict
            print("Loaded weight of MobilenetV2 : {}".format(weight_path))
        self.__extractor = FeatureExtractor(self.__submodule, extract_list)

    def forward(self, x):
        return self.__extractor(x)

if __name__=='__main__':
    #model = MobilenetV2(extract_list=["6", "13", "conv"])
    #model.eval()
    #print(model)
    #input = torch.randn(32,3,224,224)
    #y = model(input)
    #print(y)

    from model.get_model_complexity import get_model_complexity_info
    from torchstat import stat
    net = MobilenetV2(extract_list=["6", "13", "conv"],width_mult=1.0).cuda()
    #stat(net, (3, 544, 544))
    flops, params = get_model_complexity_info(net, (3, 544, 544), as_strings=False, print_per_layer_stat=True)
    print('GFlops: %.3fG' % (flops / 1e9))
    print('Params: %.2fM' % (params / 1e6))