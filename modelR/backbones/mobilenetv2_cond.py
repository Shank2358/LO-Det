
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['cond_mobilenetv2']

class route_func(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        c_in (int): Number of channels in the input image
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, c_in, num_experts):
        super(route_func, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class CondConv2d(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, h, w = x.size()
        k, c_out, c_in, kh, kw = self.weight.size()
        x = x.view(1, -1, h, w)
        weight = self.weight.view(k, -1)
        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kh, kw)
        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv2d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-2), output.size(-1))
        return output


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


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, num_experts=None):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.expand_ratio = expand_ratio
        self.cond = num_experts is not None
        Conv2d = functools.partial(CondConv2d, num_experts=num_experts) if num_experts else nn.Conv2d

        if expand_ratio != 1:
            self.pw = Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            self.bn_pw = nn.BatchNorm2d(hidden_dim)
        self.dw = Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn_dw = nn.BatchNorm2d(hidden_dim)
        self.pw_linear = Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn_pw_linear = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU6(inplace=True)

        if num_experts:
            self.route = route_func(inp, num_experts)

    def forward(self, x):
        identity = x
        if self.cond:
            routing_weight = self.route(x)
            if self.expand_ratio != 1:
                x = self.relu(self.bn_pw(self.pw(x, routing_weight)))
            x = self.relu(self.bn_dw(self.dw(x, routing_weight)))
            x = self.bn_pw_linear(self.pw_linear(x, routing_weight))
        else:
            if self.expand_ratio != 1:
                x = self.relu(self.bn_pw(self.pw(x)))
            x = self.relu(self.bn_dw(self.dw(x)))
            x = self.bn_pw_linear(self.pw_linear(x))

        if self.identity:
            return x + identity
        else:
            return x


class CondMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., num_experts=8):
        super(CondMobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        self.num_experts = None
        for j, (t, c, n, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, self.num_experts))
                input_channel = output_channel
                if j == 4 and i == 0: # CondConv layers in the final 6 inverted residual blocks
                    self.num_experts = num_experts
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_route = route_func(output_channel, num_experts)
        self.classifier = CondConv2d(output_channel, num_classes, kernel_size=1, bias=False, num_experts=num_experts)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        routing_weight = self.classifier_route(x)
        x = self.classifier(x, routing_weight)
        x = x.squeeze_()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def cond_mobilenetv2(**kwargs):
    """
    Constructs a CondConv-based MobileNet V2 model
    """
    return CondMobileNetV2(**kwargs)

if __name__=='__main__':
    #model = MobilenetV2(extract_list=["6", "13", "conv"])
    #model.eval()
    #print(model)
    #input = torch.randn(32,3,224,224)
    #y = model(input)
    #print(y)

    from model.get_model_complexity import get_model_complexity_info
    from torchstat import stat
    net = CondMobileNetV2()
    stat(net, (3, 544, 544))
    flops, params = get_model_complexity_info(net, (3, 544, 544), as_strings=False, print_per_layer_stat=True)
    print('GFlops: %.3fG' % (flops / 1e9))
    print('Params: %.2fM' % (params / 1e6))