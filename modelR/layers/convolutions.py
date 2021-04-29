from .activations import *
from modelR.plugandplay.DynamicConv import Dynamic_conv2d
from modelR.plugandplay.CondConv import CondConv2d, route_func
from modelR.layers.deform_conv_v2 import DeformConv2d

norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "relu6": nn.ReLU6,
    "Mish": Mish,
    "Swish": Swish,
    "MEMish": MemoryEfficientMish,
    "MESwish": MemoryEfficientSwish,
    "FReLu": FReLU
}

class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
        super(Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=not norm, groups=groups, dilation=dila)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = activate_name[activate]()
            if activate == "Swish":
                self.__activate = activate_name[activate]()
            if activate == "MEMish":
                self.__activate = activate_name[activate]()
            if activate == "MESwish":
                self.__activate = activate_name[activate]()
            if activate == "FReLu":
                self.__activate = activate_name[activate]()

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x

class DeConvolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=4, stride=2, pad=1, output_pad=0, groups=1, dila=1, norm=None, activate=None):
        super(DeConvolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        if kernel_size == 4:
            pad = 1
            output_pad = 0
        elif kernel_size == 3:
            pad = 1
            output_pad = 1
        elif kernel_size == 2:
            pad = 0
            output_pad = 0
        self.__deconv = nn.ConvTranspose2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=output_pad)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        x = self.__deconv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x


class Separable_Conv(nn.Module):
    def __init__(self, filters_in, filters_out, stride, norm="bn", activate="relu6"):
        super(Separable_Conv, self).__init__()

        self.__dw = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                  stride=stride, pad=1, groups=filters_in, norm=norm, activate=activate)

        self.__pw = Convolutional(filters_in=filters_in, filters_out=filters_out, kernel_size=1,
                                  stride=1, pad=0, norm=norm, activate=activate)

    def forward(self, x):
        return self.__pw(self.__dw(x))


class Deformable_Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, groups=1, norm=None, activate=None):
        super(Deformable_Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__dcn = DeformConv2d(inc=filters_in, outc=filters_out, kernel_size=kernel_size, padding=pad, stride=stride, bias=None, modulation=True)
        #DCN(filters_in, filters_out, kernel_size=kernel_size, stride=stride, padding=pad, deformable_groups=groups).cuda()
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        x = self.__dcn(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x

class Separable_Conv_dila(nn.Module):
    def __init__(self, filters_in, filters_out, stride, pad, dila):
        super(Separable_Conv_dila, self).__init__()

        self.__dw = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3, stride=stride,
                                  pad=pad, groups=filters_in, dila=dila, norm="bn", activate="relu6")
        #self.__se=SELayer(filters_in)
        self.__pw = Convolutional(filters_in=filters_in, filters_out=filters_out, kernel_size=1, stride=1,
                                  pad=0, norm="bn", activate="relu6")

    def channel_shuffle(self, features, groups=2):
        batchsize, num_channels, height, width = features.data.size()
        assert (num_channels % groups == 0)
        channels_per_group = num_channels // groups
        # reshape
        features = features.view(batchsize, groups, channels_per_group, height, width)
        features = torch.transpose(features, 1, 2).contiguous()
        # flatten
        features = features.view(batchsize, -1, height, width)
        return features

    def forward(self, x):
        #return self.__pw(self.__se(self.__dw(x)))
        out = self.__pw(self.__dw(x))
        #out = self.channel_shuffle(out)
        return out


class Cond_Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride=1, pad=0, dila=1, groups=1, bias=True, num_experts=1, norm=None, activate=None):

        super(Cond_Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = CondConv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                 stride=stride, padding=pad, dilation=dila, groups=groups, bias=bias, num_experts=num_experts)
        self.__routef = route_func(filters_in, num_experts)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        routef = self.__routef(x)
        x = self.__conv(x,routef)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x


class Dynamic_Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride=1, pad=0, dila=1, groups=1, bias=True, K=4, temperature=34, norm=None, activate=None):

        super(Dynamic_Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = Dynamic_conv2d(in_planes=filters_in, out_planes=filters_out, kernel_size=kernel_size,
                                     ratio=0.25, stride=stride, padding=pad, dilation=dila, groups=groups, bias=bias, K=K, temperature=temperature, init_weight=True)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x
