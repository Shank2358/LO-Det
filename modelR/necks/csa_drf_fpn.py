import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D, LinearScheduler
from model.layers.convolutions import Convolutional, Deformable_Convolutional
from model.layers.shuffle_blocks import Shuffle_new, Shuffle_Cond_RFA, Shuffle_new_s
import config.cfg_lodet as cfg

class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out

class CSA_DRF_FPN(nn.Module):
    def __init__(self, fileters_in, model_size=1):
        super(CSA_DRF_FPN, self).__init__()

        fi_0, fi_1, fi_2 = fileters_in
        self.__fo = (cfg.DATA["NUM"]+5 +5)*cfg.MODEL["ANCHORS_PER_SCLAE"]
        fm_0 = int(1024*model_size)
        fm_1 = fm_0//2
        fm_2 = fm_0 // 4

        self.__dcn2_1 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        self.__routdcn2_1 = Route()

        self.__dcn1_0 = Deformable_Convolutional(fi_1+fi_2, fi_1, kernel_size=3, stride=2, pad=1, groups=1)

        self.__routdcn1_0 = Route()
        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0 + fi_1, filters_out=fm_0, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=8),
            Shuffle_Cond_RFA(filters_in=fm_0, filters_out=fm_0, groups=8, dila_l=4, dila_r=6),#, dila_l=4, dila_r=6
            Shuffle_new_s(filters_in=fm_0//2, filters_out=fm_0, groups=8),
        )
        self.__conv0_0 = Shuffle_new(filters_in=fm_0, filters_out=fm_0, groups=4)
        self.__conv0_1 = Convolutional(filters_in=fm_0, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        self.__conv0up1 = nn.Conv2d(fm_0, fm_1, kernel_size=1, stride=1, padding=0)
        self.__upsample0_1 = Upsample(scale_factor=2)

        # medium
        self.__pw1 = Convolutional(filters_in=fi_2+fi_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")#, groups=fi_2+fi_1
        self.__shuffle10 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__route0_1 = Route()
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fm_1*2, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Shuffle_Cond_RFA(filters_in=fm_1, filters_out=fm_1, groups=4, dila_l=2, dila_r=3),#, dila_l=2, dila_r=3
            #Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new_s(filters_in=fm_1//2, filters_out=fm_1, groups=4),
        )
        self.__conv1_0 = Shuffle_new(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__conv1_1 = Convolutional(filters_in=fm_1, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        self.__conv1up2 = nn.Conv2d(fm_1, fm_2, kernel_size=1, stride=1, padding=0)
        self.__upsample1_2 = Upsample(scale_factor=2)


        # small
        #self.__dcn2 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=1, pad=1, groups=1, norm="bn")
        self.__pw2 = Convolutional(filters_in=fi_2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__shuffle20 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)
        self.__route1_2 = Route()
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fm_2*2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
            #Shuffle_Cond_RFA(filters_in=fm_2, filters_out=fm_2, groups=4, dila_l=1, dila_r=2),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4),
        )
        self.__conv2_0 = Shuffle_new(filters_in=fm_2, filters_out=fm_2, groups=4)
        self.__conv2_1 = Convolutional(filters_in=fm_2, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        self.__initialize_weights()


    def __initialize_weights(self):
        print("**" * 10, "Initing FPN_YOLOV3 weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

    def forward(self, x0, x1, x2):

        dcn2_1 = self.__dcn2_1(x2)
        routdcn2_1 = self.__routdcn2_1(x1, dcn2_1)

        dcn1_0  = self.__dcn1_0(routdcn2_1)
        routdcn1_0 = self.__routdcn1_0(x0, dcn1_0)

        # large
        conv_set_0 = self.__conv_set_0(routdcn1_0)
        conv0up1 = self.__conv0up1(conv_set_0)
        upsample0_1 = self.__upsample0_1(conv0up1)

        # medium
        pw1 = self.__pw1(routdcn2_1)
        shuffle10 = self.__shuffle10(pw1)
        route0_1 = self.__route0_1(shuffle10,upsample0_1)
        conv_set_1 = self.__conv_set_1(route0_1)

        conv1up2 = self.__conv1up2(conv_set_1)
        upsample1_2 = self.__upsample1_2(conv1up2)

        # small
        pw2 = self.__pw2(x2)
        shuffle20 = self.__shuffle20(pw2)
        route1_2 = self.__route1_2(shuffle20, upsample1_2)
        conv_set_2 = self.__conv_set_2(route1_2)

        out0 = self.__conv0_0(conv_set_0)
        out0 = self.__conv0_1(out0)

        out1 = self.__conv1_0(conv_set_1)
        out1 = self.__conv1_1(out1)

        out2 = self.__conv2_0(conv_set_2)
        out2 = self.__conv2_1(out2)

        return out2, out1, out0  # small, medium, large