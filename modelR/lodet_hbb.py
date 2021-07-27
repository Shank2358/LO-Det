import sys
sys.path.append("..")
import torch.nn as nn
from modelR.backbones.mobilenetv2 import MobilenetV2
from modelR.necks.csa_drf_fpn import CSA_DRF_FPN
from modelR.head.dsc_head import DSC_Head
from modelR.head.dsc_head_hbb import Ordinary_Head
from utils.utils_basic import *

class LODet(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, pre_weights=None):
        super(LODet, self).__init__()
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = CSA_DRF_FPN(fileters_in=[1280, 96, 32])
        # small
        self.__head_s = DSC_Head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium
        self.__head_m = DSC_Head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = DSC_Head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

    def forward(self, x):
        out = []
        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l = self.__neck(x_l, x_m, x_s)
        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)


