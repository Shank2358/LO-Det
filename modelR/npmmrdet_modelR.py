import sys
sys.path.append("..")

import torch.nn as nn
from modelR.backbones.darknet53_npattention import Darknet53_NPAttention
from modelR.backbones.darknet53 import Darknet53
from modelR.backbones.cspdarknet53__npattention import CSPDarknet53_NPAttention
from modelR.necks.panet_fpn import PANet_FPN
from modelR.necks.msr_fpn import MSR_FPN
from modelR.head.mtr_head_gcn import MTR_Head2
from modelR.layers.convolutions import Convolutional
from utils.utils_basic import *

class NPMMRDetR(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, init_weights=True):
        super(NPMMRDetR, self).__init__()

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]

        self.__backnone = Darknet53_NPAttention()
        self.__neck = MSR_FPN(fileters_in=[1024, 512, 256])

        # small
        self.__head_s = MTR_Head2(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium
        self.__head_m = MTR_Head2(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = MTR_Head2(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        if init_weights:
            self.__init_weights()


    def forward(self, x):
        out=[]
        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l= self.__neck(x_l, x_m, x_s)
        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)

    def __init_weights(self):
        " Note ：nn.Conv2d nn.BatchNorm2d 'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                #torch.nn.init.xavier_normal_(m.weight.data, gain=1)
                #torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)
                print("initing {}".format(m))

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                #torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"
        print("load darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1
                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                print("loading weight {}".format(conv_layer))

if __name__ == '__main__':
    from modelR.get_model_complexity import get_model_complexity_info
    #from torchstat import stat
    net = NPMMRDetR().cuda()
    #print(net)

    #for m in net.modules():
            #modules():
        #if 'Convolutional' in m:
            #print("aa",module_list[idx])

        #if isinstance(m, nn.BatchNorm2d):
            #print("aa",m)

    flops, params = get_model_complexity_info(net,(3, 544, 544), as_strings=False, print_per_layer_stat=True)
    print('GFlops: %.3fG' % (flops / 1e9))
    print('Params: %.2fM' % (params / 1e6))
    #stat(net.cuda(), (3, 544, 544))
    #
    #in_img = torch.randn(1, 3, 544, 544).cuda()
    #p, p_d = net(in_img)
    #print("Output Size of Each Head (Num_Classes: %d)" % cfg.DATA["NUM"])
    #for i in range(3):
        #print(p[i].shape)