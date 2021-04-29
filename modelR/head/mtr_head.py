import torch.nn as nn
import torch
import torch.nn.functional as F
#from dcn_v2 import DCNv2
from modelR.layers.deform_conv_v2 import DeformConv2d, DeformConv2d_offset


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class MTR_Head1(nn.Module):
    def __init__(self, filters_in, anchor_num, fo_class, temp=False):
        super(MTR_Head1, self).__init__()
        self.fo_class = fo_class
        self.anchor_num = anchor_num
        self.temp = temp

        self.__conv_conf = nn.Conv2d(in_channels=filters_in, out_channels=self.anchor_num * 2, kernel_size=1, stride=1,
                                     padding=0)###############conf 和 r

        # self.__conv_offset_mask1 = Convolutional(filters_in, self.anchor_num*4, kernel_size=1, stride=1, pad=0)
        self.__conv_offset_mask = nn.Conv2d(in_channels=filters_in, out_channels=3 * 9, kernel_size=1, stride=1,
                                            padding=0, bias=True)

        self.__dconv_loc = DeformConv2d_offset(inc=filters_in, outc=filters_in, kernel_size=3, padding=1, stride=1, bias=None)
        #DCNv2(filters_in, filters_in, kernel_size=3, stride=1, padding=1)

        self.__bnloc = nn.BatchNorm2d(filters_in)
        self.__reluloc = nn.LeakyReLU(inplace=True)
        self.__dconv_locx = nn.Conv2d(filters_in, self.anchor_num * 8, kernel_size=1, stride=1, padding=0)

        self.__dconv_cla = DeformConv2d_offset(inc=filters_in, outc=filters_in, kernel_size=3, padding=1, stride=1, bias=None)
        #DCNv2(filters_in, filters_in, kernel_size=3, stride=1, padding=1)
        self.__bncla = nn.BatchNorm2d(filters_in)
        self.__relucla = nn.LeakyReLU(inplace=True)
        self.__dconv_clax = nn.Conv2d(filters_in, self.anchor_num * self.fo_class, kernel_size=1, stride=1, padding=0)

        self.init_offset()

    def init_offset(self):
        self.__conv_offset_mask.weight.data.zero_()
        self.__conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out_conf = self.__conv_conf(x)

        out_offset_mask = self.__conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out_offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # print(offset.shape)
        # if self.temp == True:
        # mask = torch.sigmoid(mask*edge)
        # else:
        #print(offset.shape, mask.shape)
        out_loc = self.__dconv_locx(self.__reluloc(self.__bnloc(self.__dconv_loc(x, offset, mask))))
        out_cla = self.__dconv_clax(self.__relucla(self.__bncla(self.__dconv_cla(x, offset, mask))))

        out_loc1 = out_loc.view(x.shape[0], self.anchor_num, 8, x.shape[2], x.shape[3]).cuda()
        out_conf1 = out_conf.view(x.shape[0], self.anchor_num, 2, x.shape[2], x.shape[3]).cuda()#######
        out_cla1 = out_cla.view(x.shape[0], self.anchor_num, self.fo_class, x.shape[2], x.shape[3]).cuda()
        out = torch.cat((out_loc1, out_conf1, out_cla1), 2).cuda()
        return out

class MTR_Head2(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(MTR_Head2, self).__init__()
        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        p = p.permute(0, 3, 4, 1, 2)
        #print(p.shape)
        p_de = self.__decode(p.clone())
        return (p, p_de)
    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_a = p[:, :, :, :, 4:8]
        conv_raw_r = p[:, :, :, :, 8:9]
        conv_raw_conf = p[:, :, :, :, 9:10]
        conv_raw_prob = p[:, :, :, :, 10:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)
        #pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_xy = (torch.sigmoid(conv_raw_dxdy)*1.05 - ((1.05-1)/2) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        #pred_a = torch.sigmoid(conv_raw_a)
        #pred_r = torch.sigmoid(conv_raw_r)
        #pred_a = F.relu6(conv_raw_a + 3, inplace=True)/6
        pred_a = (torch.clamp(torch.sigmoid(conv_raw_a), 0.011, 1) - 0.01) / (1 - 0.01)
        pred_r = F.relu6(conv_raw_r + 3, inplace=True)/6


        maskr = pred_r
        zero = torch.zeros_like(maskr)
        one = torch.ones_like(maskr)
        maskr = torch.where(maskr > 0.8, zero, one)
        pred_a[:, :, :, :, 0:1] = pred_a[:, :, :, :, 0:1]*maskr
        pred_a[:, :, :, :, 1:2] = pred_a[:, :, :, :, 1:2] * maskr
        pred_a[:, :, :, :, 2:3] = pred_a[:, :, :, :, 2:3] * maskr
        pred_a[:, :, :, :, 3:4] = pred_a[:, :, :, :, 3:4] * maskr

        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_a, pred_r, pred_conf, pred_prob], dim=-1)
        return pred_bbox.view(-1, 5 + 5 + self.__nC) if not self.training else pred_bbox
