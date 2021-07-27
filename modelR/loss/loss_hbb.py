import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils_basic
import config.cfg_lodet as cfg

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)
        return loss

class Loss(nn.Module):
    def __init__(self, anchors, strides, iou_threshold_loss=0.5):
        super(Loss, self).__init__()
        self.__iou_threshold_loss = iou_threshold_loss
        self.__strides = strides
        self.__scale_factor = cfg.SCALE_FACTOR

    def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes):
        strides = self.__strides
        loss_s, loss_s_iou, loss_s_conf, loss_s_cls = self.__cal_loss_per_layer(p[0], p_d[0], label_sbbox,
                                                               sbboxes, strides[0])
        loss_m, loss_m_iou, loss_m_conf, loss_m_cls = self.__cal_loss_per_layer(p[1], p_d[1], label_mbbox,
                                                               mbboxes, strides[1])
        loss_l, loss_l_iou, loss_l_conf, loss_l_cls = self.__cal_loss_per_layer(p[2], p_d[2], label_lbbox,
                                                               lbboxes, strides[2])
        loss = loss_l + loss_m + loss_s
        loss_iou = loss_s_iou + loss_m_iou + loss_l_iou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls
        return loss, loss_iou, loss_conf, loss_cls

    def smooth_l1_loss(self, input, target, beta=1. / 9, size_average=True):
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def __cal_loss_per_layer(self, p, p_d, label, bboxes, stride):
        batch_size, grid = p.shape[:2]
        img_size = stride * grid
        p_d_xywh = p_d[..., :4]
        p_d_a = p_d[..., 4:8]
        p_d_r = p_d[..., 8:9]
        p_conf = p[..., 9:10]
        p_cls = p[..., 10:]

        label_xywh = label[..., :4]
        label_a = label[..., 4:8]
        label_r = label[...,8:9]
        label_s13 = label[...,9:10]
        label_s24 = label[..., 10:11]
        label_obj_mask = label[..., 11:12]
        label_mix = label[..., 12:13]
        label_cls = label[..., 13:]

        if cfg.TRAIN["IOU_TYPE"] == 'GIOU':
            xiou = utils_basic.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        elif cfg.TRAIN["IOU_TYPE"] == 'CIOU':
            xiou = utils_basic.CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        bbox_loss_scale = self.__scale_factor - (self.__scale_factor-1.0) * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)
        loss_iou = label_obj_mask * bbox_loss_scale * (1.0 - xiou) * label_mix

        #loss r
        loss_r = label_obj_mask * self.smooth_l1_loss (p_d_r, label_r) * label_mix * 16
        a_sum = self.smooth_l1_loss(p_d_a, label_a)
        a_loss_scale = 1 + (self.__scale_factor_a -1)* (label_xywh[..., 2:3] * label_xywh[...,3:4] / (img_size ** 2))
        loss_a = label_obj_mask * a_sum * label_mix * a_loss_scale
        onesa = torch.ones_like(p_d_r)
        d13 = p_d_xywh[..., 2:3] * torch.abs(onesa - p_d_a[..., 0:1] - p_d_a[..., 2:3])
        s13 = p_d_xywh[..., 3:4] / torch.sqrt(torch.mul(d13, d13) + torch.mul(p_d_xywh[..., 3:4], p_d_xywh[..., 3:4]))
        d24 = p_d_xywh[..., 3:4] * torch.abs(onesa - p_d_a[..., 1:2] - p_d_a[..., 3:4])
        s24 = p_d_xywh[..., 2:3] / torch.sqrt(torch.mul(d24, d24) + torch.mul(p_d_xywh[..., 2:3], p_d_xywh[..., 2:3]))
        s1234sum = self.smooth_l1_loss(s13, label_s13)*(1.0/(label_s13+1e-8)) + self.smooth_l1_loss(s24, label_s24)*(1.0/(label_s24+1e-8))
        loss_s = label_obj_mask * s1234sum * label_mix

        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")
        iou = utils_basic.iou_xywh_torch(p_d_xywh.unsqueeze(4), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        iou_max = iou.max(-1, keepdim=True)[0]
        label_noobj_mask = (1.0 - label_obj_mask) * (iou_max < self.__iou_threshold_loss).float()

        loss_conf = (label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) +
                    label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)) * label_mix

        # loss classes
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        loss_cls = label_obj_mask * BCE(input=p_cls, target=label_cls) * label_mix

        loss_iou = (torch.sum(loss_iou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss_a = (torch.sum(loss_a)) / batch_size
        loss_r = (torch.sum(loss_r)) / batch_size
        loss_s = (torch.sum(loss_s)) / batch_size

        loss = loss_iou + (loss_a + loss_r) + loss_conf + loss_cls + loss_s

        return loss, loss_iou, loss_conf, loss_cls, loss_a, loss_r, loss_s
