# coding=utf-8
import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset

import config.cfg_npmmr as cfg
import dataloadR.augmentations as DataAug
import utils.utils_basic as tools
from PIL import Image
import matplotlib.pyplot as plt
class Construct_Dataset(Dataset):
    def __init__(self, anno_file_type, img_size=448):
        self.img_size = img_size  # For Multi-training
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file_type)

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):

        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.__annotations) - 1)
        img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = DataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        #####bboxes xyxy
        del img_org, bboxes_org, img_mix, bboxes_mix
        #print(self.__annotations[item])
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(bboxes)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __load_annotations(self, anno_type):
        assert anno_type in ['train', 'val', 'test']
        anno_path = os.path.join(cfg.PROJECT_PATH, 'dataR', anno_type + ".txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))

        assert len(annotations) > 0, "No images found in {}".format(anno_path)
        return annotations

    def __parse_annotation(self, annotation):
        anno = annotation.strip().split(' ')

        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]]) ####xmin,ymin,xmax,ymax,c,x1,y1,x2,y2,x3,y3,x4,y4,r
        #############
        img, bboxes = DataAug.RandomVerticalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.HSV()(np.copy(img), np.copy(bboxes))

        #img, bboxes = DataAug.Blur()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Grayscale()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Gamma()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Noise()(np.copy(img), np.copy(bboxes))
        # img, bboxes = DataAug.Sharpen()(np.copy(img), np.copy(bboxes))
        # img, bboxes = DataAug.Contrast()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))

        return img, bboxes

    def __creat_label(self, bboxes):
        anchors = np.array(cfg.MODEL["ANCHORS"])
        strides = np.array(cfg.MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]

        label = [np.zeros((int(train_output_size[i]), int(train_output_size[i]), anchors_per_scale, 6 + 5 + 2+ self.num_classes)) for i in range(3)]####a r
        for i in range(3):
            label[i][..., 5+5+2] = 1.0 
        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]  # Darknet the max_num is 30
        bbox_count = np.zeros((3,))
        for bbox in bboxes:
            bbox_coor = bbox[:4] 
            bbox_class_ind = int(bbox[4])#######################从1开始的话要-1
            bbox_coor_in = bbox[5:13]
            bbox_r = bbox[13]
            bbox_mix = bbox[14]
            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0
            one_hot_smooth = DataAug.LabelSmooth()(one_hot, self.num_classes)

            # convert "xyxy" to "xywh"
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]],axis=-1)

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            # convert x1-y4 to a1-a4
            a1 = (bbox_coor_in[0]-bbox_coor[0])/(bbox_coor[2]-bbox_coor[0])
            a2 = (bbox_coor_in[3]-bbox_coor[1])/(bbox_coor[3]-bbox_coor[1])
            a3 = (bbox_coor[2]-bbox_coor_in[4])/(bbox_coor[2]-bbox_coor[0])
            a4 = (bbox_coor[3]-bbox_coor_in[7])/(bbox_coor[3]-bbox_coor[1])
            bbox_a = np.concatenate([[a1],[a2],[a3],[a4]],axis=-1)
            s13 = np.array(bbox_xywh[3]/np.sqrt((bbox_coor_in[0] - bbox_coor_in[4])**2 + (bbox_coor_in[1] - bbox_coor_in[5])**2))
            s24 = np.array(bbox_xywh[2]/np.sqrt((bbox_coor_in[2] - bbox_coor_in[6])**2 + (bbox_coor_in[3] - bbox_coor_in[7])**2))

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((anchors_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:8] = bbox_a
                    label[i][yind, xind, iou_mask, 8:9] = bbox_r

                    label[i][yind, xind, iou_mask, 9:10] = s13
                    label[i][yind, xind, iou_mask, 10:11] = s24


                    label[i][yind, xind, iou_mask, 11:12] = 1.0
                    label[i][yind, xind, iou_mask, 12:13] = bbox_mix
                    label[i][yind, xind, iou_mask, 13:] = one_hot_smooth
                    bbox_ind = int(bbox_count[i] % 150)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchors_per_scale)
                best_anchor = int(best_anchor_ind % anchors_per_scale)

                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                label[best_detect][yind, xind, best_anchor, 0:4]  = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:8]  = bbox_a
                label[best_detect][yind, xind, best_anchor, 8:9]  = bbox_r


                label[best_detect][yind, xind, best_anchor, 9:10] = s13
                label[best_detect][yind, xind, best_anchor, 10:11] = s24

                label[best_detect][yind, xind, best_anchor, 11:12]  = 1.0
                label[best_detect][yind, xind, best_anchor, 12:13]  = bbox_mix
                label[best_detect][yind, xind, best_anchor, 13:]  = one_hot_smooth

                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_dataset=Construct_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
    train_dataloader = DataLoader(train_dataset,batch_size=1, num_workers=cfg.TRAIN["NUMBER_WORKERS"],shuffle=False)
    for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(train_dataloader):
        continue