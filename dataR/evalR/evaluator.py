import shutil
import time
from tqdm import tqdm

from dataloadR.augmentations import *
from evalR import voc_eval
from utils.utils_basic import *
from utils.visualize import *
from utils.heatmap import Show_Heatmap

current_milli_time = lambda: int(round(time.time() * 1000))

class Evaluator(object):
    def __init__(self, model, visiual=True):
        self.classes = cfg.DATA["CLASSES"]
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'predictionR')#预测结果的保存路径
        self.val_data_path = cfg.DATA_PATH
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.val_shape = cfg.TEST["TEST_IMG_SIZE"]
        self.__visiual = visiual
        self.__visual_imgs = cfg.TEST["NUM_VIS_IMG"]
        self.model = model
        self.device = next(model.parameters()).device
        self.inference_time = 0.
        self.showheatmap = cfg.SHOW_HEATMAP
        self.iouthresh_test = cfg.TEST["IOU_THRESHOLD"]

    def APs_voc(self, multi_test=False, flip_test=False):
        filename = cfg.TEST["EVAL_NAME"]+'.txt'
        img_inds_file = os.path.join(self.val_data_path, 'ImageSets', filename)
        with open(img_inds_file, 'r') as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]  # 读取文件名

        #if os.path.exists(self.pred_result_path):
            #shutil.rmtree(self.pred_result_path)
        #os.mkdir(self.pred_result_path)
        # for classname in self.classes:
        # with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + classname + '.txt'), 'a') as f:
        # f.close()
        #E:\NPMMR1\data\results
        #E:\NPMMR1\data\results
        #rewritepath_pic = os.path.join(self.pred_result_path, 'results')
        #if os.path.exists(rewritepath_pic):
            #shutil.rmtree(rewritepath_pic)
        #os.mkdir(rewritepath_pic)

        rewritepath = os.path.join(self.pred_result_path, 'voc')
        if os.path.exists(rewritepath):
            shutil.rmtree(rewritepath)
        os.mkdir(rewritepath)
        for img_ind in tqdm(img_inds):
            img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind + '.tif')  # 路径+JPEG+文件名############png
            # 目标： 直接改成读txt的文件名，每一行读取
            img = cv2.imread(img_path)
            bboxes_prd = self.get_bbox(img, multi_test, flip_test)

            '''
            if bboxes_prd.shape[0] != 0 and self.__visiual and self.__visual_imgs < 100:
                boxes = bboxes_prd[..., :4]
                boxes_rota = bboxes_prd[..., 4:8]

                scores = bboxes_prd[..., 8]

                class_inds = bboxes_prd[..., 9].astype(np.int32)

                visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
                path = os.path.join(cfg.PROJECT_PATH, "data/results/{}.jpg".format(self.__visual_imgs))
                cv2.imwrite(path, img)

                self.__visual_imgs += 1
            '''

            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                a_rota = np.array(bbox[4:8], dtype=np.float64)

                x1 = a_rota[0] * (coor[2]-coor[0]) + coor[0]
                y1 = coor[1]

                x2 = coor[2]
                y2 = a_rota[1] * (coor[3]-coor[1]) + coor[1]

                x3 = coor[2] - a_rota[2] * (coor[2]-coor[0])
                y3 = coor[3]

                x4 = coor[0]
                y4 = coor[3] - a_rota[3] * (coor[3]-coor[1])

                #coor_rota = np.array(bbox[4:8], dtype=np.float64)
                score = bbox[8]
                class_ind = int(bbox[9])
                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                #x1, y1, x2, y2, x3, y3, x4, y4 = map(str, coor_rota)
                #a1, a2, a3, a4 = map(str, a_rota)
                #print(a_rota)
                img_ind_out = img_ind + ".tif"
                s = ' '.join([img_ind, score, str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)),
                              str(int(x3)), str(int(y3)), str(int(x4)), str(int(y4))]) + '\n'
                s1 = ' '.join([img_ind_out, class_name, score, str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)),
                              str(int(x3)), str(int(y3)), str(int(x4)), str(int(y4))]) + '\n'

                with open(os.path.join(self.pred_result_path, 'voc', 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(s)  # 写用于voc测试的文件，pred路径
                with open(os.path.join(self.pred_result_path, 'voc', '科目四_九霄瞰.txt'), 'a') as f1:
                    f1.write(s1)  # 写用于voc测试的文件，pred路径
                color = np.zeros(3)
                points = np.array(
                    [[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]])
                if int(class_name) == 1:
                    # 25 black
                    color = (0, 0, 0)
                elif int(class_name) == 2:
                    # 1359 blue
                    color = (255, 0, 0)
                elif int(class_name) == 3:
                    # 639 Yellow
                    color = (0, 255, 255)
                elif int(class_name) == 4:
                    # 4371 red
                    color = (0, 0, 255)
                elif int(class_name) == 5:
                    # 3025 green
                    color = (0, 255, 0)
                cv2.polylines(img, [points], 1, color, 2)
                #print(points)
                #cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                # c1 左上角 c2 右下角

            store_path = os.path.join(cfg.PROJECT_PATH, 'dataR/results/', img_ind + '.tif')
            #print(store_path)
            cv2.imwrite(store_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



        self.inference_time = 1.0 * self.inference_time / len(img_inds)
        return self.__calc_APs(iou_thresh=self.iouthresh_test), self.inference_time

    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(cfg.TEST["MULTI_TEST_RANGE"][0], cfg.TEST["MULTI_TEST_RANGE"][1], cfg.TEST["MULTI_TEST_RANGE"][2])
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))

        ############glid_普通iou
        #print(bboxes.shape)
        #bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)
        #print(bboxes.shape)
        #bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)
        bboxes = nms_glid(bboxes, self.conf_thresh, self.nms_thresh)#################################

        return bboxes

    def __predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            start_time = current_milli_time()
            if self.showheatmap: _, p_d, beta = self.model(img)
            else: _, p_d = self.model(img)
            self.inference_time += (current_milli_time() - start_time)
        pred_bbox = p_d.squeeze().cpu().numpy()

        bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)

        if self.showheatmap and len(img):
            self.__show_heatmap(beta[2], org_img)
        return bboxes

    def __show_heatmap(self, beta, img):
        Show_Heatmap(beta, img)

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        预测框进行过滤，去除尺度不合理的框
        """
        #label: xywh(没有scale, 0 - 4), a1 - a4(4 - 8), r(8), conf(9), one_hot_smooth(10...)
        pred_coor = xywh2xyxy(pred_bbox[:, :4]) #xywh2xyxy

        pred_conf = pred_bbox[:, 9]
        #print(pred_bbox.shape,np.max(pred_conf))
        #print(pred_bbox[:, 10:].shape)
        pred_prob = pred_bbox[:, 10:]
        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
###########################
        #pred_x1 = 1.0*(pred_bbox[:, 4:5] * pred_bbox[:, 2:3] + pred_coor[:, 0:1]- dw) / resize_ratio
        #pred_y1 = 1.0*(pred_coor[:, 1:2]-dh) / resize_ratio

        #pred_x2 = 1.0*(pred_coor[:, 2:3]-dw) / resize_ratio
        #pred_y2 = 1.0*(pred_bbox[:, 5:6] * pred_bbox[:, 3:4] + pred_coor[:, 1:2] -dh) / resize_ratio

        #pred_x3 = 1.0*(pred_coor[:, 2:3] - pred_bbox[:, 6:7] * pred_bbox[:, 2:3] - dw) / resize_ratio
        #pred_y3 = 1.0*(pred_coor[:, 3:4]-dh) / resize_ratio

        #pred_x4 = 1.0*(pred_coor[:, 0:1]-dw) / resize_ratio
        #pred_y4 = 1.0*(pred_coor[:, 3:4] - pred_bbox[:, 7:8] * pred_bbox[:, 3:4]- dh) / resize_ratio
###################################
        #pred_rotaxy = np.concatenate([pred_x1, pred_y1, pred_x2, pred_y2, pred_x3, pred_y3, pred_x4, pred_y4], axis=-1)###########
        pred_rotaxy = pred_bbox[:, 4:8]
        pred_r = pred_bbox[:,8:9]
        zero = np.zeros_like(pred_rotaxy)
        pred_rotaxy = np.where(pred_r > 0.85, zero, pred_rotaxy)
        # (2)将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        ###################################

        #point1 = np.concatenate([np.minimum(np.maximum(pred_rotaxy[:, :2], [0, 0]), [org_w - 1, org_h - 1])], axis=-1)
        #point2 = np.concatenate([np.minimum(np.maximum(pred_rotaxy[:, 2:4], [0, 0]), [org_w - 1, org_h - 1])], axis=-1)
        #point3 = np.concatenate([np.minimum(np.maximum(pred_rotaxy[:, 4:6], [0, 0]), [org_w - 1, org_h - 1])], axis=-1)
        #point4 = np.concatenate([np.minimum(np.maximum(pred_rotaxy[:, 6:8], [0, 0]), [org_w - 1, org_h - 1])], axis=-1)
        #pred_coor_rota = np.concatenate([point1, point2, point3, point4], axis=-1)

        # (3)将无效bbox的coor置为0
        #flag0 = pred_bbox[:, 4]*0
        #flag1 = flag0 + 1

        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        #invalid_mask_rota1 = np.logical_or((pred_bbox[:, 4] > flag1), (pred_bbox[:, 4] < flag0))
        #invalid_mask_rota2 = np.logical_or((pred_bbox[:, 5] > flag1), (pred_bbox[:, 5] < flag0))
        #invalid_mask_rota3 = np.logical_or((pred_bbox[:, 6] > flag1), (pred_bbox[:, 6] < flag0))
        #invalid_mask_rota4 = np.logical_or((pred_bbox[:, 7] > flag1), (pred_bbox[:, 7] < flag0))

        #invalid_mask = np.logical_or(invalid_mask, invalid_mask_rota1)
        #invalid_mask = np.logical_or(invalid_mask, invalid_mask_rota2)
        #invalid_mask = np.logical_or(invalid_mask, invalid_mask_rota3)
        #invalid_mask = np.logical_or(invalid_mask, invalid_mask_rota4)

        pred_coor[invalid_mask] = 0
        pred_rotaxy[invalid_mask] = 0############################

        # (4)去掉不在有效范围内的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # (5)将score低于score_threshold的bbox去掉
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        #print(pred_conf)
##########################################################
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)


        coors = pred_coor[mask]
        coors_rota = pred_rotaxy[mask]
        #coors_rota = pred_coor_rota[mask]#######################

        scores = scores[mask]

        classes = classes[mask]

        bboxes = np.concatenate([coors, coors_rota, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)#######################
        return bboxes


    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        计算每个类别的ap值
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        filename = os.path.join(self.pred_result_path, 'voc', 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'voc', 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations/{:s}.txt')
        imagesetfile = os.path.join(self.val_data_path, 'ImageSets', cfg.TEST["EVAL_NAME"]+'.txt')
        APs = {}
        Recalls = {}
        Precisions = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)#调用voc_eval.py的函数进行计算
            APs[cls] = AP
            Recalls[cls] = R
            Precisions[cls] = P
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs
