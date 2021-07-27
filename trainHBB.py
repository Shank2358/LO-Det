import logging
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataloadR.datasets as data
import utils.gpu as gpu
from utils import cosine_lr_scheduler
from utils.log import Logger
from modelR.lodet_hbb import LODet
from modelR.loss.loss_hbb import Loss
from evalR.evaluator import *
from evalR.coco_eval import COCOEvaluator
from torch.cuda.amp import autocast as autocast

class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id):
        init_seeds(0)
        self.prune=0
        self.sr=True
        self.device = gpu.select_device(gpu_id)
        print(self.device)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train: print('Using multi scales training')
        else: print('train img size is {}'.format(cfg.TRAIN["TRAIN_IMG_SIZE"]))

        self.train_dataset = data.Construct_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=cfg.TRAIN["BATCH_SIZE"],
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           shuffle=True,
                                           pin_memory=True)

        net_model = LODet()
        if torch.cuda.device_count() >1: ## multi GPUs
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net_model = torch.nn.DataParallel(net_model)
            self.model = net_model.to(self.device)
        elif torch.cuda.device_count() ==1:
            self.model = net_model.to(self.device) ## Single GPU

        #self.optimizer = optim.SGD(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"], momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"])


        self.criterion = Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                              iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        if resume:
            self.__load_model_weights(weight_path)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                           T_max=self.epochs*len(self.train_dataloader),
                                                           lr_init=cfg.TRAIN["LR_INIT"],
                                                           lr_min=cfg.TRAIN["LR_END"],
                                                           warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader))




    def __load_model_weights(self, weight_path):
        last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
        chkpt = torch.load(last_weight, map_location=self.device)
        self.model.load_state_dict(chkpt['model'])#, False
        self.start_epoch = chkpt['epoch'] + 1
        if chkpt['optimizer'] is not None:
            self.optimizer.load_state_dict(chkpt['optimizer'])
            self.best_mAP = chkpt['best_mAP']
        del chkpt


    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight,_use_new_zipfile_serialization=False)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight,_use_new_zipfile_serialization=False)
        if epoch > 0 and epoch % 5 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt'%epoch))
            #
        del chkpt

    def __save_model_weights1(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best1.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last1.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight,_use_new_zipfile_serialization=False)

        torch.save(chkpt['model'], best_weight, _use_new_zipfile_serialization=False)
        torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt'%epoch))
            #
        del chkpt

    def train(self):
        global writer
        logger.info(self.model)
        logger.info(" Training start!  Img size:{:d},  Batchsize:{:d},  Number of workers:{:d}".format(
            cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["BATCH_SIZE"], cfg.TRAIN["NUMBER_WORKERS"]))
        logger.info(" Train datasets number is : {}".format(len(self.train_dataset)))

        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.model.train()

            '''
            ##################################################################################
            sr_flag = get_sr_flag(epoch, self.sr)
            if self.prune == 1:
                CBL_idx, _, prune_idx, shortcut_idx, _ = parse_module_defs2(self.model)  ############
                if self.sr:
                    print('shortcut sparse training')
            elif self.prune == 0:
                CBL_idx, _, prune_idx = parse_module_defs(self.model)  ############ model.cfg -> idx
                if self.sr:
                    print('normal sparse training ')
            print(prune_idx)#[1, 3, 7, 10, 14, 17, 20, 23, 26, 29, 32, 35, 39, 42, 45, 48, 51, 54, 57, 60, 64, 67, 70, 73, 76, 77, 78, 79, 80, 81, 88, 89, 90, 91, 92, 93, 100, 101, 102, 103, 104, 105]
            ###################################################################################
            '''

            mloss = torch.zeros(4)
            mAP = 0
            self.__save_model_weights1(epoch, mAP)
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox,
                    sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):

                self.scheduler.step(len(self.train_dataloader)*epoch + i)
                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)
                p, p_d = self.model(imgs)

                loss, loss_iou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                  label_lbbox, sbboxes, mbboxes, lbboxes)
                self.optimizer.zero_grad()


                loss.backward()
                self.optimizer.step()

                '''
                ########################
                idx2mask = None
                # if opt.sr and opt.prune==1 and epoch > opt.epochs * 0.5:
                # idx2mask = get_mask2(model, prune_idx, 0.85)
                ##self.model.module_list = self.model.module.module_list
                BNOptimizer.updateBN(sr_flag, self.model, 0.001, prune_idx, epoch, idx2mask)  ###########实际剪枝更新的部分
                ###################################################
                '''

                loss_items = torch.tensor([loss_iou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                if i % 50 == 0:
                    logger.info(
                        " Epoch:[{:3}/{}]  Batch:[{:3}/{}]  Img_size:[{:3}]  Loss:{:.4f}  "
                        "Loss_IoU:{:.4f} | Loss_Conf:{:.4f} | Loss_Cls:{:.4f}  LR:{:g}".format(
                            epoch, self.epochs, i, len(self.train_dataloader) - 1, self.train_dataset.img_size,
                            mloss[3], mloss[0], mloss[1], mloss[2], self.optimizer.param_groups[0]['lr']
                        ))
                    writer.add_scalar('loss_iou', mloss[0], len(self.train_dataloader)
                                      / (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_conf', mloss[1], len(self.train_dataloader)
                                      / (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_cls', mloss[2], len(self.train_dataloader)
                                      / (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('train_loss', mloss[3], len(self.train_dataloader)
                                      / (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)

                if self.multi_scale_train and (i+1) % 10 == 0:
                    self.train_dataset.img_size = random.choice(range(
                        cfg.TRAIN["MULTI_TRAIN_RANGE"][0], cfg.TRAIN["MULTI_TRAIN_RANGE"][1],
                        cfg.TRAIN["MULTI_TRAIN_RANGE"][2])) * 32


            if epoch >= 60 and epoch % 5 == 0 and cfg.TRAIN["EVAL_TYPE"] == 'VOC':
                logger.info("===== Validate =====".format(epoch, self.epochs))
                with torch.no_grad():
                    APs, inference_time = Evaluator(self.model).APs_voc()
                    for i in APs:
                        logger.info("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.train_dataset.num_classes
                    logger.info("mAP : {}".format(mAP))
                    logger.info("inference time: {:.2f} ms".format(inference_time))
                    writer.add_scalar('mAP', mAP, epoch)

            elif epoch >= 60 and epoch % 5 == 0 and cfg.TRAIN["EVAL_TYPE"] == 'COCO':
                logger.info("===== Validate =====".format(epoch, self.epochs))
                with torch.no_grad():
                    evaluator = COCOEvaluator(data_dir=cfg.DATA_PATH,
                                              img_size=cfg.TEST["TEST_IMG_SIZE"],
                                              confthre=cfg.TEST["CONF_THRESH"],
                                              nmsthre=cfg.TEST["NMS_THRESH"])
                    ap50_95, ap50, inference_time = evaluator.evaluate(self.model)
                    mAP = ap50
                    logger.info('ap50_95:{} | ap50:{}'.format(ap50_95, ap50))
                    logger.info("inference time: {:.2f} ms".format(inference_time))
                    writer.add_scalar('val/COCOAP50', ap50, epoch)
                    writer.add_scalar('val/COCOAP50_95', ap50_95, epoch)

            self.__save_model_weights(epoch, mAP)
            logger.info('Save weights Done')
            logger.info("mAP: {:.3f}".format(mAP))
            end = time.time()
            logger.info("Inference time: {:.4f}s".format(end - start))

        logger.info("Training finished.  Best_mAP: {:.3f}%".format(self.best_mAP))

if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/mobilenetv2_1.0-0c6065bc.pth',
                        help='weight file path') #default=None
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + '/event')
    logger = Logger(log_file_name=opt.log_path + '/log.txt', log_level=logging.DEBUG, logger_name='NPMMRDet').get_log()

    Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id).train()