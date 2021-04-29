# coding=utf-8

PROJECT_PATH = "./"
DATA_PATH = "/mnt/Datasets/DIOR/"


DATA = {"CLASSES":['airplane','airport','baseballfield','basketballcourt','bridge','chimney',
        'dam','Expressway-Service-area','Expressway-toll-station','golffield','groundtrackfield','harbor',
        'overpass','ship','stadium','storagetank','tenniscourt','trainstation','vehicle','windmill'],
        "NUM":20}
'''        
MODEL = {
        "ANCHORS":[[(1.494992296, 1.772419808), (2.550184278, 5.105188103), (4.511253175, 2.041398611)], # Anchors for small obj
                   [(3.852394468, 3.413543783), (3.827394513, 9.012606993), (7.569651633, 7.192874667)], # Anchors for medium obj
                   [(5.269568089, 8.068825014), (10.13079538, 3.44005408), (10.41848982, 10.60006263)]], # Anchors for big obj
        "STRIDES":[8, 16, 32],
        "ANCHORS_PER_SCLAE":3
        }#544
'''
MODEL = {
        "ANCHORS":[[(3.18524223, 1.57625129), (1.95394566,4.29178376), (6.65929852, 2.8841753)], # Anchors for small obj
                   [(1.9038, 4.42035), (6.712, 3.29255), (6.645, 12.7675)], # Anchors for medium obj
                   [(5.513875, 14.38123), (11.66746, 4.2333), (15.70345, 11.94367)]], # Anchors for big obj
        "STRIDES":[8, 16, 32],
        "ANCHORS_PER_SCLAE":3
        }#800

MAX_LABEL = 500
SHOW_HEATMAP = False
SCALE_FACTOR=2.0

TRAIN = {
         "EVAL_TYPE":'VOC', #['VOC', 'COCO']
         "TRAIN_IMG_SIZE":800,
         "TRAIN_IMG_NUM":11759,#11759,
         "AUGMENT":True,
         "MULTI_SCALE_TRAIN":True,
         "MULTI_TRAIN_RANGE":[12,25,1],
         "BATCH_SIZE":10,
         "IOU_THRESHOLD_LOSS":0.5,
         "EPOCHS":121,
         "NUMBER_WORKERS":16,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         "LR_INIT":1.5e-4,
         "LR_END":1e-6,
         "WARMUP_EPOCHS":5,
         "IOU_TYPE":'CIOU' #['GIOU','CIOU']
         }

TEST = {
        "EVAL_TYPE":'VOC', #['VOC', 'COCO', 'BOTH']
        "EVAL_JSON":'test.json',
        "EVAL_NAME":'test',
        "NUM_VIS_IMG":0,
        "TEST_IMG_SIZE":800,
        "BATCH_SIZE":1,
        "NUMBER_WORKERS":4,
        "CONF_THRESH":0.05,
        "NMS_THRESH":0.45,
        "NMS_METHODS":'NMS', #['NMS', 'SOFT_NMS', 'NMS_DIOU', #'NMS_DIOU_SCALE']
        "MULTI_SCALE_TEST":False,
        "MULTI_TEST_RANGE":[320,640,96],
        "FLIP_TEST":False
      }


'''
DOTA_cfg
DATA = {"CLASSES": ['plane',
                    'baseball-diamond',
                    'bridge',
                    'ground-track-field',
                    'small-vehicle',
                    'large-vehicle',
                    'ship',
                    'tennis-court',
                    'basketball-court',
                    'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'],
        "NUM": 15}

MODEL = {"ANCHORS":[[(1.625, 2.656), ( 3.652, 3.981), (4.493, 1.797)],
        [(4.358,3.123), (2.000, 4.558), (6.077, 6.688)],
        [(2.443, 7.848), (6.237, 4.750), (9.784, 10.291)]] ,# Anchors for big obj 608
"STRIDES":[8, 16, 32],
"ANCHORS_PER_SCLAE":3
}#544

MODEL = {"ANCHORS":[[(2.80340246, 2.87380792), (4.23121697, 6.44043634), (7.38428433, 3.82613533)],
        [(4.2460819, 4.349495965), (4.42917327, 10.59395029), (8.24772929, 6.224761455)],
        [(6.02687863, 5.92446062), (7.178407523, 10.86361071), (15.30253702, 12.62863728)]] ,# Anchors for big obj 608
"STRIDES":[8, 16, 32],
"ANCHORS_PER_SCLAE":3
}#800
'''
