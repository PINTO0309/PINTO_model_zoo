# coding:utf-8

# yolo
TRAIN_INPUT_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
TEST_INPUT_SIZE = 544
STRIDES = [8, 16, 32]
IOU_LOSS_THRESH = 0.5

# train
BATCH_SIZE = 12
LEARN_RATE_INIT = 1e-4 * BATCH_SIZE / 6
LEARN_RATE_END = 1e-6
WARMUP_PERIODS = 2
MAX_PERIODS = int(1.0 * 50 * BATCH_SIZE / 6)

GT_PER_GRID = 3

# test
SCORE_THRESHOLD = 0.6 #0.01    # The threshold of the probability of the classes
IOU_THRESHOLD = 0.45     # The threshold of the IOU when implement NMS

# name and path
DATASET_PATH = '/media/b920405/Windows/datasets/VOC/test/VOCdevkit/VOC2007/ImageSets/Main/test.txt /VOC'
PROJECT_PATH = '/home/xzh/doc/code/python_code/Stronger-yolo/v3'
WEIGHTS_DIR = 'weights'
WEIGHTS_INIT = 'weights/mobilenet_v2_1.0_224.ckpt'
LOG_DIR = 'log'
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

