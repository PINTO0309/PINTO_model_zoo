#!/usr/bin/env python

# ----------------------------------------------------------------
# Configurations for Training and Testing Process
# Written by Haiyang Jiang
# Mar 1st 2019
# ----------------------------------------------------------------

# file lists ================================================================
FILE_LIST = 'file_list'
VALID_LIST = 'valid_list'
TEST_LIST = 'test_list'
CUSOMIZED_LIST = 'customized_list'

# network.py ================================================================
DEBUG = False


# train.py ================================================================
EXP_NAME = '16_bit_HE_to_HE_gt'
CHECKPOINT_DIR = './1_checkpoint/' + EXP_NAME + '/'
RESULT_DIR = './2_result/' + EXP_NAME + '/'
LOGS_DIR = RESULT_DIR
TRAIN_LOG_DIR = 'train'
VAL_LOG_DIR = 'val'
# training settings
ALL_FRAME = 200
SAVE_FRAMES = list(range(0, ALL_FRAME, 32))
CROP_FRAME = 16
CROP_HEIGHT = 256
CROP_WIDTH = 256

SAVE_FREQ = 5
MAX_EPOCH = 50

FRAME_FREQ = 1
GROUP_NUM = 4

INIT_LR = 1e-4
DECAY_LR = 1e-5
DECAY_EPOCH = 30

# test.py ================================================================
# TEST_CROP_FRAME = 32
# TEST_CROP_HEIGHT = 512
# TEST_CROP_WIDTH= 512

TEST_CROP_FRAME = 16
TEST_CROP_HEIGHT = 256
TEST_CROP_WIDTH= 256

MAX_FRAME = 800

OVERLAP = 0.01
OUT_MAX = 255.0

