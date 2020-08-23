# coding: utf-8

import numpy as np
import config as cfg
import cv2
import os
import tensorflow as tf
from model.head.yolov3 import YOLOV3
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import argparse
from utils import tools
from eval.evaluator import Evaluator


class Yolo_test(Evaluator):
    def __init__(self, test_weight):
        log_dir = os.path.join(cfg.LOG_DIR, 'test')
        test_weight_path = os.path.join(cfg.WEIGHTS_DIR, test_weight)

        with tf.name_scope('input'):
            input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            training = tf.placeholder(dtype=tf.bool, name='training')
        _, _, _, pred_sbbox, pred_mbbox, pred_lbbox = YOLOV3(training).build_nework(input_data)
        with tf.name_scope('summary'):
            tf.summary.FileWriter(log_dir).add_graph(tf.get_default_graph())
        self.__sess = tf.Session()
        net_vars = tf.get_collection('YoloV3')
        saver = tf.train.Saver(net_vars)
        saver.restore(self.__sess, test_weight_path)
        super(Yolo_test, self).__init__(self.__sess, input_data, training, pred_sbbox, pred_mbbox, pred_lbbox)
        print("input_data.name=", input_data.name)
        print("pred_sbbox=", pred_sbbox.name)
        print("pred_mbbox=", pred_mbbox.name)
        print("pred_lbbox=", pred_lbbox.name)

    def detect_image(self, image):
        original_image = np.copy(image)
        bboxes = self.get_bbox(image)
        image = tools.draw_bbox(original_image, bboxes, self._classes)
        self.__sess.close()
        return image

    def test(self, year=2007, multi_test=False, flip_test=False):
        APs, ave_times = self.APs_voc(year, multi_test, flip_test)
        APs_file = os.path.join(self._project_path, 'eval', 'APs.txt')
        with file(APs_file, 'w') as f:
            for cls in APs:
                AP_mess = 'AP for %s = %.4f\n' % (cls, APs[cls])
                print(AP_mess.strip())
                f.write(AP_mess)
            mAP = np.mean([APs[cls] for cls in APs])
            mAP_mess = 'mAP = %.4f\n' % mAP
            print(mAP_mess.strip())
            f.write(mAP_mess)
        for key in ave_times:
            print('Average time for %s :\t%.2f ms' % (key, ave_times[key]))
        self.__sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test model')
    parser.add_argument('--test_weight', help='name of test weights file', default='', type=str)
    parser.add_argument('--gpu', help='select a gpu for test', default='0', type=str)
    parser.add_argument('-mt', help='multi scale test', dest='mt', action='store_true', default=False)
    parser.add_argument('-ft', help='flip test', dest='ft', action='store_true', default=False)
    parser.add_argument('-t07', help='test voc 2007', dest='t07', action='store_true', default=False)
    parser.add_argument('-t12', help='test voc 2012', dest='t12', action='store_true', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    T = Yolo_test(args.test_weight)
    if args.t07:
        T.test(2007, args.mt, args.ft)
    elif args.t12:
        T.test(2012, args.mt, args.ft)
    else:
        #test_set_path = os.path.join(cfg.DATASET_PATH, '%d_test' % 2007)
        test_set_path = '/media/xxxx/Windows/datasets/VOC/test/VOCdevkit/VOC2007'
        #img_inds_file = os.path.join(test_set_path, 'ImageSets', 'Main', 'test.txt')
        img_inds_file = '/media/xxxx/Windows/datasets/VOC/test/VOCdevkit/VOC2007/ImageSets/Main/test.txt'
        with file(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]
        image_ind = np.random.choice(image_inds)
        image_path = os.path.join(test_set_path, 'JPEGImages', image_ind + '.jpg')
        image = cv2.imread(image_path)
        image = T.detect_image(image)
        cv2.imwrite('detect_result.jpg', image)


