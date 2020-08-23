# coding: utf-8

import sys
import os
sys.path.append(os.path.abspath('..'))
import cv2
import random
import tensorflow as tf
import logging
import numpy as np
import config as cfg
import tools
import dataAug
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class Data(object):
    def __init__(self):
        """
        需始终记住：
        small_detector对应下标索引0， medium_detector对应下标索引1，big_detector对应下标索引2
        :param dataset_type: 选择加载训练样本或测试样本，必须是'train' or 'test'
        """
        self.__dataset_path = cfg.DATASET_PATH
        self.__train_input_sizes = cfg.TRAIN_INPUT_SIZES
        self.__strides = np.array(cfg.STRIDES)
        self.__batch_size = cfg.BATCH_SIZE
        self.__classes = cfg.CLASSES
        self.__num_classes = len(self.__classes)
        self.__gt_per_grid = cfg.GT_PER_GRID
        self.__class_to_ind = dict(zip(self.__classes, range(self.__num_classes)))

        annotations_2007 = self.__load_annotations(os.path.join(self.__dataset_path, '2007_trainval'))
        annotations_2012 = self.__load_annotations(os.path.join(self.__dataset_path, '2012_trainval'))
        self.__annotations = annotations_2007 + annotations_2012
        self.__num_samples = len(self.__annotations)
        logging.info(('The number of image for train is:').ljust(50) + str(self.__num_samples))
        self.__num_batchs = np.ceil(self.__num_samples / self.__batch_size)
        self.__batch_count = 0

    def batch_size_change(self, batch_size_new):
        self.__batch_size = batch_size_new
        self.__num_batchs = np.ceil(self.__num_samples / self.__batch_size)
        logging.info('Use the new batch size: %d' % self.__batch_size)

    def __load_annotations(self, data_path, use_difficult_bbox=False):
        """
        :param data_path: 数据集的路径，如'/home/xzh/doc/code/python_code/data/VOC/2012_trainval'
        :return: 一个标注列表，元素为：'image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...'
        """
        img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', 'trainval.txt')
        with file(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]

        annotations = []
        for image_ind in image_inds:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.jpg')
            annotation = image_path
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = self.__classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            annotations.append(annotation)
        return annotations

    def __iter__(self):
        return self

    def next(self):
        """
        使得pascal_voc对象变为可迭代对象
        :return: 每次迭代返回一个batch的图片、标签
        batch_image: shape为(batch_size, input_size, input_size, 3)
        batch_label_sbbox: shape为(batch_size, input_size / 8, input_size / 8, 6 + num_classes)
        batch_label_mbbox: shape为(batch_size, input_size / 16, input_size / 16, 6 + num_classes)
        batch_label_lbbox: shape为(batch_size, input_size / 32, input_size / 32, 6 + num_classes)
        batch_sbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        batch_mbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        batch_lbboxes: shape为(batch_size, max_bbox_per_scale, 4)
        """
        with tf.device('/cpu:0'):
            self.__train_input_size = random.choice(self.__train_input_sizes)
            self.__train_output_sizes = self.__train_input_size / self.__strides

            batch_image = np.zeros((self.__batch_size, self.__train_input_size, self.__train_input_size, 3))
            batch_label_sbbox = np.zeros((self.__batch_size, self.__train_output_sizes[0], self.__train_output_sizes[0],
                                          self.__gt_per_grid, 6 + self.__num_classes))
            batch_label_mbbox = np.zeros((self.__batch_size, self.__train_output_sizes[1], self.__train_output_sizes[1],
                                          self.__gt_per_grid, 6 + self.__num_classes))
            batch_label_lbbox = np.zeros((self.__batch_size, self.__train_output_sizes[2], self.__train_output_sizes[2],
                                          self.__gt_per_grid, 6 + self.__num_classes))
            temp_batch_sbboxes = []
            temp_batch_mbboxes = []
            temp_batch_lbboxes = []
            max_sbbox_per_img = 0
            max_mbbox_per_img = 0
            max_lbbox_per_img = 0
            num = 0
            if self.__batch_count < self.__num_batchs:
                while num < self.__batch_size:
                    index = self.__batch_count * self.__batch_size + num
                    if index >= self.__num_samples:
                        index -= self.__num_samples
                    annotation = self.__annotations[index]
                    image_org, bboxes_org = self.__parse_annotation(annotation)

                    # mixup
                    if random.random() < 0.5:
                        index_mix = random.randint(0, self.__num_samples - 1)
                        annotation_mix = self.__annotations[index_mix]
                        image_mix, bboxes_mix = self.__parse_annotation(annotation_mix)

                        lam = np.random.beta(1.5, 1.5)
                        image = lam * image_org + (1 - lam) * image_mix
                        bboxes_org = np.concatenate(
                            [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=-1)
                        bboxes_mix = np.concatenate(
                            [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=-1)
                        bboxes = np.concatenate([bboxes_org, bboxes_mix])
                    else:
                        image = image_org
                        bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=-1)

                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__create_label(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox

                    zeros = np.zeros((1, 4), dtype=np.float32)
                    sbboxes = sbboxes if len(sbboxes) != 0 else zeros
                    mbboxes = mbboxes if len(mbboxes) != 0 else zeros
                    lbboxes = lbboxes if len(lbboxes) != 0 else zeros
                    temp_batch_sbboxes.append(sbboxes)
                    temp_batch_mbboxes.append(mbboxes)
                    temp_batch_lbboxes.append(lbboxes)
                    max_sbbox_per_img = max(max_sbbox_per_img, len(sbboxes))
                    max_mbbox_per_img = max(max_mbbox_per_img, len(mbboxes))
                    max_lbbox_per_img = max(max_lbbox_per_img, len(lbboxes))
                    num += 1
                batch_sbboxes = np.array([np.concatenate([sbboxes, np.zeros((max_sbbox_per_img + 1 - len(sbboxes), 4), dtype=np.float32)], axis=0)
                                          for sbboxes in temp_batch_sbboxes])
                batch_mbboxes = np.array([np.concatenate([mbboxes, np.zeros((max_mbbox_per_img + 1 - len(mbboxes), 4), dtype=np.float32)], axis=0)
                                          for mbboxes in temp_batch_mbboxes])
                batch_lbboxes = np.array([np.concatenate([lbboxes, np.zeros((max_lbbox_per_img + 1 - len(lbboxes), 4), dtype=np.float32)], axis=0)
                                          for lbboxes in temp_batch_lbboxes])
                self.__batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.__batch_count = 0
                np.random.shuffle(self.__annotations)
                raise StopIteration

    def __parse_annotation(self, annotation):
        """
        读取annotation中image_path对应的图片，并将该图片进行resize(不改变图片的高宽比)
        获取annotation中所有的bbox，并将这些bbox的坐标(xmin, ymin, xmax, ymax)进行纠正，
        使得纠正后bbox在resize后的图片中的相对位置与纠正前bbox在resize前的图片中的相对位置相同
        :param annotation: 图片地址和bbox的坐标、类别，
        如：'image_path xmin,ymin,xmax,ymax,class_ind xmin,ymin,xmax,ymax,class_ind ...'
        :return: image和bboxes
        bboxes的shape为(N, 5)，其中N表示一站图中有N个bbox，5表示(xmin, ymin, xmax, ymax, class_ind)
        """
        line = annotation.split()
        image_path = line[0]
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([map(int, box.split(',')) for box in line[1:]])

        image, bboxes = dataAug.random_horizontal_flip(np.copy(image), np.copy(bboxes))
        image, bboxes = dataAug.random_crop(np.copy(image), np.copy(bboxes))
        image, bboxes = dataAug.random_translate(np.copy(image), np.copy(bboxes))
        image, bboxes = tools.img_preprocess2(np.copy(image), np.copy(bboxes),
                                              (self.__train_input_size, self.__train_input_size), True)
        return image, bboxes

    def __create_label(self, bboxes):
        """
        :param bboxes: 一张图对应的所有bbox和每个bbox所属的类别，以及mixup的权重，
        bbox的坐标为(xmin, ymin, xmax, ymax, class_ind, mixup_weight)
        :return:
        label_sbbox: shape为(input_size / 8, input_size / 8, anchor_per_scale, 6 + num_classes)
        label_mbbox: shape为(input_size / 16, input_size / 16, anchor_per_scale, 6 + num_classes)
        label_lbbox: shape为(input_size / 32, input_size / 32, anchor_per_scale, 6 + num_classes)
        只要某个GT落入grid中，那么这个grid就负责预测它，最多负责预测gt_per_grid个GT，
        那么该grid中对应位置的数据为(xmin, ymin, xmax, ymax, 1, classes, mixup_weights),
        其他grid对应位置的数据都为(0, 0, 0, 0, 0, 0..., 1)
        sbboxes：shape为(max_bbox_per_scale, 4)
        mbboxes：shape为(max_bbox_per_scale, 4)
        lbboxes：shape为(max_bbox_per_scale, 4)
        存储的坐标为(xmin, ymin, xmax, ymax)，大小都是bbox纠正后的原始大小
        """
        label = [np.zeros((self.__train_output_sizes[i], self.__train_output_sizes[i],
                           self.__gt_per_grid, 6 + self.__num_classes)) for i in range(3)]
        # mixup weight位默认为1.0
        for i in range(3):
            label[i][:, :, :, -1] = 1.0
        bboxes_coor = [[] for _ in range(3)]
        bboxes_count = [np.zeros((self.__train_output_sizes[i], self.__train_output_sizes[i])) for i in range(3)]

        for bbox in bboxes:
            # (1)获取bbox在原图上的顶点坐标、类别索引、mix up权重、中心坐标、高宽、尺度
            bbox_coor = bbox[:4]
            bbox_class_ind = int(bbox[4])
            bbox_mixw = bbox[5]
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
                                        bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_scale = np.sqrt(np.multiply.reduce(bbox_xywh[2:]))

            # label smooth
            onehot = np.zeros(self.__num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.__num_classes, 1.0 / self.__num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            if bbox_scale <= 30:
                match_branch = 0
            elif (30 < bbox_scale) and (bbox_scale <= 90):
                match_branch = 1
            else:
                match_branch = 2

            xind, yind = np.floor(1.0 * bbox_xywh[:2] / self.__strides[match_branch]).astype(np.int32)
            gt_count = int(bboxes_count[match_branch][yind, xind])
            if gt_count < self.__gt_per_grid:
                if gt_count == 0:
                    gt_count = slice(None)
                bbox_label = np.concatenate([bbox_coor, [1.0], smooth_onehot, [bbox_mixw]], axis=-1)
                label[match_branch][yind, xind, gt_count, :] = 0
                label[match_branch][yind, xind, gt_count, :] = bbox_label
                bboxes_count[match_branch][yind, xind] += 1
                bboxes_coor[match_branch].append(bbox_coor)
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_coor
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.__num_batchs


if __name__ == '__main__':
    data_obj = Data()
    for batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes in data_obj:
        print batch_image.shape
        print batch_label_sbbox.shape
        print batch_label_mbbox.shape
        print batch_label_lbbox.shape
        print batch_sbboxes.shape
        print batch_mbboxes.shape
        print batch_lbboxes.shape